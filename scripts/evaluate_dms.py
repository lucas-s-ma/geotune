"""
Evaluation script for trained GeoTune models on Protein Gym DMS substitution benchmarks.

This script evaluates a trained model on all DMS substitution files in a directory,
computing Spearman correlation between predicted and experimental scores for each file,
then reporting the average Spearman correlation across all files.

Usage:
    # With trained LoRA model
    python scripts/evaluate_dms.py --model_path outputs/run_20260302_143521/best_model/lora_adapters --dms_dir DMS_ProteinGym_substitutions
    
    # With raw base model (no LoRA)
    python scripts/evaluate_dms.py --dms_dir DMS_ProteinGym_substitutions --model_name facebook/esm2_t30_150M_UR50D
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
import pickle

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.geotune_esm_model import load_esm_with_lora, GeoTuneESMModel
from transformers import EsmModel, EsmConfig, EsmTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained GeoTune model on DMS substitution benchmarks")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to trained model (LoRA adapters directory). If not provided, uses raw base model.")
    parser.add_argument("--dms_dir", type=str, required=True,
                       help="Path to directory containing DMS substitution CSV files")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D",
                       help="Base model name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Path to save results (default: auto-generated based on model)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (default: cuda if available)")
    parser.add_argument("--max_sequences", type=int, default=None,
                       help="Maximum number of sequences to evaluate per DMS file (for debugging)")

    return parser.parse_args()


def get_masked_sequence_scores(model, sequences, tokenizer, device, batch_size=8):
    """
    Get model scores for sequences by computing masked marginal likelihood.
    
    For each position in the sequence, we mask it and compute the log probability
    of the actual amino acid at that position. The score is the average log probability
    across all positions.
    
    Args:
        model: The trained model
        sequences: List of protein sequences
        tokenizer: ESM tokenizer
        device: torch device
        batch_size: batch size for processing
    
    Returns:
        List of scores (one per sequence)
    """
    model.eval()
    all_scores = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Computing scores"):
            batch_seqs = sequences[i:i + batch_size]
            
            # Tokenize sequences
            batch_tokens = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
            input_ids = batch_tokens['input_ids'].to(device)
            attention_mask = batch_tokens['attention_mask'].to(device)
            
            # Get model outputs
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs['sequence_output']
            
            # Get logits from LM head
            logits = model.lm_head(sequence_output)  # (batch, seq_len, vocab_size)
            
            # Compute log probabilities for each position
            log_probs = torch.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)
            
            # Get the log probability of the actual amino acid at each position
            # Shift input_ids to align with output positions (accounting for BOS token)
            batch_scores = []
            for b in range(input_ids.shape[0]):
                seq_len = attention_mask[b].sum().item()
                # Skip BOS token (position 0) and only score actual sequence positions
                positions_score = []
                for pos in range(1, seq_len - 1):  # Exclude BOS and EOS
                    actual_aa = input_ids[b, pos].item()
                    log_prob = log_probs[b, pos, actual_aa].item()
                    positions_score.append(log_prob)
                
                if positions_score:
                    avg_score = np.mean(positions_score)
                    batch_scores.append(avg_score)
                else:
                    batch_scores.append(0.0)
            
            all_scores.extend(batch_scores)
    
    return all_scores


def get_masked marginal log_probs(model, sequence, tokenizer, device, batch_size=16):
    """
    Compute masked marginal log probabilities for ALL positions in a sequence efficiently.
    
    This is the key optimization: instead of one forward pass per mutation,
    we do one forward pass per masked position in the sequence, and get
    log probabilities for ALL 20 amino acids at once.
    
    For a sequence of length L, we create L masked versions (each with one position masked),
    batch them together, and get all log probs in a single batched forward pass.
    
    Args:
        model: The trained model
        sequence: Protein sequence string
        tokenizer: ESM tokenizer
        device: torch device
        batch_size: batch size for processing masked positions
        
    Returns:
        log_probs_all: numpy array of shape (seq_len, vocab_size) with log probs for each position
    """
    model.eval()
    seq_len = len(sequence)
    vocab_size = 33  # ESM-2 vocabulary size (including special tokens)
    
    # Storage for log probabilities at each position
    log_probs_all = np.full((seq_len, vocab_size), np.nan)
    
    with torch.no_grad():
        # Process masked positions in batches
        for batch_start in range(0, seq_len, batch_size):
            batch_end = min(batch_start + batch_size, seq_len)
            masked_seqs = []
            masked_positions = []
            
            # Create masked versions of the sequence for each position in this batch
            for pos in range(batch_start, batch_end):
                masked_seq = sequence[:pos] + '<mask>' + sequence[pos+1:]
                masked_seqs.append(masked_seq)
                masked_positions.append(pos)
            
            # Tokenize all masked sequences together
            tokens = tokenizer(masked_seqs, return_tensors="pt", padding=True)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            
            # Single forward pass for all masked positions in this batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = model.lm_head(outputs['sequence_output'])
            log_probs = torch.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)
            
            # Extract log probs for each masked position
            for batch_idx, pos in enumerate(masked_positions):
                # Find the mask token position in the tokenized output
                # The mask should be at position pos + 1 (accounting for BOS token)
                token_pos = pos + 1
                if token_pos < log_probs.shape[1]:
                    log_probs_all[pos] = log_probs[batch_idx, token_pos].cpu().numpy()
    
    return log_probs_all


def compute_mutational_effect_fast(log_probs_all, wt_seq, mut_seq, tokenizer):
    """
    Compute mutational effect from pre-computed log probabilities.
    
    Args:
        log_probs_all: numpy array of shape (seq_len, vocab_size) from get_masked_marginal_log_probs
        wt_seq: Wild type protein sequence
        mut_seq: Mutated protein sequence  
        tokenizer: ESM tokenizer
        
    Returns:
        Mutational effect score (higher = more beneficial)
    """
    # Find the mutation position
    if len(wt_seq) != len(mut_seq):
        return None
        
    diff_positions = [(i, wt_seq[i], mut_seq[i]) 
                      for i in range(len(wt_seq)) 
                      if wt_seq[i] != mut_seq[i]]
    
    if len(diff_positions) == 0:
        return 0.0
    
    pos, wt_aa, mut_aa = diff_positions[0]
    
    # Get token IDs
    aa_to_token = {
        'A': 6, 'R': 12, 'N': 15, 'D': 13, 'C': 29, 'Q': 23, 'E': 17, 'G': 20,
        'H': 24, 'I': 25, 'L': 18, 'K': 19, 'M': 21, 'F': 26, 'P': 27, 'S': 28,
        'T': 30, 'W': 31, 'Y': 32, 'V': 22
    }
    
    wt_token_id = aa_to_token.get(wt_aa)
    mut_token_id = aa_to_token.get(mut_aa)
    
    if wt_token_id is None or mut_token_id is None:
        return None
    
    if pos >= log_probs_all.shape[0]:
        return None
    
    # Check if we have valid log probs for this position
    if np.any(np.isnan(log_probs_all[pos])):
        return None
    
    # Mutational effect: log P(mutant) - log P(wild-type)
    log_prob_wt = log_probs_all[pos, wt_token_id]
    log_prob_mut = log_probs_all[pos, mut_token_id]
    
    return log_prob_mut - log_prob_wt


def compute_mutational_effect(model, wild_type_seq, mutant_seq, tokenizer, device):
    """
    Compute the mutational effect score using Masked Marginal Likelihood (MML).
    
    DEPRECATED: This function is kept for backwards compatibility but is inefficient.
    Use get_masked_marginal_log_probs + compute_mutational_effect_fast instead.
    
    For each mutation:
    1. Mask the mutated position in the wild-type sequence
    2. Compute log P(mutant | masked context) - log P(wild-type | masked context)
    """
    model.eval()
    
    # Find the mutation position
    if len(wild_type_seq) != len(mutant_seq):
        return None
    
    diff_positions = [(i, wild_type_seq[i], mutant_seq[i])
                      for i in range(len(wild_type_seq))
                      if wild_type_seq[i] != mutant_seq[i]]
    
    if len(diff_positions) == 0:
        return 0.0
    
    if len(diff_positions) > 1:
        # Multiple mutations - use first one for now
        pass
    
    pos, wt_aa, mut_aa = diff_positions[0]
    
    # Create a masked version of the wild-type sequence
    masked_seq = wild_type_seq[:pos] + '<mask>' + wild_type_seq[pos+1:]
    
    # Tokenize masked sequence
    masked_tokens = tokenizer(masked_seq, return_tensors="pt")
    masked_input_ids = masked_tokens['input_ids'].to(device)
    masked_attention_mask = masked_tokens['attention_mask'].to(device)
    
    # Get model outputs for masked sequence
    with torch.no_grad():
        outputs = model(input_ids=masked_input_ids, attention_mask=masked_attention_mask)
        logits = model.lm_head(outputs['sequence_output'])
        log_probs = torch.log_softmax(logits, dim=-1)[0]
        
        # Get token IDs for amino acids
        try:
            wt_token_id = tokenizer.encode(wt_aa, add_special_tokens=False)[0]
            mut_token_id = tokenizer.encode(mut_aa, add_special_tokens=False)[0]
        except (IndexError, ValueError):
            aa_to_token = {
                'A': 6, 'R': 12, 'N': 15, 'D': 13, 'C': 29, 'Q': 23, 'E': 17, 'G': 20,
                'H': 24, 'I': 25, 'L': 18, 'K': 19, 'M': 21, 'F': 26, 'P': 27, 'S': 28,
                'T': 30, 'W': 31, 'Y': 32, 'V': 22
            }
            wt_token_id = aa_to_token.get(wt_aa, None)
            mut_token_id = aa_to_token.get(mut_aa, None)
        
        if wt_token_id is None or mut_token_id is None:
            return None
        
        token_pos = pos + 1
        
        if token_pos >= log_probs.shape[0]:
            return None
        
        log_prob_wt = log_probs[token_pos, wt_token_id].item()
        log_prob_mut = log_probs[token_pos, mut_token_id].item()
        
        return log_prob_mut - log_prob_wt


def evaluate_dms_file(model, dms_file, tokenizer, device, max_sequences=None):
    """
    Evaluate model on a single DMS file.
    
    OPTIMIZED VERSION: Uses a single set of forward passes per wild-type sequence.
    All mutants sharing the same wild-type sequence are evaluated from the same
    pre-computed masked marginal log probabilities.

    Args:
        model: The trained model
        dms_file: Path to DMS CSV file
        tokenizer: ESM tokenizer
        device: torch device
        max_sequences: Maximum number of sequences to evaluate

    Returns:
        Dictionary with results
    """
    # Load DMS data
    df = pd.read_csv(dms_file)

    # Extract mutant info and group by wild-type sequence
    mutants_by_wt_seq = {}
    scores = []
    score_bins = []

    for idx, row in df.iterrows():
        if max_sequences and len(scores) >= max_sequences:
            break

        mutant_str = row['mutant']
        mutated_seq = row['mutated_sequence']
        dms_score = row['DMS_score']
        dms_score_bin = row['DMS_score_bin']

        # Parse mutant string: {WT_AA}{POSITION}{MUT_AA}
        import re
        match = re.match(r'([A-Z])(\d+)([A-Z*])', mutant_str)
        if match:
            wt_aa, pos_str, mut_aa = match.groups()
            pos = int(pos_str) - 1

            # Reconstruct wild type sequence
            if pos < len(mutated_seq) and mutated_seq[pos] == mut_aa:
                wt_seq = mutated_seq[:pos] + wt_aa + mutated_seq[pos+1:]
            else:
                continue

            mut_info = {
                'mutant': mutant_str,
                'wt_seq': wt_seq,
                'mut_seq': mutated_seq,
                'pos': pos,
                'wt_aa': wt_aa,
                'mut_aa': mut_aa
            }
            
            # Group mutants by wild-type sequence
            if wt_seq not in mutants_by_wt_seq:
                mutants_by_wt_seq[wt_seq] = []
            mutants_by_wt_seq[wt_seq].append(mut_info)
            
            scores.append(dms_score)
            score_bins.append(dms_score_bin)

    if len(mutants_by_wt_seq) == 0:
        return {
            'file': os.path.basename(dms_file),
            'spearman': None,
            'num_mutants': 0,
            'error': 'No valid mutants found'
        }

    # Compute mutational effects using optimized approach
    print(f"Computing mutational effects for {len(scores)} mutants across {len(mutants_by_wt_seq)} wild-type sequences...")
    
    mut_effects = {}  # Map from (wt_seq, mutant_str) to effect
    all_mutants_flat = []  # Flat list for tracking order
    
    for wt_seq, mutants_list in tqdm(mutants_by_wt_seq.items(), desc="Processing wild-type sequences"):
        # KEY OPTIMIZATION: One set of forward passes per wild-type sequence
        # Get masked marginal log probs for ALL positions in this wild-type sequence
        log_probs_all = get_masked_marginal_log_probs(model, wt_seq, tokenizer, device, batch_size=16)
        
        # Compute effects for all mutants with this wild-type sequence
        for mut_info in mutants_list:
            effect = compute_mutational_effect_fast(
                log_probs_all,
                wt_seq,
                mut_info['mut_seq'],
                tokenizer
            )
            mut_effects[(wt_seq, mut_info['mutant'])] = effect
            all_mutants_flat.append(mut_info)

    # Collect predicted effects in the same order as scores
    predicted_effects = []
    for mut_info in all_mutants_flat:
        effect = mut_effects.get((mut_info['wt_seq'], mut_info['mutant']))
        if effect is not None:
            predicted_effects.append(effect)
        else:
            predicted_effects.append(np.nan)

    # Convert to numpy arrays
    predicted_effects = np.array(predicted_effects)
    experimental_scores = np.array(scores)

    # Remove NaN values
    valid_mask = ~np.isnan(predicted_effects)
    predicted_effects = predicted_effects[valid_mask]
    experimental_scores = experimental_scores[valid_mask]

    if len(predicted_effects) < 2:
        return {
            'file': os.path.basename(dms_file),
            'spearman': None,
            'num_mutants': len(all_mutants_flat),
            'valid_mutants': len(predicted_effects),
            'error': 'Too few valid predictions'
        }

    # Compute Spearman correlation
    spearman_corr, p_value = spearmanr(predicted_effects, experimental_scores)

    return {
        'file': os.path.basename(dms_file),
        'spearman': spearman_corr,
        'p_value': p_value,
        'num_mutants': len(all_mutants_flat),
        'valid_mutants': len(predicted_effects),
        'predicted_effects': predicted_effects.tolist(),
        'experimental_scores': experimental_scores.tolist()
    }


def main():
    args = parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model - either with LoRA adapters or raw base model
    print("Loading model...")
    if args.model_path:
        # Load trained model with LoRA adapters
        print(f"Loading LoRA adapters from: {args.model_path}")
        model, tokenizer = load_esm_with_lora(
            model_name=args.model_name,
            lora_weights_path=args.model_path
        )
        model_type = "LoRA-finetuned"
    else:
        # Load raw base model without LoRA
        print(f"Loading raw base model: {args.model_name}")
        config = EsmConfig.from_pretrained(args.model_name)
        base_model = EsmModel.from_pretrained(args.model_name, config=config)
        model = GeoTuneESMModel(base_model=base_model, config=config, lora_config=None)
        tokenizer = EsmTokenizer.from_pretrained(args.model_name)
        model_type = "Raw base model"
    
    model.to(device)
    model.eval()
    
    print(f"Model type: {model_type}")

    # Find all DMS files
    dms_files = []
    for f in os.listdir(args.dms_dir):
        if f.endswith('.csv'):
            dms_files.append(os.path.join(args.dms_dir, f))

    print(f"Found {len(dms_files)} DMS files to evaluate")

    # Evaluate on each DMS file
    all_results = []
    spearman_correlations = []

    for dms_file in tqdm(dms_files, desc="Evaluating DMS files"):
        print(f"\n{'='*80}")
        print(f"Evaluating: {os.path.basename(dms_file)}")
        print(f"{'='*80}")

        result = evaluate_dms_file(
            model,
            dms_file,
            tokenizer,
            device,
            max_sequences=args.max_sequences
        )
        all_results.append(result)

        if result['spearman'] is not None:
            spearman_correlations.append(result['spearman'])
            print(f"Spearman correlation: {result['spearman']:.4f} (p={result['p_value']:.2e})")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")

    # Compute average Spearman correlation
    if spearman_correlations:
        avg_spearman = np.mean(spearman_correlations)
        std_spearman = np.std(spearman_correlations)
        median_spearman = np.median(spearman_correlations)
        
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Number of DMS files evaluated: {len(spearman_correlations)}")
        print(f"Average Spearman correlation: {avg_spearman:.4f} ± {std_spearman:.4f}")
        print(f"Median Spearman correlation: {median_spearman:.4f}")
        print(f"Min Spearman correlation: {np.min(spearman_correlations):.4f}")
        print(f"Max Spearman correlation: {np.max(spearman_correlations):.4f}")
    else:
        avg_spearman = None
        std_spearman = None
        median_spearman = None
        print("\nNo valid Spearman correlations computed")
    
    # Save results
    if args.output_file:
        output_file = args.output_file
    else:
        # Default: save to model directory or current directory for raw model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.model_path:
            output_dir = os.path.dirname(args.model_path)
            output_file = os.path.join(output_dir, f"dms_results_{timestamp}.pkl")
        else:
            # For raw base model, save to dms_results directory
            model_short = args.model_name.split('/')[-1].replace('_', '-')
            output_dir = f"dms_results/{model_short}"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"dms_results_{timestamp}.pkl")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results_to_save = {
        'model_path': args.model_path,
        'model_type': 'LoRA-finetuned' if args.model_path else 'Raw base model',
        'dms_dir': args.dms_dir,
        'model_name': args.model_name,
        'timestamp': datetime.now().isoformat(),
        'individual_results': all_results,
        'spearman_correlations': spearman_correlations,
        'average_spearman': avg_spearman,
        'std_spearman': std_spearman,
        'median_spearman': median_spearman,
        'num_files': len(spearman_correlations)
    }

    with open(output_file, 'wb') as f:
        pickle.dump(results_to_save, f)

    print(f"\nResults saved to: {output_file}")

    # Also save a text summary
    summary_file = output_file.replace('.pkl', '.txt')
    with open(summary_file, 'w') as f:
        f.write("DMS Substitution Benchmark Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model type: {'LoRA-finetuned' if args.model_path else 'Raw base model'}\n")
        if args.model_path:
            f.write(f"Model path: {args.model_path}\n")
        f.write(f"Base model: {args.model_name}\n")
        f.write(f"DMS directory: {args.dms_dir}\n")
        f.write(f"Evaluation date: {datetime.now().isoformat()}\n\n")

        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of DMS files evaluated: {len(spearman_correlations)}\n")
        if avg_spearman is not None:
            f.write(f"Average Spearman correlation: {avg_spearman:.4f} ± {std_spearman:.4f}\n")
            f.write(f"Median Spearman correlation: {median_spearman:.4f}\n")
            f.write(f"Min Spearman correlation: {np.min(spearman_correlations):.4f}\n")
            f.write(f"Max Spearman correlation: {np.max(spearman_correlations):.4f}\n")
        else:
            f.write("No valid Spearman correlations computed\n")

        f.write("\n" + "="*80 + "\n")
        f.write("INDIVIDUAL RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'File':<60} {'Spearman':<12} {'Num mutants':<12}\n")
        f.write("-"*80 + "\n")

        for result in sorted(all_results, key=lambda x: x['spearman'] or -999, reverse=True):
            spearman_str = f"{result['spearman']:.4f}" if result['spearman'] is not None else "N/A"
            f.write(f"{result['file']:<60} {spearman_str:<12} {result.get('valid_mutants', result.get('num_mutants', 'N/A')):<12}\n")
    
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
