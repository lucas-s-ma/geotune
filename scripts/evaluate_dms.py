"""
Evaluation script for trained GeoTune models on Protein Gym DMS substitution benchmarks.

This script evaluates a trained model on all DMS substitution files in a directory,
computing Spearman correlation between predicted and experimental scores for each file,
then reporting the average Spearman correlation across all files.

Usage:
    python scripts/evaluate_dms.py --model_path outputs/run_20260302_143521/best_model/lora_adapters --dms_dir DMS_ProteinGym_substitutions
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

from models.geotune_esm_model import load_esm_with_lora


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained GeoTune model on DMS substitution benchmarks")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model (LoRA adapters directory)")
    parser.add_argument("--dms_dir", type=str, required=True,
                       help="Path to directory containing DMS substitution CSV files")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t30_150M_UR50D",
                       help="Base model name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Path to save results (default: model_path/dms_results.pkl)")
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


def compute_mutational_effect(model, wild_type_seq, mutant_seq, tokenizer, device):
    """
    Compute the mutational effect score using masked marginal likelihood.
    
    The score is the difference in log likelihood between the mutant and wild type.
    More precisely, we compute the log probability of the mutant amino acid at the 
    mutated position minus the log probability of the wild type amino acid.
    
    Args:
        model: The trained model
        wild_type_seq: Wild type protein sequence
        mutant_seq: Mutated protein sequence
        tokenizer: ESM tokenizer
        device: torch device
    
    Returns:
        Mutational effect score (higher = more beneficial)
    """
    model.eval()
    
    # Find the mutation position
    if len(wild_type_seq) != len(mutant_seq):
        # For now, skip sequences with length differences
        return None
    
    diff_positions = [(i, wild_type_seq[i], mutant_seq[i]) 
                      for i in range(len(wild_type_seq)) 
                      if wild_type_seq[i] != mutant_seq[i]]
    
    if len(diff_positions) == 0:
        return 0.0  # No mutation
    
    if len(diff_positions) > 1:
        # Multiple mutations - for now, just use the first one
        # Could be extended to handle multiple mutations
        pass
    
    pos, wt_aa, mut_aa = diff_positions[0]
    
    # Tokenize wild type sequence
    wt_tokens = tokenizer(wild_type_seq, return_tensors="pt")
    wt_input_ids = wt_tokens['input_ids'].to(device)
    wt_attention_mask = wt_tokens['attention_mask'].to(device)
    
    # Get model outputs for wild type
    with torch.no_grad():
        wt_outputs = model(input_ids=wt_input_ids, attention_mask=wt_attention_mask)
        wt_logits = model.lm_head(wt_outputs['sequence_output'])
        wt_log_probs = torch.log_softmax(wt_logits, dim=-1)[0]  # (seq_len, vocab_size)
        
        # Get token IDs for the amino acids using the tokenizer
        # ESM tokenizers encode amino acids as single characters
        try:
            wt_token_id = tokenizer.encode(wt_aa, add_special_tokens=False)[0]
            mut_token_id = tokenizer.encode(mut_aa, add_special_tokens=False)[0]
        except (IndexError, ValueError):
            # Fallback to hardcoded mapping if tokenizer encoding fails
            aa_to_token = {
                'A': 6, 'R': 12, 'N': 15, 'D': 13, 'C': 29, 'Q': 23, 'E': 17, 'G': 20, 
                'H': 24, 'I': 25, 'L': 18, 'K': 19, 'M': 21, 'F': 26, 'P': 27, 'S': 28, 
                'T': 30, 'W': 31, 'Y': 32, 'V': 22
            }
            wt_token_id = aa_to_token.get(wt_aa, None)
            mut_token_id = aa_to_token.get(mut_aa, None)
        
        if wt_token_id is None or mut_token_id is None:
            return None
        
        # Position in tokenized sequence (accounting for BOS token)
        # ESM2 adds BOS token at the beginning
        token_pos = pos + 1  # +1 for BOS token
        
        # Make sure token_pos is within bounds
        if token_pos >= wt_log_probs.shape[0]:
            return None
        
        # Get log probabilities
        log_prob_wt = wt_log_probs[token_pos, wt_token_id].item()
        log_prob_mut = wt_log_probs[token_pos, mut_token_id].item()
        
        # Mutational effect: log P(mutant) - log P(wild_type)
        # Positive = mutation is favored, Negative = mutation is disfavored
        mut_effect = log_prob_mut - log_prob_wt
    
    return mut_effect


def evaluate_dms_file(model, dms_file, tokenizer, device, max_sequences=None):
    """
    Evaluate model on a single DMS file.
    
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
    
    # Get wild type sequence from first row (all rows have the same mutated_sequence with single mutation)
    # We need to reconstruct the wild type by finding the consensus or using the first mutation
    # For ProteinGym, we can infer wild type from the mutant notation
    
    # Extract mutant info
    mutants = []
    scores = []
    score_bins = []
    
    for idx, row in df.iterrows():
        if max_sequences and len(mutants) >= max_sequences:
            break
            
        mutant_str = row['mutant']  # e.g., "I291A"
        mutated_seq = row['mutated_sequence']
        dms_score = row['DMS_score']
        dms_score_bin = row['DMS_score_bin']
        
        # Parse mutant string to get wild type sequence
        # Format: {WT_AA}{POSITION}{MUT_AA}
        import re
        match = re.match(r'([A-Z])(\d+)([A-Z*])', mutant_str)
        if match:
            wt_aa, pos_str, mut_aa = match.groups()
            pos = int(pos_str) - 1  # Convert to 0-indexed
            
            # Reconstruct wild type sequence
            # The mutated_sequence has the mutation at position pos
            # We need to replace mut_aa with wt_aa at that position
            if pos < len(mutated_seq) and mutated_seq[pos] == mut_aa:
                wt_seq = mutated_seq[:pos] + wt_aa + mutated_seq[pos+1:]
            else:
                # Position doesn't match, skip
                continue
            
            mutants.append({
                'mutant': mutant_str,
                'wt_seq': wt_seq,
                'mut_seq': mutated_seq,
                'pos': pos,
                'wt_aa': wt_aa,
                'mut_aa': mut_aa
            })
            scores.append(dms_score)
            score_bins.append(dms_score_bin)
    
    if len(mutants) == 0:
        return {
            'file': os.path.basename(dms_file),
            'spearman': None,
            'num_mutants': 0,
            'error': 'No valid mutants found'
        }
    
    # Compute mutational effects
    print(f"Computing mutational effects for {len(mutants)} mutants...")
    predicted_effects = []
    
    for mut_info in tqdm(mutants, desc="Evaluating mutants"):
        effect = compute_mutational_effect(
            model, 
            mut_info['wt_seq'], 
            mut_info['mut_seq'], 
            tokenizer, 
            device
        )
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
            'num_mutants': len(mutants),
            'valid_mutants': len(predicted_effects),
            'error': 'Too few valid predictions'
        }
    
    # Compute Spearman correlation
    spearman_corr, p_value = spearmanr(predicted_effects, experimental_scores)
    
    return {
        'file': os.path.basename(dms_file),
        'spearman': spearman_corr,
        'p_value': p_value,
        'num_mutants': len(mutants),
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
    
    # Load model with LoRA adapters
    print("Loading model...")
    model, tokenizer = load_esm_with_lora(
        model_name=args.model_name,
        lora_weights_path=args.model_path
    )
    model.to(device)
    model.eval()
    
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
        # Default: save to model directory
        output_dir = os.path.dirname(args.model_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"dms_results_{timestamp}.pkl")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    results_to_save = {
        'model_path': args.model_path,
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
        f.write(f"Model: {args.model_path}\n")
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
