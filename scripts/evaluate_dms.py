"""
Evaluate trained models on DMS (Deep Mutational Scanning) benchmarks.
Uses canonical Masked-Marginals scoring and strict UniProt Macro-Averaging
to exactly match ProteinGym ESM-2 benchmarks.

This implementation follows ProteinGym's compute_fitness.py exactly:
- Scores ALL positions sequentially (not just mutated positions)
- Uses optimal windowing for sequences > 1024 tokens
- Computes effect as: log P(mutant | masked_context) - log P(wildtype | masked_context)
- Uses +1 offset for BOS token in token_probs array
"""
import os
import re
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import EsmForMaskedLM, EsmTokenizer
import logging

from peft import PeftModel, LoraConfig, TaskType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

MUTATION_RE = re.compile(r"([A-Za-z*])(\d+)([A-Za-z*])")
ESM2_MAX_LEN = 1024  # Model context window including BOS/EOS


def load_model(model_name, model_path, device):
    """Load ESM model with optional LoRA adapter."""
    logger.info(f"Loading base model: {model_name}")
    model = EsmForMaskedLM.from_pretrained(model_name)
    tokenizer = EsmTokenizer.from_pretrained(model_name)

    if model_path is not None:
        logger.info(f"Loading LoRA adapter from: {model_path}")
        import json
        with open(os.path.join(model_path, "adapter_config.json")) as f:
            adapter_config = json.load(f)

        lora_config = LoraConfig(
            r=adapter_config.get("r", 8),
            lora_alpha=adapter_config.get("lora_alpha", 16),
            lora_dropout=adapter_config.get("lora_dropout", 0.1),
            target_modules=adapter_config.get("target_modules", ["query", "key", "value", "dense"]),
            bias=adapter_config.get("bias", "none"),
            task_type=TaskType.TOKEN_CLS,
        )

        peft_model = PeftModel(model, lora_config)
        from safetensors.torch import load_file
        adapter_weights = load_file(os.path.join(model_path, "adapter_model.safetensors"))

        current_state = peft_model.state_dict()
        new_state_dict = {}
        for saved_key, saved_value in adapter_weights.items():
            new_key = saved_key.replace("base_model.model.encoder", "base_model.model.esm.encoder")
            new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
            new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")

            if new_key in current_state:
                new_state_dict[new_key] = saved_value

        peft_model.load_state_dict(new_state_dict, strict=False)
        model = peft_model.merge_and_unload()
        logger.info(f"Loaded {len(new_state_dict)} LoRA weights")

    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_metadata(metadata_path):
    """Load WT sequences, MSA bounds, UniProt IDs, and coarse selection types."""
    df = pd.read_csv(metadata_path)
    meta_dict = {}

    # Case-insensitive column search
    uid_col = next((c for c in df.columns if c.lower() == 'uniprot_id'), None)
    sel_col = next((c for c in df.columns if 'coarse_selection_type' in c.lower()), None)

    for _, row in df.iterrows():
        uid = row[uid_col] if uid_col else "Unknown"
        if pd.isna(uid) or str(uid).strip() == "":
            uid = "Unknown"

        sel_type = row[sel_col] if sel_col else "Unknown"
        if pd.isna(sel_type) or str(sel_type).strip() == "":
            sel_type = "Unknown"

        meta_dict[row["DMS_id"]] = {
            "target_seq": row["target_seq"],
            "MSA_start": row.get("MSA_start", np.nan),
            "MSA_end": row.get("MSA_end", np.nan),
            "uniprot_id": str(uid).strip(),
            "coarse_selection_type": str(sel_type).strip()
        }
    return meta_dict


def build_valid_aa_set(tokenizer):
    vocab = tokenizer.get_vocab()
    return {k for k in vocab if len(k) == 1 and k.isalpha()}


# ---------------------------------------------------------------------------
# OPTIMAL WINDOWING (Matches ProteinGym's get_optimal_window exactly)
# ---------------------------------------------------------------------------

def get_optimal_window(mutation_position_relative, seq_len_wo_special, model_window):
    """
    Matches ProteinGym's scoring_utils.get_optimal_window exactly.
    
    For sequences longer than model_window, center the window around the
    mutation position. Near boundaries, clamp to edges.
    
    Args:
        mutation_position_relative: 0-indexed position in the sequence (without special tokens)
        seq_len_wo_special: Length of sequence without BOS/EOS tokens
        model_window: Model context window (1024 for ESM-2)
    
    Returns:
        [start, end] indices for the window (relative to sequence without special tokens)
    """
    half_model_window = model_window // 2
    if seq_len_wo_special <= model_window:
        return [0, seq_len_wo_special]
    elif mutation_position_relative < half_model_window:
        return [0, model_window]
    elif mutation_position_relative >= seq_len_wo_special - half_model_window:
        return [seq_len_wo_special - model_window, seq_len_wo_special]
    else:
        return [max(0, mutation_position_relative - half_model_window),
                min(seq_len_wo_special, mutation_position_relative + half_model_window)]


# ---------------------------------------------------------------------------
# MASKED MARGINALS (Matches canonical ESM zero-shot evaluation)
# ---------------------------------------------------------------------------

def compute_masked_marginals(model, tokenizer, wt_seq, device, start_idx=0, end_idx=None):
    """
    Compute masked marginals for positions in wt_seq[start_idx:end_idx].
    
    This follows ProteinGym's compute_fitness.py exactly:
    1. For each position i in the cropped sequence:
       a. Determine optimal window centered on position i
       b. Extract that window from the full sequence
       c. Mask position i within the window
       d. Run through model, extract log_probs at position i
    2. Store log_probs for each position
    
    Returns:
        dict: {position (0-indexed relative to full wt_seq): numpy array of log_probs (vocab size)}
    """
    if end_idx is None:
        end_idx = len(wt_seq)
    
    seq_len = end_idx - start_idx  # Length of the region we're scoring
    vocab_size = model.config.vocab_size
    
    # Store log_probs for each position (keyed by 0-indexed position in full wt_seq)
    log_probs_map = {}
    
    # Tokenize the FULL sequence (we'll crop windows from it)
    full_tokens = tokenizer(wt_seq, return_tensors="pt", add_special_tokens=True)
    full_input_ids = full_tokens["input_ids"].to(device)  # Shape: [1, seq_len+2] (with BOS/EOS)
    
    logger.info(f"Computing masked marginals for positions {start_idx} to {end_idx-1} "
                f"(seq_len={seq_len}, full_seq_len={len(wt_seq)})")
    
    for i in tqdm(range(seq_len), desc="Computing masked marginals"):
        # i is 0-indexed position within the cropped region
        # absolute_pos is the 0-indexed position in the full wt_seq
        absolute_pos = start_idx + i
        
        # Get optimal window for this position
        # ProteinGym uses token positions (including BOS at position 0)
        # Token position for absolute_pos = absolute_pos + 1 (BOS offset)
        absolute_token_pos = absolute_pos + 1
        seq_len_wo_special = len(wt_seq) + 2  # +2 for BOS and EOS
        
        window_start, window_end = get_optimal_window(
            mutation_position_relative=absolute_token_pos,
            seq_len_wo_special=seq_len_wo_special,
            model_window=ESM2_MAX_LEN
        )
        
        # Extract the window tokens from full_input_ids
        window_input_ids = full_input_ids[:, window_start:window_end].clone()
        
        # Find where the target position maps to within the window
        pos_in_window = absolute_token_pos - window_start
        
        # Mask the target position
        window_input_ids[0, pos_in_window] = tokenizer.mask_token_id
        
        # Run through model
        with torch.no_grad():
            outputs = model(input_ids=window_input_ids)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
        
        # Extract log_probs at the target position
        log_probs_map[absolute_pos] = log_probs[0, pos_in_window].cpu().numpy()
    
    return log_probs_map


def compute_effect_from_mutant_str(log_probs_map, mutant_str, tokenizer, wt_seq, offset_idx=1):
    """
    Compute mutation effect from mutant string (e.g., 'I291A' or 'I291A:K292R').
    
    Effect = sum over mutations of: log P(mutant_AA | context) - log P(wildtype_AA | context)
    
    This matches ProteinGym's label_row function exactly:
      idx = int(mutation[1:-1]) - offset_idx
      score += token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
    """
    effect = 0.0
    for mut in re.split(r"[:,]", mutant_str):
        m = MUTATION_RE.fullmatch(mut.strip())
        if not m:
            return None

        wt_aa, pos_str, mut_aa = m.groups()
        wt_aa, mut_aa = wt_aa.upper(), mut_aa.upper()
        
        # Match ProteinGym's label_row: idx = int(mutation[1:-1]) - offset_idx
        idx = int(pos_str) - offset_idx

        if idx < 0 or idx >= len(wt_seq):
            return None
        if wt_seq[idx].upper() != wt_aa:
            return None
        if wt_aa == mut_aa:
            continue

        wt_id = tokenizer.convert_tokens_to_ids(wt_aa)
        mut_id = tokenizer.convert_tokens_to_ids(mut_aa)

        if wt_id is None or mut_id is None:
            return None
        
        # log_probs_map is keyed by 0-indexed position in the cropped sequence
        if idx not in log_probs_map:
            return None
        
        # Effect = log P(mutant) - log P(wildtype)
        effect += log_probs_map[idx][mut_id] - log_probs_map[idx][wt_id]

    return effect


def compute_effect_from_sequence(log_probs_map, wt_seq, mut_seq, tokenizer):
    """Compute effect from full mutant sequence (when mutant column not available)."""
    effect = 0.0
    for i in range(len(wt_seq)):
        if wt_seq[i] != mut_seq[i]:
            if i not in log_probs_map:
                return None
            wt_id = tokenizer.convert_tokens_to_ids(wt_seq[i])
            mut_id = tokenizer.convert_tokens_to_ids(mut_seq[i])
            if wt_id is None or mut_id is None:
                return None
            effect += log_probs_map[i][mut_id] - log_probs_map[i][wt_id]
    return effect


def evaluate_file(model, tokenizer, device, dms_file, meta_dict, valid_aas):
    fname = os.path.basename(dms_file)
    dms_id = fname.replace(".csv", "")

    if dms_id not in meta_dict:
        return None, "Metadata not found"

    meta = meta_dict[dms_id]
    wt_seq_full = meta["target_seq"]

    try:
        df = pd.read_csv(dms_file)
    except Exception as e:
        return None, f"CSV read error: {e}"

    if len(df) == 0:
        return None, "Empty CSV"

    # ProteinGym's approach: crop sequence to MSA bounds first, then score the cropped sequence
    # This matches compute_fitness.py lines 322-324:
    #   if ((target_seq_start_index!=msa_start_index) or (target_seq_end_index!=msa_end_index)):
    #       args.sequence = args.sequence[msa_start_index-1:msa_end_index]
    #       target_seq_start_index = msa_start_index
    
    msa_start = int(meta["MSA_start"]) if pd.notna(meta["MSA_start"]) else 1
    msa_end = int(meta["MSA_end"]) if pd.notna(meta["MSA_end"]) else len(wt_seq_full)
    
    # Crop the sequence to MSA bounds (1-indexed -> 0-indexed slicing)
    wt_seq_cropped = wt_seq_full[msa_start - 1:msa_end]
    
    # offset_idx is the 1-indexed position where the cropped sequence starts
    # This is used to map mutation coordinates to the cropped sequence
    offset_idx = msa_start

    # Compute masked marginals for the cropped sequence
    # ProteinGym iterates over ALL positions in the cropped sequence
    log_probs_map = compute_masked_marginals(
        model, tokenizer, wt_seq_cropped, device,
        start_idx=0, end_idx=len(wt_seq_cropped)
    )

    use_mutant_col = "mutant" in df.columns
    predicted, experimental = [], []

    for _, row in df.iterrows():
        score = row.get("DMS_score")
        if pd.isna(score):
            continue

        # Skip rows with invalid sequences
        if "mutated_sequence" in row and isinstance(row["mutated_sequence"], str):
            if "*" in row["mutated_sequence"]:
                continue

        mutant_val = row.get("mutant", "")
        if isinstance(mutant_val, str) and mutant_val.upper() == "WT":
            continue

        if use_mutant_col and isinstance(mutant_val, str):
            effect = compute_effect_from_mutant_str(
                log_probs_map, mutant_val, tokenizer, wt_seq_cropped, offset_idx
            )
        else:
            mut_seq = row.get("mutated_sequence")
            if not isinstance(mut_seq, str) or any(aa not in valid_aas for aa in mut_seq):
                continue
            effect = compute_effect_from_sequence(log_probs_map, wt_seq_cropped, mut_seq, tokenizer)

        if effect is None or np.isnan(effect):
            continue

        predicted.append(effect)
        experimental.append(score)

    if len(predicted) < 2:
        return None, f"Not enough valid predictions ({len(predicted)} < 2)"

    corr, _ = spearmanr(predicted, experimental)
    if np.isnan(corr):
        return None, "Correlation evaluates to NaN"

    logger.info(f"{fname} -> Spearman: {corr:.4f} ({len(predicted)} mutants scored)")
    return corr, "Success"


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on DMS benchmarks")
    parser.add_argument("--dms_dir", type=str, required=True, help="Directory containing DMS CSV files")
    parser.add_argument("--metadata", type=str, required=True, help="Metadata CSV file with WT sequences")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D", help="Base model name")
    parser.add_argument("--model_path", type=str, default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output_csv", type=str, default="benchmark_results.csv", help="Save detailed results to this CSV")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(args.model_name, args.model_path, device)
    meta_dict = load_metadata(args.metadata)
    valid_aas = build_valid_aa_set(tokenizer)

    files = [f for f in os.listdir(args.dms_dir) if f.endswith(".csv")]
    results = []
    results_by_file = {}
    failed_files = [] 
    
    # Store records for the spreadsheet export
    csv_records = []

    for f in sorted(files):
        path = os.path.join(args.dms_dir, f)
        corr, status = evaluate_file(model, tokenizer, device, path, meta_dict, valid_aas)

        dms_id = f.replace(".csv", "")
        uid = meta_dict.get(dms_id, {}).get("uniprot_id", "Unknown")
        sel_type = meta_dict.get(dms_id, {}).get("coarse_selection_type", "Unknown")

        if corr is not None:
            results.append(corr)
            results_by_file[f] = corr
            csv_records.append({
                "DMS_id": dms_id,
                "UniProt_ID": uid,
                "coarse_selection_type": sel_type,
                "Spearman_Correlation": corr,
                "Status": "Success"
            })
        else:
            failed_files.append((f, status))
            csv_records.append({
                "DMS_id": dms_id,
                "UniProt_ID": uid,
                "coarse_selection_type": sel_type,
                "Spearman_Correlation": np.nan,
                "Status": status
            })

    # Save to Spreadsheet
    out_df = pd.DataFrame(csv_records)
    out_df.to_csv(args.output_csv, index=False)

    if results:
        # 1. Simple Arithmetic Mean
        simple_mean = np.mean(results)
        simple_std = np.std(results)

        # 2. ProteinGym Hierarchical Macro-Average (3-level aggregation)
        # Matches performance_DMS_benchmarks.py exactly:
        #   Level 1: Group by UniProt_ID -> mean within each protein
        #   Level 2: Group by UniProt_ID + Selection_Type -> mean within each protein+function
        #   Level 3: Average across the 5 Selection Types -> final leaderboard score

        # Build a DataFrame for aggregation
        perf_df = pd.DataFrame([
            {
                "DMS_id": f.replace(".csv", ""),
                "UniProt_ID": meta_dict.get(f.replace(".csv", ""), {}).get("uniprot_id", "Unknown"),
                "coarse_selection_type": meta_dict.get(f.replace(".csv", ""), {}).get("coarse_selection_type", "Unknown"),
                "Spearman": corr
            }
            for f, corr in results_by_file.items()
        ])

        # Level 1: Mean within each UniProt_ID
        uniprot_means = perf_df.groupby("UniProt_ID")["Spearman"].mean()
        uniprot_mean_overall = uniprot_means.mean()
        uniprot_std_overall = uniprot_means.std()

        # Level 2: Mean within each UniProt_ID + coarse_selection_type group
        uniprot_function_means = perf_df.groupby(["UniProt_ID", "coarse_selection_type"])["Spearman"].mean()

        # Level 3: Average within each coarse_selection_type, then average across types
        function_level = uniprot_function_means.groupby("coarse_selection_type").mean()
        final_leaderboard = function_level.mean()

        logger.info("\n" + "="*60)
        logger.info(f"Successfully evaluated: {len(results)} / {len(files)} files")
        logger.info(f"Unique Proteins (UniProt IDs): {uniprot_means.shape[0]}")
        logger.info("-" * 60)
        logger.info(f"Simple Average ({len(results)} Assays):         {simple_mean:.4f} ± {simple_std:.4f}")
        logger.info(f"UniProt Macro-Average:                {uniprot_mean_overall:.4f} ± {uniprot_std_overall:.4f}")
        logger.info(f"Leaderboard Average (3-level):        {final_leaderboard:.4f}")
        logger.info("-" * 60)
        logger.info("  By Functional Category:")
        for sel_type in sorted(function_level.index):
            logger.info(f"    {sel_type:25s}: {function_level[sel_type]:.4f}")
        logger.info("="*60)
        logger.info(f"Detailed results saved to: {args.output_csv}")
    else:
        logger.warning("\nNo valid results obtained across all files.")

    if failed_files:
        logger.error("\n" + "!"*60)
        logger.error(f" FAILED TO COMPUTE CORRELATION FOR {len(failed_files)} FILE(S):")
        logger.error("!"*60)
        for fname, reason in failed_files:
            logger.error(f"  --> {fname}: {reason}")
        logger.error("!"*60 + "\n")


if __name__ == "__main__":
    main()