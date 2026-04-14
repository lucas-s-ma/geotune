import argparse
import re
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import EsmForMaskedLM, EsmTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MUTATION_RE = re.compile(r"([A-Za-z*])(\d+)([A-Za-z*])")
CLS_OFFSET = 1
ESM2_MAX_LEN = 1022


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: torch.device):
    print(f"Loading model: {model_name}")
    model = EsmForMaskedLM.from_pretrained(model_name)
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, tokenizer

def get_token_id(tokenizer, aa: str):
    aa = aa.upper()
    token_id = tokenizer.convert_tokens_to_ids(aa)
    if token_id == tokenizer.unk_token_id and aa != tokenizer.unk_token:
        return None
    return token_id


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def get_context_window(wt_seq, meta_row, df):
    """
    ProteinGym `--use_msa_bounds` equivalent.
    For polyproteins (like ZIKV), strictly crop to the MSA bounds to avoid 
    feeding the model neighboring cleaved proteins.
    """
    wt_len = len(wt_seq)

    msa_start = meta_row.get("MSA_start", None)
    msa_end = meta_row.get("MSA_end", None)

    # 1. Strict MSA Bounds (Matches ProteinGym's viral benchmarks)
    if pd.notna(msa_start) and pd.notna(msa_end):
        start_idx = int(msa_start) - 1
        end_idx = int(msa_end)
        
        if (end_idx - start_idx) <= ESM2_MAX_LEN:
            print(
                f"[Windowing] Sequence is {wt_len} AA. "
                f"Using strict MSA bounds: {int(msa_start)}-{int(msa_end)}. "
                f"Crop: residues {start_idx + 1}-{end_idx} "
                f"(0-indexed: {start_idx}-{end_idx}, length: {end_idx - start_idx})"
            )
            return start_idx, end_idx

    # 2. Fallback to inferred positions padded to 1022 (your old logic)
    positions = []
    for mutant_str in df["mutant"].dropna():
        if mutant_str.upper() == "WT" or ":" in mutant_str:
            continue
        m = MUTATION_RE.fullmatch(mutant_str.strip())
        if m:
            positions.append(int(m.group(2)))

    if positions:
        min_pos = min(positions)
        max_pos = max(positions)
        start_idx = max(1, min_pos - (ESM2_MAX_LEN - (max_pos - min_pos)) // 2) - 1
        end_idx = min(wt_len, start_idx + ESM2_MAX_LEN)
        print(f"[Windowing] Inferred Crop (0-indexed): {start_idx}-{end_idx}")
        return start_idx, end_idx

    return 0, min(wt_len, ESM2_MAX_LEN)


# ---------------------------------------------------------------------------
# Masked marginals
# ---------------------------------------------------------------------------

def compute_masked_marginals(model, tokenizer, sequence, device):
    """Your original (and correct for ESM-2) strategy."""
    tokens = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    input_ids_full = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    seq_len = len(sequence)
    vocab_size = model.config.vocab_size
    log_probs_all = np.zeros((seq_len, vocab_size))

    with torch.no_grad():
        for pos in tqdm(range(seq_len), desc="Computing masked marginals"):
            input_ids = input_ids_full.clone()
            input_ids[0, CLS_OFFSET + pos] = tokenizer.mask_token_id
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
            log_probs_all[pos] = log_probs[0, CLS_OFFSET + pos].cpu().numpy()

    return log_probs_all


# ---------------------------------------------------------------------------
# Effect computation
# ---------------------------------------------------------------------------

def effect_from_mutant_str(log_probs, mutant_str, tokenizer, cropped_seq, start_idx):
    effect = 0.0
    for mut in mutant_str.split(":"):
        mut = mut.strip()
        m = MUTATION_RE.fullmatch(mut)
        if not m:
            return None

        wt_aa, pos_str, mut_aa = m.groups()
        wt_aa, mut_aa = wt_aa.upper(), mut_aa.upper()

        pos = int(pos_str) - 1
        pos_in_window = pos - start_idx

        if pos_in_window < 0 or pos_in_window >= len(cropped_seq):
            return None
        if cropped_seq[pos_in_window].upper() != wt_aa:
            return None
        if wt_aa == mut_aa:
            continue

        wt_id = get_token_id(tokenizer, wt_aa)
        mut_id = get_token_id(tokenizer, mut_aa)
        if wt_id is None or mut_id is None:
            return None

        effect += log_probs[pos_in_window, mut_id] - log_probs[pos_in_window, wt_id]

    return effect


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dms_file", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--dms_id", required=True)
    parser.add_argument("--model_name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--single_mutants_only", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Metadata ---
    meta = pd.read_csv(args.metadata)
    rows = meta[meta["DMS_id"] == args.dms_id]
    if rows.empty:
        print(f"Error: DMS_id '{args.dms_id}' not found in metadata.", file=sys.stderr)
        sys.exit(1)
    meta_row = rows.iloc[0]

    wt_seq_full = meta_row["target_seq"]

    # --- DMS data ---
    df = pd.read_csv(args.dms_file)
    if args.single_mutants_only:
        df = df[~df["mutant"].str.contains(":")]

    # --- Windowing ---
    start_idx, end_idx = get_context_window(wt_seq_full, meta_row, df)
    cropped_seq = wt_seq_full[start_idx:end_idx]

    # --- Model & Marginals ---
    model, tokenizer = load_model(args.model_name, device)
    log_probs = compute_masked_marginals(model, tokenizer, cropped_seq, device)

    # --- Scoring ---
    results = []
    dropped = 0

    for _, df_row in df.iterrows():
        dms_score = df_row.get("DMS_score")
        if pd.isna(dms_score):
            continue

        mutant = df_row.get("mutant")

        # WT rows
        if not isinstance(mutant, str) or mutant.upper() == "WT":
            results.append({
                "mutant": mutant if isinstance(mutant, str) else "WT",
                "esm2_predicted_effect": 0.0,
                "DMS_score": float(dms_score),
            })
            continue

        effect = effect_from_mutant_str(log_probs, mutant, tokenizer, cropped_seq, start_idx)

        if effect is None or np.isnan(effect):
            dropped += 1
            continue

        results.append({
            "mutant": mutant,
            "esm2_predicted_effect": effect,
            "DMS_score": float(dms_score),
        })

    print(f"\nScored: {len(results)} | Dropped: {dropped}")

    if len(results) < 2:
        print("Not enough data to compute correlation.", file=sys.stderr)
        sys.exit(1)

    res_df = pd.DataFrame(results)
    
    # We correlate against the harmonized DMS_score without applying directionality logic
    corr, pval = spearmanr(res_df["esm2_predicted_effect"], res_df["DMS_score"])
    print(f"Spearman rho = {corr:.4f}  (p = {pval:.2e})")

    # Re-implemented exactly as requested
    if args.output_csv:
        res_df.to_csv(args.output_csv, index=False)
        print(f"Results written to {args.output_csv}")


if __name__ == "__main__":
    main()