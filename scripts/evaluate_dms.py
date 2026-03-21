import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import EsmForMaskedLM, EsmTokenizer


# =========================
# MODEL LOADING (FIXED)
# =========================
def load_model(model_name, device):
    model = EsmForMaskedLM.from_pretrained(model_name).to(device)
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


# =========================
# CORE: MASKED MARGINAL LOG PROBS (CORRECT)
# =========================
def get_masked_marginal_log_probs(model, sequence, tokenizer, device, batch_size=16):
    """
    Correct ProteinGym-style masked marginal log probabilities.

    Returns:
        log_probs_all: (L, vocab_size)
    """
    tokens = tokenizer(sequence, return_tensors="pt")
    input_ids_full = tokens["input_ids"].to(device)

    seq_len = len(sequence)
    vocab_size = model.config.vocab_size

    log_probs_all = np.zeros((seq_len, vocab_size))

    with torch.no_grad():
        for start in range(0, seq_len, batch_size):
            end = min(start + batch_size, seq_len)
            batch_size_actual = end - start

            # Repeat base sequence
            input_ids = input_ids_full.repeat(batch_size_actual, 1)

            # Apply masking at token level
            for i, pos in enumerate(range(start, end)):
                input_ids[i, pos + 1] = tokenizer.mask_token_id  # +1 for BOS

            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)

            # Extract mask positions safely
            for i in range(batch_size_actual):
                mask_pos = (input_ids[i] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                if len(mask_pos) == 0:
                    continue
                mp = mask_pos.item()
                log_probs_all[start + i] = log_probs[i, mp].cpu().numpy()

    return log_probs_all


# =========================
# MUTATIONAL EFFECT (CORRECT)
# =========================
def compute_mutational_effect(log_probs_all, wt_seq, mut_seq, tokenizer):
    """
    Δ log P(mut) - log P(wt) at mutated positions
    """

    if len(wt_seq) != len(mut_seq):
        return None

    effect = 0.0
    found = False

    for pos in range(len(wt_seq)):
        if wt_seq[pos] != mut_seq[pos]:
            wt_id = tokenizer.convert_tokens_to_ids(wt_seq[pos])
            mut_id = tokenizer.convert_tokens_to_ids(mut_seq[pos])

            if wt_id is None or mut_id is None:
                return None

            effect += log_probs_all[pos, mut_id] - log_probs_all[pos, wt_id]
            found = True

    if not found:
        return 0.0

    return effect


# =========================
# EVALUATE SINGLE DMS FILE
# =========================
def evaluate_dms_file(model, tokenizer, device, dms_file, single_only=True):
    df = pd.read_csv(dms_file)

    predicted = []
    experimental = []

    # Group by WT sequence for efficiency
    grouped = {}

    for _, row in df.iterrows():
        wt_seq = row.get("wildtype_sequence", None)
        mut_seq = row["mutated_sequence"]

        if wt_seq is None:
            # fallback reconstruction
            mutant = row["mutant"]
            import re
            m = re.match(r'([A-Z])(\d+)([A-Z*])', mutant)
            if not m:
                continue
            wt_aa, pos_str, mut_aa = m.groups()
            pos = int(pos_str) - 1

            wt_seq = mut_seq[:pos] + wt_aa + mut_seq[pos+1:]

        # count mutations
        diff = sum(1 for a, b in zip(wt_seq, mut_seq) if a != b)
        if single_only and diff != 1:
            continue

        grouped.setdefault(wt_seq, []).append(row)

    # Evaluate per WT
    for wt_seq, rows in tqdm(grouped.items(), desc="WT sequences"):
        log_probs_all = get_masked_marginal_log_probs(model, wt_seq, tokenizer, device)

        for row in rows:
            mut_seq = row["mutated_sequence"]
            score = row["DMS_score"]

            effect = compute_mutational_effect(
                log_probs_all,
                wt_seq,
                mut_seq,
                tokenizer
            )

            if effect is not None:
                predicted.append(effect)
                experimental.append(score)

    if len(predicted) < 2:
        return None

    corr, _ = spearmanr(predicted, experimental)
    return corr


# =========================
# MAIN
# =========================
def run_evaluation(dms_dir, model_name="facebook/esm2_t6_8M_UR50D", device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(model_name, device)

    files = [f for f in os.listdir(dms_dir) if f.endswith(".csv")]

    results = []

    for f in files:
        path = os.path.join(dms_dir, f)
        print(f"\nEvaluating {f}")

        corr = evaluate_dms_file(model, tokenizer, device, path)

        if corr is not None:
            print(f"Spearman: {corr:.4f}")
            results.append(corr)
        else:
            print("Skipped")

    print("\n====================")
    print(f"Average Spearman: {np.mean(results):.4f}")
    print("====================")

    return results

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--dms_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    run_evaluation(
        dms_dir=args.dms_dir,
        model_name=args.model_name,
        device=args.device
    )