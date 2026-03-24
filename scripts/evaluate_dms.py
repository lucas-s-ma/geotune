import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import EsmForMaskedLM, EsmTokenizer


# =========================
# LOAD MODEL
# =========================
def load_model(model_name, device):
    model = EsmForMaskedLM.from_pretrained(model_name).to(device)
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


# =========================
# LOAD METADATA
# =========================
def load_wt_mapping(metadata_path):
    df = pd.read_csv(metadata_path)
    return dict(zip(df["DMS_id"], df["target_seq"]))


# =========================
# VALID AA SET
# =========================
def build_valid_aa_set(tokenizer):
    vocab = tokenizer.get_vocab()
    return {k for k in vocab if len(k) == 1 and k.isalpha()}


# =========================
# MASKED MARGINALS (ONE PASS)
# =========================
def compute_masked_marginals(model, tokenizer, sequence, device, batch_size=32):
    tokens = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    input_ids_full = tokens["input_ids"].to(device)

    seq_len = len(sequence)
    vocab_size = model.config.vocab_size

    log_probs_all = np.zeros((seq_len, vocab_size))
    CLS_OFFSET = 1

    with torch.no_grad():
        for start in range(0, seq_len, batch_size):
            end = min(start + batch_size, seq_len)
            B = end - start

            input_ids = input_ids_full.repeat(B, 1)

            for i, pos in enumerate(range(start, end)):
                input_ids[i, CLS_OFFSET + pos] = tokenizer.mask_token_id

            outputs = model(input_ids=input_ids)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)

            for i, pos in enumerate(range(start, end)):
                mp = CLS_OFFSET + pos
                log_probs_all[pos] = log_probs[i, mp].cpu().numpy()

    return log_probs_all


# =========================
# FAST MUTATION SCORING
# =========================
def compute_effect(log_probs_all, wt_seq, mut_seq, tokenizer):
    effect = 0.0

    for i in range(len(wt_seq)):
        if wt_seq[i] != mut_seq[i]:
            wt_id = tokenizer.convert_tokens_to_ids(wt_seq[i])
            mut_id = tokenizer.convert_tokens_to_ids(mut_seq[i])

            if wt_id is None or mut_id is None:
                return None

            effect += log_probs_all[i, mut_id] - log_probs_all[i, wt_id]

    return effect


# =========================
# EVALUATE FILE
# =========================
def evaluate_file(model, tokenizer, device, dms_file, wt_mapping, valid_aas):
    fname = os.path.basename(dms_file)
    dms_id = fname.replace(".csv", "")

    print(f"\n[FILE] {fname}")

    if dms_id not in wt_mapping:
        print("[ERROR] WT not found")
        return None

    wt_seq = wt_mapping[dms_id]
    df = pd.read_csv(dms_file)

    print(f"[INFO] WT length: {len(wt_seq)}")
    print(f"[INFO] Total rows: {len(df)}")

    print("[INFO] Computing masked marginals (ONE TIME)...")
    log_probs_all = compute_masked_marginals(model, tokenizer, wt_seq, device)

    predicted = []
    experimental = []

    dropped = 0

    print("[INFO] Scoring mutations (FAST)...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        mut_seq = row["mutated_sequence"]
        score = row["DMS_score"]

        if pd.isna(score):
            continue

        if len(mut_seq) != len(wt_seq):
            continue

        if "*" in mut_seq:
            continue

        if any(aa not in valid_aas for aa in mut_seq):
            dropped += 1
            continue

        effect = compute_effect(log_probs_all, wt_seq, mut_seq, tokenizer)

        if effect is None or np.isnan(effect):
            dropped += 1
            continue

        predicted.append(effect)
        experimental.append(score)

    print(f"[DEBUG] Used: {len(predicted)}")
    print(f"[DEBUG] Dropped invalid: {dropped}")

    if len(predicted) < 2:
        return None

    corr, _ = spearmanr(predicted, experimental)
    print(f"[RESULT] Spearman: {corr:.4f}")

    return corr


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dms_dir", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(args.model_name, device)
    wt_mapping = load_wt_mapping(args.metadata)
    valid_aas = build_valid_aa_set(tokenizer)

    files = [f for f in os.listdir(args.dms_dir) if f.endswith(".csv")]

    results = []

    for f in files:
        path = os.path.join(args.dms_dir, f)
        corr = evaluate_file(model, tokenizer, device, path, wt_mapping, valid_aas)

        if corr is not None:
            results.append(corr)

    if results:
        print("\n========================")
        print(f"[FINAL] Avg Spearman: {np.mean(results):.4f}")
        print("========================")


if __name__ == "__main__":
    main()