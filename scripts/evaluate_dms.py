import os
import re
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import EsmForMaskedLM, EsmTokenizer

# LoRA support
from peft import PeftModel


# =========================
# LOAD MODEL (BASE + LoRA)
# =========================
def load_model(model_name, model_path, device):
    print(f"[INFO] Loading base model: {model_name}")

    model = EsmForMaskedLM.from_pretrained(model_name)
    tokenizer = EsmTokenizer.from_pretrained(model_name)

    if model_path is not None:
        print(f"[INFO] Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        
        # --- DEBUG CHECK ---
        test_weight_sum = model.esm.encoder.layer[0].attention.self.query.weight.sum().item()
        print(f"[DEBUG] LoRA Model Query Weight Sum: {test_weight_sum:.4f}")
    else:
        # --- DEBUG CHECK ---
        test_weight_sum = model.esm.encoder.layer[0].attention.self.query.weight.sum().item()
        print(f"[DEBUG] Base Model Query Weight Sum: {test_weight_sum:.4f}")

    model = model.to(device)
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
# MASKED MARGINALS
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
# MUTATION PARSING
# =========================
mutation_pattern = re.compile(r"([A-Z])(\d+)([A-Z])")


def compute_effect_from_mutant_str(log_probs_all, mutant_str, tokenizer):
    effect = 0.0

    mutations = re.split(r"[:,]", mutant_str)

    for mut in mutations:
        mut = mut.strip()
        match = mutation_pattern.fullmatch(mut)
        if not match:
            return None

        wt_aa, pos, mut_aa = match.groups()
        pos = int(pos) - 1

        wt_id = tokenizer.convert_tokens_to_ids(wt_aa)
        mut_id = tokenizer.convert_tokens_to_ids(mut_aa)

        if wt_id is None or mut_id is None:
            return None

        if pos < 0 or pos >= log_probs_all.shape[0]:
            return None

        effect += log_probs_all[pos, mut_id] - log_probs_all[pos, wt_id]

    return effect


# =========================
# SEQUENCE DIFF FALLBACK
# =========================
def compute_effect_from_sequence(log_probs_all, wt_seq, mut_seq, tokenizer):
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

    print("[INFO] Computing masked marginals...")
    log_probs_all = compute_masked_marginals(model, tokenizer, wt_seq, device)

    use_mutant_col = "mutant" in df.columns

    predicted = []
    experimental = []
    dropped = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        score = row["DMS_score"]

        if pd.isna(score):
            continue

        if "mutated_sequence" in row and isinstance(row["mutated_sequence"], str):
            if "*" in row["mutated_sequence"]:
                continue

        # --- use mutation string if available ---
        if use_mutant_col and isinstance(row["mutant"], str):
            effect = compute_effect_from_mutant_str(
                log_probs_all,
                row["mutant"],
                tokenizer
            )
        else:
            mut_seq = row["mutated_sequence"]

            if len(mut_seq) != len(wt_seq):
                continue

            if any(aa not in valid_aas for aa in mut_seq):
                continue

            effect = compute_effect_from_sequence(
                log_probs_all,
                wt_seq,
                mut_seq,
                tokenizer
            )

        if effect is None or np.isnan(effect):
            dropped += 1
            continue

        predicted.append(effect)
        experimental.append(score)

    print(f"[DEBUG] Used: {len(predicted)}")
    print(f"[DEBUG] Dropped: {dropped}")

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
    parser.add_argument("--model_path", type=str, default=None)  
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(args.model_name, args.model_path, device)
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