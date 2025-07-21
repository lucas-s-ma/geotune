#!/usr/bin/env python3
import numpy as np
from transformers import AutoTokenizer
import os

# 0) set up
OUTDIR = "important_data"
os.makedirs(OUTDIR, exist_ok=True)

# 1) your toy sequences
sequences = [
    "MSTAAALYIYSANQ",
    "MVLSPADKTNVKAAW",
    "MNIKDTLLDGVVAEY",
    "MITLRVGVPTLKRSF",
    "MDKDLKVDLKKAAL",
    "MFRHRIYGTLAPVSR",
    "MNKATIVGDPTLAAN",
    "MVYAKRKAAKPGNYL",
    "MKVKLFFWMRNVKAI",
    "MNQPLVPAANHGSKE"
]
keys = [f"protein{i}" for i in range(len(sequences))]

# 2) init tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "chandar-lab/AMPLIFY_120M",
    trust_remote_code=True
)

# 3) save your train/valid splits
all_keys = np.array(keys)
np.save(os.path.join(OUTDIR, "key_names_valid_train.npy"),
        all_keys, allow_pickle=True)
np.save(os.path.join(OUTDIR, "key_names_valid_valid.npy"),
        all_keys[:1], allow_pickle=True)  # 1‐example valid split

# 4) build and save raw‐sequence map
#    so that collate_fn can call tokenizer on List[str]
key_name2seq_token = {
    key: seq
    for key, seq in zip(keys, sequences)
}
np.save(os.path.join(OUTDIR, "key_name2seq_token.npy"),
        key_name2seq_token, allow_pickle=True)

# 5) dummy structural “foldseek” tokens (still numeric maps)
dummy_foldseek = {
    key: np.random.randint(0, 10, size=len(seq)).astype(np.int64)
    for key, seq in zip(keys, sequences)
}
np.save(os.path.join(OUTDIR, "key_name2foldseek_token.npy"),
        dummy_foldseek, allow_pickle=True)

# 6) dummy labels
np.save(os.path.join(OUTDIR, "seq_labels.npy"),
        np.random.randint(0, 2, size=len(keys)), allow_pickle=True)
np.save(os.path.join(OUTDIR, "struc_labels.npy"),
        np.random.randint(0, 2, size=len(keys)), allow_pickle=True)

# 7) generate dummy structural embeddings aligned with actual token count
AF2_DIR = "af2_embedding"
GEARNET_DIR = "gearnet_embedding"
os.makedirs(AF2_DIR, exist_ok=True)
os.makedirs(GEARNET_DIR, exist_ok=True)

# For each key, tokenize raw seq and count valid tokens
for key, seq in key_name2seq_token.items():
    enc = tokenizer(seq, add_special_tokens=True)
    ids = enc["input_ids"]
    # mask out PAD/BOS/EOS if defined
    invalid_ids = set()
    if tokenizer.pad_token_id is not None:
        invalid_ids.add(tokenizer.pad_token_id)
    if tokenizer.bos_token_id is not None:
        invalid_ids.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        invalid_ids.add(tokenizer.eos_token_id)
    valid_length = sum(1 for tid in ids if tid not in invalid_ids)

    # generate random embeddings of shape (valid_length, D)
    emb_af2    = np.random.randn(valid_length, 384).astype(np.float32)
    emb_gearnet = np.random.randn(valid_length, 512).astype(np.float32)

    # save per-key
    np.save(os.path.join(AF2_DIR,    f"{key}.npy"), emb_af2)
    np.save(os.path.join(GEARNET_DIR, f"{key}.npy"), emb_gearnet)

print("important_data/, af2_embedding/, and gearnet_embedding/ now populated.")
