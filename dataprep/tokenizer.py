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
#    (tweak the slicing as you like)
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

print("important_data/ now contains:")
for f in os.listdir(OUTDIR):
    print("  ", f)
