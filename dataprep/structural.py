#!/usr/bin/env python3
import os
import numpy as np

# Adjust this if your folder is named differently
PREFIX = "dataprep"
IMP_DATA = os.path.join(PREFIX, "important_data")
AF2_DIR = os.path.join(PREFIX, "af2_embedding")
GEARNET_DIR = os.path.join(PREFIX, "gearnet_embedding")

# Create embedding folders if they don't exist
os.makedirs(AF2_DIR, exist_ok=True)
os.makedirs(GEARNET_DIR, exist_ok=True)

# Load your split key lists
train_keys = np.load(os.path.join(IMP_DATA, "key_names_valid_train.npy"), allow_pickle=True)
valid_keys = np.load(os.path.join(IMP_DATA, "key_names_valid_valid.npy"), allow_pickle=True)

# Load the sequence‐token mapping
seq_token_map = np.load(os.path.join(IMP_DATA, "key_name2seq_token.npy"), allow_pickle=True).item()

# Combine all keys
all_keys = np.concatenate([train_keys, valid_keys])

for key in all_keys:
    seq = seq_token_map[key]
    L = len(seq)
    # AF2 embeddings are size 384, Gearnet embeddings size 512
    emb_af2 = np.random.randn(L, 384).astype(np.float32)
    emb_gearnet = np.random.randn(L, 512).astype(np.float32)

    # Save out one file per key
    np.save(os.path.join(AF2_DIR,    f"{key}.npy"), emb_af2)
    np.save(os.path.join(GEARNET_DIR, f"{key}.npy"), emb_gearnet)

print(f"Generated dummy embeddings for {len(all_keys)} sequences.")
