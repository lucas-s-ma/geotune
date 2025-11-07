#!/usr/bin/env python3
"""
Debug script to check token IDs in the processed dataset.
Run this on the cluster to verify the dataset has correct ESM2 token IDs.
"""
import pickle
import torch
import numpy as np
from pathlib import Path

def check_dataset_tokens(dataset_path):
    """Check token IDs in processed dataset"""
    print(f"Loading dataset from: {dataset_path}")

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    print(f"\nDataset size: {len(data)} proteins")

    # Check token IDs across all proteins
    all_tokens = []
    invalid_samples = []

    for idx, item in enumerate(data):
        tokens = item.get('token_ids', [])
        all_tokens.extend(tokens)

        # Check for invalid tokens (ESM2 vocab: 0-23 for amino acids + special, 32 for mask)
        # Valid tokens: 0 (<cls>), 1 (<pad>), 2 (<eos>), 3 (<unk>), 4-23 (amino acids), 32 (<mask>)
        invalid = [t for t in tokens if t > 32 or (t > 23 and t < 32)]
        if invalid:
            invalid_samples.append({
                'idx': idx,
                'sequence': item.get('sequence', 'N/A')[:50],
                'invalid_tokens': invalid[:10],  # First 10 invalid
                'all_tokens': tokens[:30]  # First 30 tokens
            })

    # Statistics
    all_tokens = np.array(all_tokens)
    print(f"\n{'='*60}")
    print("TOKEN ID STATISTICS")
    print(f"{'='*60}")
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Token range: {all_tokens.min()} - {all_tokens.max()}")
    print(f"Unique tokens: {sorted(np.unique(all_tokens).tolist())}")

    # Expected ESM2 tokens
    expected_tokens = set([0, 1, 2, 3] + list(range(4, 24)) + [32])
    actual_tokens = set(all_tokens.tolist())
    unexpected = actual_tokens - expected_tokens

    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")
    if unexpected:
        print(f"❌ INVALID TOKENS FOUND: {sorted(unexpected)}")
        print(f"   These tokens are NOT in ESM2 vocabulary!")
        print(f"   Number of samples with invalid tokens: {len(invalid_samples)}")
    else:
        print(f"✅ All tokens are valid ESM2 tokens")

    # Show token distribution
    print(f"\n{'='*60}")
    print("TOKEN DISTRIBUTION")
    print(f"{'='*60}")
    unique, counts = np.unique(all_tokens, return_counts=True)
    token_names = {
        0: '<cls>', 1: '<pad>', 2: '<eos>', 3: '<unk>',
        4: 'L', 5: 'A', 6: 'G', 7: 'V', 8: 'S', 9: 'E', 10: 'R', 11: 'T',
        12: 'I', 13: 'D', 14: 'P', 15: 'K', 16: 'Q', 17: 'N', 18: 'F', 19: 'Y',
        20: 'M', 21: 'H', 22: 'W', 23: 'C', 32: '<mask>'
    }

    print(f"{'Token ID':<10} {'Token':<8} {'Count':<12} {'Percentage':<10}")
    print("-" * 50)
    for tok, cnt in zip(unique, counts):
        tok_name = token_names.get(tok, f'INVALID')
        pct = (cnt / len(all_tokens)) * 100
        symbol = "✅" if tok in expected_tokens else "❌"
        print(f"{symbol} {tok:<10} {tok_name:<8} {cnt:<12} {pct:.2f}%")

    # Show samples with invalid tokens
    if invalid_samples:
        print(f"\n{'='*60}")
        print(f"SAMPLES WITH INVALID TOKENS (showing first 5)")
        print(f"{'='*60}")
        for sample in invalid_samples[:5]:
            print(f"\nSample {sample['idx']}:")
            print(f"  Sequence: {sample['sequence']}...")
            print(f"  Invalid tokens: {sample['invalid_tokens']}")
            print(f"  First 30 tokens: {sample['all_tokens']}")

    # Check if dataset has OLD token mapping (5-24 instead of 4-23)
    print(f"\n{'='*60}")
    print("CHECKING FOR OLD TOKEN MAPPING")
    print(f"{'='*60}")

    # Old mapping would have amino acids at 5-24, new mapping at 4-23
    has_token_24 = 24 in actual_tokens
    has_token_4 = 4 in actual_tokens

    if has_token_24:
        print("❌ Found token ID 24 - this suggests OLD token mapping!")
        print("   OLD mapping: A=5, R=6, ..., V=24")
        print("   Dataset needs to be regenerated with updated data_utils.py")
    elif has_token_4 and not has_token_24:
        print("✅ Token ID 4 found, 24 not found - consistent with NEW mapping")
        print("   NEW mapping: L=4, A=5, ..., C=23")
    else:
        print("⚠️  Unusual token distribution - manual inspection needed")

    return len(invalid_samples) == 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "data/processed/processed_dataset.pkl"

    print(f"{'='*60}")
    print("ESM2 DATASET TOKEN VALIDATOR")
    print(f"{'='*60}\n")

    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("\nUsage: python debug_dataset_tokens.py [path/to/processed_dataset.pkl]")
        sys.exit(1)

    is_valid = check_dataset_tokens(dataset_path)

    print(f"\n{'='*60}")
    if is_valid:
        print("✅ VALIDATION PASSED - Dataset is ready for training")
    else:
        print("❌ VALIDATION FAILED - Dataset has invalid token IDs")
        print("\nNext steps:")
        print("1. Verify data_utils.py has the correct ESM2 token mapping")
        print("2. Clear Python cache: find . -type d -name __pycache__ -exec rm -rf {} +")
        print("3. Regenerate dataset: python data_pipeline/process_dataset.py ...")
    print(f"{'='*60}\n")
