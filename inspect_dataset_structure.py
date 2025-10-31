#!/usr/bin/env python3
"""
Inspect the actual structure of processed_dataset.pkl
"""
import pickle
import sys
from pathlib import Path

def inspect_dataset(dataset_path):
    print(f"Loading: {dataset_path}\n")

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Dataset type: {type(data)}")
    print(f"Dataset length: {len(data)}")
    print("\n" + "="*60)
    print("FIRST ITEM STRUCTURE")
    print("="*60)

    first_item = data[0]
    print(f"Item type: {type(first_item)}")
    print(f"\nKeys in first item:")

    if isinstance(first_item, dict):
        for key in first_item.keys():
            value = first_item[key]
            if hasattr(value, 'shape'):
                print(f"  '{key}': {type(value).__name__} shape={value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"  '{key}': {type(value).__name__} len={len(value)}")
            else:
                print(f"  '{key}': {type(value).__name__} = {str(value)[:100]}")
    else:
        print(f"First item: {first_item}")

    print("\n" + "="*60)
    print("CHECKING FOR TOKENS")
    print("="*60)

    # Check if any items have token_ids
    has_token_ids = sum(1 for item in data if 'token_ids' in item)
    has_input_ids = sum(1 for item in data if 'input_ids' in item)
    has_sequence = sum(1 for item in data if 'sequence' in item)

    print(f"Items with 'token_ids': {has_token_ids}/{len(data)}")
    print(f"Items with 'input_ids': {has_input_ids}/{len(data)}")
    print(f"Items with 'sequence': {has_sequence}/{len(data)}")

    if has_sequence > 0:
        # Show a sample sequence
        for item in data:
            if 'sequence' in item:
                seq = item['sequence']
                print(f"\nSample sequence: {seq[:100]}...")
                print(f"Sequence length: {len(seq)}")
                break

    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    if has_token_ids == 0:
        print("❌ CRITICAL: No 'token_ids' field found!")
        print("   This dataset was NOT created with sequence_to_tokens()")
        print("   The dataset might be using a different format")
        print("\n   Possible causes:")
        print("   1. Wrong processing script was used")
        print("   2. Dataset is for a different model/pipeline")
        print("   3. Data processing didn't complete successfully")
        print("\n   Solution: Check process_data.py and regenerate dataset")
    else:
        print(f"✅ Found 'token_ids' in {has_token_ids} items")

if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/processed_dataset.pkl"

    if not Path(dataset_path).exists():
        print(f"❌ File not found: {dataset_path}")
        sys.exit(1)

    inspect_dataset(dataset_path)
