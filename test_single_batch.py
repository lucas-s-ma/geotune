#!/usr/bin/env python3
"""
Test a single training batch to identify CUDA assert error source.
Run with: CUDA_LAUNCH_BLOCKING=1 python test_single_batch.py
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable synchronous CUDA for better error messages

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.data_utils import EfficientProteinDataset, collate_fn
from torch.utils.data import DataLoader
import yaml

def test_single_batch():
    """Test a single batch through the model to find CUDA error source"""
    print("="*60)
    print("SINGLE BATCH CUDA ERROR TEST")
    print("="*60)

    # Load config
    config_path = project_root / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load dataset
    data_path = config['data']['processed_data_path']
    print(f"\nLoading dataset from: {data_path}")

    dataset = EfficientProteinDataset(
        processed_dataset_path=os.path.join(data_path, "processed_dataset.pkl"),
        max_seq_len=config['training']['max_seq_len'],
        include_structural_tokens=False
    )

    print(f"Dataset size: {len(dataset)}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,  # Don't shuffle so we get same batch each time
        collate_fn=collate_fn,
        num_workers=0  # Single worker to avoid multiprocessing issues
    )

    # Get first batch
    print("\n" + "="*60)
    print("LOADING FIRST BATCH")
    print("="*60)

    batch = next(iter(dataloader))

    print(f"\nBatch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")

    # Check tokens in batch
    input_ids = batch['input_ids']
    print(f"\nInput IDs range: {input_ids.min().item()} - {input_ids.max().item()}")
    print(f"Unique tokens in batch: {torch.unique(input_ids).tolist()}")

    # Check for invalid tokens
    invalid_tokens = input_ids[(input_ids > 32) | ((input_ids > 23) & (input_ids < 32))]
    if len(invalid_tokens) > 0:
        print(f"\n❌ FOUND INVALID TOKENS: {invalid_tokens.tolist()}")
        print("These will cause CUDA assert error!")
        return False
    else:
        print(f"\n✅ All tokens in valid range")

    # Test masking (this is where train.py applies mask token)
    print("\n" + "="*60)
    print("TESTING MASKING")
    print("="*60)

    masked_input_ids = input_ids.clone()
    mask_prob = 0.15
    mask_positions = torch.rand(input_ids.shape, device=input_ids.device) < mask_prob
    # Don't mask padding tokens
    mask_positions = mask_positions & (input_ids != 1)

    print(f"Masking {mask_positions.sum().item()} positions...")
    masked_input_ids[mask_positions] = 32  # ESM2 mask token

    print(f"Masked IDs range: {masked_input_ids.min().item()} - {masked_input_ids.max().item()}")
    print(f"Unique tokens after masking: {torch.unique(masked_input_ids).tolist()}")

    # Check for invalid tokens after masking
    invalid_after_mask = masked_input_ids[(masked_input_ids > 32) | ((masked_input_ids > 23) & (masked_input_ids < 32))]
    if len(invalid_after_mask) > 0:
        print(f"\n❌ INVALID TOKENS AFTER MASKING: {invalid_after_mask.tolist()}")
        return False
    else:
        print(f"\n✅ All tokens valid after masking")

    # Try to load model and run forward pass
    print("\n" + "="*60)
    print("TESTING MODEL FORWARD PASS")
    print("="*60)

    try:
        from transformers import EsmModel, EsmConfig
        from peft import LoraConfig

        # Load ESM model
        print("Loading ESM2 model...")
        model_name = "facebook/esm2_t30_150M_UR50D"
        esm_model = EsmModel.from_pretrained(model_name)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        if not torch.cuda.is_available():
            print("⚠️  CUDA not available - cannot test GPU-specific errors")
            print("   Run this on the cluster with CUDA_LAUNCH_BLOCKING=1")
            return True

        esm_model = esm_model.to(device)
        masked_input_ids = masked_input_ids.to(device)
        attention_mask = batch['attention_mask'].to(device)

        print(f"\nRunning forward pass...")
        print(f"  Input shape: {masked_input_ids.shape}")
        print(f"  Token range: {masked_input_ids.min().item()} - {masked_input_ids.max().item()}")

        with torch.no_grad():
            outputs = esm_model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask
            )

        print(f"\n✅ Forward pass successful!")
        print(f"   Output shape: {outputs.last_hidden_state.shape}")

        return True

    except Exception as e:
        print(f"\n❌ ERROR DURING FORWARD PASS:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Setting CUDA_LAUNCH_BLOCKING=1 for synchronous error reporting\n")

    success = test_single_batch()

    print("\n" + "="*60)
    if success:
        print("✅ TEST PASSED - No CUDA errors detected")
        print("\nIf you're still getting errors during training,")
        print("the issue might be in:")
        print("  1. Loss calculation")
        print("  2. Later batches (try different batch indices)")
        print("  3. Gradient computation")
    else:
        print("❌ TEST FAILED - Found the error source")
    print("="*60 + "\n")
