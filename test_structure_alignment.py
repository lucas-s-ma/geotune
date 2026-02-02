"""
Test script to debug the StructureAlignmentLoss and check for NaN handling issues
"""
import torch
import numpy as np
import pickle
import os
from utils.structure_alignment_utils import StructureAlignmentLoss, PretrainedGNNWrapper
from utils.data_utils import EfficientProteinDataset, collate_fn
from torch.utils.data import DataLoader


def test_structure_alignment_loss():
    print("Testing StructureAlignmentLoss...")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define dimensions
    batch_size = 4
    seq_len = 100
    esm_hidden_size = 320  # ESM model hidden size
    pgnn_hidden_dim = 512  # Pre-computed embedding dimension
    
    # Create random embeddings to test
    pLM_embeddings = torch.randn(batch_size, seq_len, esm_hidden_size, device=device, requires_grad=True)
    pGNN_embeddings = torch.randn(batch_size, seq_len, pgnn_hidden_dim, device=device, requires_grad=True)
    structure_tokens = torch.randint(0, 21, (batch_size, seq_len), device=device)  # 21 structural classes
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    
    print(f"pLM embeddings shape: {pLM_embeddings.shape}")
    print(f"pGNN embeddings shape: {pGNN_embeddings.shape}")
    print(f"Structure tokens shape: {structure_tokens.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Check for NaN in inputs
    print(f"pLM embeddings has NaN: {torch.isnan(pLM_embeddings).any()}")
    print(f"pGNN embeddings has NaN: {torch.isnan(pGNN_embeddings).any()}")
    print(f"Structure tokens has NaN: {torch.isnan(structure_tokens).any()}")
    
    # Create StructureAlignmentLoss module
    structure_alignment_loss = StructureAlignmentLoss(
        hidden_dim=esm_hidden_size,
        pgnn_hidden_dim=pgnn_hidden_dim,
        num_structural_classes=21,
        shared_projection_dim=512,
        latent_weight=0.5,
        physical_weight=0.5
    ).to(device)
    
    print("Forward pass...")
    # Forward pass
    results = structure_alignment_loss(
        pLM_embeddings=pLM_embeddings,
        pGNN_embeddings=pGNN_embeddings,
        structure_tokens=structure_tokens,
        attention_mask=attention_mask
    )
    
    print(f"Total loss: {results['total_loss']}")
    print(f"Latent loss: {results['latent_loss']}")
    print(f"Physical loss: {results['physical_loss']}")
    print(f"Total loss is zero: {results['total_loss'].item() == 0.0}")
    
    # Check if any of the individual components are zero
    print(f"Latent loss is zero: {results['latent_loss'].item() == 0.0}")
    print(f"Physical loss is zero: {results['physical_loss'].item() == 0.0}")
    
    # Test with a batch that might trigger NaN handling
    print("\nTesting with potential NaN values...")
    pLM_nan = pLM_embeddings.clone()
    pLM_nan[0, 0, 0] = float('nan')  # Introduce a NaN
    
    results_nan = structure_alignment_loss(
        pLM_embeddings=pLM_nan,
        pGNN_embeddings=pGNN_embeddings,
        structure_tokens=structure_tokens,
        attention_mask=attention_mask
    )
    
    print(f"With NaN input - Total loss: {results_nan['total_loss']}")
    print(f"With NaN input - Latent loss: {results_nan['latent_loss']}")
    print(f"With NaN input - Physical loss: {results_nan['physical_loss']}")
    
    return results, results_nan


def test_with_real_data():
    print("\n" + "="*60)
    print("Testing with real data from your dataset...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load a small subset of your data
    data_path = "data/processed"  # Adjust path as needed
    dataset = EfficientProteinDataset(
        data_path,
        max_seq_len=128,  # Small for testing
        include_structural_tokens=True,
        load_embeddings=True  # This should load precomputed embeddings
    )
    
    # Create a small dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    
    # Move to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"N coords shape: {batch['n_coords'].shape}")
    print(f"Has precomputed embeddings: {'precomputed_embeddings' in batch}")
    if 'precomputed_embeddings' in batch:
        print(f"Precomputed embeddings shape: {batch['precomputed_embeddings'].shape}")
    print(f"Has structural tokens: {'structural_tokens' in batch}")
    if 'structural_tokens' in batch:
        print(f"Structural tokens shape: {batch['structural_tokens'].shape}")
    
    # Check for NaN in the batch
    if 'precomputed_embeddings' in batch:
        has_nan = torch.isnan(batch['precomputed_embeddings']).any().item()
        print(f"Precomputed embeddings has NaN: {has_nan}")
    
    if 'structural_tokens' in batch:
        has_nan = torch.isnan(batch['structural_tokens']).any().item()
        print(f"Structural tokens has NaN: {has_nan}")
    
    # Test the structure alignment loss with real data
    esm_hidden_size = 320
    pgnn_hidden_dim = 512
    
    structure_alignment_loss = StructureAlignmentLoss(
        hidden_dim=esm_hidden_size,
        pgnn_hidden_dim=pgnn_hidden_dim,
        num_structural_classes=21,
        shared_projection_dim=512,
        latent_weight=0.5,
        physical_weight=0.5
    ).to(device)
    
    # Simulate pLM embeddings (random for now)
    pLM_embeddings = torch.randn(batch['input_ids'].shape[0], batch['input_ids'].shape[1], esm_hidden_size, device=device)
    
    if 'precomputed_embeddings' in batch and 'structural_tokens' in batch:
        print("Computing structure alignment loss with real data...")
        results = structure_alignment_loss(
            pLM_embeddings=pLM_embeddings,
            pGNN_embeddings=batch['precomputed_embeddings'],
            structure_tokens=batch['structural_tokens'],
            attention_mask=batch['attention_mask']
        )
        
        print(f"Real data - Total loss: {results['total_loss']}")
        print(f"Real data - Latent loss: {results['latent_loss']}")
        print(f"Real data - Physical loss: {results['physical_loss']}")
        print(f"Real data - Total loss is zero: {results['total_loss'].item() == 0.0}")
    else:
        print("Could not test with real data - missing required tensors")


if __name__ == "__main__":
    print("Running Structure Alignment Loss Debug Test")
    print("="*60)
    
    # Test with synthetic data
    synthetic_results, nan_results = test_structure_alignment_loss()
    
    # Test with real data if available
    try:
        test_with_real_data()
    except Exception as e:
        print(f"Could not test with real data: {e}")
        print("Make sure your data/processed directory contains the required files.")
    
    print("\nTest completed!")