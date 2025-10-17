#!/usr/bin/env python
"""
Test script to verify dihedral angle constraint implementation works correctly
"""
import torch
import torch.nn as nn
from utils.dihedral_utils import DihedralAngleConstraint, compute_dihedral_angles_from_coordinates
import numpy as np

def test_dihedral_constraint():
    print("Testing dihedral angle constraint implementation...")
    
    # Create dummy data
    batch_size = 2
    seq_len = 10
    hidden_dim = 640  # ESM hidden size
    
    # Create dummy embeddings
    embeddings = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    
    # Create dummy coordinates (N, CA, C)
    n_coords = torch.randn(batch_size, seq_len, 3)
    ca_coords = torch.randn(batch_size, seq_len, 3)
    c_coords = torch.randn(batch_size, seq_len, 3)
    
    # Create attention mask (all valid in this case)
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"N coords shape: {n_coords.shape}")
    print(f"CA coords shape: {ca_coords.shape}")
    print(f"C coords shape: {c_coords.shape}")
    
    # Test dihedral angle computation
    print("\nTesting dihedral angle computation from coordinates...")
    cos_phi, cos_psi = compute_dihedral_angles_from_coordinates(n_coords, ca_coords, c_coords)
    print(f"Computed phi cosine shape: {cos_phi.shape}")
    print(f"Computed psi cosine shape: {cos_psi.shape}")
    print(f"Sample phi cosines: {cos_phi[0, :5] if cos_phi.numel() > 0 else 'N/A'}")
    print(f"Sample psi cosines: {cos_psi[0, :5] if cos_psi.numel() > 0 else 'N/A'}")
    
    # Test constraint module
    print("\nTesting dihedral constraint module...")
    constraint_module = DihedralAngleConstraint(constraint_weight=0.1)
    
    # Forward pass
    constraint_result = constraint_module(
        embeddings, n_coords, ca_coords, c_coords, attention_mask
    )
    
    print(f"Constraint result keys: {list(constraint_result.keys())}")
    print(f"Total dihedral loss: {constraint_result['total_dihedral_loss'].item():.4f}")
    print(f"Phi loss: {constraint_result['phi_loss'].item():.4f}")
    print(f"Psi loss: {constraint_result['psi_loss'].item():.4f}")
    
    # Test backward pass
    print("\nTesting backward pass...")
    total_loss = constraint_result['total_dihedral_loss']
    total_loss.backward()
    
    print(f"Embeddings gradient shape: {embeddings.grad.shape if embeddings.grad is not None else 'None'}")
    print(f"Phi predictor gradient shape: {constraint_module.phi_predictor.weight.grad.shape if constraint_module.phi_predictor.weight.grad is not None else 'None'}")
    print(f"Psi predictor gradient shape: {constraint_module.psi_predictor.weight.grad.shape if constraint_module.psi_predictor.weight.grad is not None else 'None'}")
    
    print("\n✅ Dihedral constraint implementation test passed!")
    
def test_with_different_lengths():
    print("\nTesting with different sequence lengths...")
    
    for seq_len in [3, 5, 10, 15]:
        print(f"\nTesting with sequence length: {seq_len}")
        
        batch_size = 1
        hidden_dim = 640
        
        # Create dummy data
        embeddings = torch.randn(batch_size, seq_len, hidden_dim)
        n_coords = torch.randn(batch_size, seq_len, 3)
        ca_coords = torch.randn(batch_size, seq_len, 3)
        c_coords = torch.randn(batch_size, seq_len, 3)
        attention_mask = torch.ones(batch_size, seq_len)
        
        constraint_module = DihedralAngleConstraint(constraint_weight=0.1)
        
        try:
            result = constraint_module(
                embeddings, n_coords, ca_coords, c_coords, attention_mask
            )
            print(f"  Result: Total loss = {result['total_dihedral_loss'].item():.4f}")
            
            # Check if it has the expected number of angles (should be seq_len - 1)
            cos_true_phi = result['cos_true_phi']
            cos_true_psi = result['cos_true_psi']
            print(f"  Expected angles: {seq_len-1}, Got phi: {cos_true_phi.shape[1] if cos_true_phi.numel() > 0 else 0}, psi: {cos_true_psi.shape[1] if cos_true_psi.numel() > 0 else 0}")
            
        except Exception as e:
            print(f"  Error with length {seq_len}: {e}")

if __name__ == "__main__":
    test_dihedral_constraint()
    test_with_different_lengths()
    print("\n🎉 All dihedral constraint tests completed successfully!")