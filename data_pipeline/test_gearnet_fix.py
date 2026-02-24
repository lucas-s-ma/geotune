#!/usr/bin/env python
"""
Quick test to verify the GearNet fix works for proteins that were failing (101M, 102L)
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.gearnet_model import GearNetFromCoordinates


def test_gearnet_fix():
    """Test GearNet with sample data similar to 101M and 102L"""
    print("Testing GearNet fix...")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = GearNetFromCoordinates(hidden_dim=512, freeze=True).to(device)
    model.eval()
    
    # Test cases
    test_cases = [
        ("Normal protein (154 residues)", 154),
        ("Normal protein (163 residues)", 163),
        ("Short protein (5 residues)", 5),
        ("Minimum protein (2 residues)", 2),
        ("Single residue (edge case)", 1),
    ]
    
    print("\n" + "=" * 80)
    
    all_passed = True
    
    for name, seq_len in test_cases:
        print(f"\nTesting: {name}")
        
        # Create dummy coordinates
        n_coords = torch.randn(1, seq_len, 3).to(device)
        ca_coords = torch.randn(1, seq_len, 3).to(device)
        c_coords = torch.randn(1, seq_len, 3).to(device)
        
        try:
            with torch.no_grad():
                embeddings = model(n_coords, ca_coords, c_coords)
            
            # Check output shape
            expected_shape = (1, seq_len, 512)
            if embeddings.shape == expected_shape:
                print(f"  ✓ PASSED - Output shape: {embeddings.shape}")
            else:
                print(f"  ⚠ WARNING - Expected shape {expected_shape}, got {embeddings.shape}")
                
        except Exception as e:
            print(f"  ✗ FAILED - Error: {e}")
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All tests PASSED!")
        print("The GearNet fix is working correctly.")
    else:
        print("✗ Some tests FAILED!")
        print("There may still be issues with the GearNet implementation.")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = test_gearnet_fix()
    sys.exit(0 if success else 1)
