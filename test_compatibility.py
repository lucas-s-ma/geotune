#!/usr/bin/env python
"""
Test script to verify that process_data.py output is compatible with the trainer
"""
import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from utils.data_utils import ProteinStructureDataset, EfficientProteinDataset, collate_fn
from torch.utils.data import DataLoader


def create_test_pdb_content():
    """Create minimal PDB content for testing"""
    return """HEADER    TEST PROTEIN
TITLE     Test Structure
ATOM      1  N   ALA A   1      20.920  17.210   8.520  1.00 10.00           N  
ATOM      2  CA  ALA A   1      20.840  16.330   9.680  1.00 10.00           C  
ATOM      3  C   ALA A   1      19.550  15.540   9.640  1.00 10.00           C  
ATOM      4  O   ALA A   1      18.780  15.840   8.780  1.00 10.00           O  
ATOM      5  N   GLY A   2      19.320  14.590  10.530  1.00 10.00           N  
ATOM      6  CA  GLY A   2      18.150  13.760  10.640  1.00 10.00           C  
ATOM      7  C   GLY A   2      17.010  14.520  11.320  1.00 10.00           C  
ATOM      8  O   GLY A   2      16.860  15.700  11.640  1.00 10.00           O  
TER
END
"""


def test_original_dataset():
    """Test the original dataset functionality"""
    print("Testing original ProteinStructureDataset...")
    
    # Create a temporary directory and test PDB file
    with tempfile.TemporaryDirectory() as temp_dir:
        pdb_file = os.path.join(temp_dir, "test.pdb")
        with open(pdb_file, 'w') as f:
            f.write(create_test_pdb_content())
        
        # Test original dataset
        try:
            dataset = ProteinStructureDataset(temp_dir)
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"✓ Original dataset works. Sample keys: {list(sample.keys())}")
                
                # Check if the sample structure matches what trainer expects
                expected_keys = ['input_ids', 'attention_mask', 'coordinates', 'seq_len', 'protein_id']
                if all(key in sample for key in expected_keys):
                    print("✓ Original dataset produces expected structure for trainer")
                else:
                    print(f"✗ Original dataset missing expected keys. Got: {list(sample.keys())}")
                    return False
            else:
                print("✗ Original dataset returned empty")
                return False
        except Exception as e:
            print(f"✗ Original dataset failed: {e}")
            return False
    
    return True


def test_efficient_dataset():
    """Test the efficient dataset functionality"""
    print("\nTesting EfficientProteinDataset...")
    
    # First, create some processed data to work with
    with tempfile.TemporaryDirectory() as temp_raw_dir:
        # Create a test PDB
        pdb_file = os.path.join(temp_raw_dir, "test.pdb")
        with open(pdb_file, 'w') as f:
            f.write(create_test_pdb_content())
        
        # Use ProteinStructureDataset to extract data (simulating process_data.py)
        temp_dataset = ProteinStructureDataset(temp_raw_dir)
        
        # Create a temporary processed directory
        with tempfile.TemporaryDirectory() as temp_processed_dir:
            import pickle
            import json
            
            # Save the proteins as if process_data.py did it
            dataset_file = os.path.join(temp_processed_dir, "processed_dataset.pkl")
            with open(dataset_file, 'wb') as f:
                pickle.dump(temp_dataset.proteins, f)
            
            # Create ID mapping
            id_mapping = {i: protein['id'] for i, protein in enumerate(temp_dataset.proteins)}
            mapping_file = os.path.join(temp_processed_dir, "id_mapping.json")
            with open(mapping_file, 'w') as f:
                json.dump(id_mapping, f)
            
            # Now test the EfficientProteinDataset
            try:
                efficient_dataset = EfficientProteinDataset(temp_processed_dir)
                if len(efficient_dataset) > 0:
                    sample = efficient_dataset[0]
                    print(f"✓ Efficient dataset works. Sample keys: {list(sample.keys())}")
                    
                    # Check if the sample structure matches what trainer expects
                    expected_keys = ['input_ids', 'attention_mask', 'coordinates', 'seq_len', 'protein_id']
                    if all(key in sample for key in expected_keys):
                        print("✓ Efficient dataset produces expected structure for trainer")
                    else:
                        print(f"✗ Efficient dataset missing expected keys. Got: {list(sample.keys())}")
                        return False
                else:
                    print("✗ Efficient dataset returned empty")
                    return False
            except Exception as e:
                print(f"✗ Efficient dataset failed: {e}")
                return False
    
    return True


def test_collate_function():
    """Test the collate function used in DataLoader"""
    print("\nTesting collate function...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        pdb_file = os.path.join(temp_dir, "test.pdb")
        with open(pdb_file, 'w') as f:
            f.write(create_test_pdb_content())
        
        dataset = ProteinStructureDataset(temp_dir)
        
        if len(dataset) > 0:
            # Test with a small batch
            batch = [dataset[0] for _ in range(2)]  # Create a batch of 2 identical samples
            try:
                batched = collate_fn(batch)
                print(f"✓ Collate function works. Batch keys: {list(batched.keys())}")
                
                # Expected batched format
                expected_batch_keys = ['input_ids', 'attention_mask', 'coordinates', 'seq_lens', 'protein_ids']
                if all(key in batched for key in expected_batch_keys):
                    print("✓ Collate function produces expected batch format for trainer")
                else:
                    print(f"✗ Collate function missing expected keys. Got: {list(batched.keys())}")
                    return False
            except Exception as e:
                print(f"✗ Collate function failed: {e}")
                return False
        else:
            print("✗ Dataset is empty for collate test")
            return False
    
    return True


def main():
    print("Testing compatibility between process_data.py output and trainer script...\n")
    
    success = True
    success &= test_original_dataset()
    success &= test_efficient_dataset()
    success &= test_collate_function()
    
    print(f"\n{'='*50}")
    if success:
        print("✓ All compatibility tests passed!")
        print("The process_data.py output and both dataset classes are compatible with the trainer script.")
    else:
        print("✗ Some compatibility tests failed!")
    print(f"{'='*50}")
    
    return success


if __name__ == "__main__":
    main()