#!/usr/bin/env python
"""
Diagnostic script to check for corrupted or invalid protein files in processed_dataset.pkl
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_protein_files(processed_dir: str, max_to_check: int = None):
    """
    Check all protein files for validity
    
    Args:
        processed_dir: Directory containing processed_dataset.pkl
        max_to_check: Maximum number of proteins to check (None = all)
    """
    dataset_file = os.path.join(processed_dir, "processed_dataset.pkl")
    
    if not os.path.exists(dataset_file):
        print(f"ERROR: Dataset file not found at {dataset_file}")
        return
    
    print(f"Loading dataset from {dataset_file}...")
    with open(dataset_file, 'rb') as f:
        proteins = pickle.load(f)
    
    total = len(proteins)
    if max_to_check:
        proteins = proteins[:max_to_check]
        print(f"Checking first {max_to_check} of {total} proteins...")
    else:
        print(f"Checking all {total} proteins...")
    
    print("\n" + "=" * 80)
    
    valid_count = 0
    invalid_count = 0
    issues = []
    
    for idx, protein in enumerate(proteins):
        protein_id = protein.get('id', f'idx_{idx}')
        issue = None
        
        # Check required keys
        required_keys = ['sequence', 'n_coords', 'ca_coords', 'c_coords']
        for key in required_keys:
            if key not in protein:
                issue = f"Missing key: {key}"
                break
        
        if not issue:
            sequence = protein['sequence']
            n_coords = protein['n_coords']
            ca_coords = protein['ca_coords']
            c_coords = protein['c_coords']
            
            # Check sequence length
            seq_len = len(sequence)
            if seq_len == 0:
                issue = "Empty sequence"
            elif seq_len < 2:
                issue = f"Sequence too short (length={seq_len}), need at least 2 residues"
            
            # Check coordinate shapes
            if not issue:
                if len(n_coords) != seq_len:
                    issue = f"n_coords length ({len(n_coords)}) != sequence length ({seq_len})"
                elif len(ca_coords) != seq_len:
                    issue = f"ca_coords length ({len(ca_coords)}) != sequence length ({seq_len})"
                elif len(c_coords) != seq_len:
                    issue = f"c_coords length ({len(c_coords)}) != sequence length ({seq_len})"
            
            # Check for NaN or Inf values
            if not issue:
                if np.any(np.isnan(n_coords)):
                    issue = "n_coords contains NaN values"
                elif np.any(np.isnan(ca_coords)):
                    issue = "ca_coords contains NaN values"
                elif np.any(np.isnan(c_coords)):
                    issue = "c_coords contains NaN values"
                
                if np.any(np.isinf(n_coords)):
                    issue = "n_coords contains Inf values"
                elif np.any(np.isinf(ca_coords)):
                    issue = "ca_coords contains Inf values"
                elif np.any(np.isinf(c_coords)):
                    issue = "c_coords contains Inf values"
            
            # Check for zero coordinates (potential padding or corruption)
            if not issue and seq_len > 0:
                zero_rows_ca = np.all(ca_coords == 0, axis=1).sum()
                if zero_rows_ca > 0:
                    issue = f"ca_coords has {zero_rows_ca} rows of all zeros (potential corruption)"
        
        if issue:
            invalid_count += 1
            issues.append((idx, protein_id, issue))
            print(f"  [{idx}] {protein_id}: ❌ {issue}")
        else:
            valid_count += 1
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total checked: {valid_count + invalid_count}")
    print(f"Valid: {valid_count} ({valid_count/(valid_count+invalid_count)*100:.1f}%)")
    print(f"Invalid: {invalid_count} ({invalid_count/(valid_count+invalid_count)*100:.1f}%)")
    
    if issues:
        print("\n" + "=" * 80)
        print("INVALID PROTEINS (first 20)")
        print("=" * 80)
        for idx, protein_id, issue in issues[:20]:
            print(f"  [{idx}] {protein_id}: {issue}")
        
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
        
        # Save full list to file
        issues_file = os.path.join(processed_dir, "invalid_proteins.txt")
        with open(issues_file, 'w') as f:
            f.write("Invalid Proteins Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total invalid: {len(issues)}\n\n")
            for idx, protein_id, issue in issues:
                f.write(f"[{idx}] {protein_id}: {issue}\n")
        print(f"\nFull list saved to: {issues_file}")
    
    # Check specifically for the proteins that failed
    print("\n" + "=" * 80)
    print("CHECKING SPECIFIC FAILED PROTEINS (101M, 102L)")
    print("=" * 80)
    
    for target_id in ['101M', '102L']:
        found = False
        for idx, protein in enumerate(proteins):
            if protein.get('id') == target_id:
                found = True
                print(f"\n{target_id} (index {idx}):")
                print(f"  Sequence length: {len(protein.get('sequence', ''))}")
                print(f"  Has 'sequence': {'sequence' in protein}")
                print(f"  Has 'n_coords': {'n_coords' in protein}")
                print(f"  Has 'ca_coords': {'ca_coords' in protein}")
                print(f"  Has 'c_coords': {'c_coords' in protein}")
                
                if 'n_coords' in protein:
                    coords = protein['n_coords']
                    print(f"  n_coords shape: {coords.shape if hasattr(coords, 'shape') else 'N/A'}")
                    print(f"  n_coords dtype: {coords.dtype if hasattr(coords, 'dtype') else 'N/A'}")
                    if hasattr(coords, 'shape') and len(coords.shape) != 2:
                        print(f"  ⚠️  n_coords has wrong number of dimensions!")
                
                if 'ca_coords' in protein:
                    coords = protein['ca_coords']
                    print(f"  ca_coords shape: {coords.shape if hasattr(coords, 'shape') else 'N/A'}")
                    if hasattr(coords, 'shape') and coords.shape[0] < 2:
                        print(f"  ⚠️  ca_coords has only {coords.shape[0]} residues - TOO FEW for edge computation!")
                
                break
        
        if not found:
            print(f"{target_id}: Not found in dataset")
    
    return len(issues)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check protein files for corruption")
    parser.add_argument("--processed_dir", type=str, default="data/processed",
                        help="Directory containing processed_dataset.pkl")
    parser.add_argument("--max_to_check", type=int, default=None,
                        help="Maximum number of proteins to check (default: all)")
    
    args = parser.parse_args()
    
    invalid_count = check_protein_files(args.processed_dir, args.max_to_check)
    
    if invalid_count > 0:
        print(f"\n⚠️  Found {invalid_count} invalid proteins!")
        print("These proteins should be removed or fixed before generating embeddings.")
        sys.exit(1)
    else:
        print("\n✓ All proteins are valid!")
        sys.exit(0)
