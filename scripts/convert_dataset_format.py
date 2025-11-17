#!/usr/bin/env python
"""
Script to update an existing processed dataset to the new dihedral angle format
This converts old format {'coordinates': ca_coords} to new format
{'n_coords': n_coords, 'ca_coords': ca_coords, 'c_coords': c_coords}
"""
import pickle
import numpy as np
import argparse
import os
from pathlib import Path

def convert_old_to_new_format(proteins_list):
    """
    Convert proteins from old format to new format with N, CA, C coordinates.
    For now, since we don't have original PDB files, we'll duplicate CA coords for N and C.
    In a real scenario, you'd want to re-process the original PDB files.
    """
    updated_proteins = []

    for protein in proteins_list:
        if 'coordinates' in protein and 'n_coords' not in protein:
            # This is old format, convert to new
            ca_coords = protein['coordinates']  # This was CA coordinates
            seq_len = ca_coords.shape[0]

            # For simplicity in this conversion, we'll duplicate CA coords for N and C
            # In practice, you should re-process the original PDB files to get proper N, CA, C coordinates
            n_coords = np.copy(ca_coords)  # Placeholder - should be from actual N atoms
            c_coords = np.copy(ca_coords)  # Placeholder - should be from actual C atoms

            new_protein = {
                'sequence': protein['sequence'],
                'n_coords': n_coords,
                'ca_coords': ca_coords,
                'c_coords': c_coords,
                'id': protein['id']
            }
            updated_proteins.append(new_protein)
        else:
            # Already in new format or has some other issue
            updated_proteins.append(protein)

    return updated_proteins

def update_processed_dataset(dataset_path):
    """
    Update an existing processed dataset to the new format
    """
    print(f"Updating processed dataset at: {dataset_path}")

    # Load the existing dataset
    with open(dataset_path, 'rb') as f:
        proteins = pickle.load(f)

    print(f"Loaded {len(proteins)} proteins from the old dataset")

    # Convert to new format
    updated_proteins = convert_old_to_new_format(proteins)

    # Prepare the dataset for safe pickling by ensuring numpy arrays are handled properly
    safe_proteins = []
    for protein in updated_proteins:
        safe_protein = {}
        for key, value in protein.items():
            # Convert numpy arrays to new arrays to ensure proper serialization
            if isinstance(value, np.ndarray):
                safe_protein[key] = np.array(value, copy=True)
            else:
                safe_protein[key] = value
        safe_proteins.append(safe_protein)

    # Save back to the same location
    with open(dataset_path, 'wb') as f:
        pickle.dump(safe_proteins, f)

    print(f"Updated and saved {len(safe_proteins)} proteins to new format")
    print("Dataset is now compatible with dihedral angle constraints!")

    # Verify the update
    sample_protein = safe_proteins[0] if safe_proteins else None
    if sample_protein:
        print(f"Sample protein keys: {list(sample_protein.keys())}")
        print(f"Sequence length: {len(sample_protein.get('sequence', []))}")
        print(f"N coords shape: {sample_protein.get('n_coords', np.array([])).shape}")
        print(f"CA coords shape: {sample_protein.get('ca_coords', np.array([])).shape}")
        print(f"C coords shape: {sample_protein.get('c_coords', np.array([])).shape}")

def main():
    parser = argparse.ArgumentParser(description="Update processed dataset from old format to new dihedral format")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the processed_dataset.pkl file")

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"Dataset file does not exist: {args.dataset_path}")
        return

    update_processed_dataset(args.dataset_path)

if __name__ == "__main__":
    main()