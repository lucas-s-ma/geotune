"""
Data pipeline utilities for protein structure processing
"""

import os
import sys
from pathlib import Path
import pickle
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def validate_processed_data(processed_dir):
    """
    Validate the processed data to ensure it meets requirements
    
    Args:
        processed_dir: Directory containing processed data files
    """
    print(f"Validating processed data in {processed_dir}...")
    
    # Check for the single large dataset file first
    efficient_dataset_file = Path(processed_dir) / "processed_dataset.pkl"
    if efficient_dataset_file.exists():
        print(f"Validating efficient dataset file: {efficient_dataset_file}")
        with open(efficient_dataset_file, 'rb') as f:
            all_proteins = pickle.load(f)
        pkl_files = all_proteins[:10] # Sample from the list of dicts
        source_is_file = True
    else:
        pkl_files = list(Path(processed_dir).glob("*.pkl"))
        source_is_file = False

    if not pkl_files:
        print("No processed files found!")
        return False
    
    valid_count = 0
    invalid_count = 0
    
    # Check first 10 as a sample
    for item in tqdm(pkl_files[:10], desc="Validating samples"):
        try:
            if source_is_file:
                data = item # item is already the data dict
            else:
                with open(item, 'rb') as f:
                    data = pickle.load(f)
                
            # Validate required fields exist (new format with N, CA, C coordinates)
            required_keys = ['sequence', 'n_coords', 'ca_coords', 'c_coords', 'id']
            if all(key in data for key in required_keys):
                # Validate data shapes
                seq_len = len(data['sequence'])
                n_coord_shape = data['n_coords'].shape
                ca_coord_shape = data['ca_coords'].shape
                c_coord_shape = data['c_coords'].shape
                
                if (seq_len > 0 and 
                    n_coord_shape[0] == seq_len and n_coord_shape[1] == 3 and
                    ca_coord_shape[0] == seq_len and ca_coord_shape[1] == 3 and
                    c_coord_shape[0] == seq_len and c_coord_shape[1] == 3):
                    valid_count += 1
                else:
                    print(f"Invalid data shape in {data['id']}: seq_len={seq_len}, "
                          f"n_coord_shape={n_coord_shape}, ca_coord_shape={ca_coord_shape}, c_coord_shape={c_coord_shape}")
                    invalid_count += 1
            else:
                # Also check for old format for backward compatibility
                old_required_keys = ['sequence', 'coordinates', 'id']
                if all(key in data for key in old_required_keys):
                    # Old format detected
                    seq_len = len(data['sequence'])
                    coord_shape = data['coordinates'].shape
                    if seq_len > 0 and coord_shape[0] == seq_len and coord_shape[1] == 3:
                        print(f"Warning: Found data in old format (with 'coordinates' key instead of N/CA/C). "
                              f"Please regenerate the processed dataset to use the new dihedral angle constraints.")
                        valid_count += 1  # Still count as valid but with warning
                    else:
                        invalid_count += 1
                else:
                    print(f"Missing required keys in {data.get('id', 'Unknown ID')}")
                    invalid_count += 1
        except Exception as e:
            print(f"Error validating item {item if not source_is_file else item.get('id', 'Unknown ID')}: {e}")
            invalid_count += 1
    
    print(f"Validation complete: {valid_count} valid, {invalid_count} invalid samples (sampled first 10 of {len(pkl_files)})")
    return valid_count > 0


def three_to_one(three_letter):
    """Convert three-letter amino acid code to one-letter (standalone function)"""
    mapping = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    return mapping.get(three_letter, 'X')


def load_structure_from_pdb(pdb_path, chain_id=None):
    """
    Load structure information directly from a PDB file
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        
        # Get first model
        model = structure[0]
        
        # Select chain if specified, otherwise use first chain
        if chain_id:
            chain = model[chain_id]
        else:
            chain = list(model.get_chains())[0]  # First chain
            
        sequence = ""
        coordinates = []
        residue_ids = []
        
        for residue in chain:
            # Check if it's a protein residue
            if residue.get_resname() in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
                                         'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                                         'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                                         'SER', 'THR', 'TRP', 'TYR', 'VAL']:
                
                aa_code = three_to_one(residue.get_resname())
                if aa_code != 'X':  # Unknown amino acid
                    sequence += aa_code
                    residue_ids.append(residue.get_id()[1])  # Residue number
                    
                    # Get CA coordinate
                    try:
                        ca_atom = residue['CA']
                        coordinates.append(list(ca_atom.get_coord()))
                    except KeyError:
                        # Fallback to other atoms
                        atom_coords = []
                        for atom in residue:
                            if atom.get_name() in ['N', 'CA', 'C', 'O']:
                                atom_coords.append(atom.get_coord())
                        
                        if atom_coords:
                            avg_coord = np.mean(atom_coords, axis=0)
                            coordinates.append(avg_coord.tolist())
                        else:
                            # Use first atom
                            first_atom = list(residue.get_atoms())[0]
                            coordinates.append(list(first_atom.get_coord()))
        
        return {
            'sequence': sequence,
            'coordinates': np.array(coordinates),
            'residue_ids': residue_ids
        }
    
    except Exception as e:
        print(f"Error loading structure from {pdb_path}: {e}")
        return None