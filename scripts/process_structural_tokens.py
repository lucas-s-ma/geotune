#!/usr/bin/env python
"""
Script to process protein structures to generate structural tokens using AlphaFold/PDB data
This script creates structural tokens that can be used in the structure alignment loss
"""

import os
import sys
from pathlib import Path
import pickle
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import argparse
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_dssp_assignment(structure, model_id=0):
    """
    Get DSSP secondary structure assignments which can be used as structural tokens
    """
    try:
        model = structure[model_id]
        dssp = DSSP(model, structure.filename)  # Note: needs the original filename
        
        ss_assignments = []
        for i in range(len(dssp)):
            # DSSP codes: H=helix, E=extended (strand), C=coil/loop, S=bend, T=turn, G=3-10 helix
            ss_code = dssp[i][2]  # Secondary structure code
            ss_assignments.append(ss_code)
        
        return ss_assignments
    except Exception as e:
        print(f"DSSP computation failed: {e}")
        return None

def get_simplified_ss_tokens(ss_assignments):
    """
    Convert DSSP assignments to simplified structural tokens (0-20)
    """
    # Map secondary structure to tokens
    ss_to_token = {
        'H': 0,  # Helix
        'E': 1,  # Strand
        'C': 2,  # Coil
        'T': 3,  # Turn
        'S': 4,  # Bend
        'G': 5,  # 3-10 helix
        'I': 6,  # Pi helix
        'B': 7   # Bridge
    }
    
    tokens = []
    for ss in ss_assignments:
        token = ss_to_token.get(ss, 2)  # Default to coil if unknown
        tokens.append(token)
    
    return tokens

def get_contact_based_tokens(pdb_path, distance_threshold=8.0):
    """
    Generate structural tokens based on contact patterns (simplified approach)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    model = structure[0]  # First model
    
    # Extract CA coordinates for contact calculation
    ca_coords = []
    residues = []
    
    for chain in model:
        for residue in chain:
            if residue.get_resname() in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
                                         'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                                         'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                                         'SER', 'THR', 'TRP', 'TYR', 'VAL']:
                try:
                    ca_atom = residue['CA']
                    ca_coords.append(ca_atom.get_coord())
                    residues.append(residue)
                except KeyError:
                    continue  # Skip residues without CA atom
    
    if len(ca_coords) == 0:
        return []
    
    # Calculate contact frequency for each residue
    ca_coords = np.array(ca_coords)
    n_res = len(ca_coords)
    
    contact_counts = []
    for i in range(n_res):
        distances = np.linalg.norm(ca_coords[i] - ca_coords, axis=1)
        contacts = np.sum(distances < distance_threshold) - 1  # Exclude self
        contact_counts.append(contacts)
    
    # Discretize contact counts into tokens (0-20)
    max_contacts = max(contact_counts) if contact_counts else 1
    tokens = [int(c * 19 / max_contacts) if max_contacts > 0 else 0 for c in contact_counts]
    
    return tokens

def process_pdb_file(pdb_path: str, method: str = "dssp") -> List[int]:
    """
    Process a single PDB file to generate structural tokens
    
    Args:
        pdb_path: Path to PDB file
        method: Method to use ('dssp', 'contact', or 'simple')
    
    Returns:
        List of structural tokens
    """
    if method == "dssp":
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_path)
            structure.filename = pdb_path  # Required for DSSP
            
            ss_assignments = get_dssp_assignment(structure)
            if ss_assignments:
                return get_simplified_ss_tokens(ss_assignments)
        except Exception:
            pass
    
    elif method == "contact":
        return get_contact_based_tokens(pdb_path)
    
    elif method == "simple":
        # Simple approach: use secondary structure elements based on geometry
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        model = structure[0]
        
        tokens = []
        for chain in model:
            for residue in chain:
                if residue.get_resname() in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
                                             'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                                             'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                                             'SER', 'THR', 'TRP', 'TYR', 'VAL']:
                    # For simplicity, assign tokens based on residue position in sequence
                    tokens.append(len(tokens) % 20)  # Simple modulo-based token
        
        return tokens
    
    # Fallback: return an empty list
    print(f"Could not process {pdb_path} with {method} method")
    return []

def process_pdb_directory(pdb_dir: str, output_pickle: str, method: str = "dssp"):
    """
    Process all PDB files in a directory to generate structural tokens
    
    Args:
        pdb_dir: Directory containing PDB files
        output_pickle: Path to save structural tokens as pickle file
        method: Method for generating tokens
    """
    print(f"Processing PDB files in {pdb_dir} using {method} method")
    
    # Find all PDB files
    pdb_files = []
    for file in os.listdir(pdb_dir):
        if file.lower().endswith('.pdb'):
            pdb_files.append(os.path.join(pdb_dir, file))
    
    print(f"Found {len(pdb_files)} PDB files")
    
    all_structural_tokens = {}
    
    for i, pdb_file in enumerate(pdb_files):
        print(f"Processing {os.path.basename(pdb_file)} ({i+1}/{len(pdb_files)})")
        
        try:
            tokens = process_pdb_file(pdb_file, method)
            
            if tokens:
                # Use PDB filename (without extension) as key
                protein_id = os.path.splitext(os.path.basename(pdb_file))[0]
                all_structural_tokens[protein_id] = {
                    'structural_tokens': tokens,
                    'protein_id': protein_id
                }
                print(f"  Generated {len(tokens)} structural tokens")
            else:
                print(f"  Failed to generate tokens for {pdb_file}")
                
        except Exception as e:
            print(f"  Error processing {pdb_file}: {e}")
            continue
    
    # Save structural tokens to pickle file
    os.makedirs(os.path.dirname(output_pickle), exist_ok=True)
    with open(output_pickle, 'wb') as f:
        pickle.dump(list(all_structural_tokens.values()), f)  # Save as list to match dataset format
    print(f"Structural tokens saved to {output_pickle} for {len(all_structural_tokens)} proteins")
    
    return all_structural_tokens

def main():
    parser = argparse.ArgumentParser(description="Generate structural tokens from PDB files")
    parser.add_argument("--pdb_dir", type=str, required=True, 
                        help="Directory containing PDB files to process")
    parser.add_argument("--output_pickle", type=str, default="data/processed/structural_tokens.pkl",
                        help="Path to save structural tokens as pickle file")
    parser.add_argument("--method", type=str, choices=["dssp", "contact", "simple"], 
                        default="contact", help="Method for generating tokens")
    
    args = parser.parse_args()
    
    print("Starting structural token generation...")
    
    # Process all PDB files
    structural_tokens = process_pdb_directory(
        args.pdb_dir,
        args.output_pickle,
        args.method
    )
    
    print(f"Generated structural tokens for {len(structural_tokens)} proteins")
    
    # Print some statistics
    if structural_tokens:
        token_lengths = [len(data['structural_tokens']) for data in structural_tokens.values()]
        print(f"Average token sequence length: {np.mean(token_lengths):.2f}")
        print(f"Min token sequence length: {np.min(token_lengths)}")
        print(f"Max token sequence length: {np.max(token_lengths)}")
        print("First few protein IDs:", list(structural_tokens.keys())[:5])

if __name__ == "__main__":
    main()