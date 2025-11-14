#!/usr/bin/env python
"""
Script to fix structural token alignment issues by ensuring sequence and token lengths match
"""
import pickle
import numpy as np
import os
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from utils.data_utils import three_to_one


def clean_structural_tokens(processed_dataset_path, structural_tokens_path, output_path):
    """
    Clean structural tokens to ensure they align with the amino acid sequence
    """
    print("Loading processed dataset and structural tokens...")
    
    # Load the datasets
    with open(processed_dataset_path, 'rb') as f:
        proteins = pickle.load(f)
    
    with open(structural_tokens_path, 'rb') as f:
        structural_tokens = pickle.load(f)
    
    print(f"Loaded {len(proteins)} proteins and {len(structural_tokens)} structural token sets")
    
    # Create a mapping for quick lookup
    token_dict = {}
    for item in structural_tokens:
        token_dict[item['protein_id']] = item['structural_tokens']
    
    cleaned_structural_tokens = []
    valid_count = 0
    invalid_count = 0
    
    for protein in tqdm(proteins, desc="Processing proteins"):
        protein_id = protein['id']
        sequence = protein['sequence']
        seq_len = len(sequence)
        
        if protein_id not in token_dict:
            print(f"Warning: No structural tokens found for protein {protein_id}")
            invalid_count += 1
            continue
        
        struct_tokens = token_dict[protein_id]
        token_len = len(struct_tokens)
        
        if seq_len != token_len:
            print(f"Length mismatch for {protein_id}: sequence len={seq_len}, tokens len={token_len}")
            
            # Truncate or pad tokens to match sequence length
            if token_len > seq_len:
                # Truncate tokens to match sequence
                struct_tokens = struct_tokens[:seq_len]
                print(f"  Truncated tokens from {token_len} to {seq_len}")
            else:
                # Pad tokens with a special value (e.g., 20 which might represent 'unknown')
                padding_needed = seq_len - token_len
                struct_tokens = struct_tokens + [20] * padding_needed  # Use 20 as 'unknown' token
                print(f"  Padded tokens from {token_len} to {seq_len}")
        
        # Validate token values are in range [0, 20] (for 21 classes)
        token_array = np.array(struct_tokens)
        invalid_tokens = np.where((token_array < 0) | (token_array > 20))[0]
        
        if len(invalid_tokens) > 0:
            print(f"Invalid tokens found for {protein_id}: {invalid_tokens[:10]}... (first 10)")
            invalid_count += 1
            continue
        
        # Add to cleaned list
        cleaned_structural_tokens.append({
            'protein_id': protein_id,
            'structural_tokens': struct_tokens
        })
        valid_count += 1
    
    print(f"\nCleaning Results:")
    print(f"  Valid proteins after cleaning: {valid_count}")
    print(f"  Invalid proteins skipped: {invalid_count}")
    
    # Save cleaned structural tokens
    with open(output_path, 'wb') as f:
        pickle.dump(cleaned_structural_tokens, f)
    
    print(f"Cleaned structural tokens saved to {output_path}")
    
    return cleaned_structural_tokens


def regenerate_structural_tokens_for_problematic_pdbs(processed_dataset_path, pdb_directory, output_path):
    """
    Alternative approach: regenerate structural tokens by ensuring proper alignment
    """
    print(f"Regenerating structural tokens from PDB directory: {pdb_directory}")
    
    # Load the processed dataset
    with open(processed_dataset_path, 'rb') as f:
        proteins = pickle.load(f)
    
    print(f"Loaded {len(proteins)} proteins")
    
    cleaned_structural_tokens = []
    valid_count = 0
    invalid_count = 0
    
    for protein in tqdm(proteins, desc="Processing proteins from PDBs"):
        protein_id = protein['id']
        sequence = protein['sequence']
        seq_len = len(sequence)
        
        # Find corresponding PDB file
        pdb_file = None
        for ext in ['.pdb', '.ent']:
            potential_file = os.path.join(pdb_directory, f"{protein_id}{ext}")
            if os.path.exists(potential_file):
                pdb_file = potential_file
                break
        
        if not pdb_file or not os.path.exists(pdb_file):
            print(f"Warning: PDB file not found for protein {protein_id}")
            invalid_count += 1
            continue
        
        try:
            # Parse PDB and extract structural information
            dssp_tokens = extract_dssp_tokens(pdb_file)
            
            if not dssp_tokens:
                print(f"Warning: Could not extract DSSP tokens for {protein_id}")
                invalid_count += 1
                continue
            
            # Align DSSP tokens with sequence - only keep tokens for residues that match the sequence
            aligned_tokens = align_dssp_with_sequence(pdb_file, sequence, dssp_tokens)
            
            if not aligned_tokens or len(aligned_tokens) != seq_len:
                print(f"Alignment failed for {protein_id}: seq_len={seq_len}, aligned_len={len(aligned_tokens) if aligned_tokens else 0}")
                invalid_count += 1
                continue
            
            # Add to cleaned list
            cleaned_structural_tokens.append({
                'protein_id': protein_id,
                'structural_tokens': aligned_tokens
            })
            valid_count += 1
            
        except Exception as e:
            print(f"Error processing {protein_id}: {e}")
            invalid_count += 1
            continue
    
    print(f"\nRegeneration Results:")
    print(f"  Valid proteins after regeneration: {valid_count}")
    print(f"  Invalid proteins skipped: {invalid_count}")
    
    # Save cleaned structural tokens
    with open(output_path, 'wb') as f:
        pickle.dump(cleaned_structural_tokens, f)
    
    print(f"Regenerated structural tokens saved to {output_path}")
    
    return cleaned_structural_tokens


def extract_dssp_tokens(pdb_file):
    """
    Extract DSSP structural tokens from a PDB file
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        
        # Get the first model
        model = structure[0]
        
        # Run DSSP
        try:
            dssp = DSSP(model, pdb_file)
        except Exception as e:
            print(f"Error running DSSP on {pdb_file}: {e}")
            return None
        
        # Extract structural information
        dssp_tokens = []
        for i, (res_id, dssp_data) in enumerate(dssp):
            # dssp_data[2] is secondary structure (SS), dssp_data[1] is accessible surface area (ACC)
            # We can use secondary structure information or other DSSP outputs
            # Map to structural alphabet tokens (simplified approach)
            ss = dssp_data[2]  # Secondary structure
            # Map secondary structure to tokens (example mapping)
            ss_to_token = {
                'H': 0,  # Helix
                'E': 1,  # Strand
                'C': 2,  # Coil
                'T': 3,  # Turn
                'S': 4,  # Bend
                'G': 5,  # 3-10 helix
                'I': 6,  # Pi helix
                'B': 7,  # Bridge
            }
            
            token = ss_to_token.get(ss, 2)  # Default to coil (2) if not mapped
            dssp_tokens.append(token)
        
        return dssp_tokens
        
    except Exception as e:
        print(f"Error extracting DSSP tokens from {pdb_file}: {e}")
        return None


def align_dssp_with_sequence(pdb_file, sequence, dssp_tokens):
    """
    Align DSSP tokens with the amino acid sequence
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        model = structure[0]
        
        # Get sequence from PDB file (structure)
        pdb_sequence = ""
        for chain in model:
            for residue in chain:
                if residue.get_resname() in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                                           'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                                           'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                                           'SER', 'THR', 'TRP', 'TYR', 'VAL']:
                    aa_code = three_to_one(residue.get_resname())
                    if aa_code != 'X':  # Unknown amino acid
                        pdb_sequence += aa_code

        # Align the sequence from PDB with the provided sequence
        # This is a simplified alignment approach - you might want to use a proper sequence alignment algorithm
        if len(pdb_sequence) != len(dssp_tokens):
            print(f"Length mismatch: pdb_seq={len(pdb_sequence)}, dssp_tokens={len(dssp_tokens)}")
            return None

        # Check if sequences match or are similar enough to align
        # For now, we'll return the dssp_tokens if lengths match
        if len(sequence) <= len(dssp_tokens):
            # Truncate DSSP tokens to match sequence length
            return dssp_tokens[:len(sequence)]
        else:
            # This shouldn't happen if DSSP tokens come from the same structure as the sequence
            print(f"Warning: Sequence is longer than PDB structure for alignment")
            return dssp_tokens[:len(pdb_sequence)]
        
    except Exception as e:
        print(f"Error aligning DSSP with sequence for {pdb_file}: {e}")
        return None


if __name__ == "__main__":
    # Paths - adjust these to your setup
    processed_dataset_path = input("Enter path to processed_dataset.pkl (default: data/processed/processed_dataset.pkl): ").strip()
    if not processed_dataset_path:
        processed_dataset_path = "data/processed/processed_dataset.pkl"
    
    structural_tokens_path = input("Enter path to structural_tokens.pkl (default: data/processed/structural_tokens.pkl): ").strip()
    if not structural_tokens_path:
        structural_tokens_path = "data/processed/structural_tokens.pkl"
    
    output_path = input("Enter output path for cleaned tokens (default: data/processed/cleaned_structural_tokens.pkl): ").strip()
    if not output_path:
        output_path = "data/processed/cleaned_structural_tokens.pkl"
    
    # Choose cleaning method
    print("\nChoose cleaning method:")
    print("1. Clean existing tokens (truncate/pad to match sequence lengths)")
    print("2. Regenerate tokens from PDB files using DSSP (better alignment)")
    choice = input("Enter choice (1 or 2, default 1): ").strip() or "1"
    
    if choice == "2":
        pdb_directory = input("Enter path to PDB directory: ").strip()
        cleaned_tokens = regenerate_structural_tokens_for_problematic_pdbs(
            processed_dataset_path,
            pdb_directory,
            output_path
        )
    else:
        cleaned_tokens = clean_structural_tokens(
            processed_dataset_path,
            structural_tokens_path,
            output_path
        )
    
    print(f"\nDone! Cleaned tokens saved to {output_path}")