#!/usr/bin/env python
"""
Script to process protein structures with Foldseek to generate structural tokens
This script will convert PDB files to Foldseek structural alphabet tokens
"""

import os
import subprocess
import sys
import pickle
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict

def check_foldseek_installation():
    """Check if foldseek is installed and available in PATH"""
    try:
        result = subprocess.run(['foldseek', '--version'], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("Foldseek is available")
            return True
        else:
            print("Foldseek is not available. Please install foldseek from: https://github.com/steineggerlab/foldseek")
            return False
    except FileNotFoundError:
        print("Foldseek is not found in your system. Please install foldseek from: https://github.com/steineggerlab/foldseek")
        return False

def run_foldseek_structure_search(pdb_file: str, output_dir: str, foldseek_db: str = None):
    """
    Run foldseek search to convert structure to structural alphabet
    
    Args:
        pdb_file: Path to input PDB file
        output_dir: Directory to store results
        foldseek_db: Path to foldseek database (optional)
    """
    if not check_foldseek_installation():
        raise RuntimeError("Foldseek is required but not installed. Please install from https://github.com/steineggerlab/foldseek")
    
    # Create database from PDB file first
    pdb_base = os.path.splitext(os.path.basename(pdb_file))[0]
    db_path = os.path.join(output_dir, f"{pdb_base}_db")
    
    # Create foldseek database
    subprocess.run([
        'foldseek', 'createdb', pdb_file, db_path
    ], check=True)
    
    # Convert to structural alphabet (3di format)
    output_file = os.path.join(output_dir, f"{pdb_base}_3di")
    subprocess.run([
        'foldseek', 'convertalis', db_path, db_path, output_file, 
        '--format-output', 'query,target,f,qseq,tseq'
    ], check=True)
    
    return output_file

def parse_foldseek_output_to_tokens(output_file: str) -> Dict[str, List[int]]:
    """
    Parse Foldseek output to extract structural alphabet tokens
    Converts 3di (3D intermediate) format to integer tokens
    
    Args:
        output_file: Path to foldseek output file
    
    Returns:
        Dictionary mapping protein IDs to their structural token sequences
    """
    # Map 3di characters to integer tokens (20 standard tokens)
    char_to_token = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
        'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        'X': 20  # Unknown/other
    }
    
    protein_tokens = {}
    
    with open(output_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                query_id = parts[0]
                target_id = parts[1]
                query_seq = parts[2]
                target_seq = parts[3]
                
                # Convert target sequence (structural alphabet) to tokens
                struct_tokens = [char_to_token.get(char, 20) for char in target_seq]
                
                protein_tokens[query_id] = struct_tokens
    
    return protein_tokens

def process_pdb_directory_with_foldseek(pdb_dir: str, output_dir: str, output_pickle: str = None):
    """
    Process all PDB files in a directory with Foldseek
    
    Args:
        pdb_dir: Directory containing PDB files
        output_dir: Directory to store intermediate results
        output_pickle: Path to save structural tokens as pickle file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDB files in the directory
    pdb_files = []
    for file in os.listdir(pdb_dir):
        if file.lower().endswith('.pdb'):
            pdb_files.append(os.path.join(pdb_dir, file))
    
    print(f"Found {len(pdb_files)} PDB files to process")
    
    all_structural_tokens = {}
    
    for i, pdb_file in enumerate(pdb_files):
        print(f"Processing {pdb_file} ({i+1}/{len(pdb_files)})")
        
        try:
            # Run foldseek on the PDB file
            output_file = run_foldseek_structure_search(pdb_file, output_dir)
            
            # Parse the output to get structural tokens
            tokens = parse_foldseek_output_to_tokens(output_file)
            
            # Add to the overall dictionary
            all_structural_tokens.update(tokens)
            
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
            continue
    
    # Save structural tokens to pickle file if requested
    if output_pickle:
        os.makedirs(os.path.dirname(output_pickle), exist_ok=True)
        with open(output_pickle, 'wb') as f:
            pickle.dump(all_structural_tokens, f)
        print(f"Structural tokens saved to {output_pickle}")
    
    return all_structural_tokens

def main():
    parser = argparse.ArgumentParser(description="Process PDB files with Foldseek to generate structural tokens")
    parser.add_argument("--pdb_dir", type=str, required=True, 
                        help="Directory containing PDB files to process")
    parser.add_argument("--output_dir", type=str, default="foldseek_output", 
                        help="Directory to store foldseek intermediate results")
    parser.add_argument("--output_pickle", type=str, default="data/processed/structural_tokens.pkl",
                        help="Path to save structural tokens as pickle file")
    
    args = parser.parse_args()
    
    print("Starting Foldseek structural token generation...")
    
    # Process all PDB files
    structural_tokens = process_pdb_directory_with_foldseek(
        args.pdb_dir, 
        args.output_dir, 
        args.output_pickle
    )
    
    print(f"Generated structural tokens for {len(structural_tokens)} proteins")
    
    # Print some statistics
    token_lengths = [len(tokens) for tokens in structural_tokens.values()]
    print(f"Average token sequence length: {np.mean(token_lengths):.2f}")
    print(f"Min token sequence length: {np.min(token_lengths)}")
    print(f"Max token sequence length: {np.max(token_lengths)}")

if __name__ == "__main__":
    main()