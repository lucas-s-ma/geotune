#!/usr/bin/env python
"""
Script to generate 3Di structural tokens from PDB files using Foldseek's direct structure-to-3Di capability.
"""

import os
import subprocess
import tempfile
import argparse
from pathlib import Path
from typing import List, Optional

def check_foldseek_installation() -> bool:
    """
    Check if foldseek is installed and available in PATH.
    """
    try:
        result = subprocess.run(['foldseek'], capture_output=True, text=True, check=False)
        if 'Foldseek' in result.stdout or 'Foldseek' in result.stderr:
            print("Foldseek is available.")
            return True
        else:
            print(f"Foldseek may not be correctly installed. Stdout: {result.stdout}, Stderr: {result.stderr}")
            return False
    except FileNotFoundError:
        print("Foldseek command not found. Please ensure it is installed and in your system's PATH.")
        return False

def generate_foldseek_tokens(pdb_file_path: str) -> Optional[List[int]]:
    """
    Generates 3Di structural tokens from a PDB file using Foldseek.
    """
    if not check_foldseek_installation():
        return None

    pdb_path = Path(pdb_file_path)
    if not pdb_path.exists():
        print(f"Error: PDB file does not exist: {pdb_file_path}")
        return None

    with tempfile.TemporaryDirectory() as temp_dir:
        db_name = pdb_path.stem
        db_path = os.path.join(temp_dir, db_name)
        
        cmd_createdb = ['foldseek', 'createdb', str(pdb_path), db_path]
        print(f"Running Foldseek createdb: {' '.join(cmd_createdb)}")
        result_createdb = subprocess.run(cmd_createdb, capture_output=True, text=True, check=False)

        if result_createdb.returncode != 0:
            print(f"Error creating Foldseek database for {pdb_path.name}.")
            print(f"Stderr: {result_createdb.stderr}")
            return None

        # Link the headers
        ss_db_path = f"{db_path}_ss"
        cmd_lndb = ['foldseek', 'lndb', f"{db_path}_h", f"{ss_db_path}_h"]
        print(f"Running Foldseek lndb: {' '.join(cmd_lndb)}")
        result_lndb = subprocess.run(cmd_lndb, capture_output=True, text=True, check=False)

        if result_lndb.returncode != 0:
            print(f"Error linking database headers for {pdb_path.name}.")
            print(f"Stderr: {result_lndb.stderr}")
            return None

        fasta_path = os.path.join(temp_dir, f"{db_name}_ss.fasta")
        cmd_convert = ['foldseek', 'convert2fasta', ss_db_path, fasta_path]
        print(f"Running Foldseek convert2fasta: {' '.join(cmd_convert)}")
        result_convert = subprocess.run(cmd_convert, capture_output=True, text=True, check=False)

        if result_convert.returncode != 0:
            print(f"Error converting 3Di database to FASTA for {pdb_path.name}.")
            print(f"Stderr: {result_convert.stderr}")
            return None
            
        try:
            with open(fasta_path, 'r') as f:
                lines = f.readlines()
                seq3di = "".join([line.strip() for line in lines if not line.startswith('>')])

            if not seq3di:
                print(f"Foldseek generated an empty 3Di sequence for {pdb_path.name}.")
                return None
            
            tokens = convert_3di_to_ints(seq3di)
            print(f"Successfully generated {len(tokens)} structural tokens for {pdb_path.name}.")
            return tokens

        except FileNotFoundError:
            print(f"Error: Output file not found at {fasta_path}.")
            return None
        except Exception as e:
            print(f"An error occurred while reading the 3Di sequence: {e}")
            return None

def convert_3di_to_ints(ascii_seq: str) -> List[int]:
    """
    Convert 3Di ASCII sequence to integer tokens.

    Foldseek 3Di alphabet uses 20 structural states, represented by the 20 standard
    amino acids, plus 'X' for unknown.
    """
    struct_to_int = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
        'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20
    }
    tokens = []
    for char in ascii_seq:
        token = struct_to_int.get(char.upper(), 20)  # Default to 'X' for unknown
        if char.upper() not in struct_to_int:
            print(f"Warning: Unknown 3Di character '{char}' mapped to token 20 (X)")
        tokens.append(token)
    return tokens

def main():
    parser = argparse.ArgumentParser(
        description="Generate Foldseek 3Di structural tokens directly from a PDB file."
    )
    parser.add_argument(
        "--pdb_file", 
        type=str, 
        required=True, 
        help="Path to the input PDB file."
    )
    
    args = parser.parse_args()
    
    tokens = generate_foldseek_tokens(args.pdb_file)
    
    if tokens:
        print(f"\nSuccessfully generated {len(tokens)} tokens.")
        if len(tokens) > 40:
            print(f"First 20 tokens: {tokens[:20]}")
            print(f"Last 20 tokens: {tokens[-20:]}")
        else:
            print(f"Tokens: {tokens}")
    else:
        print(f"\nFailed to generate structural tokens for {args.pdb_file}.")
        exit(1)

if __name__ == "__main__":
    main()
