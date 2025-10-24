#!/usr/bin/env python
"""
Script to generate Foldseek structural tokens from PDB files
This script calls the Foldseek tool directly using subprocess
"""

import os
import subprocess
import tempfile
import argparse
from pathlib import Path
from typing import List, Optional


def check_foldseek_installation() -> bool:
    """
    Check if foldseek is installed and available in PATH
    
    Returns:
        bool: True if foldseek is available, False otherwise
    """
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


def generate_foldseek_tokens(pdb_file_path: str) -> Optional[List[int]]:
    """
    Generate Foldseek structural tokens from a PDB file
    
    Args:
        pdb_file_path: Path to the input PDB file
    
    Returns:
        List of structural tokens (integers) or None if failed
    """
    if not check_foldseek_installation():
        raise RuntimeError("Foldseek is required but not installed. Please install from https://github.com/steineggerlab/foldseek")
    
    pdb_path = Path(pdb_file_path)
    
    if not pdb_path.exists():
        print(f"Error: PDB file does not exist: {pdb_file_path}")
        return None
    
    # Create a temporary directory for Foldseek output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define output file path
        output_file = os.path.join(temp_dir, f"{pdb_path.stem}_output.alph")
        
        try:
            # Call Foldseek structure2alph command
            cmd = [
                'foldseek', 
                'structure2alph', 
                str(pdb_path), 
                output_file
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Check the output file
            if not os.path.exists(output_file):
                print(f"Error: Foldseek output file was not created: {output_file}")
                return None
            
            # Read and parse the structural tokens from the output file
            tokens = parse_foldseek_output(output_file)
            
            if tokens is None:
                print(f"Error: Could not parse structural tokens from {output_file}")
                return None
            
            print(f"Successfully generated {len(tokens)} structural tokens for {pdb_path.name}")
            return tokens
            
        except subprocess.CalledProcessError as e:
            print(f"Error running Foldseek on {pdb_file_path}: {e}")
            print(f"Stderr: {e.stderr}")
            return None
        except Exception as e:
            print(f"Unexpected error processing {pdb_file_path}: {e}")
            return None


def parse_foldseek_output(output_file: str) -> Optional[List[int]]:
    """
    Parse the Foldseek output file to extract structural tokens.
    The structure2alph command generates a file with structural alphabet representation.
    
    Args:
        output_file: Path to the Foldseek output file
    
    Returns:
        List of structural tokens (integers) or None if parsing failed
    """
    try:
        with open(output_file, 'r') as f:
            content = f.read().strip()
        
        if not content:
            print(f"Warning: Empty output file {output_file}")
            return []
        
        # For Foldseek structure2alph output, the structural alphabet is represented as a sequence
        # of letters that need to be mapped to integers (0-19 for the 20-letter structural alphabet)
        # The output is typically a single line with the structural sequence
        
        # Split the content into lines and get the structural sequence
        lines = content.split('\n')
        struct_seq = ""
        
        # Look for the structural sequence (skip header lines if present)
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('>'):
                struct_seq = line
                break
        
        if not struct_seq:
            print(f"Warning: No structural sequence found in {output_file}")
            return []
        
        # Map structural alphabet letters to integers (0-19)
        # Foldseek uses 20-letter structural alphabet: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
        struct_to_int = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
            'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
        }
        
        # Convert structural sequence to integer tokens
        tokens = []
        for char in struct_seq:
            if char in struct_to_int:
                tokens.append(struct_to_int[char])
            else:
                # Unknown character, map to a default token (e.g., 20)
                tokens.append(20)
        
        return tokens
        
    except Exception as e:
        print(f"Error parsing Foldseek output file {output_file}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate Foldseek structural tokens from PDB files")
    parser.add_argument("--pdb_file", type=str, required=True, 
                        help="Path to input PDB file")
    
    args = parser.parse_args()
    
    if not check_foldseek_installation():
        print("Foldseek is not installed. Please install from https://github.com/steineggerlab/foldseek")
        return 1
    
    print(f"Generating Foldseek structural tokens for {args.pdb_file}")
    
    tokens = generate_foldseek_tokens(args.pdb_file)
    
    if tokens is not None:
        print(f"Generated {len(tokens)} structural tokens:")
        print(tokens)
        if len(tokens) > 50:
            print(f"First 20 tokens: {tokens[:20]}")
            print(f"Last 20 tokens: {tokens[-20:]}")
        else:
            print(f"All tokens: {tokens}")
    else:
        print("Failed to generate structural tokens")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())