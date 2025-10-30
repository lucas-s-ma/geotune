#!/usr/bin/env python
"""
Script to generate Foldseek structural tokens (3Di) from PDB files
"""

import argparse
from typing import List, Optional

from scripts.generate_foldseek_3di import generate_foldseek_tokens, check_foldseek_installation


def main():
    parser = argparse.ArgumentParser(description="Generate Foldseek structural tokens from PDB files")
    parser.add_argument("--pdb_file", type=str, required=True, 
                        help="Path to input PDB file")
    
    args = parser.parse_args()
    
    if not check_foldseek_installation():
        return 1
    
    print(f"Generating structural tokens for {args.pdb_file}")
    
    tokens = generate_foldseek_tokens(args.pdb_file)
    
    if tokens:
        print(f"Generated {len(tokens)} structural tokens:")
        if len(tokens) > 50:
            print(f"First 20: {tokens[:20]}")
            print(f"Last 20: {tokens[-20:]}")
        else:
            print(f"All tokens: {tokens}")
    else:
        print("Failed to generate structural tokens")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
