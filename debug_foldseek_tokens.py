#!/usr/bin/env python
"""
Debug script to check structural token alignment and validity
"""
import pickle
import numpy as np
import os
from collections import Counter

def validate_structural_tokens(data_path):
    """
    Validate that structural tokens align properly with sequences and have valid values
    """
    print("Validating structural tokens...")
    
    # Load processed dataset
    dataset_file = os.path.join(data_path, "processed_dataset.pkl")
    tokens_file = os.path.join(data_path, "structural_tokens.pkl")
    
    if not os.path.exists(dataset_file):
        print(f"Error: Processed dataset not found at {dataset_file}")
        return
    
    if not os.path.exists(tokens_file):
        print(f"Error: Structural tokens not found at {tokens_file}")
        return
        
    # Load the datasets
    with open(dataset_file, 'rb') as f:
        proteins = pickle.load(f)
    
    with open(tokens_file, 'rb') as f:
        structural_tokens = pickle.load(f)
    
    print(f"Loaded {len(proteins)} proteins and {len(structural_tokens)} structural token sets")
    
    # Create a mapping for quick lookup
    token_dict = {}
    for item in structural_tokens:
        token_dict[item['protein_id']] = item['structural_tokens']
    
    # Validate alignment and token values
    valid_count = 0
    invalid_count = 0
    length_mismatch = 0
    invalid_token_problems = 0
    
    all_token_counts = Counter()
    
    for protein in proteins:
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
            length_mismatch += 1
            invalid_count += 1
            continue
        
        # Check token values (should be in range [0, 20] for 21 classes)
        token_array = np.array(struct_tokens)
        invalid_tokens = np.where((token_array < 0) | (token_array > 20))[0]
        
        if len(invalid_tokens) > 0:
            print(f"Invalid tokens found for {protein_id}: {invalid_tokens[:10]}... (first 10)")
            invalid_token_problems += 1
            invalid_count += 1
            continue
        
        # Count token distributions
        unique, counts = np.unique(token_array, return_counts=True)
        for u, c in zip(unique, counts):
            all_token_counts[int(u)] += c
        
        valid_count += 1
    
    print(f"\nValidation Results:")
    print(f"  Valid proteins: {valid_count}")
    print(f"  Invalid proteins: {invalid_count}")
    print(f"  Length mismatches: {length_mismatch}")
    print(f"  Invalid tokens: {invalid_token_problems}")
    
    print(f"\nToken distribution:")
    for token_id, count in sorted(all_token_counts.items()):
        print(f"  Token {token_id}: {count} occurrences")
    
    # Check if tokens are balanced or skewed
    if all_token_counts:
        most_common = all_token_counts.most_common(3)
        print(f"\nMost common tokens: {most_common}")
        
        # Calculate entropy to check class balance
        total_tokens = sum(all_token_counts.values())
        probabilities = [count/total_tokens for count in all_token_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        print(f"Entropy of token distribution: {entropy:.3f} (higher is more balanced)")
        
        if entropy < 2.0:  # Very low entropy indicates high skew
            print("Warning: Token distribution is highly skewed, which may make training difficult")


def inspect_token_generation():
    """
    Generate some diagnostic information about foldseek token generation
    """
    print("\n" + "="*60)
    print("SUGGESTED DEBUG STEPS:")
    print("="*60)
    print("1. Regenerate structural tokens with validation:")
    print("   python data_pipeline/generate_foldseek_tokens.py --pdb_file /path/to/one_pdb_file.pdb")
    print("\n2. Check a few PDB files to see if they have proper backbone atoms:")
    print("   grep -E \"^ATOM.*(N|CA|C)  \" your_pdb_file.pdb | head -20")
    print("\n3. Inspect token generation quality manually")
    print("\n4. Consider increasing the weight of the physical loss if it's too low")
    print("   In configs/config.yaml, try increasing 'physical_weight' in structure_alignment_utils")
    print("\n5. Make sure the number of structural classes (21) matches your token range")


if __name__ == "__main__":
    # Default path - adjust as needed
    data_path = input("Enter data path (or press Enter for default 'data/processed'): ").strip()
    if not data_path:
        data_path = "data/processed"
    
    validate_structural_tokens(data_path)
    inspect_token_generation()