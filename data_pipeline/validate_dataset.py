"""
Validation script for the data pipeline
Checks if processed data, structural tokens, and embeddings are all properly generated
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def validate_processed_dataset(dataset_path):
    """Validate the main processed dataset"""
    print(f"Validating processed dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset file not found at {dataset_path}")
        return False
    
    try:
        with open(dataset_path, 'rb') as f:
            proteins = pickle.load(f)
        
        if not proteins:
            print("❌ Error: Dataset is empty")
            return False
        
        print(f"✅ Found {len(proteins)} proteins in dataset")
        
        # Check first few proteins for required fields
        for i, protein in enumerate(proteins[:5]):
            required_keys = ['sequence', 'n_coords', 'ca_coords', 'c_coords', 'id']
            missing_keys = [key for key in required_keys if key not in protein]
            
            if missing_keys:
                print(f"❌ Missing keys in protein {i}: {missing_keys}")
                return False
            
            # Check shapes
            seq_len = len(protein['sequence'])
            if (protein['n_coords'].shape[0] != seq_len or 
                protein['ca_coords'].shape[0] != seq_len or 
                protein['c_coords'].shape[0] != seq_len):
                print(f"❌ Coordinate sequence length mismatch in protein {i}")
                return False
        
        print("✅ Dataset structure is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False


def validate_structural_tokens(token_path):
    """Validate the structural tokens file"""
    print(f"Validating structural tokens: {token_path}")
    
    if not os.path.exists(token_path):
        print(f"❌ Error: Structural tokens file not found at {token_path}")
        return False
    
    try:
        with open(token_path, 'rb') as f:
            tokens_data = pickle.load(f)
        
        if not tokens_data:
            print("❌ Error: Structural tokens file is empty")
            return False
        
        print(f"✅ Found structural tokens for {len(tokens_data)} proteins")
        
        # Check token validity
        total_invalid = 0
        for i, item in enumerate(tokens_data[:10]):  # Check first 10
            if 'structural_tokens' not in item:
                print(f"❌ Missing 'structural_tokens' key in item {i}")
                return False
            
            tokens = item['structural_tokens']
            invalid_count = sum(1 for token in tokens if token < 0 or token > 19)
            total_invalid += invalid_count
            
            if invalid_count > 0:
                print(f"⚠️  Found {invalid_count} invalid tokens in protein {i} (expected range: 0-19)")
        
        if total_invalid == 0:
            print("✅ All checked tokens are in valid range [0, 19]")
        else:
            print(f"⚠️  Found {total_invalid} total invalid tokens in first 10 proteins")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading structural tokens: {e}")
        return False


def validate_gearnet_embeddings(embeddings_path, expected_dim=None):
    """Validate the GearNet embeddings directory"""
    print(f"Validating GearNet embeddings: {embeddings_path}")
    
    if not os.path.exists(embeddings_path):
        print(f"❌ Error: Embeddings directory not found at {embeddings_path}")
        return False
    
    embedding_files = [f for f in os.listdir(embeddings_path) if f.endswith('_gearnet_embeddings.pkl')]
    if not embedding_files:
        print("⚠️ Warning: No individual embedding files found.")
        return True # Not a failure, but a warning

    print(f"✅ Found {len(embedding_files)} individual embedding files")

    if expected_dim is not None:
        # Check dimension of a sample embedding
        try:
            sample_file = os.path.join(embeddings_path, embedding_files[0])
            with open(sample_file, 'rb') as f:
                sample_data = pickle.load(f)
                embedding_dim = sample_data['embeddings'].shape[-1]
                if embedding_dim != expected_dim:
                    print(f"❌ Error: Embedding dimension mismatch. Expected {expected_dim}, found {embedding_dim}")
                    return False
            print(f"✅ Embedding dimension is correct ({expected_dim})")
        except Exception as e:
            print(f"❌ Error checking embedding dimension: {e}")
            return False
            
    return True


def validate_dataset_integrity(dataset_path, embeddings_path):
    """Validate the integrity between the processed dataset and embeddings"""
    print("Validating dataset integrity...")
    
    try:
        with open(dataset_path, 'rb') as f:
            proteins = pickle.load(f)
        protein_ids = {p['id'] for p in proteins}
        
        embedding_files = [f for f in os.listdir(embeddings_path) if f.endswith('_gearnet_embeddings.pkl')]
        embedding_ids = {f.replace('_gearnet_embeddings.pkl', '') for f in embedding_files}
        
        missing_embeddings = protein_ids - embedding_ids
        if missing_embeddings:
            print(f"❌ Error: {len(missing_embeddings)} proteins are missing embeddings (e.g., {list(missing_embeddings)[:5]})")
            return False
            
        extra_embeddings = embedding_ids - protein_ids
        if extra_embeddings:
            print(f"⚠️ Warning: {len(extra_embeddings)} embeddings do not have a corresponding protein in the dataset (e.g., {list(extra_embeddings)[:5]})")
            
        print("✅ Dataset integrity is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error validating dataset integrity: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate data pipeline outputs")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing processed data files")
    parser.add_argument("--embedding_dim", type=int, default=512,
                        help="Expected dimension of GearNet embeddings")
    
    args = parser.parse_args()
    
    print("🔍 Starting data pipeline validation...")
    
    # Paths to check
    dataset_path = os.path.join(args.data_dir, "processed_dataset.pkl")
    token_path = os.path.join(args.data_dir, "structural_tokens.pkl")
    embeddings_path = os.path.join(args.data_dir, "embeddings")
    
    # Validate each component
    dataset_valid = validate_processed_dataset(dataset_path)
    tokens_valid = validate_structural_tokens(token_path)
    embeddings_valid = validate_gearnet_embeddings(embeddings_path, expected_dim=args.embedding_dim)
    integrity_valid = validate_dataset_integrity(dataset_path, embeddings_path)
    
    print("\n📊 Validation Summary:")
    print(f"Processed dataset: {'✅ VALID' if dataset_valid else '❌ INVALID'}")
    print(f"Structural tokens: {'✅ VALID' if tokens_valid else '❌ INVALID'}")
    print(f"GearNet embeddings: {'✅ VALID' if embeddings_valid else '❌ INVALID'}")
    print(f"Dataset integrity: {'✅ VALID' if integrity_valid else '❌ INVALID'}")
    
    if all([dataset_valid, tokens_valid, embeddings_valid, integrity_valid]):
        print("\n🎉 All pipeline components are valid!")
        return 0
    else:
        print("\n❌ Some pipeline components are invalid. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())