
"""
A script to validate pre-computed GNN embeddings.

This script samples a subset of pre-computed embedding files, loads them,
and checks for corruption, NaNs, and infinite values to ensure data integrity.
"""
import os
import pickle
import numpy as np
import argparse
import random

def validate_embedding_file(file_path):
    """
    Validates a single embedding file.

    Args:
        file_path (str): The absolute path to the .pkl embedding file.

    Returns:
        A tuple (is_valid, message).
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        if 'embeddings' not in data:
            return False, "Corrupted: 'embeddings' key not found."

        embeddings = data['embeddings']
        if not isinstance(embeddings, (np.ndarray, list)):
            return False, f"Corrupted: Embeddings are of unexpected type {type(embeddings)}."

        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        if embeddings.size == 0:
            return False, "Corrupted: Embeddings array is empty."

        if np.isnan(embeddings).any():
            return False, "Corrupted: Contains NaN values."

        if np.isinf(embeddings).any():
            return False, "Corrupted: Contains Inf values."

        return True, f"OK (Shape: {embeddings.shape}, DType: {embeddings.dtype})"

    except (pickle.UnpicklingError, EOFError):
        return False, "Corrupted: Failed to unpickle file."
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"

def main():
    parser = argparse.ArgumentParser(description="Validate pre-computed GNN embeddings.")
    parser.add_argument(
        "--path",
        type=str,
        default="data/processed/embeddings",
        help="Path to the directory containing pre-computed embedding files."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=20,
        help="Number of random embedding files to sample and validate."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Error: Directory not found at '{args.path}'")
        return

    all_files = [f for f in os.listdir(args.path) if f.endswith('_gearnet_embeddings.pkl')]

    if not all_files:
        print(f"No embedding files ('*_gearnet_embeddings.pkl') found in '{args.path}'")
        return

    print(f"Found {len(all_files)} total embedding files.")
    
    # Take a random sample
    sample_size = min(args.sample_size, len(all_files))
    sampled_files = random.sample(all_files, sample_size)
    
    print(f"--- Validating a random sample of {sample_size} files ---")

    num_corrupted = 0
    for filename in sampled_files:
        file_path = os.path.join(args.path, filename)
        is_valid, message = validate_embedding_file(file_path)
        
        if is_valid:
            print(f"  [VALID] {filename}: {message}")
        else:
            print(f"  [CORRUPTED] {filename}: {message}")
            num_corrupted += 1

    print("\n--- Validation Summary ---")
    if num_corrupted == 0:
        print(f"✅ All {sample_size} sampled files appear to be valid.")
    else:
        print(f"❌ Found {num_corrupted} corrupted files in the sample of {sample_size}.")
        print("There may be issues with your pre-computed embedding data.")
    print("-" * 28)

if __name__ == "__main__":
    main()
