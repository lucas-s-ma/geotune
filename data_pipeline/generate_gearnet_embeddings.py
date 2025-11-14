#!/usr/bin/env python
"""
Script to generate GearNet embeddings from protein structure data
This script processes PDB files and extracts structural embeddings using a proper GearNet model
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import argparse
import gc  # garbage collection

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.structure_alignment_utils import PretrainedGNNWrapper


def generate_gearnet_embeddings_for_protein(n_coords, ca_coords, c_coords, model, device):
    """
    Generate GearNet embeddings for a single protein

    Args:
        n_coords: (seq_len, 3) N atom coordinates
        ca_coords: (seq_len, 3) CA atom coordinates
        c_coords: (seq_len, 3) C atom coordinates
        model: PretrainedGNNWrapper instance
        device: torch.device to move tensors to

    Returns:
        embeddings: (seq_len, hidden_dim) structural embeddings
    """
    # Add batch dimension and move to device
    n_coords = torch.tensor(n_coords, dtype=torch.float32).unsqueeze(0).to(device)
    ca_coords = torch.tensor(ca_coords, dtype=torch.float32).unsqueeze(0).to(device)
    c_coords = torch.tensor(c_coords, dtype=torch.float32).unsqueeze(0).to(device)

    # Generate embeddings using the GearNet model
    with torch.no_grad():
        embeddings = model(n_coords, ca_coords, c_coords)

    # Remove batch dimension and convert to numpy
    embeddings = embeddings.squeeze(0).cpu().numpy()  # (seq_len, hidden_dim)
    
    # Clean up GPU memory
    del n_coords, ca_coords, c_coords
    
    return embeddings


def generate_gearnet_embeddings_for_dataset(processed_dataset_path, output_dir, model_path=None, hidden_dim=512, batch_size=10):
    """
    Generate GearNet embeddings for an entire processed dataset

    Args:
        processed_dataset_path: Path to the processed dataset directory containing processed_dataset.pkl
        output_dir: Directory to save GearNet embeddings
        model_path: Path to pre-trained model (uses proper GearNet implementation if available)
        hidden_dim: Hidden dimension for the model
        batch_size: Number of proteins to process before clearing memory
    """
    print(f"Generating GearNet embeddings from {processed_dataset_path}")

    # Load the processed dataset
    dataset_file = os.path.join(processed_dataset_path, "processed_dataset.pkl")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Processed dataset not found at {dataset_file}")

    # Handle potential NumPy compatibility issues when loading
    import numpy.core.multiarray
    with open(dataset_file, 'rb') as f:
        proteins = pickle.load(f)

    print(f"Loaded {len(proteins)} proteins from dataset")

    # Initialize the GearNet model (will try to use proper implementation first)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PretrainedGNNWrapper(
        model_path=model_path,
        hidden_dim=hidden_dim,
        freeze=True,
        use_gearnet_stub=False  # Try to use proper implementation first
    ).to(device)

    model.eval()  # Set to evaluation mode

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process proteins in batches to manage memory
    processed_count = 0
    failed_count = 0
    
    for i, protein in enumerate(tqdm(proteins, desc="Generating GearNet embeddings")):
        protein_id = protein['id']

        try:
            n_coords = protein['n_coords']  # (seq_len, 3)
            ca_coords = protein['ca_coords']  # (seq_len, 3)
            c_coords = protein['c_coords']  # (seq_len, 3)

            # Generate embeddings for this protein
            embeddings = generate_gearnet_embeddings_for_protein(
                n_coords, ca_coords, c_coords, model, device
            )

            # Save individual embedding file
            output_file = os.path.join(output_dir, f"{protein_id}_gearnet_embeddings.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump({
                    'protein_id': protein_id,
                    'embeddings': embeddings,
                    'sequence_length': embeddings.shape[0]
                }, f)

            processed_count += 1
            
            # Clear memory periodically
            if processed_count % batch_size == 0:
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing protein {protein_id}: {e}")
            failed_count += 1
            continue

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(proteins)} proteins (successful: {processed_count}, failed: {failed_count})")

    print(f"Completed! Generated embeddings for {processed_count} proteins")
    print(f"Failed to process {failed_count} proteins")
    
    # Clean up
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return processed_count, failed_count


def main():
    parser = argparse.ArgumentParser(description="Generate GearNet embeddings for proteins")
    parser.add_argument("--processed_dataset_path", type=str, required=True,
                        help="Path to processed dataset directory containing processed_dataset.pkl")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated GearNet embeddings")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pre-trained model (optional)")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension for the model (default: 512)")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of proteins to process before clearing memory (default: 10)")

    args = parser.parse_args()

    print(f"Generating GearNet embeddings from {args.processed_dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size for memory management: {args.batch_size}")

    # Generate embeddings
    processed_count, failed_count = generate_gearnet_embeddings_for_dataset(
        args.processed_dataset_path,
        args.output_dir,
        args.model_path,
        args.hidden_dim,
        args.batch_size
    )

    print(f"Completed! Successfully processed {processed_count} proteins")
    print(f"Failed to process {failed_count} proteins")
    print(f"Embeddings saved to: {args.output_dir}")


if __name__ == "__main__":
    main()