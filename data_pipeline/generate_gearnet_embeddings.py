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

    return embeddings


def generate_gearnet_embeddings_for_dataset(processed_dataset_path, output_dir, model_path=None, hidden_dim=512):
    """
    Generate GearNet embeddings for an entire processed dataset

    Args:
        processed_dataset_path: Path to the processed dataset directory containing processed_dataset.pkl
        output_dir: Directory to save GearNet embeddings
        model_path: Path to pre-trained model (uses proper GearNet implementation if available)
        hidden_dim: Hidden dimension for the model
    """
    print(f"Generating GearNet embeddings from {processed_dataset_path}")

    # Load the processed dataset
    dataset_file = os.path.join(processed_dataset_path, "processed_dataset.pkl")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Processed dataset not found at {dataset_file}")

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

    # Process each protein and generate embeddings
    embeddings_dict = {}

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

            # Ensure embeddings are properly handled to avoid pickling issues
            if isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings, copy=True)
            else:
                embeddings = embeddings

            # Save embeddings with protein ID
            embeddings_dict[protein_id] = {
                'protein_id': protein_id,
                'embeddings': embeddings,  # (seq_len, hidden_dim)
                'sequence_length': int(embeddings.shape[0])  # Convert to Python int to avoid pickling issues
            }

            # Also save individual embedding file - ensure proper serialization
            output_file = os.path.join(output_dir, f"{protein_id}_gearnet_embeddings.pkl")
            embedding_data = {
                'protein_id': protein_id,
                'embeddings': np.array(embeddings, copy=True) if isinstance(embeddings, np.ndarray) else embeddings,
                'sequence_length': int(embeddings.shape[0])
            }
            with open(output_file, 'wb') as f:
                pickle.dump(embedding_data, f)

            if (i + 1) % 100 == 0:
                print(f"Generated embeddings for {i + 1}/{len(proteins)} proteins")

        except Exception as e:
            print(f"Error processing protein {protein_id}: {e}")
            continue

    # Save all embeddings as a single file - ensure proper serialization
    all_embeddings_file = os.path.join(output_dir, "gearnet_embeddings.pkl")
    # Prepare data with proper numpy array handling
    all_embeddings_data = {}
    for protein_id, data in embeddings_dict.items():
        all_embeddings_data[protein_id] = {
            'protein_id': data['protein_id'],
            'embeddings': np.array(data['embeddings'], copy=True) if isinstance(data['embeddings'], np.ndarray) else data['embeddings'],
            'sequence_length': int(data['sequence_length'])
        }

    with open(all_embeddings_file, 'wb') as f:
        pickle.dump(all_embeddings_data, f)

    print(f"Saved all GearNet embeddings to {all_embeddings_file}")
    print(f"Generated embeddings for {len(embeddings_dict)} proteins")

    return embeddings_dict


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

    args = parser.parse_args()

    print(f"Generating GearNet embeddings from {args.processed_dataset_path}")
    print(f"Output directory: {args.output_dir}")

    # Generate embeddings
    embeddings_dict = generate_gearnet_embeddings_for_dataset(
        args.processed_dataset_path,
        args.output_dir,
        args.model_path,
        args.hidden_dim
    )

    print(f"Completed! Generated embeddings for {len(embeddings_dict)} proteins")
    print(f"Embeddings saved to: {args.output_dir}")


if __name__ == "__main__":
    main()