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
import gc

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


def generate_gearnet_embeddings_for_dataset(processed_dataset_path, output_dir, model_path=None, hidden_dim=512, chunk_size=50):
    """
    Generate GearNet embeddings for an entire processed dataset with memory-efficient processing

    Args:
        processed_dataset_path: Path to the processed dataset directory containing processed_dataset.pkl
        output_dir: Directory to save GearNet embeddings
        model_path: Path to pre-trained model (uses proper GearNet implementation if available)
        hidden_dim: Hidden dimension for the model
        chunk_size: Number of proteins to process before clearing GPU cache
    """
    print(f"Generating GearNet embeddings from {processed_dataset_path}")
    print(f"Processing in chunks of {chunk_size} proteins to manage memory")

    # Load the processed dataset metadata
    dataset_file = os.path.join(processed_dataset_path, "processed_dataset.pkl")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Processed dataset not found at {dataset_file}")

    print("Loading dataset metadata...")
    with open(dataset_file, 'rb') as f:
        proteins = pickle.load(f)

    total_proteins = len(proteins)
    print(f"Total proteins to process: {total_proteins}")

    # Initialize the GearNet model (will try to use proper implementation first)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PretrainedGNNWrapper(
        model_path=model_path,
        hidden_dim=hidden_dim,
        freeze=True,
        use_gearnet_stub=False  # Try to use proper implementation first
    ).to(device)

    model.eval()  # Set to evaluation mode

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process proteins in chunks to manage memory
    successful_count = 0
    failed_count = 0
    chunk_count = 0

    for start_idx in tqdm(range(0, total_proteins, chunk_size), desc="Processing chunks"):
        end_idx = min(start_idx + chunk_size, total_proteins)
        chunk = proteins[start_idx:end_idx]

        # Process each protein in the current chunk
        for protein in tqdm(chunk, desc=f"Processing chunk {chunk_count + 1}", leave=False):
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
                embeddings = np.array(embeddings, copy=True) if isinstance(embeddings, np.ndarray) else embeddings

                # Save individual embedding file immediately to free memory
                output_file = os.path.join(output_dir, f"{protein_id}_gearnet_embeddings.pkl")
                embedding_data = {
                    'protein_id': protein_id,
                    'embeddings': embeddings,
                    'sequence_length': int(embeddings.shape[0])
                }

                with open(output_file, 'wb') as f:
                    pickle.dump(embedding_data, f)

                successful_count += 1

                # Show progress
                current_total = start_idx + (successful_count + failed_count)
                if current_total % 100 == 0:
                    print(f"Progress: {current_total}/{total_proteins} proteins processed ({successful_count} successful, {failed_count} failed)")

            except Exception as e:
                failed_count += 1
                print(f"Error processing protein {protein_id}: {e}")

                # Continue to next protein without stopping
                continue

        # Clear GPU cache and garbage collect after each chunk
        chunk_count += 1
        if hasattr(torch.cuda, 'empty_cache') and device.type == 'cuda':
            torch.cuda.empty_cache()

        # Force garbage collection after each chunk to free up memory
        gc.collect()

    print(f"\nProcessing completed!")
    print(f"Successful: {successful_count} proteins")
    print(f"Failed: {failed_count} proteins")
    print(f"Total attempts: {successful_count + failed_count}")

    # Create a simple summary file
    summary_file = os.path.join(output_dir, "generation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"GearNet Embeddings Generation Summary\n")
        f.write(f"Total proteins in input: {total_proteins}\n")
        f.write(f"Successfully processed: {successful_count}\n")
        f.write(f"Failed to process: {failed_count}\n")
        f.write(f"Success rate: {successful_count/total_proteins*100:.2f}% if total_proteins > 0 else 0%\n")
        f.write(f"Output directory: {output_dir}\n")

    return successful_count


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
    parser.add_argument("--chunk_size", type=int, default=50,
                        help="Number of proteins to process in each memory chunk (default: 50)")

    args = parser.parse_args()

    print(f"Generating GearNet embeddings from {args.processed_dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Memory management: Processing in chunks of {args.chunk_size} proteins")

    # Generate embeddings
    successful_count = generate_gearnet_embeddings_for_dataset(
        args.processed_dataset_path,
        args.output_dir,
        args.model_path,
        args.hidden_dim,
        args.chunk_size
    )

    print(f"Completed! Successfully generated embeddings for {successful_count} proteins")
    print(f"Individual embeddings saved to: {args.output_dir}")

    # Inform user about the summary
    print(f"A summary file has been created at: {os.path.join(args.output_dir, 'generation_summary.txt')}")


if __name__ == "__main__":
    main()