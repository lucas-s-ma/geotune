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
    import time
    start = time.time()

    # Add batch dimension and move to device
    n_coords = torch.tensor(n_coords, dtype=torch.float32).unsqueeze(0).to(device)
    ca_coords = torch.tensor(ca_coords, dtype=torch.float32).unsqueeze(0).to(device)
    c_coords = torch.tensor(c_coords, dtype=torch.float32).unsqueeze(0).to(device)

    print(f"  [DEBUG] Tensor preparation: {time.time() - start:.3f}s")
    start = time.time()

    # Generate embeddings using the GearNet model
    with torch.no_grad():
        embeddings = model(n_coords, ca_coords, c_coords)

    print(f"  [DEBUG] Model forward pass: {time.time() - start:.3f}s")
    start = time.time()

    # Remove batch dimension and convert to numpy
    embeddings = embeddings.squeeze(0).cpu().numpy()  # (seq_len, hidden_dim)

    print(f"  [DEBUG] Post-processing: {time.time() - start:.3f}s")

    return embeddings


def save_embeddings_file(output_file, protein_id, embeddings, sequence_length):
    """
    Save embeddings to file, with pickling-safe format
    """
    # Ensure embeddings are properly handled to avoid pickling issues
    if isinstance(embeddings, np.ndarray):
        # Convert to a "clean" numpy array to avoid pickling issues
        embeddings = np.asarray(embeddings).copy()
    else:
        embeddings = np.asarray(embeddings)

    # Save individual embedding file immediately to free memory
    embedding_data = {
        'protein_id': protein_id,
        'embeddings': embeddings.tolist(),  # Convert to list to avoid pickling issues
        'sequence_length': int(sequence_length)
    }

    with open(output_file, 'wb') as f:
        pickle.dump(embedding_data, f)


def load_embeddings_file(embedding_file):
    """
    Load embeddings from file and return as numpy array
    """
    with open(embedding_file, 'rb') as f:
        embedding_data = pickle.load(f)

    embeddings = embedding_data['embeddings']
    if isinstance(embeddings, list):
        # Convert list back to numpy array
        embeddings = np.array(embeddings, dtype=np.float32)
    elif isinstance(embeddings, np.ndarray):
        # Create a new array to avoid reference issues
        embeddings = np.array(embeddings, copy=True, dtype=np.float32)
    else:
        embeddings = np.array(embeddings, dtype=np.float32)

    return embeddings, embedding_data['protein_id'], embedding_data['sequence_length']


def generate_gearnet_embeddings_for_dataset(processed_dataset_path, output_dir, model_path=None, hidden_dim=512, chunk_size=50, max_proteins=None):
    """
    Generate GearNet embeddings for an entire processed dataset with memory-efficient processing

    Args:
        processed_dataset_path: Path to the processed dataset directory containing processed_dataset.pkl
        output_dir: Directory to save GearNet embeddings
        model_path: Path to pre-trained model (uses proper GearNet implementation if available)
        hidden_dim: Hidden dimension for the model
                    IMPORTANT: This MUST match your ESM2 model's hidden_size:
                    - ESM2-8M (facebook/esm2_t6_8M_UR50D): 320
                    - ESM2-35M (facebook/esm2_t12_35M_UR50D): 480
                    - ESM2-150M (facebook/esm2_t30_150M_UR50D): 640
                    - ESM2-650M (facebook/esm2_t33_650M_UR50D): 1280
        chunk_size: Number of proteins to process before clearing GPU cache
        max_proteins: Maximum number of proteins to process (for testing). If None, process all.
    """
    print(f"Generating GearNet embeddings from {processed_dataset_path}")
    print(f"Processing in chunks of {chunk_size} proteins to manage memory")
    print(f"Hidden dimension: {hidden_dim}")

    # Validate hidden_dim matches common ESM2 models
    esm_dims = {320: 'ESM2-8M', 480: 'ESM2-35M', 640: 'ESM2-150M', 1280: 'ESM2-650M'}
    if hidden_dim in esm_dims:
        print(f"✓ Hidden dimension {hidden_dim} matches {esm_dims[hidden_dim]}")
    else:
        print(f"⚠ WARNING: Hidden dimension {hidden_dim} does not match standard ESM2 models!")
        print(f"  Standard dimensions: {list(esm_dims.keys())}")
        print(f"  Make sure this matches your ESM2 model's hidden_size!")

    # Load the processed dataset metadata
    dataset_file = os.path.join(processed_dataset_path, "processed_dataset.pkl")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Processed dataset not found at {dataset_file}")

    print("Loading dataset metadata...")
    with open(dataset_file, 'rb') as f:
        proteins = pickle.load(f)

    total_proteins = len(proteins)

    # Limit to max_proteins if specified
    if max_proteins is not None and max_proteins < total_proteins:
        print(f"⚠ Limiting to first {max_proteins} proteins (out of {total_proteins} total) for testing")
        proteins = proteins[:max_proteins]
        total_proteins = max_proteins

    print(f"Total proteins to process: {total_proteins}")

    # Initialize the GearNet model (will try to use proper implementation first)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ============================================================================
    # IMPORTANT: TorchDrug GearNet Hanging Issue (2025-12-12)
    # ============================================================================
    # TorchDrug's GeometryAwareRelationalGraphNeuralNetwork hangs indefinitely
    # during forward pass (even on CPU, even with 10-node graphs).
    #
    # WORKAROUND: Using simple_structural_encoder instead (see models/simple_structural_encoder.py)
    # - Faster: ~0.1-0.5s per protein vs 1s+ for GearNet
    # - Reliable: No hanging issues
    # - Still captures structural info via k-NN distances + coordinates
    #
    # TO FIX LATER: See TORCHDRUG_ISSUE.md for detailed troubleshooting
    # TO RE-ENABLE GEARNET: Set use_simple = False (after fixing TorchDrug)
    # ============================================================================
    use_simple = True  # CURRENT: Using simple encoder due to TorchDrug issue

    model = PretrainedGNNWrapper(
        hidden_dim=hidden_dim,
        use_simple_encoder=use_simple
    ).to(device)

    model.eval()  # Set to evaluation mode

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process proteins in chunks to manage memory
    successful_count = 0
    failed_count = 0
    chunk_count = 0

    import time
    start_time = time.time()

    for start_idx in tqdm(range(0, total_proteins, chunk_size), desc="Processing chunks"):
        end_idx = min(start_idx + chunk_size, total_proteins)
        chunk = proteins[start_idx:end_idx]

        # Process each protein in the current chunk
        for idx, protein in enumerate(tqdm(chunk, desc=f"Processing chunk {chunk_count + 1}", leave=False)):
            protein_id = protein['id']
            protein_start_time = time.time()

            try:
                n_coords = protein['n_coords']  # (seq_len, 3)
                ca_coords = protein['ca_coords']  # (seq_len, 3)
                c_coords = protein['c_coords']  # (seq_len, 3)

                # Generate embeddings for this protein
                embeddings = generate_gearnet_embeddings_for_protein(
                    n_coords, ca_coords, c_coords, model, device
                )

                # Save individual embedding file immediately to free memory
                output_file = os.path.join(output_dir, f"{protein_id}_gearnet_embeddings.pkl")
                save_embeddings_file(output_file, protein_id, embeddings, embeddings.shape[0])

                successful_count += 1
                protein_time = time.time() - protein_start_time

                # Show progress with timing every 10 proteins
                if (successful_count + failed_count) % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time_per_protein = elapsed / (successful_count + failed_count)
                    remaining_proteins = total_proteins - (successful_count + failed_count)
                    eta_seconds = avg_time_per_protein * remaining_proteins
                    eta_hours = eta_seconds / 3600

                    print(f"\nProgress: {successful_count + failed_count}/{total_proteins} proteins")
                    print(f"  Success: {successful_count}, Failed: {failed_count}")
                    print(f"  Avg time/protein: {avg_time_per_protein:.2f}s")
                    print(f"  ETA: {eta_hours:.2f} hours ({eta_seconds/60:.1f} minutes)")

            except Exception as e:
                failed_count += 1
                print(f"\n{'='*80}")
                print(f"ERROR processing protein {protein_id}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                import traceback
                print(f"Full traceback:")
                traceback.print_exc()
                print(f"{'='*80}\n")

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
    parser.add_argument("--max_proteins", type=int, default=None,
                        help="Maximum number of proteins to process (for testing). If not specified, process all.")

    args = parser.parse_args()

    print(f"Generating GearNet embeddings from {args.processed_dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Memory management: Processing in chunks of {args.chunk_size} proteins")
    if args.max_proteins:
        print(f"Testing mode: Processing only first {args.max_proteins} proteins")

    # Generate embeddings
    successful_count = generate_gearnet_embeddings_for_dataset(
        args.processed_dataset_path,
        args.output_dir,
        args.model_path,
        args.hidden_dim,
        args.chunk_size,
        args.max_proteins
    )

    print(f"Completed! Successfully generated embeddings for {successful_count} proteins")
    print(f"Individual embeddings saved to: {args.output_dir}")

    # Inform user about the summary
    print(f"A summary file has been created at: {os.path.join(args.output_dir, 'generation_summary.txt')}")


if __name__ == "__main__":
    main()