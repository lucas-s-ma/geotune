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
import tempfile
import gc  # garbage collection

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.structure_alignment_utils import PretrainedGNNWrapper
from torchdrug.data import Protein


def clean_pdb_file(pdb_path, cleaned_pdb_path):
    """
    Create a cleaned version of a PDB file by keeping only ATOM records.
    This helps to remove HETATM records that can cause parsing errors.
    """
    with open(pdb_path, 'r') as f_in, open(cleaned_pdb_path, 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM'):
                f_out.write(line)


def generate_gearnet_embeddings_for_protein(protein_graph, model):
    """
    Generate GearNet embeddings for a single protein

    Args:
        protein_graph (torchdrug.data.Protein): A protein graph object from TorchDrug.
        model: PretrainedGNNWrapper instance

    Returns:
        embeddings: (seq_len, hidden_dim) structural embeddings
    """
    # Generate embeddings using the GearNet model
    with torch.no_grad():
        embeddings = model(protein_graph)

    # Convert to numpy
    embeddings = embeddings.cpu().numpy()  # (seq_len, hidden_dim)

    return embeddings


def generate_gearnet_embeddings_for_dataset(processed_dataset_path, output_dir, raw_pdb_dir, model_path=None, hidden_dim=512, batch_size=10):
    """
    Generate GearNet embeddings for an entire processed dataset

    Args:
        processed_dataset_path: Path to the processed dataset directory containing processed_dataset.pkl
        output_dir: Directory to save GearNet embeddings
        raw_pdb_dir: Directory containing raw PDB files
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
    # This is a common issue when pickle files were created with different NumPy versions
    try:
        with open(dataset_file, 'rb') as f:
            proteins = pickle.load(f)
    except AttributeError as e:
        if "'numpy.ndarray' object has no attribute" in str(e) or "it's not the same object as numpy.ndarray" in str(e):
            # Handle NumPy version mismatch by temporarily modifying the import path
            print(f"NumPy compatibility issue detected: {e}")
            print("Attempting to resolve by using numpy.core.multiarray...")

            # Temporarily add numpy.core.multiarray to handle the old format
            import numpy.core.multiarray
            import sys

            # Ensure numpy.ndarray exists in the expected location
            if not hasattr(numpy.core.multiarray, 'ndarray'):
                numpy.core.multiarray.ndarray = numpy.ndarray

            with open(dataset_file, 'rb') as f:
                # Use pickle with a custom Unpickler to handle NumPy compatibility
                class NumpyUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module == 'numpy.core.multiarray' and name == 'ndarray':
                            return numpy.ndarray
                        if module == 'numpy' and name == 'ndarray':
                            return numpy.ndarray
                        return super().find_class(module, name)

                proteins = NumpyUnpickler(f).load()
        else:
            raise

    print(f"Loaded {len(proteins)} proteins from dataset")

    # Initialize the GearNet model (will try to use proper implementation first)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PretrainedGNNWrapper(
        model_path=model_path,
        hidden_dim=hidden_dim,
        freeze=True
    ).to(device)

    model.eval()  # Set to evaluation mode

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process proteins in batches to manage memory
    processed_count = 0
    failed_count = 0

    for i, protein_info in enumerate(tqdm(proteins, desc="Generating GearNet embeddings")):
        protein_id = protein_info['id']
        pdb_file = os.path.join(raw_pdb_dir, f"{protein_id}.pdb")

        if not os.path.exists(pdb_file):
            print(f"PDB file not found for protein {protein_id} at {pdb_file}")
            failed_count += 1
            continue

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=True) as tmp_pdb:
                # Clean the PDB file by removing HETATM records
                clean_pdb_file(pdb_file, tmp_pdb.name)

                # Create a torchdrug.data.Protein object from the cleaned PDB file
                protein_graph = Protein.from_pdb(tmp_pdb.name, atom_feature="position", bond_feature="length", residue_feature="symbol")
                
                if protein_graph is None:
                    raise ValueError("Protein graph could not be created from PDB, it might be empty after cleaning.")

                protein_graph = protein_graph.to(device)

                # Generate embeddings for this protein
                embeddings = generate_gearnet_embeddings_for_protein(
                    protein_graph, model
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
    parser.add_argument("--raw_pdb_dir", type=str, required=True,
                        help="Directory containing raw PDB files")
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
        args.raw_pdb_dir,
        args.model_path,
        args.hidden_dim,
        args.batch_size
    )

    print(f"Completed! Successfully processed {processed_count} proteins")
    print(f"Failed to process {failed_count} proteins")
    print(f"Embeddings saved to: {args.output_dir}")


if __name__ == "__main__":
    main()