"""
Process raw protein structure data for training
This script processes PDB files into a format suitable for the GeoTune model.
"""
import os
import sys
import argparse
import numpy as np
# Note: Bio.PDB imports are likely inside your data_utils, which is good practice.
# from Bio.PDB import PDBParser
# from Bio.PDB.DSSP import DSSP
from pathlib import Path
import pickle
import json
from tqdm import tqdm

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.data_utils import ProteinStructureDataset
# Handle import from either relative or absolute context
try:
    from .generate_foldseek_tokens import generate_foldseek_tokens, check_foldseek_installation
except ImportError:
    from generate_foldseek_tokens import generate_foldseek_tokens, check_foldseek_installation


# The inefficient 'process_pdb_to_features' function has been removed.

def create_efficient_dataset(raw_dir, output_dir, include_structural_tokens=True):
    """
    Create an efficient dataset file that contains all processed protein data in one file

    Args:
        raw_dir: Directory containing PDB files
        output_dir: Directory to save the processed dataset file
        include_structural_tokens: Whether to generate structural tokens from PDB files
    """
    print(f"Creating efficient dataset from {raw_dir}")

    # Use the ProteinStructureDataset to process all files at once
    temp_dataset = ProteinStructureDataset(raw_dir)

    print(f"Processed {len(temp_dataset.proteins)} proteins from PDB files")

    # Generate structural tokens if requested
    if include_structural_tokens:
        print("Generating Foldseek structural tokens for each protein...")
        structural_tokens_list = []

        # Get list of PDB files to process
        pdb_files = []
        for file in os.listdir(raw_dir):
            if file.lower().endswith('.pdb'):
                pdb_files.append(os.path.join(raw_dir, file))

        print(f"Found {len(pdb_files)} PDB files to process for structural tokens")

        for i, pdb_file in enumerate(tqdm(pdb_files, desc="Generating Foldseek structural tokens")):
            pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]

            # Find corresponding protein data in the processed dataset
            protein_data = None
            for protein in temp_dataset.proteins:
                if protein['id'] == pdb_name:
                    protein_data = protein
                    break

            if protein_data is not None:
                # Generate structural tokens for this PDB file using Foldseek
                try:
                    tokens = generate_foldseek_tokens(pdb_file)
                    if tokens is not None and len(tokens) > 0:
                        structural_tokens_list.append({
                            'protein_id': pdb_name,
                            'structural_tokens': tokens
                        })
                        print(f"  Generated {len(tokens)} tokens for {pdb_name}")
                    else:
                        print(f"  Failed to generate tokens for {pdb_name}")
                except Exception as e:
                    print(f"  Error generating tokens for {pdb_name}: {e}")
            else:
                print(f"No protein data found for {pdb_name}")

    # Save the entire processed dataset to a single file
    # Handle numpy arrays properly to avoid pickling issues
    os.makedirs(output_dir, exist_ok=True)
    dataset_file = os.path.join(output_dir, "processed_dataset.pkl")

    # Prepare the dataset for safe pickling by ensuring numpy arrays are handled properly
    safe_proteins = []
    for protein in temp_dataset.proteins:
        safe_protein = {}
        for key, value in protein.items():
            # Convert numpy arrays to new arrays to ensure proper serialization
            if isinstance(value, np.ndarray):
                # Create a completely new array with the same content
                safe_protein[key] = np.array(value, copy=True)
            else:
                safe_protein[key] = value
        safe_proteins.append(safe_protein)

    with open(dataset_file, 'wb') as f:
        pickle.dump(safe_proteins, f)

    print(f"Saved processed dataset to {dataset_file} with {len(temp_dataset.proteins)} proteins")

    # Also create a mapping file that maps indices to protein IDs
    id_mapping = {}
    for i, protein in enumerate(temp_dataset.proteins):
        id_mapping[i] = protein['id']

    mapping_file = os.path.join(output_dir, "id_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(id_mapping, f)

    print(f"Saved ID mapping to {mapping_file}")

    # Save structural tokens if generated
    if include_structural_tokens and len(structural_tokens_list) > 0:
        struct_token_file = os.path.join(output_dir, "structural_tokens.pkl")
        # Handle structural tokens with the same approach for safe pickling
        safe_structural_tokens = []
        for item in structural_tokens_list:
            safe_item = {}
            for key, value in item.items():
                if isinstance(value, np.ndarray):
                    safe_item[key] = np.array(value, copy=True)
                else:
                    safe_item[key] = value
            safe_structural_tokens.append(safe_item)
        with open(struct_token_file, 'wb') as f:
            pickle.dump(safe_structural_tokens, f)
        print(f"Saved structural tokens for {len(structural_tokens_list)} proteins to {struct_token_file}")

    return dataset_file, mapping_file


def process_directory(input_dir, output_dir, pdb_extensions=['.pdb', '.ent']):
    """
    Process all PDB files in a directory by initializing the dataset once.
    This is much more efficient than processing files one by one.

    Args:
        input_dir: Directory containing PDB files
        output_dir: Directory to save processed features
        pdb_extensions: List of file extensions to process (Note: This is now likely handled by ProteinStructureDataset)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- EFFICIENT REFACTOR ---
    # Initialize the dataset ONCE. This will process all PDBs in the input_dir internally,
    # avoiding the massive overhead of calling DSSP for each file in a separate process.
    print("Initializing dataset and processing all PDB files in memory...")
    # This assumes your ProteinStructureDataset class finds and processes all valid PDB files in the given directory.
    dataset = ProteinStructureDataset(input_dir)
    print(f"Found and processed {len(dataset.proteins)} proteins. Now saving features to disk...")

    success_count = 0
    fail_count = 0
    processed_files_list = []

    # Loop through the pre-processed protein data and save each to a separate file
    for protein_info in tqdm(dataset.proteins, desc="Saving processed features"):
        try:
            # Ensure protein_info is not None and has an 'id'
            if protein_info and 'id' in protein_info:
                protein_id = protein_info['id']
                output_file = os.path.join(output_dir, f"{protein_id}_features.pkl")

                # Handle numpy arrays properly for individual file saving
                safe_protein = {}
                for key, value in protein_info.items():
                    if isinstance(value, np.ndarray):
                        safe_protein[key] = np.array(value, copy=True)
                    else:
                        safe_protein[key] = value

                with open(output_file, 'wb') as f:
                    pickle.dump(safe_protein, f)

                processed_files_list.append(os.path.basename(output_file))
                success_count += 1
            else:
                # This case handles if the dataset processing returned a None entry
                print("Warning: A processed item was empty or lacked an ID.")
                fail_count += 1

        except Exception as e:
            # Get protein ID for a more informative error message if possible
            pid_for_error = protein_info.get('id', 'UNKNOWN') if isinstance(protein_info, dict) else 'UNKNOWN'
            print(f"Error saving features for protein {pid_for_error}: {e}")
            fail_count += 1

    print(f"\nProcessing completed! Success: {success_count}, Failed: {fail_count}")

    # Create a summary file
    summary = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "total_files_processed": len(dataset.proteins),
        "successful_saves": success_count,
        "failed_saves": fail_count,
        "processed_files": processed_files_list
    }

    summary_path = os.path.join(output_dir, "processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {summary_path}")
    return summary


def validate_processed_data(processed_dir):
    """
    Validate the processed data to ensure it meets requirements

    Args:
        processed_dir: Directory containing processed data files
    """
    print(f"Validating processed data in {processed_dir}...")

    # Check for the single large dataset file first
    efficient_dataset_file = Path(processed_dir) / "processed_dataset.pkl"
    if efficient_dataset_file.exists():
        print(f"Validating efficient dataset file: {efficient_dataset_file}")
        with open(efficient_dataset_file, 'rb') as f:
            all_proteins = pickle.load(f)
        pkl_files = all_proteins[:10] # Sample from the list of dicts
        source_is_file = True
    else:
        pkl_files = list(Path(processed_dir).glob("*.pkl"))
        source_is_file = False

    if not pkl_files:
        print("No processed files found!")
        return False

    valid_count = 0
    invalid_count = 0

    # Check first 10 as a sample
    for item in tqdm(pkl_files[:10], desc="Validating samples"):
        try:
            if source_is_file:
                data = item # item is already the data dict
            else:
                with open(item, 'rb') as f:
                    data = pickle.load(f)

            # Validate required fields exist (new format with N, CA, C coordinates)
            required_keys = ['sequence', 'n_coords', 'ca_coords', 'c_coords', 'id']
            if all(key in data for key in required_keys):
                # Validate data shapes
                seq_len = len(data['sequence'])
                n_coord_shape = data['n_coords'].shape
                ca_coord_shape = data['ca_coords'].shape
                c_coord_shape = data['c_coords'].shape

                if (seq_len > 0 and
                    n_coord_shape[0] == seq_len and n_coord_shape[1] == 3 and
                    ca_coord_shape[0] == seq_len and ca_coord_shape[1] == 3 and
                    c_coord_shape[0] == seq_len and c_coord_shape[1] == 3):
                    valid_count += 1
                else:
                    print(f"Invalid data shape in {data['id']}: seq_len={seq_len}, "
                          f"n_coord_shape={n_coord_shape}, ca_coord_shape={ca_coord_shape}, c_coord_shape={c_coord_shape}")
                    invalid_count += 1
            else:
                # Also check for old format for backward compatibility
                old_required_keys = ['sequence', 'coordinates', 'id']
                if all(key in data for key in old_required_keys):
                    # Old format detected
                    seq_len = len(data['sequence'])
                    coord_shape = data['coordinates'].shape
                    if seq_len > 0 and coord_shape[0] == seq_len and coord_shape[1] == 3:
                        print(f"Warning: Found data in old format (with 'coordinates' key instead of N/CA/C). "
                              f"Please regenerate the processed dataset to use the new dihedral angle constraints.")
                        valid_count += 1  # Still count as valid but with warning
                    else:
                        invalid_count += 1
                else:
                    print(f"Missing required keys in {data.get('id', 'Unknown ID')}")
                    invalid_count += 1
        except Exception as e:
            print(f"Error validating item {item if not source_is_file else item.get('id', 'Unknown ID')}: {e}")
            invalid_count += 1

    print(f"Validation complete: {valid_count} valid, {invalid_count} invalid samples (sampled first 10 of {len(pkl_files)})")
    return valid_count > 0


def main():
    parser = argparse.ArgumentParser(description="Process raw protein structure data for GeoTune training")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Directory containing raw PDB files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save processed features")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation on processed data")
    parser.add_argument("--pdb_extensions", nargs="+", default=['.pdb', '.ent'],
                        help="PDB file extensions to process (default: .pdb .ent)")
    parser.add_argument("--create_efficient_dataset", action="store_true",
                        help="Create a single efficient dataset file for fast loading during training")
    parser.add_argument("--no_structural_tokens", action="store_true",
                        help="Skip generation of structural tokens (for faster processing)")
    parser.add_argument("--generate_gearnet_embeddings", action="store_true",
                        help="Generate GearNet embeddings for processed proteins")

    args = parser.parse_args()

    print(f"Processing data from {args.raw_dir} to {args.output_dir}")

    # The --create_efficient_dataset flag provides the fastest experience for both processing and training.
    if args.create_efficient_dataset:
        print("Creating single efficient dataset file for fast loading...")
        include_tokens = not args.no_structural_tokens
        create_efficient_dataset(args.raw_dir, args.output_dir, include_structural_tokens=include_tokens)
    else:
        # Otherwise, process into individual files using the now-efficient method.
        # Note: Individual file processing doesn't include structural tokens for consistency
        process_directory(args.raw_dir, args.output_dir, args.pdb_extensions)

    # Optionally validate the processed data
    if args.validate:
        validate_processed_data(args.output_dir)

    # If requested, generate GearNet embeddings
    if args.generate_gearnet_embeddings:
        print("Generating GearNet embeddings...")

        # Import here to avoid issues if the dependencies are not available
        try:
            from scripts.generate_gearnet_embeddings import generate_gearnet_embeddings_for_dataset

            # Generate GearNet embeddings for the processed dataset
            embeddings_output_dir = os.path.join(project_root, "embeddings")
            generate_gearnet_embeddings_for_dataset(
                processed_dataset_path=args.output_dir,
                output_dir=embeddings_output_dir
            )
            print(f"GearNet embeddings saved to {embeddings_output_dir}")
        except ImportError as e:
            print(f"Could not import generate_gearnet_embeddings: {e}")
            print("Make sure all dependencies are installed.")
        except Exception as e:
            print(f"Error generating GearNet embeddings: {e}")


if __name__ == "__main__":
    main()