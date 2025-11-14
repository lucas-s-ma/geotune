#!/usr/bin/env python
"""
Script to regenerate structural tokens correctly aligned with the sequence/coordinate extraction
"""
import os
import pickle
import numpy as np
import tempfile
from Bio.PDB import PDBParser
from tqdm import tqdm
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import from local modules
from data_pipeline.generate_foldseek_tokens import generate_foldseek_tokens, convert_3di_to_ints


def extract_sequence_and_coords_from_pdb(pdb_path):
    """
    Extract sequence and coordinates exactly as done in the data processing pipeline
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)

        # Get first model
        model = structure[0]  # First model

        sequence = ""
        n_coords = []
        ca_coords = []
        c_coords = []

        for chain in model:
            for residue in chain:
                # Check if it's a protein residue
                if residue.get_resname() in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                                             'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                                             'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                                             'SER', 'THR', 'TRP', 'TYR', 'VAL']:
                    # Get residue name
                    aa_code = three_to_one(residue.get_resname())
                    if aa_code != 'X':  # Unknown amino acid
                        sequence += aa_code

                        # Try to get backbone atom coordinates
                        n_coord = ca_coord = c_coord = None

                        try:
                            n_atom = residue['N']
                            n_coord = [n_atom.get_coord()[0], n_atom.get_coord()[1], n_atom.get_coord()[2]]
                        except KeyError:
                            pass  # N coordinate not available

                        try:
                            ca_atom = residue['CA']
                            ca_coord = [ca_atom.get_coord()[0], ca_atom.get_coord()[1], ca_atom.get_coord()[2]]
                        except KeyError:
                            pass  # CA coordinate not available

                        try:
                            c_atom = residue['C']
                            c_coord = [c_atom.get_coord()[0], c_atom.get_coord()[1], c_atom.get_coord()[2]]
                        except KeyError:
                            pass  # C coordinate not available

                        n_coords.append(n_coord if n_coord is not None else [0, 0, 0])  # Use zero if not available
                        ca_coords.append(ca_coord if ca_coord is not None else [0, 0, 0])  # Use zero if not available
                        c_coords.append(c_coord if c_coord is not None else [0, 0, 0])  # Use zero if not available

        if len(sequence) > 0:
            # Ensure all coordinate lists have the same length
            min_len = min(len(sequence), len(n_coords), len(ca_coords), len(c_coords))
            sequence = sequence[:min_len]
            n_coords = n_coords[:min_len]
            ca_coords = ca_coords[:min_len]
            c_coords = c_coords[:min_len]

            return {
                'sequence': sequence,
                'n_coords': np.array(n_coords),
                'ca_coords': np.array(ca_coords),
                'c_coords': np.array(c_coords),
                'id': os.path.basename(pdb_path).replace('.pdb', '').replace('.ent', '')
            }

    except Exception as e:
        print(f"Error processing {pdb_path}: {e}")
        return None


def three_to_one(three_letter):
    """Convert three-letter amino acid code to one-letter (standalone function)"""
    mapping = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    return mapping.get(three_letter, 'X')


def create_canonical_only_pdb(original_pdb_path, output_pdb_path):
    """
    Create a new PDB with only canonical amino acid residues
    This will ensure Foldseek tokens align with sequence extraction
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', original_pdb_path)
        
        # Get first model
        model = structure[0]
        
        with open(output_pdb_path, 'w') as output_file:
            atom_counter = 1
            residue_counter = 1
            current_chain = None
            
            canonical_residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                                  'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                                  'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                                  'SER', 'THR', 'TRP', 'TYR', 'VAL']
            
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in canonical_residues:
                        # Output each atom in the canonical residue
                        for atom in residue:
                            # Get coordinates
                            x, y, z = atom.get_coord()
                            
                            # Get occupancy and B-factor (default to 1.0 and 0.0 if not available)
                            occupancy = atom.get_occupancy()
                            if occupancy is None:
                                occupancy = 1.0
                            bfactor = atom.get_bfactor()
                            if bfactor is None:
                                bfactor = 0.0
                            
                            # Format ATOM line
                            atom_name = atom.get_fullname().strip()
                            residue_name = residue.get_resname()
                            element = atom.element
                            
                            atom_line = (
                                f"ATOM  {atom_counter:>5} {atom_name:>4} {residue_name} {chain.id}{residue.id[1]:>4}    "
                                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{bfactor:>6.2f}          {element:>2}  "
                            )
                            
                            output_file.write(atom_line + '\n')
                            atom_counter += 1
                        
                        # Add TER record after each chain residue to maintain proper structure
                        if residue_counter > 1:  # Don't add TER before first residue
                            pass  # We'll add TER after all canonical residues
                        residue_counter += 1
            
            # Add TER to indicate end of chain
            output_file.write("TER\n")
            output_file.write("END\n")
            
        return True
    except Exception as e:
        print(f"Error creating canonical-only PDB for {original_pdb_path}: {e}")
        return False


def regenerate_structural_tokens_for_directory(pdb_directory, processed_dataset_path, output_path):
    """
    Regenerate structural tokens ensuring proper alignment with sequence extraction
    """
    print("Loading processed dataset to get protein IDs...")
    
    # Load processed dataset to know which proteins we have
    with open(processed_dataset_path, 'rb') as f:
        proteins = pickle.load(f)
    
    protein_ids = {protein['id'] for protein in proteins}
    print(f"Found {len(protein_ids)} proteins in processed dataset")
    
    # Find corresponding PDB files
    pdb_files = []
    for filename in os.listdir(pdb_directory):
        if filename.lower().endswith(('.pdb', '.ent')):
            protein_id = os.path.splitext(filename)[0]
            if protein_id in protein_ids:
                pdb_files.append(os.path.join(pdb_directory, filename))
    
    print(f"Found {len(pdb_files)} PDB files matching processed proteins")
    
    regenerated_tokens = []
    success_count = 0
    failure_count = 0
    
    for pdb_path in tqdm(pdb_files, desc="Regenerating structural tokens"):
        protein_id = os.path.splitext(os.path.basename(pdb_path))[0]
        
        try:
            # Extract sequence as would be done in processing pipeline
            pdb_data = extract_sequence_and_coords_from_pdb(pdb_path)
            if pdb_data is None:
                print(f"Could not extract sequence from {pdb_path}")
                failure_count += 1
                continue
            
            original_seq_len = len(pdb_data['sequence'])
            
            # Create a temporary PDB with only canonical residues
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as temp_file:
                canonical_success = create_canonical_only_pdb(pdb_path, temp_file.name)
                if not canonical_success:
                    failure_count += 1
                    continue
                canonical_pdb_path = temp_file.name
            
            try:
                # Generate Foldseek tokens for the canonical-only PDB
                foldseek_tokens = generate_foldseek_tokens(canonical_pdb_path)
                
                if foldseek_tokens is None or len(foldseek_tokens) == 0:
                    print(f"Failed to generate Foldseek tokens for {protein_id}")
                    failure_count += 1
                else:
                    # Now we have tokens aligned with canonical amino acids
                    
                    # If there's a mismatch (though there shouldn't be if our extraction is correct)
                    if len(foldseek_tokens) != original_seq_len:
                        print(f"Length mismatch for {protein_id}: seq={original_seq_len}, tokens={len(foldseek_tokens)}")
                        
                        # Try to align by taking the first N tokens or padding
                        if len(foldseek_tokens) > original_seq_len:
                            foldseek_tokens = foldseek_tokens[:original_seq_len]
                        else:
                            # Pad with a reasonable value (e.g., 20 for unknown)
                            padding_needed = original_seq_len - len(foldseek_tokens)
                            foldseek_tokens.extend([20] * padding_needed)
                    
                    # Add to results
                    regenerated_tokens.append({
                        'protein_id': protein_id,
                        'structural_tokens': foldseek_tokens
                    })
                    
                    success_count += 1
                    if success_count % 50 == 0:
                        print(f"Processed {success_count}: {protein_id} - {len(foldseek_tokens)} tokens for {original_seq_len} amino acids")
            
            finally:
                # Clean up temporary file
                if os.path.exists(canonical_pdb_path):
                    os.unlink(canonical_pdb_path)
                    
        except Exception as e:
            print(f"Error processing {protein_id}: {e}")
            failure_count += 1
    
    print(f"\nRegeneration completed:")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failure_count}")
    
    if regenerated_tokens:
        # Save the regenerated tokens
        with open(output_path, 'wb') as f:
            pickle.dump(regenerated_tokens, f)
        print(f"Regenerated tokens saved to {output_path}")
    
    return regenerated_tokens


import tempfile

def main():
    # Paths - adjust as needed for your setup
    pdb_directory = input("Enter path to PDB directory (with raw PDB files): ").strip()
    processed_dataset_path = input("Enter path to processed_dataset.pkl (e.g., data/processed/processed_dataset.pkl): ").strip()
    output_path = input("Enter output path for fixed tokens (e.g., data/processed/fixed_structural_tokens.pkl): ").strip()
    
    regenerate_structural_tokens_for_directory(pdb_directory, processed_dataset_path, output_path)


if __name__ == "__main__":
    main()