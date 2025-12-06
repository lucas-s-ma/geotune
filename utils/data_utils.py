"""
Data utilities for protein sequences and structures
"""
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import warnings
import pickle
import json
warnings.filterwarnings("ignore", category=UserWarning)


class ProteinStructureDataset(Dataset):
    """
    Dataset for protein sequences with structural information
    """
    def __init__(self, data_path, max_seq_len=1024, include_3d_coords=True):
        """
        Args:
            data_path: Path to directory containing protein data
            max_seq_len: Maximum sequence length
            include_3d_coords: Whether to include 3D coordinates
        """
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.include_3d_coords = include_3d_coords

        # Load protein data
        self.proteins = self.load_protein_data()

    def load_protein_data(self):
        """Load protein sequences and structures from data directory"""
        proteins = []

        # Look for PDB files
        for filename in os.listdir(self.data_path):
            if filename.lower().endswith('.pdb'):
                pdb_path = os.path.join(self.data_path, filename)
                protein_data = self.extract_protein_info(pdb_path)
                if protein_data is not None:
                    proteins.append(protein_data)

        return proteins

    def extract_protein_info(self, pdb_path):
        """Extract sequence and structural information from PDB file including N, CA, C coordinates"""
        try:
            # Parse PDB file
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_path)

            # Get first model
            model = structure[0]  # First model

            # Extract sequence and coordinates
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
                        aa_code = self.three_to_one(residue.get_resname())
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
                    'id': os.path.basename(pdb_path).replace('.pdb', '')
                }

        except Exception as e:
            print(f"Error processing {pdb_path}: {e}")
            return None

        return None

    def three_to_one(self, three_letter):
        """Convert three-letter amino acid code to one-letter"""
        mapping = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        return mapping.get(three_letter, 'X')

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        protein = self.proteins[idx]

        sequence = protein['sequence']
        n_coords = protein['n_coords']
        ca_coords = protein['ca_coords']
        c_coords = protein['c_coords']

        # Truncate if necessary
        if len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
            n_coords = n_coords[:self.max_seq_len]
            ca_coords = ca_coords[:self.max_seq_len]
            c_coords = c_coords[:self.max_seq_len]

        # Create token mapping (simple mapping for now)
        token_ids = self.sequence_to_tokens(sequence)

        # Pad or truncate to max length
        if len(token_ids) < self.max_seq_len:
            padding_length = self.max_seq_len - len(token_ids)
            token_ids.extend([1] * padding_length)  # 1 for <pad> token in ESM2
            padding_coords = np.zeros((padding_length, 3))
            n_coords = np.vstack([n_coords, padding_coords])
            ca_coords = np.vstack([ca_coords, padding_coords])
            c_coords = np.vstack([c_coords, padding_coords])

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if token != 1 else 0 for token in token_ids]  # 1 is <pad> in ESM2

        result = {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'n_coords': torch.tensor(n_coords, dtype=torch.float32),
            'ca_coords': torch.tensor(ca_coords, dtype=torch.float32),
            'c_coords': torch.tensor(c_coords, dtype=torch.float32),
            'seq_len': len(protein['sequence']),
            'protein_id': protein['id'],
            'index': idx
        }

        # Add structural tokens if available
        # Use try-except to handle missing structural tokens gracefully
        if self.include_structural_tokens:
            try:
                if idx < len(self.structural_tokens):
                    struct_tokens = self.structural_tokens[idx]
                    struct_seq = struct_tokens['structural_tokens']  # Assuming it's in the same format

                    # Truncate structural tokens to match sequence length
                    if len(struct_seq) > self.max_seq_len:
                        struct_seq = struct_seq[:self.max_seq_len]

                    # Pad or truncate structural tokens to max length
                    if len(struct_seq) < self.max_seq_len:
                        padding_length = self.max_seq_len - len(struct_seq)
                        struct_seq.extend([-100] * padding_length)  # Use -100 as ignore index for padding

                    result['structural_tokens'] = torch.tensor(struct_seq, dtype=torch.long)
            except (IndexError, KeyError, TypeError) as e:
                # Skip structural tokens for this sample if there's any issue
                pass

        return result

    def sequence_to_tokens(self, sequence):
        """
        Convert amino acid sequence to token IDs using ESM2 vocabulary

        ESM2 token mapping:
        - 0: <cls>
        - 1: <pad>
        - 2: <eos>
        - 3: <unk>
        - 4-23: Standard amino acids (L, A, G, V, S, E, R, T, I, D, P, K, Q, N, F, Y, M, H, W, C)
        - 32: <mask>
        """
        # ESM2 amino acid to token ID mapping (IDs 4-23)
        # Order: L, A, G, V, S, E, R, T, I, D, P, K, Q, N, F, Y, M, H, W, C
        aa_to_id = {
            'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11,
            'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18,
            'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23
        }

        # Use <unk> token (3) for unknown amino acids, <pad> token (1) for padding
        tokens = [aa_to_id.get(aa, 3) for aa in sequence]  # 3 for unknown amino acids
        return tokens


def collate_fn(batch):
    """
    Collate function for protein dataset with dihedral angle information
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    n_coords = torch.stack([item['n_coords'] for item in batch])
    ca_coords = torch.stack([item['ca_coords'] for item in batch])
    c_coords = torch.stack([item['c_coords'] for item in batch])

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'n_coords': n_coords,
        'ca_coords': ca_coords,
        'c_coords': c_coords,
        'seq_lens': [item['seq_len'] for item in batch],
        'protein_ids': [item['protein_id'] for item in batch]
    }

    # Add indices if they exist
    if 'index' in batch[0]:
        indices = torch.tensor([item['index'] for item in batch], dtype=torch.long)
        result['indices'] = indices

    # Add pre-computed embeddings if they exist in ALL items of the batch
    if all('precomputed_embeddings' in item for item in batch):
        precomputed_embeddings = torch.stack([item['precomputed_embeddings'] for item in batch])
        result['precomputed_embeddings'] = precomputed_embeddings

    # Add structural tokens if they exist in ALL items of the batch
    if all('structural_tokens' in item for item in batch):
        structural_tokens = torch.stack([item['structural_tokens'] for item in batch])
        result['structural_tokens'] = structural_tokens

    return result


def load_structure_from_pdb(pdb_path, chain_id=None):
    """
    Load structure information directly from a PDB file
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)

        # Get first model
        model = structure[0]

        # Select chain if specified, otherwise use first chain
        if chain_id:
            chain = model[chain_id]
        else:
            chain = list(model.get_chains())[0]  # First chain

        sequence = ""
        coordinates = []
        residue_ids = []

        for residue in chain:
            # Check if it's a protein residue
            if residue.get_resname() in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                                         'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                                         'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                                         'SER', 'THR', 'TRP', 'TYR', 'VAL']:

                aa_code = three_to_one(residue.get_resname())
                if aa_code != 'X':  # Unknown amino acid
                    sequence += aa_code
                    residue_ids.append(residue.get_id()[1])  # Residue number

                    # Get CA coordinate
                    try:
                        ca_atom = residue['CA']
                        coordinates.append(list(ca_atom.get_coord()))
                    except KeyError:
                        # Fallback to other atoms
                        atom_coords = []
                        for atom in residue:
                            if atom.get_name() in ['N', 'CA', 'C', 'O']:
                                atom_coords.append(atom.get_coord())

                        if atom_coords:
                            avg_coord = np.mean(atom_coords, axis=0)
                            coordinates.append(avg_coord.tolist())
                        else:
                            # Use first atom
                            first_atom = list(residue.get_atoms())[0]
                            coordinates.append(list(first_atom.get_coord()))

        return {
            'sequence': sequence,
            'coordinates': np.array(coordinates),
            'residue_ids': residue_ids
        }

    except Exception as e:
        print(f"Error loading structure from {pdb_path}: {e}")
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


class EfficientProteinDataset(Dataset):
    """
    Efficient dataset that loads pre-processed protein data from a single file
    This avoids re-parsing PDB files every time, significantly improving performance
    """
    _protein_data = None  # Class-level cache for protein data
    _structural_tokens = None  # Class-level cache for structural tokens
    _embeddings_dict = None  # Class-level cache for embeddings

    def __init__(self, processed_data_path, max_seq_len=1024, include_structural_tokens=False, load_embeddings=False):
        """
        Args:
            processed_data_path: Path to directory containing pre-processed dataset.pkl
            max_seq_len: Maximum sequence length
            include_structural_tokens: Whether to include precomputed structural tokens (Foldseek)
            load_embeddings: Whether to load pre-computed embeddings from data/processed/embeddings
        """
        self.max_seq_len = max_seq_len
        self.include_structural_tokens = include_structural_tokens
        self.load_embeddings = load_embeddings

        # Load data only once and cache it at the class level
        if EfficientProteinDataset._protein_data is None:
            dataset_file = os.path.join(processed_data_path, "processed_dataset.pkl")
            if os.path.exists(dataset_file):
                with open(dataset_file, 'rb') as f:
                    EfficientProteinDataset._protein_data = pickle.load(f)
                print(f"Loaded and cached {len(EfficientProteinDataset._protein_data)} proteins.")
            else:
                raise FileNotFoundError(f"Processed dataset not found at {dataset_file}.")

        self.proteins = EfficientProteinDataset._protein_data

        if self.include_structural_tokens and EfficientProteinDataset._structural_tokens is None:
            struct_token_file = os.path.join(processed_data_path, "structural_tokens.pkl")
            if os.path.exists(struct_token_file):
                with open(struct_token_file, 'rb') as f:
                    EfficientProteinDataset._structural_tokens = pickle.load(f)
                print(f"Loaded and cached structural tokens for {len(EfficientProteinDataset._structural_tokens)} proteins.")
            else:
                print("Warning: Structural tokens file not found. Continuing without them.")
                self.include_structural_tokens = False
        
        self.structural_tokens = EfficientProteinDataset._structural_tokens

        if self.load_embeddings and EfficientProteinDataset._embeddings_dict is None:
            embeddings_dir = os.path.join(processed_data_path, "embeddings")
            if os.path.exists(embeddings_dir):
                EfficientProteinDataset._embeddings_dict = self._load_embeddings_from_directory(embeddings_dir)
                print(f"Loaded and cached embeddings for {len(EfficientProteinDataset._embeddings_dict)} proteins.")
            else:
                print("Warning: Embeddings directory not found. Continuing without pre-computed embeddings.")
                self.load_embeddings = False
        
        self.embeddings_dict = EfficientProteinDataset._embeddings_dict

        mapping_file = os.path.join(processed_data_path, "id_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                self.id_mapping = {int(k): v for k, v in json.load(f).items()}
        else:
            self.id_mapping = {i: p['id'] for i, p in enumerate(self.proteins)}

    def _load_embeddings_from_directory(self, embeddings_dir):
        """
        Load embeddings from the embeddings directory
        Each protein should have an embedding file named {protein_id}_gearnet_embeddings.pkl
        """
        embeddings_dict = {}

        # Look for embedding files in the directory
        for filename in os.listdir(embeddings_dir):
            if filename.endswith('_gearnet_embeddings.pkl'):
                protein_id = filename.replace('_gearnet_embeddings.pkl', '')

                # Load the embedding file
                embedding_path = os.path.join(embeddings_dir, filename)
                try:
                    with open(embedding_path, 'rb') as f:
                        embedding_data = pickle.load(f)
                        # Store only the embeddings array and ensure it's a proper numpy array
                        embeddings = embedding_data['embeddings']
                        if isinstance(embeddings, np.ndarray):
                            # Create a new array to avoid reference issues
                            embeddings_dict[protein_id] = np.array(embeddings, copy=True)
                        else:
                            embeddings_dict[protein_id] = embeddings
                except Exception as e:
                    print(f"Error loading embedding file {embedding_path}: {e}")

        return embeddings_dict

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        protein = self.proteins[idx]

        sequence = protein['sequence']
        n_coords = protein['n_coords']
        ca_coords = protein['ca_coords']
        c_coords = protein['c_coords']

        # Truncate if necessary
        if len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
            n_coords = n_coords[:self.max_seq_len]
            ca_coords = ca_coords[:self.max_seq_len]
            c_coords = c_coords[:self.max_seq_len]

        # Create token mapping (simple mapping for now)
        token_ids = self.sequence_to_tokens(sequence)

        # Pad or truncate to max length
        if len(token_ids) < self.max_seq_len:
            padding_length = self.max_seq_len - len(token_ids)
            token_ids.extend([1] * padding_length)  # 1 for <pad> token in ESM2
            padding_coords = np.zeros((padding_length, 3))
            n_coords = np.vstack([n_coords, padding_coords])
            ca_coords = np.vstack([ca_coords, padding_coords])
            c_coords = np.vstack([c_coords, padding_coords])

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if token != 1 else 0 for token in token_ids]  # 1 is <pad> in ESM2

        result = {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'n_coords': torch.tensor(n_coords, dtype=torch.float32),
            'ca_coords': torch.tensor(ca_coords, dtype=torch.float32),
            'c_coords': torch.tensor(c_coords, dtype=torch.float32),
            'seq_len': len(protein['sequence']),
            'protein_id': protein['id'],
            'index': idx
        }

        # Add pre-computed embeddings if available
        if self.load_embeddings and protein['id'] in self.embeddings_dict:
            embeddings = self.embeddings_dict[protein['id']]
            # Truncate if necessary
            if len(embeddings) > self.max_seq_len:
                embeddings = embeddings[:self.max_seq_len]
            # Pad or truncate to max length
            if len(embeddings) < self.max_seq_len:
                padding_length = self.max_seq_len - len(embeddings)
                padding_embeddings = np.zeros((padding_length, embeddings.shape[-1]))
                embeddings = np.vstack([embeddings, padding_embeddings])
            result['precomputed_embeddings'] = torch.tensor(embeddings, dtype=torch.float32)

        # Add structural tokens if available
        # Use try-except to handle missing structural tokens gracefully
        if self.include_structural_tokens:
            try:
                if idx < len(self.structural_tokens):
                    struct_tokens = self.structural_tokens[idx]
                    struct_seq = struct_tokens['structural_tokens']  # Assuming it's in the same format

                    # Truncate structural tokens to match sequence length
                    if len(struct_seq) > self.max_seq_len:
                        struct_seq = struct_seq[:self.max_seq_len]

                    # Pad or truncate structural tokens to max length
                    if len(struct_seq) < self.max_seq_len:
                        padding_length = self.max_seq_len - len(struct_seq)
                        struct_seq.extend([-100] * padding_length)  # Use -100 as ignore index for padding

                    result['structural_tokens'] = torch.tensor(struct_seq, dtype=torch.long)
            except (IndexError, KeyError, TypeError) as e:
                # Skip structural tokens for this sample if there's any issue
                pass

        return result

    def sequence_to_tokens(self, sequence):
        """
        Convert amino acid sequence to token IDs using ESM2 vocabulary

        ESM2 token mapping:
        - 0: <cls>
        - 1: <pad>
        - 2: <eos>
        - 3: <unk>
        - 4-23: Standard amino acids (L, A, G, V, S, E, R, T, I, D, P, K, Q, N, F, Y, M, H, W, C)
        - 32: <mask>
        """
        # ESM2 amino acid to token ID mapping (IDs 4-23)
        # Order: L, A, G, V, S, E, R, T, I, D, P, K, Q, N, F, Y, M, H, W, C
        aa_to_id = {
            'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11,
            'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18,
            'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23
        }

        # Use <unk> token (3) for unknown amino acids, <pad> token (1) for padding
        tokens = [aa_to_id.get(aa, 3) for aa in sequence]  # 3 for unknown amino acids
        return tokens