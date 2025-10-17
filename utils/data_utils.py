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
            token_ids.extend([0] * padding_length)  # 0 for padding token
            padding_coords = np.zeros((padding_length, 3))
            n_coords = np.vstack([n_coords, padding_coords])
            ca_coords = np.vstack([ca_coords, padding_coords])
            c_coords = np.vstack([c_coords, padding_coords])
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if token != 0 else 0 for token in token_ids]
        
        result = {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'n_coords': torch.tensor(n_coords, dtype=torch.float32),
            'ca_coords': torch.tensor(ca_coords, dtype=torch.float32),
            'c_coords': torch.tensor(c_coords, dtype=torch.float32),
            'seq_len': len(protein['sequence']),
            'protein_id': protein['id']
        }
        
        return result
    
    def sequence_to_tokens(self, sequence):
        """Convert amino acid sequence to token IDs"""
        # Simple mapping: A->5, C->6, D->7, E->8, F->9, G->10, H->11, I->12, K->13, L->14, 
        # M->15, N->16, P->17, Q->18, R->19, S->20, T->21, V->22, W->23, Y->24
        # 1 for BOS, 2 for EOS, 3 for MASK, 4 for PAD
        aa_to_id = {
            'A': 5, 'R': 6, 'N': 7, 'D': 8, 'C': 9, 'Q': 10, 'E': 11, 'G': 12,
            'H': 13, 'I': 14, 'L': 15, 'K': 16, 'M': 17, 'F': 18, 'P': 19,
            'S': 20, 'T': 21, 'W': 22, 'Y': 23, 'V': 24
        }
        
        tokens = [aa_to_id.get(aa, 0) for aa in sequence]  # 0 for unknown amino acids
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
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'n_coords': n_coords,
        'ca_coords': ca_coords,
        'c_coords': c_coords,
        'seq_lens': [item['seq_len'] for item in batch],
        'protein_ids': [item['protein_id'] for item in batch]
    }


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
    def __init__(self, processed_data_path, max_seq_len=1024):
        """
        Args:
            processed_data_path: Path to directory containing pre-processed dataset.pkl
            max_seq_len: Maximum sequence length
        """
        self.max_seq_len = max_seq_len
        
        # Load the pre-processed dataset
        dataset_file = os.path.join(processed_data_path, "processed_dataset.pkl")
        mapping_file = os.path.join(processed_data_path, "id_mapping.json")
        
        if os.path.exists(dataset_file):
            with open(dataset_file, 'rb') as f:
                self.proteins = pickle.load(f)
            print(f"Loaded {len(self.proteins)} proteins from pre-processed dataset")
        else:
            raise FileNotFoundError(f"Processed dataset not found at {dataset_file}. "
                                  f"Run process_data.py to create the processed dataset first.")
        
        # Load the ID mapping if it exists
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                self.id_mapping = json.load(f)
                # Convert string keys back to integers
                self.id_mapping = {int(k): v for k, v in self.id_mapping.items()}
        else:
            # Create ID mapping if not found
            self.id_mapping = {i: protein['id'] for i, protein in enumerate(self.proteins)}
    
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
            token_ids.extend([0] * padding_length)  # 0 for padding token
            padding_coords = np.zeros((padding_length, 3))
            n_coords = np.vstack([n_coords, padding_coords])
            ca_coords = np.vstack([ca_coords, padding_coords])
            c_coords = np.vstack([c_coords, padding_coords])
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if token != 0 else 0 for token in token_ids]
        
        result = {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'n_coords': torch.tensor(n_coords, dtype=torch.float32),
            'ca_coords': torch.tensor(ca_coords, dtype=torch.float32),
            'c_coords': torch.tensor(c_coords, dtype=torch.float32),
            'seq_len': len(protein['sequence']),
            'protein_id': protein['id']
        }
        
        return result
    
    def sequence_to_tokens(self, sequence):
        """Convert amino acid sequence to token IDs"""
        # Simple mapping: A->5, C->6, D->7, E->8, F->9, G->10, H->11, I->12, K->13, L->14, 
        # M->15, N->16, P->17, Q->18, R->19, S->20, T->21, V->22, W->23, Y->24
        # 1 for BOS, 2 for EOS, 3 for MASK, 4 for PAD
        aa_to_id = {
            'A': 5, 'R': 6, 'N': 7, 'D': 8, 'C': 9, 'Q': 10, 'E': 11, 'G': 12,
            'H': 13, 'I': 14, 'L': 15, 'K': 16, 'M': 17, 'F': 18, 'P': 19,
            'S': 20, 'T': 21, 'W': 22, 'Y': 23, 'V': 24
        }
        
        tokens = [aa_to_id.get(aa, 0) for aa in sequence]  # 0 for unknown amino acids
        return tokens