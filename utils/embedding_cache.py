"""
Utility for caching structural embeddings during training
Generates embeddings on-the-fly and saves them for future epochs
"""
import os
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict


class EmbeddingCache:
    """
    Manages caching of structural embeddings during training

    Features:
    - Generates embeddings on-the-fly when not cached
    - Saves to disk for reuse in later epochs
    - Thread-safe file operations
    - Automatic cache directory management
    """

    def __init__(self, cache_dir: str, gnn_model, device, verbose=True):
        """
        Args:
            cache_dir: Directory to store cached embeddings
            gnn_model: The GNN model (e.g., PretrainedGNNWrapper) for generating embeddings
            device: torch.device to use for generation
            verbose: Whether to print cache status messages
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.gnn_model = gnn_model
        self.device = device
        self.verbose = verbose

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.generated_count = 0

        if self.verbose:
            print(f"Embedding cache initialized at: {self.cache_dir}")

    def _get_cache_path(self, protein_id: str) -> Path:
        """Get the cache file path for a protein ID"""
        return self.cache_dir / f"{protein_id}_embedding.pkl"

    def get_embedding(self, protein_id: str, n_coords: torch.Tensor,
                      ca_coords: torch.Tensor, c_coords: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for a protein, from cache if available or generate if not

        Args:
            protein_id: Unique protein identifier
            n_coords: (seq_len, 3) N atom coordinates
            ca_coords: (seq_len, 3) CA atom coordinates
            c_coords: (seq_len, 3) C atom coordinates

        Returns:
            embeddings: (seq_len, hidden_dim) structural embeddings
        """
        cache_path = self._get_cache_path(protein_id)

        # Try to load from cache
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)

                # Verify dimensions match
                embeddings = cache_data['embeddings']
                if embeddings.shape[0] == n_coords.shape[0]:
                    self.cache_hits += 1
                    # Convert to tensor and move to device
                    return torch.from_numpy(embeddings).to(self.device)
                else:
                    if self.verbose:
                        print(f"Warning: Cached embedding dimension mismatch for {protein_id}, regenerating")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to load cache for {protein_id}: {e}, regenerating")

        # Generate embedding if not in cache or cache invalid
        self.cache_misses += 1
        self.generated_count += 1

        embedding = self._generate_embedding(n_coords, ca_coords, c_coords)

        # Save to cache
        self._save_to_cache(protein_id, embedding, n_coords.shape[0])

        return embedding

    def _generate_embedding(self, n_coords: torch.Tensor,
                           ca_coords: torch.Tensor,
                           c_coords: torch.Tensor) -> torch.Tensor:
        """
        Generate embedding using the GNN model

        Args:
            n_coords: (seq_len, 3) or (1, seq_len, 3)
            ca_coords: (seq_len, 3) or (1, seq_len, 3)
            c_coords: (seq_len, 3) or (1, seq_len, 3)

        Returns:
            embedding: (seq_len, hidden_dim)
        """
        # Ensure batch dimension
        if n_coords.dim() == 2:
            n_coords = n_coords.unsqueeze(0)
            ca_coords = ca_coords.unsqueeze(0)
            c_coords = c_coords.unsqueeze(0)

        # Move to device
        n_coords = n_coords.to(self.device)
        ca_coords = ca_coords.to(self.device)
        c_coords = c_coords.to(self.device)

        # Set GNN model to eval mode (critical for validation!)
        self.gnn_model.eval()

        # Generate embedding
        with torch.no_grad():
            embedding = self.gnn_model(n_coords, ca_coords, c_coords)

        # Remove batch dimension if added
        if embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)

        return embedding

    def _save_to_cache(self, protein_id: str, embedding: torch.Tensor, seq_len: int):
        """Save embedding to cache file"""
        cache_path = self._get_cache_path(protein_id)

        try:
            # Convert to numpy for storage (create a clean copy to avoid pickle issues)
            embedding_np = np.array(embedding.cpu().numpy(), copy=True, dtype=np.float32)

            cache_data = {
                'protein_id': protein_id,
                'embeddings': embedding_np,
                'sequence_length': seq_len,
                'shape': embedding_np.shape
            }

            # Atomic write: write to temp file then rename
            temp_path = cache_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(cache_data, f)

            # Atomic rename
            temp_path.rename(cache_path)

        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to cache embedding for {protein_id}: {e}")

    def get_batch_embeddings(self, batch_data: Dict) -> torch.Tensor:
        """
        Get embeddings for a batch of proteins

        Args:
            batch_data: Dictionary containing:
                - 'protein_ids': List of protein IDs
                - 'n_coords': (batch_size, seq_len, 3)
                - 'ca_coords': (batch_size, seq_len, 3)
                - 'c_coords': (batch_size, seq_len, 3)

        Returns:
            embeddings: (batch_size, seq_len, hidden_dim)
        """
        protein_ids = batch_data['protein_ids']
        n_coords = batch_data['n_coords']
        ca_coords = batch_data['ca_coords']
        c_coords = batch_data['c_coords']

        batch_size = len(protein_ids)
        embeddings_list = []

        for i in range(batch_size):
            embedding = self.get_embedding(
                protein_ids[i],
                n_coords[i],
                ca_coords[i],
                c_coords[i]
            )
            embeddings_list.append(embedding)

        # Stack into batch
        embeddings_batch = torch.stack(embeddings_list, dim=0)

        return embeddings_batch

    def print_statistics(self):
        """Print cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

        print("\n" + "=" * 60)
        print("Embedding Cache Statistics")
        print("=" * 60)
        print(f"Cache hits:        {self.cache_hits:,}")
        print(f"Cache misses:      {self.cache_misses:,}")
        print(f"Total requests:    {total_requests:,}")
        print(f"Hit rate:          {hit_rate:.1f}%")
        print(f"Generated:         {self.generated_count:,}")
        print(f"Cache directory:   {self.cache_dir}")
        print("=" * 60 + "\n")

    def clear_cache(self):
        """Clear all cached embeddings"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                print(f"Cache cleared: {self.cache_dir}")
