"""
Simple structural encoder as fallback when GearNet/TorchDrug doesn't work
Computes distance-based and geometric features directly
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleStructuralEncoder(nn.Module):
    """
    Simple MLP-based structural encoder that takes CA coordinates
    and produces structural embeddings without needing GearNet/TorchDrug

    Much faster and more reliable than TorchDrug when that's broken.
    """
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input: distance features + coordinate features
        # Distance features: pairwise distances to k nearest neighbors (k=10, so 10 features)
        # Coordinate features: CA coordinates (3 features)
        input_dim = 10 + 3  # 13 features per residue

        # Simple MLP to encode structural information
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, n_coords, ca_coords, c_coords):
        """
        Args:
            n_coords: (batch_size, seq_len, 3) - not used but kept for compatibility
            ca_coords: (batch_size, seq_len, 3)
            c_coords: (batch_size, seq_len, 3) - not used but kept for compatibility

        Returns:
            embeddings: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = ca_coords.shape
        device = ca_coords.device

        # Compute pairwise distances
        # dist_matrix: (batch_size, seq_len, seq_len)
        dist_matrix = torch.cdist(ca_coords, ca_coords)

        # Get k nearest neighbors (k=10)
        k = min(10, seq_len - 1)
        if k > 0:
            # topk_dists: (batch_size, seq_len, k+1)
            topk_dists, _ = torch.topk(dist_matrix, k + 1, largest=False, dim=-1)
            # Remove self-distance (first column)
            knn_distances = topk_dists[:, :, 1:]  # (batch_size, seq_len, k)

            # Pad to always have 10 features
            if k < 10:
                padding = torch.zeros(batch_size, seq_len, 10 - k, device=device)
                knn_distances = torch.cat([knn_distances, padding], dim=-1)
        else:
            knn_distances = torch.zeros(batch_size, seq_len, 10, device=device)

        # Concatenate distance features with coordinates
        # features: (batch_size, seq_len, 13)
        features = torch.cat([knn_distances, ca_coords], dim=-1)

        # Encode through MLP
        embeddings = self.encoder(features)  # (batch_size, seq_len, hidden_dim)

        return embeddings


def create_simple_structural_encoder(hidden_dim=512, freeze=True):
    """
    Factory function to create a simple structural encoder

    Args:
        hidden_dim: Output embedding dimension
        freeze: Whether to freeze the model (for compatibility, not used here)

    Returns:
        SimpleStructuralEncoder instance
    """
    model = SimpleStructuralEncoder(hidden_dim=hidden_dim)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    print(f"Created SimpleStructuralEncoder with hidden_dim={hidden_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model
