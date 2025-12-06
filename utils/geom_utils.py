"""
Geometric constraint functions for protein structure-based constraints
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GeometricConstraints(nn.Module):
    """
    Implements geometric constraints based on 3D protein structure information
    """
    def __init__(self, constraint_weight=1.0, dist_threshold=15.0):
        super().__init__()
        self.constraint_weight = constraint_weight
        self.dist_threshold = dist_threshold  # Distance threshold in Angstroms

    def forward(self, sequence_embeddings, geometric_coords, attention_mask=None):
        """
        Calculate geometric constraint loss
        
        Args:
            sequence_embeddings: (batch_size, seq_len, hidden_dim) - model embeddings
            geometric_coords: (batch_size, seq_len, 3) - 3D coordinates
            attention_mask: (batch_size, seq_len) - attention mask to ignore padding
        """
        batch_size, seq_len = sequence_embeddings.shape[:2]
        
        # Compute constraint losses
        distance_loss = self.distance_constraint_loss(sequence_embeddings, geometric_coords, attention_mask)
        angle_loss = self.angle_constraint_loss(sequence_embeddings, geometric_coords, attention_mask)
        neighborhood_loss = self.neighborhood_constraint_loss(sequence_embeddings, geometric_coords, attention_mask)
        
        # Combine losses
        total_constraint_loss = (
            distance_loss + 
            angle_loss + 
            neighborhood_loss
        )
        
        return {
            'total_constraint_loss': total_constraint_loss * self.constraint_weight,
            'distance_loss': distance_loss * self.constraint_weight,
            'angle_loss': angle_loss * self.constraint_weight,
            'neighborhood_loss': neighborhood_loss * self.constraint_weight
        }

    def distance_constraint_loss(self, sequence_embeddings, geometric_coords, attention_mask=None):
        """
        Constraint based on preserving inter-residue distances in 3D space.
        Uses Smooth L1 Loss for robustness.
        """
        batch_size, seq_len, hidden_dim = sequence_embeddings.shape
        
        # Subsample for efficiency with long sequences
        if seq_len > 128:
            sample_size = 128
            sampled_indices = torch.randperm(seq_len, device=sequence_embeddings.device)[:sample_size].sort().values
            
            sampled_embeddings = sequence_embeddings[:, sampled_indices, :]
            sampled_coords = geometric_coords[:, sampled_indices, :]
            
            if attention_mask is not None:
                sampled_mask = attention_mask[:, sampled_indices]
            else:
                sampled_mask = torch.ones(batch_size, sample_size, device=sequence_embeddings.device)
        else:
            sampled_embeddings = sequence_embeddings
            sampled_coords = geometric_coords
            sampled_mask = attention_mask if attention_mask is not None else torch.ones(batch_size, seq_len, device=sequence_embeddings.device)

        # Compute distance matrices
        true_distances = torch.cdist(sampled_coords, sampled_coords, p=2)
        
        normalized_embeddings = F.normalize(sampled_embeddings, p=2, dim=-1)
        embedding_distances = 1 - torch.matmul(normalized_embeddings, normalized_embeddings.transpose(-2, -1))

        # Create mask for valid pairs
        mask = sampled_mask.unsqueeze(1) * sampled_mask.unsqueeze(2)
        
        # Identify spatial neighbors
        spatial_neighbors = (true_distances < self.dist_threshold).float()
        valid_spatial_pairs = mask * spatial_neighbors
        
        # Normalize true distances for the loss
        normalized_true_distances = true_distances / self.dist_threshold
        
        # Use Smooth L1 Loss
        loss_fn = nn.SmoothL1Loss(reduction='none')
        distance_diff = loss_fn(embedding_distances, normalized_true_distances)
        
        # Apply mask and compute final loss
        constraint_loss = (distance_diff * valid_spatial_pairs).sum() / (valid_spatial_pairs.sum() + 1e-8)
        
        return constraint_loss

    def angle_constraint_loss(self, sequence_embeddings, geometric_coords, attention_mask=None):
        """
        Constraint based on preserving local geometric angles
        This function is O(n) and doesn't require optimization for long sequences
        """
        batch_size, seq_len, hidden_dim = sequence_embeddings.shape
        
        if seq_len < 3:
            return torch.tensor(0.0, device=sequence_embeddings.device)
        
        # Compute vectors for consecutive triplets (i-1, i, i+1)
        # coords shape: (B, L, 3)
        prev_coords = geometric_coords[:, :-2, :]  # (B, L-2, 3) - positions i-1
        curr_coords = geometric_coords[:, 1:-1, :]  # (B, L-2, 3) - positions i
        next_coords = geometric_coords[:, 2:, :]   # (B, L-2, 3) - positions i+1
        
        # Compute vectors
        v1 = prev_coords - curr_coords  # vector from i to i-1
        v2 = next_coords - curr_coords  # vector from i to i+1
        
        # Normalize vectors
        v1_norm = F.normalize(v1, p=2, dim=-1)
        v2_norm = F.normalize(v2, p=2, dim=-1)
        
        # Compute true angles (cosine of angle between vectors)
        true_angles_cos = torch.sum(v1_norm * v2_norm, dim=-1)  # (B, L-2)
        
        # Compute embedding-based angle proxies
        prev_embeds = sequence_embeddings[:, :-2, :]  # (B, L-2, H)
        curr_embeds = sequence_embeddings[:, 1:-1, :]  # (B, L-2, H)
        next_embeds = sequence_embeddings[:, 2:, :]   # (B, L-2, H)
        
        # Normalize embedding vectors
        prev_norm = F.normalize(prev_embeds, p=2, dim=-1)
        curr_norm = F.normalize(curr_embeds, p=2, dim=-1)
        next_norm = F.normalize(next_embeds, p=2, dim=-1)
        
        # Compute embedding angle proxies (cosine similarity of relative positions)
        embed_angle_1 = torch.sum(prev_norm * curr_norm, dim=-1)  # (B, L-2)
        embed_angle_2 = torch.sum(next_norm * curr_norm, dim=-1)  # (B, L-2)
        embed_angles_cos = (embed_angle_1 + embed_angle_2) / 2  # Average as proxy
        
        # Compute angle constraint loss
        angle_diff = torch.abs(embed_angles_cos - true_angles_cos)
        
        # Apply mask if provided
        if attention_mask is not None:
            valid_positions = attention_mask[:, 1:-1].float()  # (B, L-2)
            constraint_loss = torch.sum(angle_diff * valid_positions) / (torch.sum(valid_positions) + 1e-8)
        else:
            constraint_loss = torch.mean(angle_diff)
        
        return constraint_loss

    def neighborhood_constraint_loss(self, sequence_embeddings, geometric_coords, attention_mask=None):
        """
        Constraint based on preserving spatial neighborhood relationships
        Optimized to handle long sequences more efficiently by subsampling if needed
        """
        batch_size, seq_len, hidden_dim = sequence_embeddings.shape
        
        # For very long sequences, subsample to reduce computational load
        if seq_len > 100:  # If sequence is longer than 100 residues
            # Randomly sample up to 100 positions to reduce N^2 complexity
            max_sampled = 100
            sample_size = min(max_sampled, seq_len)
            
            # Randomly select indices to sample
            sampled_indices = torch.multinomial(
                torch.ones(seq_len, device=sequence_embeddings.device), 
                sample_size, 
                replacement=False
            ).sort().values  # Sort to maintain order
            
            # Sample embeddings and coordinates
            sampled_embeddings = sequence_embeddings[:, sampled_indices, :]
            sampled_coords = geometric_coords[:, sampled_indices, :]
            
            # Handle attention mask if present (sample the mask too)
            sampled_mask = None
            if attention_mask is not None:
                sampled_mask = attention_mask[:, sampled_indices]  # (B, S)
            
            # Compute distance matrix on subsampled data
            coords_expanded = sampled_coords.unsqueeze(2)  # (B, S, 1, 3)
            coords_transposed = sampled_coords.unsqueeze(1)  # (B, 1, S, 3)
            dist_matrix = torch.norm(coords_expanded - coords_transposed, dim=-1)  # (B, S, S)
        else:
            # For shorter sequences, use the full calculation
            sampled_embeddings = sequence_embeddings
            sampled_coords = geometric_coords
            sampled_mask = attention_mask
            sample_size = seq_len
            dist_matrix = None
            
            # Compute distance matrix
            coords_expanded = geometric_coords.unsqueeze(2)  # (B, L, 1, 3)
            coords_transposed = geometric_coords.unsqueeze(1)  # (B, 1, L, 3)
            dist_matrix = torch.norm(coords_expanded - coords_transposed, dim=-1)  # (B, L, L)
        
        # Find k-nearest spatial neighbors for each residue
        k = min(10, sample_size)  # Use 10 neighbors or fewer if sequence is shorter
        
        # Get k-nearest neighbors (excluding self)
        # Add large value to diagonal to exclude self-connections
        diag_mask = torch.eye(sample_size, device=sampled_coords.device).unsqueeze(0).expand(batch_size, -1, -1) * 1e9
        masked_dists = dist_matrix + diag_mask
        _, nearest_indices = torch.topk(masked_dists, k, dim=-1, largest=False)  # (B, S, k)
        
        # Gather embedding neighbors
        batch_idx = torch.arange(batch_size, device=sampled_embeddings.device).unsqueeze(1).unsqueeze(2)
        seq_idx = torch.arange(sample_size, device=sampled_embeddings.device).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, k)
        neighbor_idx = nearest_indices  # (B, S, k)
        
        neighbor_embeddings = sampled_embeddings[batch_idx, neighbor_idx, :]  # (B, S, k, H)
        center_embeddings = sampled_embeddings.unsqueeze(2).expand(-1, -1, k, -1)  # (B, S, k, H)
        
        # Compute similarity within neighborhoods in embedding space
        neighbor_similarities = F.cosine_similarity(center_embeddings, neighbor_embeddings, dim=-1)  # (B, S, k)
        avg_neighbor_similarity = torch.mean(neighbor_similarities, dim=-1)  # (B, S)
        
        # Compute expected similarity based on spatial distances
        nearest_distances = torch.gather(dist_matrix, 2, nearest_indices)  # (B, S, k)
        # Normalize distances to [0,1] range using threshold
        normalized_distances = torch.clamp(nearest_distances / self.dist_threshold, 0, 1)
        expected_similarities = 1 - normalized_distances  # Closer residues should have higher similarity
        avg_expected_similarity = torch.mean(expected_similarities, dim=-1)  # (B, S)
        
        # Compute neighborhood constraint loss
        neighbor_diff = torch.abs(avg_neighbor_similarity - avg_expected_similarity)
        
        # Apply mask if provided
        if sampled_mask is not None:
            valid_positions = sampled_mask.float()  # (B, S)
            constraint_loss = torch.sum(neighbor_diff * valid_positions) / (torch.sum(valid_positions) + 1e-8)
        else:
            constraint_loss = torch.mean(neighbor_diff)
        
        return constraint_loss

import matplotlib.pyplot as plt

def visualize_distance_violations(true_distances, embedding_distances, spatial_neighbors_mask, save_path="distance_violations.png"):
    """
    Visualizes the relationship between true and embedding distances for spatial neighbors.
    """
    # Flatten and filter data for plotting
    true_dist_flat = true_distances[spatial_neighbors_mask].detach().cpu().numpy()
    embed_dist_flat = embedding_distances[spatial_neighbors_mask].detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.scatter(true_dist_flat, embed_dist_flat, alpha=0.3, label="Spatial Neighbors")
    
    # Highlight violations (e.g., where embedding distance is high for close true distances)
    violations = (embed_dist_flat > 0.5) & (true_dist_flat < 10)
    plt.scatter(true_dist_flat[violations], embed_dist_flat[violations], color='r', alpha=0.5, label="Violations")
    
    plt.xlabel("True Distance (Angstroms)")
    plt.ylabel("Embedding Distance (1 - Cosine Similarity)")
    plt.title("True vs. Embedding Distance for Spatial Neighbors")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(save_path)
    print(f"Distance violation plot saved to {save_path}")
    plt.close()


class DistanceConstraint(nn.Module):
    """
    Simple distance-based constraint
    """
    def __init__(self, threshold=10.0, margin=1.0):
        super().__init__()
        self.threshold = threshold
        self.margin = margin

    def forward(self, embeddings, coords, mask=None):
        """
        Ensures that residues close in 3D space have similar embeddings
        """
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Normalize embeddings
        normalized_embeds = F.normalize(embeddings, p=2, dim=-1)
        
        # Compute embedding distance matrix
        embed_dist = self._compute_distance_matrix(normalized_embeds)  # (B, L, L)
        
        # Compute coordinate distance matrix
        coord_dist = self._compute_distance_matrix(coords)  # (B, L, L)
        
        # Identify spatial neighbors
        neighbors = (coord_dist < self.threshold).float()
        
        # Loss: penalize when embedding distance is large for spatial neighbors
        neighbor_loss = neighbors * F.relu(embed_dist - self.margin)
        
        if mask is not None:
            mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
            neighbor_loss = neighbor_loss * mask_2d
            valid_count = mask_2d.sum()
        else:
            valid_count = neighbor_loss.numel()
        
        return neighbor_loss.sum() / (valid_count + 1e-8)

    def _compute_distance_matrix(self, x):
        """
        Compute pairwise distance matrix
        """
        x_expanded = x.unsqueeze(2)
        x_transposed = x.unsqueeze(1)
        dist_matrix = torch.norm(x_expanded - x_transposed, p=2, dim=-1)
        return dist_matrix


class TorsionConstraint(nn.Module):
    """
    Constraint based on torsion angles in protein structure
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, embeddings, coords, mask=None):
        """
        Constraint based on preserving torsion angle information
        """
        # Compute backbone torsion information implicitly
        batch_size, seq_len, _ = coords.shape
        
        if seq_len < 4:
            return torch.tensor(0.0, device=coords.device)
        
        # Get backbone atoms: C, N, CA, C for phi/psi angles
        # For simplicity, we'll use coordinates directly
        coords_trimmed = coords[:, 2:-1, :]  # Leave space for 4-atom dihedral computation
        embeds_trimmed = embeddings[:, 2:-1, :]
        
        # Approximate torsion constraint through local geometric properties
        # This is a simplified version - a full implementation would compute actual dihedral angles
        local_geom = self._compute_local_geometry(coords_trimmed)
        local_embed_similarity = self._compute_local_similarity(embeds_trimmed)
        
        diff = torch.abs(local_geom - local_embed_similarity)
        
        if mask is not None:
            valid_mask = mask[:, 2:-1]
            diff = diff * valid_mask.unsqueeze(-1)
            valid_count = valid_mask.sum() * diff.shape[-1]
        else:
            valid_count = diff.numel()
        
        return diff.sum() / (valid_count + 1e-8)

    def _compute_local_geometry(self, coords):
        """
        Compute local geometric properties (simplified)
        """
        # Compute local curvature based on 3 consecutive vectors
        v1 = coords[:, :-2, :] - coords[:, 1:-1, :]
        v2 = coords[:, 1:-1, :] - coords[:, 2:, :]
        
        cos_angles = F.cosine_similarity(v1, v2, dim=-1).unsqueeze(-1)
        return cos_angles

    def _compute_local_similarity(self, embeddings):
        """
        Compute local embedding similarity (simplified)
        """
        norm_embeds = F.normalize(embeddings, p=2, dim=-1)
        similarity1 = torch.sum(norm_embeds[:, :-2, :] * norm_embeds[:, 1:-1, :], dim=-1).unsqueeze(-1)  # Adjacent
        similarity2 = torch.sum(norm_embeds[:, 1:-1, :] * norm_embeds[:, 2:, :], dim=-1).unsqueeze(-1)   # Adjacent
        return (similarity1 + similarity2) / 2