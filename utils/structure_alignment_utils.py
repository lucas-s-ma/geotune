"""
Structure alignment loss functions for protein language models
Implementation of the dual-task framework from 'Structure-Aligned Protein Language Model' (arXiv:2505.16896)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureAlignmentLoss(nn.Module):
    """
    Implements the structure alignment loss with both latent and physical level components
    """
    def __init__(
        self,
        hidden_dim: int,
        num_structural_classes: int = 21,  # Default to 21 for Foldseek (20 + X)
        shared_projection_dim: int = 512,
        latent_weight: float = 0.5,
        physical_weight: float = 0.5
    ):
        """
        Initialize the structure alignment loss module
        
        Args:
            hidden_dim: Dimension of the protein language model embeddings
            num_structural_classes: Number of structural alphabet classes (default 20 for Foldseek)
            shared_projection_dim: Dimension of the shared space for contrastive learning
            latent_weight: Weight for the latent-level loss
            physical_weight: Weight for the physical-level loss
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_structural_classes = num_structural_classes
        self.shared_projection_dim = shared_projection_dim
        self.latent_weight = latent_weight
        self.physical_weight = physical_weight
        
        # Projection layers for latent-level loss
        self.pLM_projection = nn.Linear(hidden_dim, shared_projection_dim)
        self.pGNN_projection = nn.Linear(hidden_dim, shared_projection_dim)  # Assuming pGNN has same hidden dim
        
        # Learnable temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(1.0))  # Initialize with value of 1.0
        
        # MLP head for physical-level loss (structural token prediction)
        self.structural_prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, num_structural_classes)
        )
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(
        self,
        pLM_embeddings,
        pGNN_embeddings,
        structure_tokens,
        attention_mask=None
    ):
        """
        Calculate the combined structure alignment loss
        
        Args:
            pLM_embeddings: (batch_size, seq_len, hidden_dim) - embeddings from protein language model
            pGNN_embeddings: (batch_size, seq_len, hidden_dim) - embeddings from pGNN (e.g. GearNet)
            structure_tokens: (batch_size, seq_len) - precomputed structural alphabet tokens
            attention_mask: (batch_size, seq_len) - attention mask to ignore padding positions
        
        Returns:
            A dictionary containing:
            - total_loss: Combined structure alignment loss
            - latent_loss: Latent-level contrastive loss
            - physical_loss: Physical-level structural token prediction loss
        """
        # Calculate latent-level loss
        latent_loss = self._calculate_latent_loss(pLM_embeddings, pGNN_embeddings, attention_mask)
        
        # Calculate physical-level loss
        physical_loss = self._calculate_physical_loss(pLM_embeddings, structure_tokens, attention_mask)
        
        # Combine losses
        total_loss = self.latent_weight * latent_loss + self.physical_weight * physical_loss
        
        return {
            'total_loss': total_loss,
            'latent_loss': latent_loss,
            'physical_loss': physical_loss
        }
    
    def _calculate_latent_loss(self, pLM_embeddings, pGNN_embeddings, attention_mask=None):
        """
        Calculate the latent-level contrastive loss between pLM and pGNN embeddings
        """
        batch_size, seq_len, hidden_dim = pLM_embeddings.shape
        
        # Project embeddings to shared space
        pLM_projected = self.pLM_projection(pLM_embeddings)  # (batch_size, seq_len, shared_dim)
        pGNN_projected = self.pGNN_projection(pGNN_embeddings)  # (batch_size, seq_len, shared_dim)
        
        # Reshape to (batch_size * seq_len, shared_dim) for easier computation
        pLM_flat = pLM_projected.view(-1, self.shared_projection_dim)  # (batch_size * seq_len, shared_dim)
        pGNN_flat = pGNN_projected.view(-1, self.shared_projection_dim)  # (batch_size * seq_len, shared_dim)
        
        # Create attention mask mask_flattened
        if attention_mask is not None:
            # Flatten attention mask to match the flattened embeddings
            mask_flat = attention_mask.view(-1).bool()  # (batch_size * seq_len,)
        else:
            mask_flat = torch.ones(batch_size * seq_len, dtype=torch.bool, device=pLM_embeddings.device)
        
        # Only compute loss for non-padded positions
        pLM_active = pLM_flat[mask_flat]  # (active_positions, shared_dim)
        pGNN_active = pGNN_flat[mask_flat]  # (active_positions, shared_dim)
        
        if pLM_active.size(0) == 0:
            return torch.tensor(0.0, device=pLM_embeddings.device, requires_grad=True)
        
        # Calculate similarity scores: (active_positions, active_positions)
        similarity_matrix = torch.matmul(pLM_active, pGNN_active.t()) * self.temperature  # Scaled dot product
        
        # Create labels for cross-entropy: diagonal positions are positive pairs
        batch_active = pLM_active.size(0)
        labels = torch.arange(batch_active, device=pLM_embeddings.device)
        
        # Loss from pLM to pGNN (a2g)
        a2g_loss = F.cross_entropy(similarity_matrix, labels)
        
        # Loss from pGNN to pLM (g2a)
        g2a_loss = F.cross_entropy(similarity_matrix.t(), labels)
        
        # Average the two directional losses
        latent_loss = 0.5 * (a2g_loss + g2a_loss)
        
        return latent_loss
    
    def _calculate_physical_loss(self, pLM_embeddings, structure_tokens, attention_mask=None):
        """
        Calculate the physical-level loss for structural token prediction
        """
        batch_size, seq_len, hidden_dim = pLM_embeddings.shape

        # Predict structural tokens
        logits = self.structural_prediction_head(pLM_embeddings)  # (batch_size, seq_len, num_classes)

        # Flatten for loss calculation
        logits_flat = logits.view(-1, self.num_structural_classes)  # (batch_size * seq_len, num_classes)
        tokens_flat = structure_tokens.view(-1)  # (batch_size * seq_len,)

        # Validate structural tokens are in the valid range [0, num_classes-1] BEFORE applying attention mask
        # Filter out invalid tokens by replacing them with -100 (ignore index)
        invalid_mask = (tokens_flat >= self.num_structural_classes) | (tokens_flat < 0) | (tokens_flat != tokens_flat)  # also check for NaN
        if invalid_mask.any():
            num_invalid = invalid_mask.sum().item()
            if num_invalid > 0:
                print(f"Warning: Found {num_invalid} invalid structural tokens (>= {self.num_structural_classes} or < 0 or NaN). Setting to ignore_index (-100).")
            # Set invalid tokens to -100 so they are ignored in loss calculation
            tokens_flat = torch.where(invalid_mask, torch.tensor(-100, device=tokens_flat.device), tokens_flat)

        # Apply attention mask by setting ignored positions to ignore_index (-100)
        if attention_mask is not None:
            mask_flat = attention_mask.view(-1)  # (batch_size * seq_len,)
            # Only mask tokens that are valid (not already marked as -100)
            # This ensures attention-masked positions are set to -100 to be ignored
            tokens_flat = torch.where(mask_flat.bool(), tokens_flat, torch.tensor(-100, device=tokens_flat.device))

        # Calculate cross-entropy loss - tokens with -100 will be ignored
        physical_loss = self.ce_loss(logits_flat, tokens_flat)

        return physical_loss



from torchdrug.models import GearNet
from torchdrug.data import Protein


class PretrainedGNNWrapper(nn.Module):
    """
    Wrapper for a frozen, pre-trained protein Graph Neural Network (e.g. GearNet)
    This module is frozen during training to provide structural embeddings
    """
    def __init__(self, model_path=None, hidden_dim=512, freeze=True):
        """
        Args:
            model_path: Path to pre-trained model (e.g. gearnet_edge.pth)
            hidden_dim: Hidden dimension for the model
            freeze: Whether to freeze the model parameters
        """
        super().__init__()

        # Use the full GearNet implementation
        self.backbone = GearNet(
            input_dim=21,  # 20 standard amino acids + 1 unknown
            hidden_dims=[hidden_dim, hidden_dim, hidden_dim],
            num_relation=7,  # Default for GearNet
            edge_input_dim=None,
            num_angle_bin=None,
            batch_norm=True,
            concat_hidden=True
        )
        print("Using full GearNet implementation from TorchDrug")

        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))['model']
                self.backbone.load_state_dict(state_dict)
                print(f"Loaded pre-trained GearNet weights from {model_path}")
            except FileNotFoundError:
                print(f"Warning: Pre-trained GearNet model not found at {model_path}. Using random initialization.")
            except Exception as e:
                print(f"Error loading pre-trained GearNet model: {e}. Using random initialization.")

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, protein_graph):
        """
        Forward pass through the frozen GNN to get structural embeddings.

        Args:
            protein_graph (torchdrug.data.Protein): A protein graph object from TorchDrug.
                This can be a single graph or a batch of graphs.

        Returns:
            pGNN_embeddings (torch.Tensor): (num_residues, hidden_dim) structural embeddings
                for all residues in the batch.
        """
        # The input to GearNet should be the graph and its residue features
        residue_features = protein_graph.residue_feature.float()
        
        # The GearNet model returns a dictionary of features
        features = self.backbone(protein_graph, residue_features)
        
        # We want the node-level (residue-level) embeddings
        pgnn_embeddings = features['node_feature']

        return pgnn_embeddings
