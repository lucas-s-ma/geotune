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
        num_structural_classes: int = 20,
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
        
        # Apply attention mask by setting ignored positions to ignore_index
        if attention_mask is not None:
            mask_flat = attention_mask.view(-1)  # (batch_size * seq_len,)
            tokens_flat = torch.where(mask_flat.bool(), tokens_flat, torch.tensor(-100, device=tokens_flat.device))
        
        # Calculate cross-entropy loss
        physical_loss = self.ce_loss(logits_flat, tokens_flat)
        
        return physical_loss


class PretrainedGNNWrapper(nn.Module):
    """
    Wrapper for a frozen, pre-trained protein Graph Neural Network (e.g. GearNet)
    This module is frozen during training to provide structural embeddings
    Can load from local path or HuggingFace hub if available
    """
    def __init__(self, model_path=None, hidden_dim=512, freeze=True, use_gearnet_stub=False):
        """
        Args:
            model_path: Path to pre-trained model (not currently available for GearNet)
            hidden_dim: Hidden dimension for the model
            freeze: Whether to freeze the model parameters
            use_gearnet_stub: Whether to use the stub implementation (avoids TorchDrug import)
        """
        super().__init__()

        # Check if we should use the stub FIRST (before trying to import TorchDrug)
        if use_gearnet_stub:
            # Use our simplified GearNet implementation - no TorchDrug import needed
            self.backbone = self._create_stub_gearnet(hidden_dim, freeze)
            print("Using stub implementation for pre-trained GNN (avoiding TorchDrug)")
        else:
            # Try to import and use the real GearNet implementation
            try:
                from models.gearnet_model import create_pretrained_gearnet
                self.backbone = create_pretrained_gearnet(
                    hidden_dim=hidden_dim,
                    pretrained_path=model_path,
                    freeze=freeze
                )
                print("Successfully loaded GearNet implementation from TorchDrug")
            except (ImportError, AttributeError) as e:
                print(f"Could not load TorchDrug GearNet implementation: {e}")
                print("Falling back to stub implementation")
                self.backbone = self._create_stub_gearnet(hidden_dim, freeze)
    
    def _create_stub_gearnet(self, hidden_dim, freeze):
        """Create a stub GearNet implementation if the real one is not available"""
        return StubGearNetWrapper(hidden_dim=hidden_dim, freeze=freeze)
    
    def forward(self, n_coords, ca_coords, c_coords):
        """
        Forward pass through the frozen GNN to get structural embeddings
        
        Args:
            n_coords: (batch_size, seq_len, 3) N atom coordinates
            ca_coords: (batch_size, seq_len, 3) CA atom coordinates
            c_coords: (batch_size, seq_len, 3) C atom coordinates
        
        Returns:
            pGNN_embeddings: (batch_size, seq_len, hidden_dim) structural embeddings
        """
        return self.backbone(n_coords, ca_coords, c_coords)


class StubGearNetWrapper(nn.Module):
    """
    Stub implementation of GearNet - this is the old implementation for fallback
    """
    def __init__(self, hidden_dim=512, num_layers=4, freeze=True):
        """
        Initialize GearNet wrapper - this is a simplified placeholder
        In practice, you would load an actual pre-trained GearNet model

        Args:
            hidden_dim: Hidden dimension for the GNN (should match ESM hidden dim)
            num_layers: Number of GNN layers
            freeze: Whether to freeze the model parameters during training
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Placeholder: This would be replaced with an actual GearNet implementation
        # For now, we'll create a simplified structure encoder
        # Input: 3D coords (9 dimensions: N, CA, C = 3*3) + hidden_dim for features
        initial_input_dim = 9  # 3 coords each for N, CA, C atoms
        self.graph_conv_layers = nn.ModuleList()

        # Create layers with compatible dimensions
        for i in range(num_layers):
            input_dim = initial_input_dim if i == 0 else hidden_dim
            self.graph_conv_layers.append(nn.Linear(input_dim, hidden_dim))

        # No additional projection needed - output hidden_dim directly to match ESM
        # The structure_alignment_loss will handle any necessary projections

        if freeze:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, n_coords, ca_coords, c_coords):
        """
        Forward pass through the GNN to get structural embeddings
        In a real implementation, this would process the 3D graph structure
        
        Args:
            n_coords: (batch_size, seq_len, 3) N atom coordinates
            ca_coords: (batch_size, seq_len, 3) CA atom coordinates
            c_coords: (batch_size, seq_len, 3) C atom coordinates
        
        Returns:
            pGNN_embeddings: (batch_size, seq_len, hidden_dim) structural embeddings
        """
        # Create initial node features from coordinates (simplified approach)
        # In a real GNN, this would include edge features and message passing
        batch_size, seq_len, _ = ca_coords.shape

        # Concatenate backbone coordinates as initial features
        initial_features = torch.cat([n_coords, ca_coords, c_coords], dim=-1)  # (B, L, 9)

        # Process through graph convolution layers (simplified)
        x = initial_features
        for i, layer in enumerate(self.graph_conv_layers):
            # Simplified message passing - in reality this would involve neighbor aggregation
            x = torch.relu(layer(x))

        # Return embeddings with shape (batch_size, seq_len, hidden_dim)
        # This should match the ESM model's hidden dimension
        return x