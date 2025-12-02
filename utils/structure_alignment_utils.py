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
            - latent_loss_per_sample: Per-sample latent-level losses
            - physical_loss_per_sample: Per-sample physical-level losses
        """
        # Calculate latent-level loss and per-sample components
        latent_loss, latent_loss_per_sample = self._calculate_latent_loss_with_per_sample(pLM_embeddings, pGNN_embeddings, attention_mask)

        # Calculate physical-level loss and per-sample components
        physical_loss, physical_loss_per_sample = self._calculate_physical_loss_with_per_sample(pLM_embeddings, structure_tokens, attention_mask)

        # Combine losses
        total_loss = self.latent_weight * latent_loss + self.physical_weight * physical_loss

        return {
            'total_loss': total_loss,
            'latent_loss': latent_loss,
            'physical_loss': physical_loss,
            'latent_loss_per_sample': latent_loss_per_sample,
            'physical_loss_per_sample': physical_loss_per_sample
        }

    def _calculate_latent_loss_with_per_sample(self, pLM_embeddings, pGNN_embeddings, attention_mask=None):
        """
        Calculate the latent-level contrastive loss between pLM and pGNN embeddings with per-sample outputs
        """
        batch_size, seq_len, hidden_dim = pLM_embeddings.shape

        # Project embeddings to shared space
        pLM_projected = self.pLM_projection(pLM_embeddings)  # (batch_size, seq_len, shared_dim)
        pGNN_projected = self.pGNN_projection(pGNN_embeddings)  # (batch_size, seq_len, shared_dim)

        # Calculate per-sample latent losses
        latent_loss_per_sample = torch.zeros(batch_size, device=pLM_embeddings.device)

        for i in range(batch_size):
            # Get embeddings for single sample
            pLM_single = pLM_projected[i]  # (seq_len, shared_dim)
            pGNN_single = pGNN_projected[i]  # (seq_len, shared_dim)

            # Apply mask for this sample if provided
            if attention_mask is not None:
                sample_mask = attention_mask[i].bool()  # (seq_len,)
                pLM_active = pLM_single[sample_mask]  # (active_len, shared_dim)
                pGNN_active = pGNN_single[sample_mask]  # (active_len, shared_dim)
            else:
                pLM_active = pLM_single  # (seq_len, shared_dim)
                pGNN_active = pGNN_single  # (seq_len, shared_dim)

            # If no active positions, continue
            if pLM_active.size(0) == 0:
                continue

            # Calculate similarity scores: (active_len, active_len)
            similarity_matrix = torch.matmul(pLM_active, pGNN_active.t()) * self.temperature  # Scaled dot product

            # Create labels for cross-entropy: diagonal positions are positive pairs
            active_len = pLM_active.size(0)
            labels = torch.arange(active_len, device=pLM_embeddings.device)

            # Loss from pLM to pGNN (a2g)
            a2g_loss = F.cross_entropy(similarity_matrix, labels)

            # Loss from pGNN to pLM (g2a)
            g2a_loss = F.cross_entropy(similarity_matrix.t(), labels)

            # Average the two directional losses for this sample
            sample_loss = 0.5 * (a2g_loss + g2a_loss)
            latent_loss_per_sample[i] = sample_loss

        # Overall latent loss is the average of per-sample losses
        latent_loss = latent_loss_per_sample.mean() if latent_loss_per_sample.numel() > 0 else torch.tensor(0.0, device=pLM_embeddings.device)

        return latent_loss, latent_loss_per_sample

    def _calculate_latent_loss(self, pLM_embeddings, pGNN_embeddings, attention_mask=None):
        """
        Calculate the latent-level contrastive loss between pLM and pGNN embeddings (original method)
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

    def _calculate_physical_loss_with_per_sample(self, pLM_embeddings, structure_tokens, attention_mask=None):
        """
        Calculate the physical-level loss for structural token prediction with per-sample outputs
        """
        batch_size, seq_len, hidden_dim = pLM_embeddings.shape

        # Predict structural tokens
        logits = self.structural_prediction_head(pLM_embeddings)  # (batch_size, seq_len, num_classes)

        # Calculate per-sample physical losses
        physical_loss_per_sample = torch.zeros(batch_size, device=pLM_embeddings.device)

        for i in range(batch_size):
            # Get logits and tokens for single sample
            logits_single = logits[i]  # (seq_len, num_classes)
            tokens_single = structure_tokens[i]  # (seq_len,)

            # Apply mask if provided
            if attention_mask is not None:
                sample_mask = attention_mask[i]  # (seq_len,)
                # Only consider valid positions (not masked)
                valid_positions = sample_mask.bool() & (tokens_single >= 0)
            else:
                # Only consider non-padded positions (tokens >= 0)
                valid_positions = tokens_single >= 0

            # Get valid logits and tokens
            valid_logits = logits_single[valid_positions]  # (valid_len, num_classes)
            valid_tokens = tokens_single[valid_positions]  # (valid_len,)

            # CRITICAL: Validate and remove invalid structural tokens to avoid corrupting the loss calculation
            invalid_mask = (valid_tokens >= self.num_structural_classes) | (valid_tokens < 0)
            if invalid_mask.any():
                num_invalid = invalid_mask.sum().item()
                print(f"Warning: Found {num_invalid} invalid structural tokens for sample {i}. Ignoring these positions.")
                # Keep only valid tokens
                valid_indices = ~invalid_mask
                valid_logits = valid_logits[valid_indices]
                valid_tokens = valid_tokens[valid_indices]

            # Calculate cross-entropy loss for this sample if we have valid tokens
            if valid_logits.size(0) > 0:
                sample_loss = F.cross_entropy(valid_logits, valid_tokens, reduction='mean', ignore_index=-100)
                physical_loss_per_sample[i] = sample_loss

        # Overall physical loss is the average of per-sample losses
        physical_loss = physical_loss_per_sample.mean() if physical_loss_per_sample.numel() > 0 else torch.tensor(0.0, device=pLM_embeddings.device)

        return physical_loss, physical_loss_per_sample

    def _calculate_physical_loss(self, pLM_embeddings, structure_tokens, attention_mask=None):
        """
        Calculate the physical-level loss for structural token prediction (original method)
        """
        batch_size, seq_len, hidden_dim = pLM_embeddings.shape

        # Predict structural tokens
        logits = self.structural_prediction_head(pLM_embeddings)  # (batch_size, seq_len, num_classes)

        # Flatten for loss calculation
        logits_flat = logits.view(-1, self.num_structural_classes)  # (batch_size * seq_len, num_classes)
        tokens_flat = structure_tokens.view(-1)  # (batch_size * seq_len,)

        # Debug: Print statistics about structural tokens
        unique_tokens = torch.unique(tokens_flat[tokens_flat >= 0])  # Only non-padded tokens
        print(f"Physical loss debug - Batch size: {batch_size}, Seq len: {seq_len}")
        print(f"  Unique tokens (non-padded): {unique_tokens.tolist()}")
        print(f"  Token range: [{tokens_flat[tokens_flat >= 0].min().item() if (tokens_flat >= 0).any() else 'N/A'}, "
              f"{tokens_flat[tokens_flat >= 0].max().item() if (tokens_flat >= 0).any() else 'N/A'}]")
        print(f"  Number of valid (non-padded) tokens: {(tokens_flat >= 0).sum().item()}")
        print(f"  Number of padding tokens: {(tokens_flat == -100).sum().item()}")

        # Show token distribution for first sample in batch if batch size is reasonable
        if batch_size > 0:
            first_sample_tokens = structure_tokens[0]
            print(f"  First sample tokens (first 50): {first_sample_tokens[:min(50, len(first_sample_tokens))].tolist()}")
            if attention_mask is not None:
                print(f"  First sample attention mask (first 50): {attention_mask[0][:min(50, len(attention_mask[0]))].tolist()}")

        # CRITICAL: Validate and remove invalid structural tokens to avoid corrupting the loss calculation
        # This handles tokens generated with old buggy code that mapped unknown chars to a value outside of the class range
        invalid_mask = (tokens_flat >= self.num_structural_classes) | (tokens_flat < 0)
        if invalid_mask.any():
            num_invalid = invalid_mask.sum().item()
            print(f"Warning: Found {num_invalid} invalid structural tokens (>= {self.num_structural_classes} or < 0). Ignoring these positions.")
            # Set invalid tokens to ignore_index so they don't contribute to the loss
            tokens_flat = torch.where(invalid_mask, torch.tensor(-100, device=tokens_flat.device), tokens_flat)

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
    def __init__(self, model_path=None, hidden_dim=512, freeze=True):
        """
        Args:
            model_path: Path to pre-trained model (not currently available for GearNet)
            hidden_dim: Hidden dimension for the model
            freeze: Whether to freeze the model parameters
        """
        super().__init__()

        from models.gearnet_model import create_pretrained_gearnet
        self.backbone = create_pretrained_gearnet(
            hidden_dim=hidden_dim,
            pretrained_path=model_path,
            freeze=freeze
        )
        print("Successfully loaded GearNet implementation from TorchDrug")

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