"""
Constrained Learning Framework for Dihedral Angle Constraints

This module implements a constrained learning framework for dihedral angles
using primal-dual optimization as described in the reference paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_dihedral_angles_from_coordinates(n_coords, ca_coords, c_coords):
    """
    Compute phi, psi angles from backbone atom coordinates (N, CA, C)

    Args:
        n_coords: (batch_size, seq_len, 3) tensor of N atom coordinates
        ca_coords: (batch_size, seq_len, 3) tensor of CA atom coordinates
        c_coords: (batch_size, seq_len, 3) tensor of C atom coordinates

    Returns:
        cos_phi, cos_psi: (batch_size, seq_len-1) tensors of cosine values
    """
    batch_size, seq_len, _ = ca_coords.shape

    if seq_len < 3:
        # Return empty tensors for insufficient length
        return (torch.empty(batch_size, 0, device=ca_coords.device),
                torch.empty(batch_size, 0, device=ca_coords.device))

    # Phi angle: C(i-1)-N(i)-CA(i)-C(i)
    # Psi angle: N(i)-CA(i)-C(i)-N(i+1)

    # For phi angles: need C(i-1), N(i), CA(i), C(i)
    # For psi angles: need N(i), CA(i), C(i), N(i+1)

    # Shift coordinates to get required atoms for each angle
    # Phi angles calculated at residue i using atoms from residue i-1 and i
    c_prev = c_coords[:, :-1]   # C from previous residues
    n_curr = n_coords[:, 1:]    # N from current residues
    ca_curr = ca_coords[:, 1:]  # CA from current residues
    c_curr = c_coords[:, 1:]    # C from current residues

    # Psi angles calculated at residue i using atoms from residue i and i+1
    n_curr_psi = n_coords[:, :-1]   # N from current residues
    ca_curr_psi = ca_coords[:, :-1] # CA from current residues
    c_curr_psi = c_coords[:, :-1]   # C from current residues
    n_next = n_coords[:, 1:]        # N from next residues

    # Calculate vectors for phi angles
    v1_phi = n_curr - c_prev    # b1 vector
    v2_phi = ca_curr - n_curr   # b2 vector
    v3_phi = c_curr - ca_curr   # b3 vector

    # Calculate vectors for psi angles
    v1_psi = ca_curr_psi - n_curr_psi  # b1 vector
    v2_psi = c_curr_psi - ca_curr_psi  # b2 vector
    v3_psi = n_next - c_curr_psi       # b3 vector

    # Calculate cross products for normal vectors
    n1_phi = torch.cross(v1_phi, v2_phi, dim=-1)  # Normal to first plane
    n2_phi = torch.cross(v2_phi, v3_phi, dim=-1)  # Normal to second plane

    n1_psi = torch.cross(v1_psi, v2_psi, dim=-1)  # Normal to first plane
    n2_psi = torch.cross(v2_psi, v3_psi, dim=-1)  # Normal to second plane

    # Ensure normals are not zero vectors by adding small epsilon
    n1_phi = F.normalize(n1_phi, p=2, dim=-1, eps=1e-8)
    n2_phi = F.normalize(n2_phi, p=2, dim=-1, eps=1e-8)
    n1_psi = F.normalize(n1_psi, p=2, dim=-1, eps=1e-8)
    n2_psi = F.normalize(n2_psi, p=2, dim=-1, eps=1e-8)

    # Calculate cosines of dihedral angles using dot products
    cos_phi = torch.clamp(torch.sum(n1_phi * n2_phi, dim=-1), -1, 1)
    cos_psi = torch.clamp(torch.sum(n1_psi * n2_psi, dim=-1), -1, 1)

    return cos_phi, cos_psi


class ConstrainedDihedralAngleConstraint(nn.Module):
    """
    Implements constrained learning framework for dihedral angles (phi/psi)
    using primal-dual optimization approach.
    """
    def __init__(self, hidden_dim: int, constraint_weight: float = 1.0):
        super().__init__()
        self.constraint_weight = constraint_weight

        # Linear layers to predict dihedral angles from embeddings
        self.phi_predictor = nn.Linear(hidden_dim, 1)
        self.psi_predictor = nn.Linear(hidden_dim, 1)

        # Initialize layers with small random weights
        nn.init.xavier_uniform_(self.phi_predictor.weight)
        nn.init.zeros_(self.phi_predictor.bias)
        nn.init.xavier_uniform_(self.psi_predictor.weight)
        nn.init.zeros_(self.psi_predictor.bias)

    def forward(self, sequence_embeddings, n_coords, ca_coords, c_coords, attention_mask=None):
        """
        Calculate dihedral angle constraint loss using constrained learning

        Args:
            sequence_embeddings: (batch_size, seq_len, hidden_dim) - model embeddings
            n_coords: (batch_size, seq_len, 3) - N atom coordinates
            ca_coords: (batch_size, seq_len, 3) - CA atom coordinates
            c_coords: (batch_size, seq_len, 3) - C atom coordinates
            attention_mask: (batch_size, seq_len) - attention mask to ignore padding
        """
        # Compute true dihedral angles from coordinates
        cos_true_phi, cos_true_psi = compute_dihedral_angles_from_coordinates(n_coords, ca_coords, c_coords)

        # Predict dihedral angles from embeddings
        cos_pred_phi, cos_pred_psi = self.predict_dihedral_angles(sequence_embeddings)

        # Calculate constraint losses (masked MSE)
        min_len_phi = min(cos_true_phi.shape[1], cos_pred_phi.shape[1])
        phi_mask = attention_mask[:, 1:1+min_len_phi].float()
        phi_loss_sq = (cos_true_phi[:, :min_len_phi] - cos_pred_phi[:, :min_len_phi])**2
        phi_loss = (phi_loss_sq * phi_mask).sum() / phi_mask.sum().clamp(min=1.0)

        min_len_psi = min(cos_true_psi.shape[1], cos_pred_psi.shape[1])
        psi_mask = attention_mask[:, :min_len_psi].float()
        psi_loss_sq = (cos_true_psi[:, :min_len_psi] - cos_pred_psi[:, :min_len_psi])**2
        psi_loss = (psi_loss_sq * psi_mask).sum() / psi_mask.sum().clamp(min=1.0)

        # Combine constraint losses
        constraint_loss = phi_loss + psi_loss

        return {
            'dihedral_loss': constraint_loss * self.constraint_weight,
            'phi_loss': phi_loss * self.constraint_weight,
            'psi_loss': psi_loss * self.constraint_weight,
            'raw_constraint_loss': constraint_loss
        }

    def predict_dihedral_angles(self, embeddings):
        """
        Predict dihedral angles from sequence embeddings
        """
        batch_size, seq_len, hidden_dim = embeddings.shape

        if seq_len < 2:
            # Not enough residues for dihedral angles
            return (torch.empty(batch_size, 0, device=embeddings.device),
                    torch.empty(batch_size, 0, device=embeddings.device))

        # Predict phi angles (available from position 1 onwards)
        phi_logits = self.phi_predictor(embeddings[:, 1:, :]).squeeze(-1)  # (B, seq_len-1)
        cos_pred_phi = torch.tanh(phi_logits)  # Constrain to [-1, 1] range

        # Predict psi angles (available up to position seq_len-2)
        psi_logits = self.psi_predictor(embeddings[:, :-1, :]).squeeze(-1)  # (B, seq_len-1)
        cos_pred_psi = torch.tanh(psi_logits)  # Constrain to [-1, 1] range

        return cos_pred_phi, cos_pred_psi

    def angle_consistency_loss(self, true_cos, pred_cos, attention_mask=None, angle_type='phi'):
        """
        Calculate loss between true and predicted angle cosines
        """
        if true_cos.numel() == 0 or pred_cos.numel() == 0:
            return torch.tensor(0.0, device=true_cos.device, requires_grad=True)

        # Ensure shapes match
        min_len = min(true_cos.shape[1], pred_cos.shape[1])
        true_cos = true_cos[:, :min_len]
        pred_cos = pred_cos[:, :min_len]

        # Calculate cosine difference loss (MSE for dihedral constraints)
        cos_diff = true_cos - pred_cos
        angle_loss = torch.mean(cos_diff ** 2)  # MSE for dihedral constraints

        # Apply mask if available
        if attention_mask is not None:
            # Different mask positions depending on angle type
            if angle_type == 'phi':
                # Phi angles calculated from position 1 (residue 1), so mask from position 1
                valid_mask = attention_mask[:, 1:1+min_len].float()
            else:  # psi
                # Psi angles calculated up to position seq_len-2, so mask from position 0
                valid_mask = attention_mask[:, :min_len].float()

            if valid_mask.sum() > 0:
                angle_loss = torch.sum((cos_diff ** 2) * valid_mask) / torch.sum(valid_mask)

        return angle_loss


class MultiConstraintLagrangian(nn.Module):
    """
    Implements constrained learning framework for multiple losses using primal-dual optimization.
    Each loss type is constrained separately with its own Lagrange multipliers and epsilon values.
    This version maintains per-sample lambdas for the entire training dataset.
    """
    def __init__(self,
                 num_training_samples,
                 dihedral_epsilon=0.076,
                 gnn_epsilon=6.38,
                 foldseek_epsilon=3.00,
                 dual_lr=1e-3):
        super().__init__()

        # Epsilon values for different constraints
        self.dihedral_epsilon = dihedral_epsilon
        self.gnn_epsilon = gnn_epsilon
        self.foldseek_epsilon = foldseek_epsilon
        self.num_training_samples = num_training_samples
        self.dual_lr = dual_lr

        # Dual variables (Lagrange multipliers, non-negative) for each sample in the dataset.
        # We register them as buffers so they are moved to the correct device with the module,
        # but are not considered model parameters by the optimizer.
        self.register_buffer('lam_dihedral', torch.full((num_training_samples,), 0.00))
        self.register_buffer('lam_gnn', torch.full((num_training_samples,), 0.00))
        self.register_buffer('lam_foldseek', torch.full((num_training_samples,), 0.00))

    def compute_lagrangian(self, primary_loss, dihedral_losses, gnn_losses, foldseek_losses, indices):
        """
        Compute the multi-constraint Lagrangian with per-sample constraints

        Args:
            primary_loss: The main task loss (e.g., MLM loss)
            dihedral_losses: Per-sample dihedral constraint losses (tensor of shape [batch_size])
            gnn_losses: Per-sample GNN alignment losses (tensor of shape [batch_size])
            foldseek_losses: Per-sample foldseek alignment losses (tensor of shape [batch_size])
            indices: The indices of the samples in the current batch (tensor of shape [batch_size])
        """
        # Select the lambdas for the current batch using the provided indices
        lam_dihedral_batch = self.lam_dihedral[indices]
        lam_gnn_batch = self.lam_gnn[indices]
        lam_foldseek_batch = self.lam_foldseek[indices]

        # Individual constraint terms for each protein in the batch.
        # The lambdas are detached so that their gradients don't flow back to the primary loss.
        dihedral_constraint_terms = lam_dihedral_batch.detach() * (dihedral_losses - self.dihedral_epsilon)
        gnn_constraint_terms = lam_gnn_batch.detach() * (gnn_losses - self.gnn_epsilon)
        foldseek_constraint_terms = lam_foldseek_batch.detach() * (foldseek_losses - self.foldseek_epsilon)

        # Sum the constraint terms for this batch
        total_constraint_term = torch.sum(dihedral_constraint_terms + gnn_constraint_terms + foldseek_constraint_terms)

        # Total Lagrangian
        lagrangian = primary_loss + total_constraint_term

        return lagrangian

    def update_dual_variables(self, dihedral_losses, gnn_losses, foldseek_losses, indices):
        """
        Update Lagrange multipliers using projected gradient ascent.
        This is done with no_grad context and updates the buffers in-place.

        Args:
            dihedral_losses: Per-sample dihedral constraint losses for the batch
            gnn_losses: Per-sample GNN alignment losses for the batch
            foldseek_losses: Per-sample foldseek alignment losses for the batch
            indices: The indices of the samples in the current batch
        """
        with torch.no_grad():
            # Calculate violations for each constraint type for the current batch
            dihedral_violations = dihedral_losses.detach() - self.dihedral_epsilon
            gnn_violations = gnn_losses.detach() - self.gnn_epsilon
            foldseek_violations = foldseek_losses.detach() - self.foldseek_epsilon

            # Update dihedral multipliers for the samples in the batch
            self.lam_dihedral[indices] += self.dual_lr * dihedral_violations
            self.lam_dihedral[indices].clamp_(min=0.0)

            # Update GNN multipliers
            self.lam_gnn[indices] += self.dual_lr * gnn_violations
            
            # Note: clam should clamp to 0 $$$$$$$$$$$$$$$$$$$$$$$$$$ $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            self.lam_gnn[indices].data.clamp_(min=0.0)

            # Update foldseek multipliers
            self.lam_foldseek[indices] += self.dual_lr * foldseek_violations
            self.lam_foldseek[indices].clamp_(min=0.0)

    def get_average_lambdas(self):
        """Get the average lambda values over the entire dataset."""
        with torch.no_grad():
            avg_lam_dihedral = self.lam_dihedral.mean().item()
            avg_lam_gnn = self.lam_gnn.mean().item()
            avg_lam_foldseek = self.lam_foldseek.mean().item()
        return avg_lam_dihedral, avg_lam_gnn, avg_lam_foldseek