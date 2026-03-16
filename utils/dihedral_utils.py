"""
Dihedral angle constraint functions for protein structure-based constraints
This module implements constraints based on phi/psi/omega dihedral angles
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
    n1_phi = F.normalize(n1_phi, p=2, dim=-1)
    n2_phi = F.normalize(n2_phi, p=2, dim=-1)
    n1_psi = F.normalize(n1_psi, p=2, dim=-1)
    n2_psi = F.normalize(n2_psi, p=2, dim=-1)
    
    # Calculate cosines of dihedral angles using dot products
    cos_phi = torch.clamp(torch.sum(n1_phi * n2_phi, dim=-1), -1, 1)
    cos_psi = torch.clamp(torch.sum(n1_psi * n2_psi, dim=-1), -1, 1)
    
    return cos_phi, cos_psi


class DihedralAngleConstraint(nn.Module):
    """
    Implements constraints based on dihedral angles (phi/psi) 
    """
    def __init__(self, hidden_dim, constraint_weight=1.0):
        super().__init__()
        self.constraint_weight = constraint_weight
        
        # Use a more robust loss function
        self.loss_fn = nn.SmoothL1Loss(reduction='none')
        
        # Linear layers to predict dihedral angles from embeddings
        self.phi_predictor = nn.Linear(hidden_dim, 1)
        self.psi_predictor = nn.Linear(hidden_dim, 1)
        
        # Standard initialization
        nn.init.xavier_uniform_(self.phi_predictor.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.phi_predictor.bias)
        nn.init.xavier_uniform_(self.psi_predictor.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.psi_predictor.bias)

    def forward(self, sequence_embeddings, n_coords, ca_coords, c_coords, attention_mask=None):
        """
        Calculate dihedral angle constraint loss

        Args:
            sequence_embeddings: (batch_size, seq_len, hidden_dim) - model embeddings
            n_coords: (batch_size, seq_len, 3) - N atom coordinates
            ca_coords: (batch_size, seq_len, 3) - CA atom coordinates
            c_coords: (batch_size, seq_len, 3) - C atom coordinates
            attention_mask: (batch_size, seq_len) - attention mask to ignore padding
        """
        batch_size, seq_len, hidden_dim = sequence_embeddings.shape

        # Compute true dihedral angles from coordinates
        cos_true_phi, cos_true_psi = compute_dihedral_angles_from_coordinates(n_coords, ca_coords, c_coords)

        # Predict dihedral angles from embeddings
        cos_pred_phi, cos_pred_psi = self.predict_dihedral_angles(sequence_embeddings)

        # Calculate per-sample losses (for consistency with constrained learning framework)
        phi_loss_per_sample, psi_loss_per_sample = self.angle_consistency_loss_per_sample(
            cos_true_phi, cos_pred_phi, cos_true_psi, cos_pred_psi, attention_mask)
        
        # Also compute global average loss (for backward compatibility)
        phi_loss_global = self.angle_consistency_loss(cos_true_phi, cos_pred_phi, attention_mask, angle_type='phi')
        psi_loss_global = self.angle_consistency_loss(cos_true_psi, cos_pred_psi, attention_mask, angle_type='psi')
        total_dihedral_loss_global = phi_loss_global + psi_loss_global

        # Combine per-sample losses
        per_sample_dihedral_losses = phi_loss_per_sample + psi_loss_per_sample

        return {
            'total_dihedral_loss': total_dihedral_loss_global * self.constraint_weight,
            'phi_loss': phi_loss_global * self.constraint_weight,
            'psi_loss': psi_loss_global * self.constraint_weight,
            'per_sample_dihedral_losses': per_sample_dihedral_losses,
            'cos_true_phi': cos_true_phi,
            'cos_true_psi': cos_true_psi,
            'cos_pred_phi': cos_pred_phi,
            'cos_pred_psi': cos_pred_psi
        }

    def angle_consistency_loss_per_sample(self, true_cos_phi, pred_cos_phi, true_cos_psi, pred_cos_psi, attention_mask=None):
        """
        Calculate per-sample loss between true and predicted angle cosines.
        Returns loss per sample (shape: [batch_size]) for use in constrained learning.
        
        Args:
            true_cos_phi, pred_cos_phi: (batch_size, seq_len-1) cosine values for phi
            true_cos_psi, pred_cos_psi: (batch_size, seq_len-1) cosine values for psi
            attention_mask: (batch_size, seq_len) attention mask
        """
        batch_size = true_cos_phi.shape[0]
        
        # Handle empty case
        if true_cos_phi.numel() == 0 or pred_cos_phi.numel() == 0:
            return (torch.zeros(batch_size, device=true_cos_phi.device),
                    torch.zeros(batch_size, device=true_cos_phi.device))
        
        min_len_phi = min(true_cos_phi.shape[1], pred_cos_phi.shape[1])
        min_len_psi = min(true_cos_psi.shape[1], pred_cos_psi.shape[1])
        
        # Phi loss per sample
        phi_mask = attention_mask[:, 1:1+min_len_phi].float() if attention_mask is not None else torch.ones(batch_size, min_len_phi, device=true_cos_phi.device)
        phi_loss = self.loss_fn(pred_cos_phi[:, :min_len_phi], true_cos_phi[:, :min_len_phi])
        phi_loss_per_sample = (phi_loss * phi_mask).sum(dim=1) / phi_mask.sum(dim=1).clamp(min=1.0)
        
        # Psi loss per sample
        psi_mask = attention_mask[:, :min_len_psi].float() if attention_mask is not None else torch.ones(batch_size, min_len_psi, device=true_cos_psi.device)
        psi_loss = self.loss_fn(pred_cos_psi[:, :min_len_psi], true_cos_psi[:, :min_len_psi])
        psi_loss_per_sample = (psi_loss * psi_mask).sum(dim=1) / psi_mask.sum(dim=1).clamp(min=1.0)
        
        return phi_loss_per_sample, psi_loss_per_sample
    
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
        Calculate loss between true and predicted angle cosines using Smooth L1 Loss.
        """
        if true_cos.numel() == 0 or pred_cos.numel() == 0:
            return torch.tensor(0.0, device=true_cos.device, requires_grad=True)
        
        min_len = min(true_cos.shape[1], pred_cos.shape[1])
        true_cos = true_cos[:, :min_len]
        pred_cos = pred_cos[:, :min_len]
        
        loss = self.loss_fn(pred_cos, true_cos)
        
        if attention_mask is not None:
            if angle_type == 'phi':
                valid_mask = attention_mask[:, 1:1+min_len].float()
            else:
                valid_mask = attention_mask[:, :min_len].float()
            
            if valid_mask.sum() > 0:
                loss = (loss * valid_mask).sum() / valid_mask.sum()
            else:
                loss = torch.tensor(0.0, device=true_cos.device, requires_grad=True)
        else:
            loss = loss.mean()

        return loss

import matplotlib.pyplot as plt

def visualize_dihedral_distribution(true_cos, pred_cos, angle_type, save_path="dihedral_distribution.png"):
    """
    Visualizes the distribution of true vs. predicted dihedral angles.
    """
    true_cos = true_cos.detach().cpu().numpy().flatten()
    pred_cos = pred_cos.detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(true_cos, pred_cos, alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlabel(f"True cos({angle_type})")
    plt.ylabel(f"Predicted cos({angle_type})")
    plt.title(f"True vs. Predicted cos({angle_type})")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid(True)
    
    # Save the plot
    plt.savefig(save_path)
    print(f"Dihedral distribution plot saved to {save_path}")
    plt.close()