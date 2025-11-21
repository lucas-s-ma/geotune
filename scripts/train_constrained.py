"""
Main training script for ESM2 with constrained geometric learning and LoRA
"""
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import transformers
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
import argparse
from omegaconf import OmegaConf

from models.geotune_esm_model import load_esm_with_lora
from utils.constrained_dihedral_utils import ConstrainedDihedralAngleConstraint, MultiConstraintLagrangian, compute_dihedral_angles_from_coordinates
from utils.data_utils import ProteinStructureDataset, EfficientProteinDataset, collate_fn
from utils.structure_alignment_utils import StructureAlignmentLoss, PretrainedGNNWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Train ESM2 with constrained geometric learning and LoRA")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to protein data directory")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t30_150M_UR50D",
                       help="ESM model name or path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--constraint_weight", type=float, default=0.1, help="Weight for constraint loss")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--dist_threshold", type=float, default=15.0, help="Distance threshold for constraints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, scheduler, dihedral_constraints, device, config, structure_alignment_loss=None, frozen_gnn=None, multi_constraint_lagrangian=None, scaler=None):
    """Train for one epoch with constrained dihedral angle learning and multiple constraint types"""
    model.train()
    total_loss = 0
    dihedral_loss_total = 0
    mlm_loss_total = 0
    structure_alignment_loss_total = 0
    gnn_loss_total = 0
    foldseek_loss_total = 0

    # Get gradient accumulation steps
    gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(progress_bar):
        # Log batch timing
        import time
        batch_start_time = time.time()

        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Check if we have the new format with N, CA, C coordinates
        if 'n_coords' in batch and 'ca_coords' in batch and 'c_coords' in batch:
            n_coords = batch['n_coords'].to(device)
            ca_coords = batch['ca_coords'].to(device)
            c_coords = batch['c_coords'].to(device)
        else:
            # Old format detected - this should not happen if dataset was processed with new code
            # But we'll try to convert if possible
            print("WARNING: Old dataset format detected. Please re-process your data using the new format.")
            print("Expected 'n_coords', 'ca_coords', 'c_coords' but found other keys.")
            print(f"Available keys: {list(batch.keys())}")
            raise KeyError("Dataset is in old format. Please re-run process_data.py with the --create_efficient_dataset flag to generate a new dataset with N, CA, C coordinates.")

        # Check if structural tokens are available in the batch
        has_structural_tokens = 'structural_tokens' in batch

        # Check if pre-computed embeddings are available
        has_precomputed_embeddings = 'precomputed_embeddings' in batch

        # Log tensor shapes for debugging
        if batch_idx == 0:
            actual_seq_lengths = torch.sum(attention_mask, dim=1)
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Attention mask shape: {attention_mask.shape}")
            print(f"N coordinates shape: {n_coords.shape}")
            print(f"CA coordinates shape: {ca_coords.shape}")
            print(f"C coordinates shape: {c_coords.shape}")
            print(f"Structural tokens available: {has_structural_tokens}")
            print(f"Pre-computed embeddings available: {has_precomputed_embeddings}")
            if has_structural_tokens:
                print(f"Structural tokens shape: {batch['structural_tokens'].shape}")
            if has_precomputed_embeddings:
                print(f"Pre-computed embeddings shape: {batch['precomputed_embeddings'].shape}")
            print(f"Actual sequence lengths in batch: {actual_seq_lengths.tolist()}")
            print(f"Max sequence length in batch: {actual_seq_lengths.max().item()}")

            # Warn if any sequence is very long (>200 residues)
            max_len = actual_seq_lengths.max().item()
            if max_len > 200:
                print(f"⚠️  WARNING: Found sequence of length {max_len}, this may cause training to be very slow.")
                print(f"   Consider using --max_seq_len 200 or shorter for faster training during debugging.")

        # Use mixed precision if scaler is provided
        use_amp = scaler is not None

        # Forward pass with automatic mixed precision
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Calculate MLM loss (masked language modeling)
            # For this example, we'll create masked tokens randomly
            seq_output = outputs['sequence_output']

            # Create random masks for MLM task
            batch_size, seq_len, hidden_dim = seq_output.shape
            mask_ratio = 0.15  # Standard ESM masking ratio

            # Create mask for masking tokens (ignore padding positions)
            mask_positions = (torch.rand(batch_size, seq_len, device=device) < mask_ratio) & (attention_mask.bool())

            # Clone input_ids to create labels
            labels = input_ids.clone()
            labels[~mask_positions] = -100  # Ignore non-masked positions in loss

            # Create masked input
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = 32  # Use <mask> token ID (32 in ESM2)

            # Recompute with masked inputs to get predictions for masked positions
            masked_outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask
            )

            # Calculate MLM loss
            seq_logits = model.lm_head(masked_outputs['sequence_output'])
            mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(seq_logits.view(-1, seq_logits.size(-1)), labels.view(-1))

        # Calculate dihedral angle constraint loss using constrained learning
        # Add timing for constraint calculation to identify bottlenecks
        constraint_start_time = time.time()

        with torch.amp.autocast('cuda', enabled=use_amp):
            # Get dihedral constraint results
            dihedral_results = dihedral_constraints(
                masked_outputs['sequence_output'],
                n_coords,
                ca_coords,
                c_coords,
                attention_mask
            )
            dihedral_loss = dihedral_results['dihedral_loss']
            raw_constraint_loss = dihedral_results['raw_constraint_loss']
            # Calculate per-sample dihedral losses instead of aggregated
            per_sample_dihedral_losses = torch.zeros(batch_size, device=device)
            if raw_constraint_loss.numel() > 0:
                # Calculate per-sample losses by averaging over sequence length per sample
                # We need to compute per-sample losses differently
                # Get phi and psi losses per sample
                cos_true_phi, cos_true_psi = compute_dihedral_angles_from_coordinates(n_coords, ca_coords, c_coords)
                cos_pred_phi, cos_pred_psi = dihedral_constraints.predict_dihedral_angles(masked_outputs['sequence_output'])

                # Calculate per-sample dihedral losses
                phi_loss_per_sample = dihedral_constraints.angle_consistency_loss(cos_true_phi, cos_pred_phi, attention_mask, angle_type='phi')
                psi_loss_per_sample = dihedral_constraints.angle_consistency_loss(cos_true_psi, cos_pred_psi, attention_mask, angle_type='psi')

                # Calculate per-sample dihedral losses by averaging over sequence length
                valid_lengths = attention_mask.sum(dim=1)  # Sequence lengths per sample
                total_phi_loss = torch.mean((cos_true_phi - cos_pred_phi) ** 2, dim=1)  # Average across sequence length
                total_psi_loss = torch.mean((cos_true_psi - cos_pred_psi) ** 2, dim=1)  # Average across sequence length
                per_sample_dihedral_losses = total_phi_loss + total_psi_loss

            constraint_time = time.time() - constraint_start_time

            # Calculate structure alignment loss if structural tokens are available
            struct_align_loss = torch.tensor(0.0, device=device, requires_grad=True)
            gnn_loss = torch.tensor(0.0, device=device, requires_grad=True)
            foldseek_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Initialize per-sample losses
            per_sample_gnn_losses = torch.zeros(batch_size, device=device)
            per_sample_foldseek_losses = torch.zeros(batch_size, device=device)
            per_sample_dihedral_losses_final = torch.zeros(batch_size, device=device)

            if structure_alignment_loss is not None and has_structural_tokens:
                # Get structural tokens
                structure_tokens = batch['structural_tokens'].to(device)

                # Use pre-computed embeddings if available, otherwise generate using frozen GNN
                if has_precomputed_embeddings:
                    # Use pre-computed embeddings
                    pGNN_embeddings = batch['precomputed_embeddings'].to(device)
                else:
                    # Generate pGNN embeddings using the frozen GNN (inference only)
                    with torch.no_grad():
                        pGNN_embeddings = frozen_gnn(n_coords, ca_coords, c_coords)

                # Get pLM embeddings
                pLM_embeddings = masked_outputs['sequence_output']

                # Calculate structure alignment loss
                struct_align_results = structure_alignment_loss(
                    pLM_embeddings=pLM_embeddings,
                    pGNN_embeddings=pGNN_embeddings,
                    structure_tokens=structure_tokens,
                    attention_mask=attention_mask
                )
                struct_align_loss = struct_align_results['total_loss']
                gnn_loss = struct_align_results.get('latent_loss', torch.tensor(0.0, device=device))
                foldseek_loss = struct_align_results.get('physical_loss', torch.tensor(0.0, device=device))

                # Extract per-sample losses for GNN and foldseek
                per_sample_gnn_losses = struct_align_results.get('latent_loss_per_sample', torch.zeros(batch_size, device=device))
                per_sample_foldseek_losses = struct_align_results.get('physical_loss_per_sample', torch.zeros(batch_size, device=device))

                # For foldseek loss, it's typically computed per amino acid then averaged per protein
                # So we need to aggregate it per sample based on sequence length

            # Use multi-constraint Lagrangian if it exists, otherwise use original approach
            if multi_constraint_lagrangian is not None:
                # Compute the multi-constraint Lagrangian with per-sample losses
                lagrangian, lagrangian_components = multi_constraint_lagrangian.compute_lagrangian(
                    primary_loss=mlm_loss,
                    dihedral_losses=per_sample_dihedral_losses,
                    gnn_losses=per_sample_gnn_losses,
                    foldseek_losses=per_sample_foldseek_losses,
                    batch_size=batch_size
                )
            else:
                # Fallback to single constraint approach
                lagrangian = dihedral_constraints.compute_lagrangian(
                    primary_loss=mlm_loss,
                    constraint_loss=raw_constraint_loss  # Use raw constraint loss without weight scaling
                )

            # Scale loss for gradient accumulation
            combined_loss = lagrangian / gradient_accumulation_steps

        # Log loss components immediately to wandb for debugging
        # Note: combined_loss is scaled by gradient_accumulation_steps, so multiply back for logging
        if config.logging.use_wandb:
            if multi_constraint_lagrangian is not None:
                # Log multi-constraint components
                wandb.log({
                    'train_batch_loss': combined_loss.item() * gradient_accumulation_steps,
                    'train_batch_mlm_loss': mlm_loss.item(),
                    'train_batch_dihedral_loss': dihedral_loss.item(),
                    'train_batch_gnn_loss': gnn_loss.item(),
                    'train_batch_foldseek_loss': foldseek_loss.item(),
                    'train_batch_struct_align_loss': struct_align_loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0],
                    'batch_idx': batch_idx,
                })
            else:
                # Log single constraint components
                wandb.log({
                    'train_batch_loss': combined_loss.item() * gradient_accumulation_steps,
                    'train_batch_mlm_loss': mlm_loss.item(),
                    'train_batch_dihedral_loss': dihedral_loss.item(),
                    'train_batch_struct_align_loss': struct_align_loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0],
                    'batch_idx': batch_idx,
                })

        # Backward pass with gradient accumulation and mixed precision (for primal update)
        if scaler is not None:
            # Mixed precision backward
            scaler.scale(combined_loss).backward()
        else:
            # Standard backward
            combined_loss.backward()

        # Only update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Unscale gradients and clip them
            if scaler is not None:
                scaler.unscale_(optimizer)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Log gradient norms for debugging
            if batch_idx % 10 == 0:
                total_norm = 0
                param_count = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_count += 1
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)

                if config.logging.use_wandb:
                    wandb.log({
                        'grad_norm': total_norm,
                        'param_count_with_grad': param_count
                    })

            # Primal optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            # Dual optimizer step (for Lagrange multipliers)
            # Calculate dual loss (negative of lagrangian for maximization)
            with torch.no_grad():
                if multi_constraint_lagrangian is not None:
                    # Update the dual variables (lambda multipliers) using the per-sample losses
                    multi_constraint_lagrangian.update_dual_variables(
                        dihedral_losses=per_sample_dihedral_losses,
                        gnn_losses=per_sample_gnn_losses,
                        foldseek_losses=per_sample_foldseek_losses,
                        batch_size=batch_size
                    )
                    # Project the slack variables to be non-negative
                    multi_constraint_lagrangian.update_slack_variables()
                else:
                    # Project the slack and dual variables to be non-negative (original approach)
                    dihedral_constraints.update_slack_variables()
                    dihedral_constraints.update_dual_variables()

        # Update progress bar
        # Note: combined_loss is scaled by gradient_accumulation_steps, so multiply back
        total_loss += combined_loss.item() * gradient_accumulation_steps
        dihedral_loss_total += dihedral_loss.item()
        mlm_loss_total += mlm_loss.item()
        structure_alignment_loss_total += struct_align_loss.item()
        gnn_loss_total += gnn_loss.item()
        foldseek_loss_total += foldseek_loss.item()

        # Calculate batch processing time
        batch_time = time.time() - batch_start_time

        if multi_constraint_lagrangian is not None:
            if batch_size > 0:
                avg_dihedral_lambda = multi_constraint_lagrangian.lam_dihedral[:batch_size].mean().item()
                avg_gnn_lambda = multi_constraint_lagrangian.lam_gnn[:batch_size].mean().item()
                avg_foldseek_lambda = multi_constraint_lagrangian.lam_foldseek[:batch_size].mean().item()

                progress_bar.set_postfix({
                    'Loss': f'{combined_loss.item() * gradient_accumulation_steps:.4f}',
                    'MLM': f'{mlm_loss.item():.4f}',
                    'Dihedral': f'{dihedral_loss.item():.4f}',
                    'GNN': f'{gnn_loss.item():.4f}',
                    'FoldSeek': f'{foldseek_loss.item():.4f}',
                    'StructAlign': f'{struct_align_loss.item():.4f}',
                    'S_dih': f'{multi_constraint_lagrangian.s_dihedral.item():.4f}',
                    'S_gnn': f'{multi_constraint_lagrangian.s_gnn.item():.4f}',
                    'S_fs': f'{multi_constraint_lagrangian.s_foldseek.item():.4f}',
                    'L_dih': f'{avg_dihedral_lambda:.3f}',
                    'L_gnn': f'{avg_gnn_lambda:.3f}',
                    'L_fs': f'{avg_foldseek_lambda:.3f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.2e}',
                    'Time': f'{batch_time:.2f}s',
                    'CnstTime': f'{constraint_time:.2f}s'
                })
            else:
                progress_bar.set_postfix({
                    'Loss': f'{combined_loss.item() * gradient_accumulation_steps:.4f}',
                    'MLM': f'{mlm_loss.item():.4f}',
                    'Dihedral': f'{dihedral_loss.item():.4f}',
                    'GNN': f'{gnn_loss.item():.4f}',
                    'FoldSeek': f'{foldseek_loss.item():.4f}',
                    'StructAlign': f'{struct_align_loss.item():.4f}',
                    'S_dih': f'{multi_constraint_lagrangian.s_dihedral.item():.4f}',
                    'S_gnn': f'{multi_constraint_lagrangian.s_gnn.item():.4f}',
                    'S_fs': f'{multi_constraint_lagrangian.s_foldseek.item():.4f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.2e}',
                    'Time': f'{batch_time:.2f}s',
                    'CnstTime': f'{constraint_time:.2f}s'
                })
        else:
            progress_bar.set_postfix({
                'Loss': f'{combined_loss.item() * gradient_accumulation_steps:.4f}',
                'MLM': f'{mlm_loss.item():.4f}',
                'Dihedral': f'{dihedral_loss.item():.4f}',
                'StructAlign': f'{struct_align_loss.item():.4f}',
                'Slack': f'{dihedral_constraints.s.item():.4f}',
                'Lambda': f'{dihedral_constraints.lam.item():.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}',
                'Time': f'{batch_time:.2f}s',
                'CnstTime': f'{constraint_time:.2f}s'
            })

    avg_loss = total_loss / len(dataloader)
    avg_dihedral_loss = dihedral_loss_total / len(dataloader)
    avg_mlm_loss = mlm_loss_total / len(dataloader)
    avg_struct_align_loss = structure_alignment_loss_total / len(dataloader)
    avg_gnn_loss = gnn_loss_total / len(dataloader)
    avg_foldseek_loss = foldseek_loss_total / len(dataloader)

    # Log epoch averages
    if config.logging.use_wandb:
        if multi_constraint_lagrangian is not None:
            wandb.log({
                'epoch_avg_loss': avg_loss,
                'epoch_avg_mlm_loss': avg_mlm_loss,
                'epoch_avg_dihedral_loss': avg_dihedral_loss,
                'epoch_avg_gnn_loss': avg_gnn_loss,
                'epoch_avg_foldseek_loss': avg_foldseek_loss,
                'epoch_avg_struct_align_loss': avg_struct_align_loss,
            })
        else:
            wandb.log({
                'epoch_avg_loss': avg_loss,
                'epoch_avg_mlm_loss': avg_mlm_loss,
                'epoch_avg_dihedral_loss': avg_dihedral_loss,
                'epoch_avg_struct_align_loss': avg_struct_align_loss,
            })

    return avg_loss, avg_mlm_loss, avg_dihedral_loss, avg_struct_align_loss, avg_gnn_loss, avg_foldseek_loss


def validate(model, dataloader, dihedral_constraints, device, config, structure_alignment_loss=None, frozen_gnn=None, multi_constraint_lagrangian=None):
    """Validate the model with constrained dihedral angle learning and multiple constraint types"""
    model.eval()
    total_loss = 0
    dihedral_loss_total = 0
    mlm_loss_total = 0
    structure_alignment_loss_total = 0
    gnn_loss_total = 0
    foldseek_loss_total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            n_coords = batch['n_coords'].to(device)
            ca_coords = batch['ca_coords'].to(device)
            c_coords = batch['c_coords'].to(device)

            # Check if structural tokens are available in the batch
            has_structural_tokens = 'structural_tokens' in batch

            # Check if pre-computed embeddings are available
            has_precomputed_embeddings = 'precomputed_embeddings' in batch

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Create random masks for MLM task
            seq_output = outputs['sequence_output']
            batch_size, seq_len, hidden_dim = seq_output.shape
            mask_ratio = 0.15

            mask_positions = (torch.rand(batch_size, seq_len, device=device) < mask_ratio) & (attention_mask.bool())
            labels = input_ids.clone()
            labels[~mask_positions] = -100

            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = 32  # Use <mask> token ID (32 in ESM2)

            masked_outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask
            )

            # Calculate MLM loss
            seq_logits = model.lm_head(masked_outputs['sequence_output'])
            mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(seq_logits.view(-1, seq_logits.size(-1)), labels.view(-1))

            # Calculate dihedral constraint loss
            dihedral_results = dihedral_constraints(
                masked_outputs['sequence_output'],
                n_coords,
                ca_coords,
                c_coords,
                attention_mask
            )

            dihedral_loss = dihedral_results['dihedral_loss']
            raw_constraint_loss = dihedral_results['raw_constraint_loss']

            # Calculate per-sample dihedral losses for validation
            per_sample_dihedral_losses = torch.zeros(batch_size, device=device)
            if raw_constraint_loss.numel() > 0:
                # Get phi and psi losses per sample
                cos_true_phi, cos_true_psi = compute_dihedral_angles_from_coordinates(n_coords, ca_coords, c_coords)
                cos_pred_phi, cos_pred_psi = dihedral_constraints.predict_dihedral_angles(masked_outputs['sequence_output'])

                # Calculate per-sample dihedral losses by averaging over sequence length
                total_phi_loss = torch.mean((cos_true_phi - cos_pred_phi) ** 2, dim=1)  # Average across sequence length
                total_psi_loss = torch.mean((cos_true_psi - cos_pred_psi) ** 2, dim=1)  # Average across sequence length
                per_sample_dihedral_losses = total_phi_loss + total_psi_loss

            # Calculate structure alignment loss if structural tokens are available
            struct_align_loss = torch.tensor(0.0, device=device)
            gnn_loss = torch.tensor(0.0, device=device)
            foldseek_loss = torch.tensor(0.0, device=device)

            # Initialize per-sample losses
            per_sample_gnn_losses = torch.zeros(batch_size, device=device)
            per_sample_foldseek_losses = torch.zeros(batch_size, device=device)

            if structure_alignment_loss is not None and has_structural_tokens and 'structural_tokens' in batch:
                # Get structural tokens
                structure_tokens = batch['structural_tokens'].to(device)

                # Use pre-computed embeddings if available, otherwise generate using frozen GNN
                if has_precomputed_embeddings:
                    # Use pre-computed embeddings
                    pGNN_embeddings = batch['precomputed_embeddings'].to(device)
                else:
                    # Generate pGNN embeddings using the frozen GNN (inference only)
                    with torch.no_grad():
                        pGNN_embeddings = frozen_gnn(n_coords, ca_coords, c_coords)

                # Get pLM embeddings
                pLM_embeddings = masked_outputs['sequence_output']

                # Calculate structure alignment loss
                struct_align_results = structure_alignment_loss(
                    pLM_embeddings=pLM_embeddings,
                    pGNN_embeddings=pGNN_embeddings,
                    structure_tokens=structure_tokens,
                    attention_mask=attention_mask
                )
                struct_align_loss = struct_align_results['total_loss']
                gnn_loss = struct_align_results.get('latent_loss', torch.tensor(0.0, device=device))
                foldseek_loss = struct_align_results.get('physical_loss', torch.tensor(0.0, device=device))

                # Extract per-sample losses for GNN and foldseek
                per_sample_gnn_losses = struct_align_results.get('latent_loss_per_sample', torch.zeros(batch_size, device=device))
                per_sample_foldseek_losses = struct_align_results.get('physical_loss_per_sample', torch.zeros(batch_size, device=device))

            # Use multi-constraint Lagrangian if it exists, otherwise use original approach
            if multi_constraint_lagrangian is not None:
                # Calculate Lagrangian for validation with multiple constraints using per-sample losses
                lagrangian, lagrangian_components = multi_constraint_lagrangian.compute_lagrangian(
                    primary_loss=mlm_loss,
                    dihedral_losses=per_sample_dihedral_losses,
                    gnn_losses=per_sample_gnn_losses,
                    foldseek_losses=per_sample_foldseek_losses,
                    batch_size=batch_size
                )
            else:
                # Fallback to single constraint approach
                lagrangian = dihedral_constraints.compute_lagrangian(
                    primary_loss=mlm_loss,
                    constraint_loss=raw_constraint_loss
                )

            combined_loss = lagrangian

            total_loss += combined_loss.item()
            dihedral_loss_total += dihedral_loss.item()
            mlm_loss_total += mlm_loss.item()
            structure_alignment_loss_total += struct_align_loss.item()
            gnn_loss_total += gnn_loss.item()
            foldseek_loss_total += foldseek_loss.item()

            # Log validation metrics to wandb
            if config.logging.use_wandb and batch_idx % 10 == 0:
                if multi_constraint_lagrangian is not None:
                    wandb.log({
                        'val_batch_loss': combined_loss.item(),
                        'val_batch_mlm_loss': mlm_loss.item(),
                        'val_batch_dihedral_loss': dihedral_loss.item(),
                        'val_batch_gnn_loss': gnn_loss.item(),
                        'val_batch_foldseek_loss': foldseek_loss.item(),
                        'val_batch_struct_align_loss': struct_align_loss.item(),
                        'val_batch': batch_idx
                    })
                else:
                    wandb.log({
                        'val_batch_loss': combined_loss.item(),
                        'val_batch_mlm_loss': mlm_loss.item(),
                        'val_batch_dihedral_loss': dihedral_loss.item(),
                        'val_batch_struct_align_loss': struct_align_loss.item(),
                        'val_batch': batch_idx
                    })

    avg_loss = total_loss / len(dataloader)
    avg_dihedral_loss = dihedral_loss_total / len(dataloader)
    avg_mlm_loss = mlm_loss_total / len(dataloader)
    avg_struct_align_loss = structure_alignment_loss_total / len(dataloader)
    avg_gnn_loss = gnn_loss_total / len(dataloader)
    avg_foldseek_loss = foldseek_loss_total / len(dataloader)

    return avg_loss, avg_mlm_loss, avg_dihedral_loss, avg_struct_align_loss, avg_gnn_loss, avg_foldseek_loss


def main():
    args = parse_args()

    # Load config from file if provided
    if args.config and os.path.exists(args.config):
        config = OmegaConf.load(args.config)
    else:
        # Use args to create config if no config file exists
        config = OmegaConf.create({
            'model': {
                'model_name': args.model_name,
                'constraint_weight': args.constraint_weight,
                'dist_threshold': args.dist_threshold
            },
            'lora': {
                'r': args.lora_r,
                'alpha': args.lora_alpha,
                'dropout': 0.1,
                'target_modules': ["query", "key", "value", "dense", "intermediate.dense", "output.dense"]
            },
            'training': {
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'num_epochs': args.num_epochs,
                'warmup_steps': args.warmup_steps,
                'max_seq_len': args.max_seq_len,
                'seed': args.seed,
                'output_dir': args.output_dir
            },
            'data': {
                'data_path': args.data_path
            },
            'logging': {
                'use_wandb': True,
                'project_name': 'geotune_new'
            }
        })

    # Set random seed
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.project_name,
            config=OmegaConf.to_container(config),
            name=f"esm2_lora_constraints_{config.model.model_name.split('/')[-1]}"
        )

    # Load model and tokenizer
    print("Loading model...")
    lora_params = {
        'r': config.lora.r,
        'lora_alpha': config.lora.alpha,
        'lora_dropout': config.lora.dropout,
        'target_modules': config.lora.target_modules
    }
    model, tokenizer = load_esm_with_lora(config.model.model_name, lora_params)
    model.to(device)

    # Enable gradient checkpointing if configured
    if hasattr(config.training, 'use_gradient_checkpointing') and config.training.use_gradient_checkpointing:
        if hasattr(model.model, 'gradient_checkpointing_enable'):
            model.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        else:
            print("Warning: Gradient checkpointing not supported by this model")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}% of total)")

    # Log model parameter info to wandb
    if config.logging.use_wandb:
        wandb.config.update({
            'trainable_params': trainable_params,
            'total_params': total_params,
            'trainable_percentage': trainable_params/total_params*100
        })

    # Initialize constrained dihedral angle constraints
    dihedral_constraints = ConstrainedDihedralAngleConstraint(
        constraint_weight=config.model.constraint_weight
    ).to(device)

    # Initialize multi-constraint Lagrangian for primal-dual optimization
    multi_constraint_lagrangian = MultiConstraintLagrangian(
        dihedral_epsilon=0.076,  # Epsilon for dihedral constraint
        gnn_epsilon=6.38,        # Epsilon for GNN loss constraint
        foldseek_epsilon=3.00,   # Epsilon for foldseek loss constraint
        alpha=1.0,               # Penalty for slack variables
        max_batch_size=config.training.batch_size  # Maximum expected batch size
    ).to(device)

    # Initialize structure alignment loss module
    # Use ESM model's hidden dimension for the projection dimensions
    esm_hidden_size = model.config.hidden_size
    structure_alignment_loss = StructureAlignmentLoss(
        hidden_dim=esm_hidden_size,
        num_structural_classes=21,  # 21 structural classes for Foldseek (20 + 'X')
        shared_projection_dim=512,
        latent_weight=0.5,
        physical_weight=0.5
    ).to(device)

    # Initialize frozen pre-trained GNN (e.g. GearNet) - use stub to avoid TorchDrug dependency
    # Set use_gearnet_stub=True to avoid TorchDrug/RDKit compatibility issues
    frozen_gnn = PretrainedGNNWrapper(hidden_dim=esm_hidden_size, use_gearnet_stub=True).to(device)
    frozen_gnn.eval()  # Set to evaluation mode to ensure no gradients

    # Load dataset
    print("Loading dataset...")

    # Use the efficient pre-processed dataset
    processed_dataset_path = os.path.join(config.data.data_path, "processed_dataset.pkl")
    if not os.path.exists(processed_dataset_path):
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_dataset_path}. "
            f"Please run 'python data_pipeline/process_dataset.py --raw_dir [path_to_pdb_files] --output_dir [path_to_save_processed_data] --create_efficient_dataset' "
            f"to create the processed dataset first."
        )

    print("Using efficient pre-processed dataset...")
    # Check if structural tokens are available
    struct_token_path = os.path.join(config.data.data_path, "structural_tokens.pkl")
    include_structural_tokens = os.path.exists(struct_token_path)
    print(f"Structural tokens available: {include_structural_tokens}")

    # Check if embeddings directory exists
    embeddings_path = os.path.join(config.data.data_path, "embeddings")
    embeddings_exist = os.path.exists(embeddings_path)

    # IMPORTANT: Disable loading pre-computed embeddings if they have wrong dimension
    # Pre-computed embeddings must match ESM model's hidden_dim
    # ESM2-150M uses 640, ESM2-35M uses 480, etc.
    # If embeddings were pre-computed with different hidden_dim, regenerate them or disable
    load_embeddings = False  # Generate on-the-fly with correct dimension
    if embeddings_exist:
        print(f"Pre-computed embeddings found but disabled (may have wrong dimension)")
        print(f"Embeddings will be generated on-the-fly with hidden_dim={esm_hidden_size}")
    else:
        print(f"Pre-computed embeddings not available, will generate on-the-fly")

    full_dataset = EfficientProteinDataset(
        config.data.data_path,
        max_seq_len=config.training.max_seq_len,
        include_structural_tokens=include_structural_tokens,
        load_embeddings=load_embeddings
    )

    # Create train-validation split (80% train, 20% validation)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    # Set seed for reproducible splits
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.training.seed)
    )

    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1
    )

    # Setup primal optimizer - for model parameters and all slack variables
    primal_params = list(model.parameters()) + [
        multi_constraint_lagrangian.s_dihedral,
        multi_constraint_lagrangian.s_gnn,
        multi_constraint_lagrangian.s_foldseek
    ]
    primal_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, primal_params),  # Only optimize trainable params
        lr=config.training.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.98)  # Standard AdamW values
    )

    # Setup dual optimizer - for all Lagrange multipliers (separate from primal optimizer)
    # Only the first few elements (up to batch size) will be actively used, but we want to optimize all of them
    dual_params = [
        multi_constraint_lagrangian.lam_dihedral,
        multi_constraint_lagrangian.lam_gnn,
        multi_constraint_lagrangian.lam_foldseek
    ]
    dual_optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, dual_params),  # Only optimize trainable params
        lr=1e-3
    )  # Smaller learning rate for dual variables

    # Setup scheduler
    # Adjust total steps for gradient accumulation
    gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    total_steps = (len(train_loader) // gradient_accumulation_steps) * config.training.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        primal_optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=total_steps
    )

    # Setup mixed precision scaler if enabled
    scaler = None
    if hasattr(config.training, 'mixed_precision') and config.training.mixed_precision:
        if torch.cuda.is_available():
            scaler = torch.amp.GradScaler('cuda')
            print("Mixed precision training enabled")
        else:
            print("Mixed precision requested but CUDA not available, using float32")

    # Training loop
    print(f"Starting training for {config.training.num_epochs} epochs...")
    print(f"Effective batch size: {config.training.batch_size * gradient_accumulation_steps}")

    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.training.num_epochs}")

        # Train with multi-constraint approach
        train_loss, train_mlm_loss, train_dihedral_loss, train_struct_align_loss, train_gnn_loss, train_foldseek_loss = train_epoch(
            model, train_loader, primal_optimizer, scheduler, dihedral_constraints, device, config,
            structure_alignment_loss=structure_alignment_loss, frozen_gnn=frozen_gnn,
            multi_constraint_lagrangian=multi_constraint_lagrangian, scaler=scaler
        )

        # Validate with multi-constraint approach
        val_loss, val_mlm_loss, val_dihedral_loss, val_struct_align_loss, val_gnn_loss, val_foldseek_loss = validate(
            model, val_loader, dihedral_constraints, device, config,
            structure_alignment_loss=structure_alignment_loss, frozen_gnn=frozen_gnn,
            multi_constraint_lagrangian=multi_constraint_lagrangian
        )

        # Log metrics
        if config.logging.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_mlm_loss': train_mlm_loss,
                'train_dihedral_loss': train_dihedral_loss,
                'train_gnn_loss': train_gnn_loss,
                'train_foldseek_loss': train_foldseek_loss,
                'train_struct_align_loss': train_struct_align_loss,
                'val_loss': val_loss,
                'val_mlm_loss': val_mlm_loss,
                'val_dihedral_loss': val_dihedral_loss,
                'val_gnn_loss': val_gnn_loss,
                'val_foldseek_loss': val_foldseek_loss,
                'val_struct_align_loss': val_struct_align_loss,
            })

        print(f"Epoch {epoch+1} completed:")
        print(f"  Train Loss: {train_loss:.4f} (MLM: {train_mlm_loss:.4f}, Dihedral: {train_dihedral_loss:.4f}, GNN: {train_gnn_loss:.4f}, FoldSeek: {train_foldseek_loss:.4f}, StructAlign: {train_struct_align_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (MLM: {val_mlm_loss:.4f}, Dihedral: {val_dihedral_loss:.4f}, GNN: {val_gnn_loss:.4f}, FoldSeek: {val_foldseek_loss:.4f}, StructAlign: {val_struct_align_loss:.4f})")

        # Save model checkpoint periodically
        if (epoch + 1) % 2 == 0:  # Save every 2 epochs
            checkpoint_dir = os.path.join(config.training.output_dir, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save LoRA adapters
            model.save_lora_adapters(os.path.join(checkpoint_dir, "lora_adapters"))

            print(f"Checkpoint saved to {checkpoint_dir}")

    # Final save
    final_dir = os.path.join(config.training.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_lora_adapters(os.path.join(final_dir, "lora_adapters"))

    print("Training completed!")
    if config.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()