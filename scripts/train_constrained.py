"""
Main training script for ESM2 with constrained geometric learning and LoRA.
Implements primal-dual optimization where all losses except MLM are treated as constraints.
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import transformers
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
import argparse
from omegaconf import OmegaConf
import pickle
import logging
import glob

from models.geotune_esm_model import load_esm_with_lora
from utils.constrained_dihedral_utils import MultiConstraintLagrangian, ConstrainedDihedralAngleConstraint, compute_dihedral_angles_from_coordinates
from utils.data_utils import EfficientProteinDataset, collate_fn
from utils.structure_alignment_utils import StructureAlignmentLoss, PretrainedGNNWrapper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parses command-line arguments.
    
    Note: Epsilon values for constrained learning are the ONLY training parameters
    that can be set via command-line. All other training parameters must come from
    the config file.
    """
    parser = argparse.ArgumentParser(description="Train ESM2 with constrained geometric learning and LoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--data_path", type=str, default=None, help="Override path to processed protein data directory.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory for checkpoints.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint directory to resume training from.")

    # Epsilon values for constrained learning - these are the ONLY training params allowed as CLI args
    parser.add_argument("--dihedral_epsilon", type=float, required=True, help="Epsilon for dihedral angle constraint.")
    parser.add_argument("--gnn_epsilon", type=float, required=True, help="Epsilon for GNN structure alignment constraint.")
    parser.add_argument("--foldseek_epsilon", type=float, required=True, help="Epsilon for Foldseek structure alignment constraint.")
    
    # Data split (not in config as it's experiment-specific)
    parser.add_argument("--train_fraction", type=float, default=0.8, help="Fraction of data to use for training.")

    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, scheduler, lagrangian_module, dihedral_module, 
                alignment_module, gnn_module, device, config, epoch, scaler):
    """Trains the model for one epoch using constrained optimization."""
    model.train()

    total_lagrangian_loss = 0
    total_mlm_loss = 0
    total_dihedral_loss = 0
    total_gnn_loss = 0
    total_foldseek_loss = 0
    total_struct_align_loss = 0

    progress_bar = tqdm(dataloader, desc="Training Epoch")
    gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    mask_ratio = getattr(config.training, 'mask_ratio', 0.15)
    struct_align_weight = getattr(config.training, 'struct_align_weight', 0.1)

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        indices = batch['indices'].to(device)
        n_coords = batch['n_coords'].to(device)
        ca_coords = batch['ca_coords'].to(device)
        c_coords = batch['c_coords'].to(device)
        batch_size = input_ids.size(0)

        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            # 1. Primary Loss (MLM)
            mask_positions = (torch.rand(input_ids.shape, device=device) < mask_ratio) & attention_mask.bool()
            labels = input_ids.clone()
            labels[~mask_positions] = -100
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = 32

            masked_outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
            pLM_embeddings = masked_outputs['sequence_output']

            seq_logits = model.lm_head(pLM_embeddings)
            mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                seq_logits.view(-1, seq_logits.size(-1)), 
                labels.view(-1)
            )

            # 2. Per-Sample Constraint Losses
            dihedral_results = dihedral_module(
                sequence_embeddings=pLM_embeddings,
                n_coords=n_coords,
                ca_coords=ca_coords,
                c_coords=c_coords,
                attention_mask=attention_mask
            )
            
            cos_true_phi, cos_true_psi = compute_dihedral_angles_from_coordinates(n_coords, ca_coords, c_coords)
            cos_pred_phi, cos_pred_psi = dihedral_module.predict_dihedral_angles(pLM_embeddings)

            min_len_phi = min(cos_true_phi.shape[1], cos_pred_phi.shape[1])
            phi_mask = attention_mask[:, 1:1+min_len_phi].float()
            phi_loss_per_sample = (dihedral_module.loss_fn(cos_pred_phi[:, :min_len_phi], cos_true_phi[:, :min_len_phi]) * phi_mask).sum(dim=1) / phi_mask.sum(dim=1).clamp(min=1.0)

            min_len_psi = min(cos_true_psi.shape[1], cos_pred_psi.shape[1])
            psi_mask = attention_mask[:, :min_len_psi].float()
            psi_loss_per_sample = (dihedral_module.loss_fn(cos_pred_psi[:, :min_len_psi], cos_true_psi[:, :min_len_psi]) * psi_mask).sum(dim=1) / psi_mask.sum(dim=1).clamp(min=1.0)

            per_sample_dihedral_losses = phi_loss_per_sample + psi_loss_per_sample

            # Structure Alignment Constraints
            per_sample_gnn_losses = torch.zeros(batch_size, device=device)
            per_sample_foldseek_losses = torch.zeros(batch_size, device=device)
            
            if alignment_module is not None and 'structural_tokens' in batch:
                structure_tokens = batch['structural_tokens'].to(device)
                if 'precomputed_embeddings' in batch:
                    pGNN_embeddings = batch['precomputed_embeddings'].to(device)
                else:
                    with torch.no_grad():
                        pGNN_embeddings = gnn_module(n_coords, ca_coords, c_coords)

                struct_align_results = alignment_module(pLM_embeddings, pGNN_embeddings, structure_tokens, attention_mask)
                per_sample_gnn_losses = struct_align_results.get('latent_loss_per_sample', per_sample_gnn_losses)
                per_sample_foldseek_losses = struct_align_results.get('physical_loss_per_sample', per_sample_foldseek_losses)

            # 3. Compute Lagrangian
            lagrangian = lagrangian_module.compute_lagrangian(
                primary_loss=mlm_loss,
                dihedral_losses=per_sample_dihedral_losses * config.model.constraint_weight,
                gnn_losses=per_sample_gnn_losses * struct_align_weight,
                foldseek_losses=per_sample_foldseek_losses * struct_align_weight,
                indices=indices
            )

            scaled_lagrangian = lagrangian / gradient_accumulation_steps

        # 4. Backward pass
        if scaler is not None:
            scaler.scale(scaled_lagrangian).backward()
        else:
            scaled_lagrangian.backward()

        # 5. Dual Update (Lagrange Multipliers) - every batch
        lagrangian_module.update_dual_variables(
            per_sample_dihedral_losses,
            per_sample_gnn_losses,
            per_sample_foldseek_losses,
            indices
        )

        # 6. Primal Update (Model Parameters) - every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            
            all_params = [p for group in optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.5)
            
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        total_lagrangian_loss += lagrangian.item()
        total_mlm_loss += mlm_loss.item()
        total_dihedral_loss += (per_sample_dihedral_losses.mean().item() * dihedral_module.constraint_weight)
        total_gnn_loss += per_sample_gnn_losses.mean().item()
        total_foldseek_loss += per_sample_foldseek_losses.mean().item()
        total_struct_align_loss += (per_sample_gnn_losses.mean().item() + per_sample_foldseek_losses.mean().item())

        if batch_idx % 10 == 0 and config.logging.use_wandb:
            lambda_stats = lagrangian_module.get_lambda_stats()
            wandb.log({
                'train_batch_mlm_loss': mlm_loss.item(),
                'train_batch_dihedral_loss': per_sample_dihedral_losses.mean().item() * dihedral_module.constraint_weight,
                'train_batch_gnn_loss': per_sample_gnn_losses.mean().item(),
                'train_batch_foldseek_loss': per_sample_foldseek_losses.mean().item(),
                'train_batch_lagrangian_loss': lagrangian.item(),
                'train_lambda_dihedral_mean': lambda_stats['lam_dihedral_mean'],
                'train_lambda_dihedral_std': lambda_stats['lam_dihedral_std'],
                'train_lambda_dihedral_min': lambda_stats['lam_dihedral_min'],
                'train_lambda_dihedral_max': lambda_stats['lam_dihedral_max'],
                'train_lambda_gnn_mean': lambda_stats['lam_gnn_mean'],
                'train_lambda_gnn_std': lambda_stats['lam_gnn_std'],
                'train_lambda_gnn_min': lambda_stats['lam_gnn_min'],
                'train_lambda_gnn_max': lambda_stats['lam_gnn_max'],
                'train_lambda_foldseek_mean': lambda_stats['lam_foldseek_mean'],
                'train_lambda_foldseek_std': lambda_stats['lam_foldseek_std'],
                'train_lambda_foldseek_min': lambda_stats['lam_foldseek_min'],
                'train_lambda_foldseek_max': lambda_stats['lam_foldseek_max'],
                'learning_rate': scheduler.get_last_lr()[0]
            })

    num_batches = len(dataloader)
    avg_struct_align_loss = (total_gnn_loss + total_foldseek_loss) / num_batches if num_batches > 0 else 0
    return (
        total_lagrangian_loss / num_batches,
        total_mlm_loss / num_batches,
        total_dihedral_loss / num_batches,
        total_gnn_loss / num_batches,
        total_foldseek_loss / num_batches,
        avg_struct_align_loss
    )


def validate(model, dataloader, dihedral_module, alignment_module, gnn_module, device, config):
    """Validates the model by computing unconstrained losses."""
    import time

    model.eval()
    dihedral_module.eval()
    if alignment_module is not None:
        alignment_module.eval()
    if gnn_module is not None:
        gnn_module.eval()

    total_mlm_loss, total_dihedral_loss, total_gnn_loss, total_foldseek_loss, total_struct_align_loss = 0, 0, 0, 0, 0
    num_batches = 0

    val_start_time = time.time()
    forward_time = 0
    loss_time = 0
    dataloader_time = 0
    precomputed_count = 0
    onthefly_count = 0

    mask_ratio = getattr(config.training, 'mask_ratio', 0.15)

    with torch.inference_mode():
        progress_bar = tqdm(dataloader, desc="Validating")
        batch_start = time.time()

        for batch_idx, batch in enumerate(progress_bar):
            dataloader_time += time.time() - batch_start

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            n_coords = batch['n_coords'].to(device)
            ca_coords = batch['ca_coords'].to(device)
            c_coords = batch['c_coords'].to(device)

            has_precomputed_embeddings = 'precomputed_embeddings' in batch
            if has_precomputed_embeddings:
                precomputed_count += 1
            else:
                onthefly_count += 1

            # Forward pass for geometric losses
            forward_start = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pLM_embeddings = outputs['sequence_output']
            forward_time += time.time() - forward_start

            # Dihedral Loss
            loss_start = time.time()
            dihedral_results = dihedral_module(
                sequence_embeddings=pLM_embeddings,
                n_coords=n_coords,
                ca_coords=ca_coords,
                c_coords=c_coords,
                attention_mask=attention_mask
            )
            
            cos_true_phi, cos_true_psi = compute_dihedral_angles_from_coordinates(n_coords, ca_coords, c_coords)
            cos_pred_phi, cos_pred_psi = dihedral_module.predict_dihedral_angles(pLM_embeddings)

            min_len_phi = min(cos_true_phi.shape[1], cos_pred_phi.shape[1])
            phi_mask = attention_mask[:, 1:1+min_len_phi].float()
            phi_loss_per_sample = (dihedral_module.loss_fn(cos_pred_phi[:, :min_len_phi], cos_true_phi[:, :min_len_phi]) * phi_mask).sum(dim=1) / phi_mask.sum(dim=1).clamp(min=1.0)

            min_len_psi = min(cos_true_psi.shape[1], cos_pred_psi.shape[1])
            psi_mask = attention_mask[:, :min_len_psi].float()
            psi_loss_per_sample = (dihedral_module.loss_fn(cos_pred_psi[:, :min_len_psi], cos_true_psi[:, :min_len_psi]) * psi_mask).sum(dim=1) / psi_mask.sum(dim=1).clamp(min=1.0)

            per_sample_dihedral_losses = phi_loss_per_sample + psi_loss_per_sample
            total_dihedral_loss += (per_sample_dihedral_losses.mean().item() * dihedral_module.constraint_weight)

            # Structure Alignment Losses
            if alignment_module is not None and 'structural_tokens' in batch:
                structure_tokens = batch['structural_tokens'].to(device)
                if has_precomputed_embeddings:
                    pGNN_embeddings = batch['precomputed_embeddings'].to(device)
                else:
                    pGNN_embeddings = gnn_module(n_coords, ca_coords, c_coords)

                struct_align_results = alignment_module(pLM_embeddings, pGNN_embeddings, structure_tokens, attention_mask)

                if torch.is_tensor(struct_align_results['latent_loss']):
                    total_gnn_loss += struct_align_results['latent_loss'].item()
                if torch.is_tensor(struct_align_results['physical_loss']):
                    total_foldseek_loss += struct_align_results['physical_loss'].item()
                total_struct_align_loss += (
                    (struct_align_results['latent_loss'].item() if torch.is_tensor(struct_align_results['latent_loss']) else 0.0) + \
                    (struct_align_results['physical_loss'].item() if torch.is_tensor(struct_align_results['physical_loss']) else 0.0)
                )
            loss_time += time.time() - loss_start

            # MLM Loss
            mask_positions = (torch.rand(input_ids.shape, device=device) < mask_ratio) & attention_mask.bool()
            labels = input_ids.clone()
            labels[~mask_positions] = -100
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = 32
            masked_outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
            seq_logits = model.lm_head(masked_outputs['sequence_output'])
            total_mlm_loss += nn.CrossEntropyLoss(ignore_index=-100)(
                seq_logits.view(-1, seq_logits.size(-1)), 
                labels.view(-1)
            ).item()

            num_batches += 1

            total_time = time.time() - val_start_time
            iter_per_sec = (batch_idx + 1) / total_time if total_time > 0 else 0
            progress_bar.set_postfix({'Loss': f'{(total_mlm_loss + total_dihedral_loss + total_gnn_loss + total_foldseek_loss) / num_batches:.4f}', 'iter/s': f'{iter_per_sec:.1f}'})
            batch_start = time.time()

    total_time = time.time() - val_start_time
    logger.info(f"Validation completed in {total_time:.2f}s ({num_batches/total_time:.1f} iter/s)")

    avg_struct_align_loss = total_struct_align_loss / num_batches if num_batches > 0 else 0
    return (
        total_mlm_loss / num_batches if num_batches > 0 else 0,
        total_dihedral_loss / num_batches if num_batches > 0 else 0,
        total_gnn_loss / num_batches if num_batches > 0 else 0,
        total_foldseek_loss / num_batches if num_batches > 0 else 0,
        avg_struct_align_loss
    )


def log_lambda_distributions(lagrangian_module, epoch, config):
    """Log lambda statistics to wandb."""
    if not config.logging.use_wandb:
        return

    lambda_stats = lagrangian_module.get_lambda_stats()
    wandb.log({
        'lambda_stats/dihedral_mean': lambda_stats['lam_dihedral_mean'],
        'lambda_stats/dihedral_std': lambda_stats['lam_dihedral_std'],
        'lambda_stats/dihedral_min': lambda_stats['lam_dihedral_min'],
        'lambda_stats/dihedral_max': lambda_stats['lam_dihedral_max'],
        'lambda_stats/gnn_mean': lambda_stats['lam_gnn_mean'],
        'lambda_stats/gnn_std': lambda_stats['lam_gnn_std'],
        'lambda_stats/gnn_min': lambda_stats['lam_gnn_min'],
        'lambda_stats/gnn_max': lambda_stats['lam_gnn_max'],
        'lambda_stats/foldseek_mean': lambda_stats['lam_foldseek_mean'],
        'lambda_stats/foldseek_std': lambda_stats['lam_foldseek_std'],
        'lambda_stats/foldseek_min': lambda_stats['lam_foldseek_min'],
        'lambda_stats/foldseek_max': lambda_stats['lam_foldseek_max'],
        'epoch': epoch
    })


def save_checkpoint(model, lagrangian_module, optimizer, scheduler, epoch, val_loss, config, checkpoint_dir, is_best=False):
    """Save training checkpoint with all necessary state"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save LoRA adapters
    lora_path = os.path.join(checkpoint_dir, "lora_adapters")
    model.save_lora_adapters(lora_path)
    
    # Save training state
    checkpoint = {
        'epoch': epoch,
        'val_loss': val_loss,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': OmegaConf.to_container(config),
        'epsilon_config': {
            'dihedral_epsilon': config.training.dihedral_epsilon,
            'gnn_epsilon': config.training.gnn_epsilon,
            'foldseek_epsilon': config.training.foldseek_epsilon,
        }
    }
    
    if lagrangian_module is not None:
        checkpoint['lagrangian_state'] = lagrangian_module.state_dict()
    
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'training_state.pt'))
    
    # Create symlink for easy access to best model
    if is_best:
        best_link_path = os.path.join(os.path.dirname(checkpoint_dir), "best_model")
        if os.path.exists(best_link_path) or os.path.islink(best_link_path):
            os.remove(best_link_path)
        os.symlink(os.path.basename(checkpoint_dir), best_link_path)
        logger.info(f"Created symlink to best model at {best_link_path}")
    
    logger.info(f"Checkpoint saved to {checkpoint_dir}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, lagrangian_module=None, device=None):
    """Load training checkpoint and restore state"""
    checkpoint = torch.load(os.path.join(checkpoint_path, 'training_state.pt'), map_location=device)
    
    # Load LoRA adapters
    lora_path = os.path.join(checkpoint_path, "lora_adapters")
    if os.path.exists(lora_path):
        from peft import PeftModel
        model.model = PeftModel.from_pretrained(model.model.base_model.model, lora_path)
        logger.info(f"Loaded LoRA adapters from {lora_path}")
    
    # Restore training state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if lagrangian_module is not None and 'lagrangian_state' in checkpoint:
        lagrangian_module.load_state_dict(checkpoint['lagrangian_state'])
        logger.info("Loaded Lagrangian state")
    
    logger.info(f"Resumed from epoch {checkpoint['epoch']} with val_loss {checkpoint['val_loss']:.4f}")
    
    return checkpoint['epoch'], checkpoint['val_loss'], checkpoint.get('epsilon_config', None)


def main():
    """Main function to set up and run constrained training."""
    args = parse_args()
    config = OmegaConf.load(args.config)

    # Override config with command-line arguments (only paths, not training params)
    if args.data_path:
        config.data.data_path = args.data_path
    if args.output_dir:
        config.training.output_dir = args.output_dir

    # Use config values for learning rates - use primal_lr and dual_lr from config
    if hasattr(config.training, 'primal_lr'):
        config.training.learning_rate = config.training.primal_lr
        logger.info(f"Using primal_lr from config: {config.training.primal_lr}")
    if hasattr(config.training, 'dual_lr'):
        config.training.dual_learning_rate = config.training.dual_lr
        logger.info(f"Using dual_lr from config: {config.training.dual_lr}")
    elif not hasattr(config.training, 'dual_learning_rate'):
        # Fallback if neither dual_lr nor dual_learning_rate is set
        config.training.dual_learning_rate = 5e-1
        logger.warning(f"dual_lr not set in config, using default: {config.training.dual_learning_rate}")

    # Epsilon values for constrained learning - MUST be provided via CLI (required in parse_args)
    # Store epsilon config
    config.training.dihedral_epsilon = args.dihedral_epsilon
    config.training.gnn_epsilon = args.gnn_epsilon
    config.training.foldseek_epsilon = args.foldseek_epsilon

    # Use separate output directory for constrained training
    if args.output_dir is None:
        args.output_dir = "outputs_constrained"

    # Generate unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.training.output_dir = os.path.join(
        args.output_dir,
        f"run_eps{args.dihedral_epsilon}-{args.gnn_epsilon}-{args.foldseek_epsilon}_{timestamp}"
    )
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Store epsilon config
    config.training.dihedral_epsilon = args.dihedral_epsilon
    config.training.gnn_epsilon = args.gnn_epsilon
    config.training.foldseek_epsilon = args.foldseek_epsilon

    logger.info(f"Output directory: {config.training.output_dir}")
    logger.info(f"Epsilon values - Dihedral: {args.dihedral_epsilon}, GNN: {args.gnn_epsilon}, Foldseek: {args.foldseek_epsilon}")

    # Setup
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize wandb
    if config.logging.use_wandb:
        model_name_parts = config.model.model_name.split('/')[-1].split('_')
        model_short = next((part for part in model_name_parts if 'M' in part), 'unknown')
        run_name = (f"constrained_{model_short}_"
                   f"plr{config.training.learning_rate}_"
                   f"dlr{config.training.dual_learning_rate}_"
                   f"eps{config.training.dihedral_epsilon}-{config.training.gnn_epsilon}-{config.training.foldseek_epsilon}_"
                   f"bs{config.training.batch_size}_"
                   f"{timestamp}")
        wandb.init(project=config.logging.project_name, config=OmegaConf.to_container(config), name=run_name)

    # Load model
    lora_params = {
        "r": config.lora.r, 
        "lora_alpha": config.lora.alpha, 
        "lora_dropout": config.lora.dropout, 
        "target_modules": list(config.lora.target_modules)
    }
    model, _ = load_esm_with_lora(config.model.model_name, lora_params)
    model.to(device)
    esm_hidden_size = model.config.hidden_size

    # Initialize modules
    dihedral_module = ConstrainedDihedralAngleConstraint(
        hidden_dim=esm_hidden_size,
        constraint_weight=config.model.constraint_weight
    ).to(device)

    use_structure_alignment = getattr(config.constraints, 'use_structure_alignment', True)
    use_precomputed = getattr(config.data, 'use_precomputed_embeddings', False)
    load_embeddings = False
    pgnn_hidden_dim = None
    gnn_module = None
    alignment_module = None

    if use_structure_alignment:
        if use_precomputed:
            embeddings_path = os.path.join(config.data.data_path, "embeddings")
            if not os.path.exists(embeddings_path) or not any(f.endswith('_gearnet_embeddings.pkl') for f in os.listdir(embeddings_path)):
                raise FileNotFoundError(f"Pre-computed embeddings not found in {embeddings_path}")

            logger.info("Using pre-computed embeddings")
            load_embeddings = True

            embedding_files = [f for f in os.listdir(embeddings_path) if f.endswith('_gearnet_embeddings.pkl')]
            sample_file = os.path.join(embeddings_path, embedding_files[0])
            with open(sample_file, 'rb') as f:
                sample_data = pickle.load(f)
                sample_embeddings = sample_data['embeddings']
                if isinstance(sample_embeddings, list):
                    sample_embeddings = np.array(sample_embeddings)
                pgnn_hidden_dim = sample_embeddings.shape[-1]
        else:
            logger.info("Generating embeddings on-the-fly")
            load_embeddings = False
            gnn_module = PretrainedGNNWrapper(hidden_dim=esm_hidden_size, use_simple_encoder=False).to(device).eval()
            pgnn_hidden_dim = gnn_module.output_dim

        alignment_module = StructureAlignmentLoss(
            hidden_dim=esm_hidden_size,
            pgnn_hidden_dim=pgnn_hidden_dim,
            num_structural_classes=21
        ).to(device)

    # Load dataset
    full_dataset = EfficientProteinDataset(
        config.data.data_path, 
        max_seq_len=config.training.max_seq_len, 
        include_structural_tokens=True, 
        load_embeddings=load_embeddings
    )
    
    train_size = int(args.train_fraction * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(config.training.seed)
    )

    logger.info(f"Dataset split: {train_size} training, {val_size} validation samples")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=2
    )

    # Constrained Learning Setup
    lagrangian_module = MultiConstraintLagrangian(
        dataset_size=len(full_dataset),
        dihedral_epsilon=config.training.dihedral_epsilon,
        gnn_epsilon=config.training.gnn_epsilon,
        foldseek_epsilon=config.training.foldseek_epsilon,
        dual_lr=config.training.dual_learning_rate,
        device=device
    )

    # Optimizer and Scheduler - use primal_lr for learning rate
    trainable_params = list(model.parameters()) + list(dihedral_module.parameters()) + list(alignment_module.parameters())
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, trainable_params),
        lr=getattr(config.training, 'primal_lr', config.training.learning_rate)
    )
    
    total_steps = (len(train_loader) // getattr(config.training, 'gradient_accumulation_steps', 1)) * config.training.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.training.warmup_steps, 
        num_training_steps=total_steps
    )

    # Mixed precision
    scaler = None
    if getattr(config.training, 'mixed_precision', False) and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
        logger.info("Mixed precision training enabled")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume_from and os.path.exists(args.resume_from):
        start_epoch, best_val_loss, epsilon_config = load_checkpoint(
            args.resume_from, model, optimizer, scheduler, 
            lagrangian_module=lagrangian_module, device=device
        )
        start_epoch += 1

    # Training Loop
    logger.info(f"Starting constrained training for {config.training.num_epochs} epochs from epoch {start_epoch}")
    
    for epoch in range(start_epoch, config.training.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.training.num_epochs}")

        train_lagrangian, train_mlm, train_dihedral, train_gnn, train_foldseek, train_struct_align = train_epoch(
            model, train_loader, optimizer, scheduler, lagrangian_module, dihedral_module, 
            alignment_module, gnn_module, device, config, epoch, scaler
        )

        val_mlm, val_dihedral, val_gnn, val_foldseek, val_struct_align = validate(
            model, val_loader, dihedral_module, alignment_module, gnn_module, device, config
        )

        if config.logging.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_lagrangian,
                'train_mlm_loss': train_mlm,
                'train_dihedral_loss': train_dihedral,
                'train_gnn_loss': train_gnn,
                'train_foldseek_loss': train_foldseek,
                'train_struct_align_loss': train_struct_align,
                'val_mlm_loss': val_mlm,
                'val_dihedral_loss': val_dihedral,
                'val_gnn_loss': val_gnn,
                'val_foldseek_loss': val_foldseek,
                'val_struct_align_loss': val_struct_align,
                'train_lagrangian_loss': train_lagrangian,
                'learning_rate': scheduler.get_last_lr()[0]
            })

        logger.info(f"Epoch {epoch+1}: Train Lagrangian={train_lagrangian:.4f}, Val MLM={val_mlm:.4f}")

        # Log lambda distributions
        lambda_log_frequency = getattr(config.logging, 'lambda_log_frequency', 1)
        if (epoch + 1) % lambda_log_frequency == 0:
            log_lambda_distributions(lagrangian_module, epoch + 1, config)

        # Save checkpoint
        is_best = val_mlm < best_val_loss
        if is_best:
            best_val_loss = val_mlm
        
        checkpoint_frequency = getattr(config.training, 'checkpoint_frequency', 2)
        if (epoch + 1) % checkpoint_frequency == 0 or is_best:
            checkpoint_dir = os.path.join(config.training.output_dir, f"checkpoint_epoch_{epoch+1}")
            save_checkpoint(
                model, lagrangian_module, optimizer, scheduler,
                epoch=epoch, val_loss=val_mlm, config=config,
                checkpoint_dir=checkpoint_dir, is_best=is_best
            )

    # Final save
    final_dir = os.path.join(config.training.output_dir, "final_model")
    save_checkpoint(
        model, lagrangian_module, optimizer, scheduler,
        epoch=config.training.num_epochs - 1, val_loss=val_mlm, config=config,
        checkpoint_dir=final_dir
    )

    if config.logging.use_wandb:
        lambda_stats = lagrangian_module.get_lambda_stats()
        wandb.log({
            "final_lambda_stats/dihedral_mean": lambda_stats['lam_dihedral_mean'],
            "final_lambda_stats/gnn_mean": lambda_stats['lam_gnn_mean'],
            "final_lambda_stats/foldseek_mean": lambda_stats['lam_foldseek_mean'],
        })

    logger.info("Training completed!")
    if config.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
