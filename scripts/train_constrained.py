"""
Main training script for ESM2 with constrained geometric learning and LoRA.
This script implements a primal-dual optimization approach where all losses
except for Masked Language Modeling (MLM) are treated as constraints.
"""
import os
import sys
from pathlib import Path
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

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.geotune_esm_model import load_esm_with_lora
from utils.constrained_dihedral_utils import ConstrainedDihedralAngleConstraint, MultiConstraintLagrangian, compute_dihedral_angles_from_coordinates
from utils.data_utils import EfficientProteinDataset, collate_fn
from utils.structure_alignment_utils import StructureAlignmentLoss, PretrainedGNNWrapper

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train ESM2 with constrained geometric learning and LoRA")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the config file.")

    # --- Overrides for config file ---
    parser.add_argument("--data_path", type=str, default=None, help="Override path to the processed protein data directory.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory for checkpoints.")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override the learning rate.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override the batch size.")

    # --- New arguments for constrained learning ---
    parser.add_argument("--dual_learning_rate", type=float, default=1e-3, help="Learning rate for the dual optimizer (Lagrange multipliers).")
    parser.add_argument("--dihedral_epsilon", type=float, default=0.1, help="Epsilon (upper bound) for the dihedral angle constraint.")
    parser.add_argument("--gnn_epsilon", type=float, default=7.0, help="Epsilon (upper bound) for the GNN (latent) structure alignment constraint.")
    parser.add_argument("--foldseek_epsilon", type=float, default=3.0, help="Epsilon (upper bound) for the Foldseek (physical) structure alignment constraint.")

    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, scheduler, lagrangian_module, dihedral_module, alignment_module, gnn_module, device, config, scaler):
    """Trains the model for one epoch using a constrained optimization framework."""
    model.train()

    total_lagrangian_loss = 0
    total_mlm_loss = 0
    total_dihedral_loss = 0
    total_gnn_loss = 0
    total_foldseek_loss = 0

    progress_bar = tqdm(dataloader, desc="Training Epoch")
    gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        indices = batch['indices'].to(device)
        n_coords = batch['n_coords'].to(device)
        ca_coords = batch['ca_coords'].to(device)
        c_coords = batch['c_coords'].to(device)
        batch_size = input_ids.size(0)

        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            # 1. Calculate Primary Loss (MLM)
            mask_ratio = 0.15
            mask_positions = (torch.rand(input_ids.shape, device=device) < mask_ratio) & attention_mask.bool()
            labels = input_ids.clone()
            labels[~mask_positions] = -100
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = 32

            masked_outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
            pLM_embeddings = masked_outputs['sequence_output']

            seq_logits = model.lm_head(pLM_embeddings)
            mlm_loss = nn.CrossEntropyLoss()(seq_logits.view(-1, seq_logits.size(-1)), labels.view(-1))

            # 2. Calculate Per-Sample Constraint Losses
            # Dihedral Constraint
            cos_true_phi, cos_true_psi = compute_dihedral_angles_from_coordinates(n_coords, ca_coords, c_coords)
            cos_pred_phi, cos_pred_psi = dihedral_module.predict_dihedral_angles(pLM_embeddings)

            min_len_phi = min(cos_true_phi.shape[1], cos_pred_phi.shape[1])
            phi_mask = attention_mask[:, 1:1+min_len_phi].float()
            phi_loss_sq = (cos_true_phi[:, :min_len_phi] - cos_pred_phi[:, :min_len_phi])**2
            phi_loss_per_sample = (phi_loss_sq * phi_mask).sum(dim=1) / phi_mask.sum(dim=1).clamp(min=1.0)

            min_len_psi = min(cos_true_psi.shape[1], cos_pred_psi.shape[1])
            psi_mask = attention_mask[:, :min_len_psi].float()
            psi_loss_sq = (cos_true_psi[:, :min_len_psi] - cos_pred_psi[:, :min_len_psi])**2
            psi_loss_per_sample = (psi_loss_sq * psi_mask).sum(dim=1) / psi_mask.sum(dim=1).clamp(min=1.0)

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

            # 3. Compute the Lagrangian
            lagrangian = lagrangian_module.compute_lagrangian(
                primary_loss=mlm_loss,
                dihedral_losses=per_sample_dihedral_losses,
                gnn_losses=per_sample_gnn_losses,
                foldseek_losses=per_sample_foldseek_losses,
                indices=indices
            )

            scaled_lagrangian = lagrangian / gradient_accumulation_steps

        # 4. Backward pass for gradients
        if scaler is not None:
            scaler.scale(scaled_lagrangian).backward()
        else:
            scaled_lagrangian.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # 5. Primal Update (Model Parameters) - UPDATE THETA FIRST
            if scaler is not None:
                scaler.unscale_(optimizer)
            # Clip gradients for all trainable parameters in optimizer
            all_params = [p for group in optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.5)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # 6. Dual Update (Lagrange Multipliers) - UPDATE LAMBDA SECOND
            # Update based on constraint violations from current batch
            lagrangian_module.update_dual_variables(
                per_sample_dihedral_losses,
                per_sample_gnn_losses,
                per_sample_foldseek_losses,
                indices
            )

        # Logging
        total_lagrangian_loss += lagrangian.item()
        total_mlm_loss += mlm_loss.item()
        total_dihedral_loss += per_sample_dihedral_losses.mean().item()
        total_gnn_loss += per_sample_gnn_losses.mean().item()
        total_foldseek_loss += per_sample_foldseek_losses.mean().item()

        if batch_idx % 10 == 0 and config.logging.use_wandb:
            lambda_stats = lagrangian_module.get_lambda_stats()
            wandb.log({
                'train_batch/lagrangian_loss': lagrangian.item(),
                'train_batch/mlm_loss': mlm_loss.item(),
                'train_batch/dihedral_loss': per_sample_dihedral_losses.mean().item(),
                'train_batch/gnn_loss': per_sample_gnn_losses.mean().item(),
                'train_batch/foldseek_loss': per_sample_foldseek_losses.mean().item(),
                'train_lambda/dihedral_mean': lambda_stats['lam_dihedral_mean'],
                'train_lambda/dihedral_std': lambda_stats['lam_dihedral_std'],
                'train_lambda/dihedral_min': lambda_stats['lam_dihedral_min'],
                'train_lambda/dihedral_max': lambda_stats['lam_dihedral_max'],
                'train_lambda/gnn_mean': lambda_stats['lam_gnn_mean'],
                'train_lambda/gnn_std': lambda_stats['lam_gnn_std'],
                'train_lambda/gnn_min': lambda_stats['lam_gnn_min'],
                'train_lambda/gnn_max': lambda_stats['lam_gnn_max'],
                'train_lambda/foldseek_mean': lambda_stats['lam_foldseek_mean'],
                'train_lambda/foldseek_std': lambda_stats['lam_foldseek_std'],
                'train_lambda/foldseek_min': lambda_stats['lam_foldseek_min'],
                'train_lambda/foldseek_max': lambda_stats['lam_foldseek_max'],
                'learning_rate': scheduler.get_last_lr()[0]
            })

    num_batches = len(dataloader)
    return (
        total_lagrangian_loss / num_batches,
        total_mlm_loss / num_batches,
        total_dihedral_loss / num_batches,
        total_gnn_loss / num_batches,
        total_foldseek_loss / num_batches
    )

def validate(model, dataloader, dihedral_module, alignment_module, gnn_module, device, config):
    """Validates the model by computing and logging unconstrained losses."""
    model.eval()
    total_mlm_loss, total_dihedral_loss, total_gnn_loss, total_foldseek_loss = 0, 0, 0, 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            n_coords = batch['n_coords'].to(device)
            ca_coords = batch['ca_coords'].to(device)
            c_coords = batch['c_coords'].to(device)

            # === Forward pass for Geometric and Structural Losses (using unmasked inputs) ===
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pLM_embeddings = outputs['sequence_output']

            # Dihedral Loss
            cos_true_phi, cos_true_psi = compute_dihedral_angles_from_coordinates(n_coords, ca_coords, c_coords)
            cos_pred_phi, cos_pred_psi = dihedral_module.predict_dihedral_angles(pLM_embeddings)
            min_len_phi = min(cos_true_phi.shape[1], cos_pred_phi.shape[1])
            phi_mask = attention_mask[:, 1:1+min_len_phi].float()
            phi_loss_sq = (cos_true_phi[:, :min_len_phi] - cos_pred_phi[:, :min_len_phi])**2
            phi_loss_per_sample = (phi_loss_sq * phi_mask).sum(dim=1) / phi_mask.sum(dim=1).clamp(min=1.0)
            min_len_psi = min(cos_true_psi.shape[1], cos_pred_psi.shape[1])
            psi_mask = attention_mask[:, :min_len_psi].float()
            psi_loss_sq = (cos_true_psi[:, :min_len_psi] - cos_pred_psi[:, :min_len_psi])**2
            psi_loss_per_sample = (psi_loss_sq * psi_mask).sum(dim=1) / psi_mask.sum(dim=1).clamp(min=1.0)
            total_dihedral_loss += (phi_loss_per_sample + psi_loss_per_sample).mean().item()

            # Structure Alignment Losses
            if alignment_module is not None and 'structural_tokens' in batch:
                structure_tokens = batch['structural_tokens'].to(device)
                if 'precomputed_embeddings' in batch:
                    pGNN_embeddings = batch['precomputed_embeddings'].to(device)
                else:
                    pGNN_embeddings = gnn_module(n_coords, ca_coords, c_coords)
                
                struct_align_results = alignment_module(pLM_embeddings, pGNN_embeddings, structure_tokens, attention_mask)
                
                # Ensure we handle cases where loss is not computed (e.g., all NaNs were skipped)
                if torch.is_tensor(struct_align_results['latent_loss']):
                    total_gnn_loss += struct_align_results['latent_loss'].item()
                if torch.is_tensor(struct_align_results['physical_loss']):
                    total_foldseek_loss += struct_align_results['physical_loss'].item()

            # === Forward pass for MLM Loss (using masked inputs) ===
            mask_ratio = 0.15
            mask_positions = (torch.rand(input_ids.shape, device=device) < mask_ratio) & attention_mask.bool()
            labels = input_ids.clone()
            labels[~mask_positions] = -100
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = 32
            masked_outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
            seq_logits = model.lm_head(masked_outputs['sequence_output'])
            total_mlm_loss += nn.CrossEntropyLoss(ignore_index=-100)(seq_logits.view(-1, seq_logits.size(-1)), labels.view(-1)).item()

            num_batches += 1

    return (
        total_mlm_loss / num_batches if num_batches > 0 else 0,
        total_dihedral_loss / num_batches if num_batches > 0 else 0,
        total_gnn_loss / num_batches if num_batches > 0 else 0,
        total_foldseek_loss / num_batches if num_batches > 0 else 0
    )

def log_lambda_distributions(lagrangian_module, epoch, config):
    """
    Log lambda statistics (mean, std, min, max) during training for per-sample lambda vectors.

    Args:
        lagrangian_module: MultiConstraintLagrangian instance with lambda vectors
        epoch: Current epoch number
        config: Config object for logging settings
    """
    if not config.logging.use_wandb:
        return

    # Get lambda statistics
    lambda_stats = lagrangian_module.get_lambda_stats()

    # Log statistics to wandb
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

def main():
    """Main function to set up and run the constrained training process."""
    args = parse_args()
    config = OmegaConf.load(args.config)

    # --- Configuration Management ---
    # Override config with command-line arguments if provided
    if args.data_path: config.data.data_path = args.data_path
    if args.output_dir: config.training.output_dir = args.output_dir
    if args.num_epochs: config.training.num_epochs = args.num_epochs
    if args.learning_rate: config.training.learning_rate = args.learning_rate
    if args.batch_size: config.training.batch_size = args.batch_size

    # Add new constrained-learning arguments to the config for logging
    config.training.dual_learning_rate = args.dual_learning_rate
    config.training.dihedral_epsilon = args.dihedral_epsilon
    config.training.gnn_epsilon = args.gnn_epsilon
    config.training.foldseek_epsilon = args.foldseek_epsilon

    # --- Setup ---
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if config.logging.use_wandb:
        # Extract short model name (e.g., "8M" from "facebook/esm2_t6_8M_UR50D")
        model_name_parts = config.model.model_name.split('/')[-1].split('_')
        model_short = next((part for part in model_name_parts if 'M' in part), 'unknown')

        # Create descriptive run name with key hyperparameters
        run_name = (f"constrained_{model_short}_"
                   f"plr{config.training.learning_rate}_"
                   f"dlr{config.training.dual_learning_rate}_"
                   f"eps{config.training.dihedral_epsilon}-{config.training.gnn_epsilon}-{config.training.foldseek_epsilon}_"
                   f"bs{config.training.batch_size}")

        wandb.init(project=config.logging.project_name, config=OmegaConf.to_container(config), name=run_name)

    # --- Model and Modules ---
    # Convert OmegaConf objects to primitive Python types to avoid JSON serialization issues
    lora_params = {"r": config.lora.r, "lora_alpha": config.lora.alpha, "lora_dropout": config.lora.dropout, "target_modules": list(config.lora.target_modules)}
    model, _ = load_esm_with_lora(config.model.model_name, lora_params)
    model.to(device)
    esm_hidden_size = model.config.hidden_size

    dihedral_module = ConstrainedDihedralAngleConstraint(
        hidden_dim=esm_hidden_size,
        constraint_weight=config.model.constraint_weight
    ).to(device)

    # Initialize GNN module first to get its output dimension
    gnn_module = PretrainedGNNWrapper(hidden_dim=esm_hidden_size, use_simple_encoder=False).to(device).eval()

    # Create alignment module with separate dimensions for PLM and GNN (Chen et al. 2025)
    alignment_module = StructureAlignmentLoss(
        hidden_dim=esm_hidden_size,
        pgnn_hidden_dim=gnn_module.output_dim,
        num_structural_classes=21
    ).to(device)

    # --- Data ---
    use_structure_alignment = getattr(config.constraints, 'use_structure_alignment', True)
    use_precomputed = getattr(config.data, 'use_precomputed_embeddings', False)
    load_embeddings = False
    pgnn_hidden_dim = None

    if use_structure_alignment:
        if use_precomputed:
            print("Attempting to use pre-computed embeddings as per config.")
            embeddings_path = os.path.join(config.data.data_path, "embeddings")
            if not os.path.exists(embeddings_path) or not any(f.endswith('_gearnet_embeddings.pkl') for f in os.listdir(embeddings_path)):
                raise FileNotFoundError(
                    f"Configuration requires pre-computed embeddings, but none were found in {embeddings_path}. "
                    "Please generate embeddings first or set 'use_precomputed_embeddings' to false in your config."
                )
            
            print("Pre-computed embeddings found. On-the-fly generation will be disabled.")
            load_embeddings = True
            
            embedding_files = [f for f in os.listdir(embeddings_path) if f.endswith('_gearnet_embeddings.pkl')]
            sample_embedding_file = os.path.join(embeddings_path, embedding_files[0])
            with open(sample_embedding_file, 'rb') as f:
                sample_data = pickle.load(f)
                sample_embeddings = sample_data['embeddings']
                if isinstance(sample_embeddings, list):
                    sample_embeddings = np.array(sample_embeddings)
                pgnn_hidden_dim = sample_embeddings.shape[-1]
                print(f"Inferred GNN embedding dimension from sample: {pgnn_hidden_dim}")
            
            gnn_module = None # No on-the-fly generation needed
        else:
            print("Generating embeddings on-the-fly.")
            load_embeddings = False
            gnn_module = PretrainedGNNWrapper(hidden_dim=esm_hidden_size, use_simple_encoder=False).to(device).eval()
            pgnn_hidden_dim = gnn_module.output_dim

        alignment_module = StructureAlignmentLoss(
            hidden_dim=esm_hidden_size,
            pgnn_hidden_dim=pgnn_hidden_dim,
            num_structural_classes=21
        ).to(device)
    else:
        gnn_module = None
        alignment_module = None

    full_dataset = EfficientProteinDataset(config.data.data_path, max_seq_len=config.training.max_seq_len, include_structural_tokens=True, load_embeddings=load_embeddings)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.training.seed))

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    print(f"Dataset split: {train_size} training, {val_size} validation samples.")

    # --- Constrained Learning Setup ---
    lagrangian_module = MultiConstraintLagrangian(
        dataset_size=train_size,
        dihedral_epsilon=config.training.dihedral_epsilon,
        gnn_epsilon=config.training.gnn_epsilon,
        foldseek_epsilon=config.training.foldseek_epsilon,
        dual_lr=config.training.dual_learning_rate,
        device=device
    )

    # --- Optimizer, Scheduler, and Scaler ---
    # Collect all trainable parameters from model, dihedral module, and alignment module
    # Note: dihedral_module MUST be trained so prediction heads can learn to map embeddings -> dihedral angles
    trainable_params = list(model.parameters()) + list(dihedral_module.parameters()) + list(alignment_module.parameters())
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, trainable_params), lr=config.training.learning_rate)
    total_steps = (len(train_loader) // getattr(config.training, 'gradient_accumulation_steps', 1)) * config.training.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.training.warmup_steps, num_training_steps=total_steps)
    
    scaler = None
    use_mixed_precision = getattr(config.training, 'mixed_precision', False)
    if use_mixed_precision:
        if torch.cuda.is_available():
            scaler = torch.amp.GradScaler('cuda')
            print("Mixed precision training enabled.")
        else:
            print("Mixed precision requested but CUDA not available, using float32.")
    else:
        print("Mixed precision training disabled.")

    # --- Training Loop ---
    print(f"Starting constrained training for {config.training.num_epochs} epochs...")
    for epoch in range(config.training.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{config.training.num_epochs} ---")

        train_lagrangian, train_mlm, train_dihedral, train_gnn, train_foldseek = train_epoch(
            model, train_loader, optimizer, scheduler, lagrangian_module, dihedral_module, alignment_module, gnn_module, device, config, scaler
        )

        val_mlm, val_dihedral, val_gnn, val_foldseek = validate(
            model, val_loader, dihedral_module, alignment_module, gnn_module, device, config
        )

        if config.logging.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_epoch/avg_lagrangian': train_lagrangian, 'train_epoch/avg_mlm_loss': train_mlm,
                'train_epoch/avg_dihedral_loss': train_dihedral, 'train_epoch/avg_gnn_loss': train_gnn,
                'train_epoch/avg_foldseek_loss': train_foldseek,
                'val_epoch/avg_mlm_loss': val_mlm, 'val_epoch/avg_dihedral_loss': val_dihedral,
                'val_epoch/avg_gnn_loss': val_gnn, 'val_epoch/avg_foldseek_loss': val_foldseek,
            })

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train | Lagrangian: {train_lagrangian:.4f}, MLM: {train_mlm:.4f}, Dihedral: {train_dihedral:.4f}, GNN: {train_gnn:.4f}, Foldseek: {train_foldseek:.4f}")
        print(f"  Val   | MLM: {val_mlm:.4f}, Dihedral: {val_dihedral:.4f}, GNN: {val_gnn:.4f}, Foldseek: {val_foldseek:.4f}")

        # Log lambda distributions periodically
        lambda_log_frequency = getattr(config.logging, 'lambda_log_frequency', 1)  # Default: log every epoch
        if (epoch + 1) % lambda_log_frequency == 0:
            log_lambda_distributions(lagrangian_module, epoch + 1, config)
            print(f"  Lambda distributions logged for epoch {epoch+1}")

        # Save periodic checkpoints to test saving and provide backup
        checkpoint_frequency = getattr(config.training, 'checkpoint_frequency', 2)  # Default: save every 2 epochs
        if (epoch + 1) % checkpoint_frequency == 0:
            checkpoint_dir = os.path.join(config.training.output_dir, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            try:
                model.save_lora_adapters(os.path.join(checkpoint_dir, "lora_adapters"))
                print(f"  Checkpoint saved to {checkpoint_dir}")
            except Exception as e:
                print(f"  Failed to save checkpoint: {e}")

    # --- Final Actions ---
    final_dir = os.path.join(config.training.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_lora_adapters(os.path.join(final_dir, "lora_adapters"))
    print(f"Final LoRA adapters saved to {final_dir}")

    if config.logging.use_wandb:
        # Log final lambda statistics
        lambda_stats = lagrangian_module.get_lambda_stats()

        wandb.log({
            "final_lambda_stats/dihedral_mean": lambda_stats['lam_dihedral_mean'],
            "final_lambda_stats/dihedral_std": lambda_stats['lam_dihedral_std'],
            "final_lambda_stats/dihedral_min": lambda_stats['lam_dihedral_min'],
            "final_lambda_stats/dihedral_max": lambda_stats['lam_dihedral_max'],
            "final_lambda_stats/gnn_mean": lambda_stats['lam_gnn_mean'],
            "final_lambda_stats/gnn_std": lambda_stats['lam_gnn_std'],
            "final_lambda_stats/gnn_min": lambda_stats['lam_gnn_min'],
            "final_lambda_stats/gnn_max": lambda_stats['lam_gnn_max'],
            "final_lambda_stats/foldseek_mean": lambda_stats['lam_foldseek_mean'],
            "final_lambda_stats/foldseek_std": lambda_stats['lam_foldseek_std'],
            "final_lambda_stats/foldseek_min": lambda_stats['lam_foldseek_min'],
            "final_lambda_stats/foldseek_max": lambda_stats['lam_foldseek_max'],
        })
        print(f"Final lambda statistics - Dihedral: mean={lambda_stats['lam_dihedral_mean']:.4f}, GNN: mean={lambda_stats['lam_gnn_mean']:.4f}, Foldseek: mean={lambda_stats['lam_foldseek_mean']:.4f}")

    if config.logging.use_wandb: wandb.finish()
    print("\nTraining completed!")

if __name__ == "__main__":
    main()