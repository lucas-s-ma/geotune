
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
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
import argparse
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

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
    parser.add_argument("--data_path", type=str, required=True, help="Path to the processed protein data directory.")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t30_150M_UR50D", help="ESM model name or path.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for checkpoints.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the primal optimizer (model parameters).")
    parser.add_argument("--dual_learning_rate", type=float, default=1e-3, help="Learning rate for the dual optimizer (Lagrange multipliers).")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for the scheduler.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    # Epsilon values for constraints
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
            # Create masked inputs for MLM
            mask_ratio = 0.15
            mask_positions = (torch.rand(input_ids.shape, device=device) < mask_ratio) & attention_mask.bool()
            labels = input_ids.clone()
            labels[~mask_positions] = -100  # -100 is the ignore_index for CrossEntropyLoss
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = 32 # ESM2's <mask> token ID

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

            # Structure Alignment Constraints (GNN and Foldseek)
            per_sample_gnn_losses = torch.zeros(batch_size, device=device)
            per_sample_foldseek_losses = torch.zeros(batch_size, device=device)
            if alignment_module is not None and 'structural_tokens' in batch:
                structure_tokens = batch['structural_tokens'].to(device)
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
            
            # Scale loss for gradient accumulation
            scaled_lagrangian = lagrangian / gradient_accumulation_steps

        # 4. Primal Update (Model Parameters)
        scaler.scale(scaled_lagrangian).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            # 5. Dual Update (Lagrange Multipliers)
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
            avg_lam_dih, avg_lam_gnn, avg_lam_fs = lagrangian_module.get_average_lambdas()
            wandb.log({
                'train_batch/lagrangian_loss': lagrangian.item(),
                'train_batch/mlm_loss': mlm_loss.item(),
                'train_batch/dihedral_loss': per_sample_dihedral_losses.mean().item(),
                'train_batch/gnn_loss': per_sample_gnn_losses.mean().item(),
                'train_batch/foldseek_loss': per_sample_foldseek_losses.mean().item(),
                'train_avg_lambda/dihedral': avg_lam_dih,
                'train_avg_lambda/gnn': avg_lam_gnn,
                'train_avg_lambda/foldseek': avg_lam_fs,
                'learning_rate': scheduler.get_last_lr()[0]
            })

    # Return average epoch losses
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
    
    total_mlm_loss = 0
    total_dihedral_loss = 0
    total_gnn_loss = 0
    total_foldseek_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            n_coords = batch['n_coords'].to(device)
            ca_coords = batch['ca_coords'].to(device)
            c_coords = batch['c_coords'].to(device)
            batch_size = input_ids.size(0)

            # MLM Loss (using the same masking strategy as training for consistency)
            mask_ratio = 0.15
            mask_positions = (torch.rand(input_ids.shape, device=device) < mask_ratio) & attention_mask.bool()
            labels = input_ids.clone()
            labels[~mask_positions] = -100
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = 32

            masked_outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
            pLM_embeddings = masked_outputs['sequence_output']
            seq_logits = model.lm_head(pLM_embeddings)
            mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(seq_logits.view(-1, seq_logits.size(-1)), labels.view(-1))
            total_mlm_loss += mlm_loss.item()

            # Dihedral Loss
            dihedral_results = dihedral_module(pLM_embeddings, n_coords, ca_coords, c_coords, attention_mask)
            total_dihedral_loss += dihedral_results['raw_constraint_loss'].item()

            # Structure Alignment Losses
            if alignment_module is not None and 'structural_tokens' in batch:
                structure_tokens = batch['structural_tokens'].to(device)
                pGNN_embeddings = gnn_module(n_coords, ca_coords, c_coords)
                struct_align_results = alignment_module(pLM_embeddings, pGNN_embeddings, structure_tokens, attention_mask)
                total_gnn_loss += struct_align_results['latent_loss'].item()
                total_foldseek_loss += struct_align_results['physical_loss'].item()
            
            num_batches += 1

    # Calculate and return averages
    avg_mlm = total_mlm_loss / num_batches
    avg_dihedral = total_dihedral_loss / num_batches
    avg_gnn = total_gnn_loss / num_batches
    avg_foldseek = total_foldseek_loss / num_batches
    return avg_mlm, avg_dihedral, avg_gnn, avg_foldseek

def main():
    """Main function to set up and run the constrained training process."""
    args = parse_args()
    config = OmegaConf.load(args.config)

    # Setup
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if config.logging.use_wandb:
        wandb.init(project=config.logging.project_name, config={**OmegaConf.to_container(config), **vars(args)})

    # Model and Modules
    lora_params = {
        "r": config.lora.r,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "target_modules": config.lora.target_modules
    }
    model, _ = load_esm_with_lora(args.model_name, lora_params)
    model.to(device)
    esm_hidden_size = model.config.hidden_size

    dihedral_module = ConstrainedDihedralAngleConstraint(hidden_dim=esm_hidden_size).to(device)
    alignment_module = StructureAlignmentLoss(hidden_dim=esm_hidden_size, num_structural_classes=21).to(device)
    gnn_module = PretrainedGNNWrapper(hidden_dim=esm_hidden_size, use_gearnet_stub=True).to(device).eval()

    # Data
    full_dataset = EfficientProteinDataset(args.data_path, max_seq_len=args.max_seq_len, include_structural_tokens=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    print(f"Dataset split: {train_size} training, {val_size} validation samples.")

    # Constrained Learning Setup
    lagrangian_module = MultiConstraintLagrangian(
        num_training_samples=train_size,
        dihedral_epsilon=args.dihedral_epsilon,
        gnn_epsilon=args.gnn_epsilon,
        foldseek_epsilon=args.foldseek_epsilon,
        dual_lr=args.dual_learning_rate
    ).to(device)

    # Optimizer, Scheduler, and Scaler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    total_steps = (len(train_loader) // getattr(config.training, 'gradient_accumulation_steps', 1)) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Training Loop
    print(f"Starting constrained training for {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
        
        train_lagrangian, train_mlm, train_dihedral, train_gnn, train_foldseek = train_epoch(
            model, train_loader, optimizer, scheduler, lagrangian_module, dihedral_module, alignment_module, gnn_module, device, config, scaler
        )
        
        val_mlm, val_dihedral, val_gnn, val_foldseek = validate(
            model, val_loader, dihedral_module, alignment_module, gnn_module, device, config
        )

        # Log epoch averages
        if config.logging.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_epoch/avg_lagrangian': train_lagrangian,
                'train_epoch/avg_mlm_loss': train_mlm,
                'train_epoch/avg_dihedral_loss': train_dihedral,
                'train_epoch/avg_gnn_loss': train_gnn,
                'train_epoch/avg_foldseek_loss': train_foldseek,
                'val_epoch/avg_mlm_loss': val_mlm,
                'val_epoch/avg_dihedral_loss': val_dihedral,
                'val_epoch/avg_gnn_loss': val_gnn,
                'val_epoch/avg_foldseek_loss': val_foldseek,
            })
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train | Lagrangian: {train_lagrangian:.4f}, MLM: {train_mlm:.4f}, Dihedral: {train_dihedral:.4f}, GNN: {train_gnn:.4f}, Foldseek: {train_foldseek:.4f}")
        print(f"  Val   | MLM: {val_mlm:.4f}, Dihedral: {val_dihedral:.4f}, GNN: {val_gnn:.4f}, Foldseek: {val_foldseek:.4f}")

    # Final Actions
    # Save model
    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_lora_adapters(os.path.join(final_dir, "lora_adapters"))
    print(f"Final LoRA adapters saved to {final_dir}")

    # Log final lambda histograms
    if config.logging.use_wandb:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)
        axs[0].hist(lagrangian_module.lam_dihedral.cpu().numpy(), bins=50, color='blue', alpha=0.7)
        axs[0].set_title('Final Dihedral Lambdas')
        axs[1].hist(lagrangian_module.lam_gnn.cpu().numpy(), bins=50, color='green', alpha=0.7)
        axs[1].set_title('Final GNN Lambdas')
        axs[2].hist(lagrangian_module.lam_foldseek.cpu().numpy(), bins=50, color='red', alpha=0.7)
        axs[2].set_title('Final Foldseek Lambdas')
        for ax in axs:
            ax.set_xlabel('Lambda Value')
            ax.set_ylabel('Frequency')
        wandb.log({"final_lambda_histograms": wandb.Image(plt)})
        plt.close(fig)
        print("Logged final lambda histograms to wandb.")

    if config.logging.use_wandb:
        wandb.finish()

    print("\nTraining completed!")

if __name__ == "__main__":
    main()
