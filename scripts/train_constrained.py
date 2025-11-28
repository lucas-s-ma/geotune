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
import matplotlib.pyplot as plt

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
    parser.add_argument("--dihedral_epsilon", type=float, default=0.076, help="Epsilon for dihedral constraint")
    parser.add_argument("--gnn_epsilon", type=float, default=6.38, help="Epsilon for GNN loss constraint")
    parser.add_argument("--foldseek_epsilon", type=float, default=3.00, help="Epsilon for foldseek loss constraint")

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

    gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        indices = batch['indices'].to(device)
        n_coords = batch['n_coords'].to(device)
        ca_coords = batch['ca_coords'].to(device)
        c_coords = batch['c_coords'].to(device)

        batch_size = input_ids.size(0)
        has_structural_tokens = 'structural_tokens' in batch
        has_precomputed_embeddings = 'precomputed_embeddings' in batch
        use_amp = scaler is not None

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            mask_ratio = 0.15
            mask_positions = (torch.rand(batch_size, outputs.sequence_output.size(1), device=device) < mask_ratio) & (attention_mask.bool())
            labels = input_ids.clone()
            labels[~mask_positions] = -100
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = 32

            masked_outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
            seq_logits = model.lm_head(masked_outputs['sequence_output'])
            mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(seq_logits.view(-1, seq_logits.size(-1)), labels.view(-1))

            # Correctly calculate per-sample dihedral losses with masking
            cos_true_phi, cos_true_psi = compute_dihedral_angles_from_coordinates(n_coords, ca_coords, c_coords)
            cos_pred_phi, cos_pred_psi = dihedral_constraints.predict_dihedral_angles(masked_outputs['sequence_output'])

            min_len_phi = min(cos_true_phi.shape[1], cos_pred_phi.shape[1])
            phi_mask = attention_mask[:, 1:1+min_len_phi].float()
            phi_loss_sq = (cos_true_phi[:, :min_len_phi] - cos_pred_phi[:, :min_len_phi]) ** 2
            phi_loss_per_sample = (phi_loss_sq * phi_mask).sum(dim=1) / phi_mask.sum(dim=1).clamp(min=1.0)

            min_len_psi = min(cos_true_psi.shape[1], cos_pred_psi.shape[1])
            psi_mask = attention_mask[:, :min_len_psi].float()
            psi_loss_sq = (cos_true_psi[:, :min_len_psi] - cos_pred_psi[:, :min_len_psi]) ** 2
            psi_loss_per_sample = (psi_loss_sq * psi_mask).sum(dim=1) / psi_mask.sum(dim=1).clamp(min=1.0)
            
            per_sample_dihedral_losses = phi_loss_per_sample + psi_loss_per_sample
            dihedral_loss = per_sample_dihedral_losses.mean() # For logging, get the batch average

            struct_align_loss = torch.tensor(0.0, device=device)
            gnn_loss = torch.tensor(0.0, device=device)
            foldseek_loss = torch.tensor(0.0, device=device)
            per_sample_gnn_losses = torch.zeros(batch_size, device=device)
            per_sample_foldseek_losses = torch.zeros(batch_size, device=device)

            if structure_alignment_loss is not None and has_structural_tokens:
                structure_tokens = batch['structural_tokens'].to(device)
                with torch.no_grad():
                    pGNN_embeddings = frozen_gnn(n_coords, ca_coords, c_coords)
                
                pLM_embeddings = masked_outputs['sequence_output']
                struct_align_results = structure_alignment_loss(pLM_embeddings, pGNN_embeddings, structure_tokens, attention_mask)
                struct_align_loss = struct_align_results['total_loss']
                gnn_loss = struct_align_results.get('latent_loss', torch.tensor(0.0, device=device))
                foldseek_loss = struct_align_results.get('physical_loss', torch.tensor(0.0, device=device))
                per_sample_gnn_losses = struct_align_results.get('latent_loss_per_sample', torch.zeros(batch_size, device=device))
                per_sample_foldseek_losses = struct_align_results.get('physical_loss_per_sample', torch.zeros(batch_size, device=device))

            if multi_constraint_lagrangian is not None:
                lagrangian, _ = multi_constraint_lagrangian.compute_lagrangian(
                    primary_loss=mlm_loss,
                    dihedral_losses=per_sample_dihedral_losses,
                    gnn_losses=per_sample_gnn_losses,
                    foldseek_losses=per_sample_foldseek_losses,
                    indices=indices
                )
                combined_loss = lagrangian / gradient_accumulation_steps
            else:
                combined_loss = (mlm_loss + dihedral_loss + struct_align_loss) / gradient_accumulation_steps

        if config.logging.use_wandb:
            log_data = {
                'train_batch_loss': combined_loss.item() * gradient_accumulation_steps,
                'train_batch_mlm_loss': mlm_loss.item(),
                'train_batch_dihedral_loss': dihedral_loss.item(),
                'train_batch_gnn_loss': gnn_loss.item(),
                'train_batch_foldseek_loss': foldseek_loss.item(),
                'learning_rate': scheduler.get_last_lr()[0],
            }
            if multi_constraint_lagrangian is not None:
                avg_lam_dih, avg_lam_gnn, avg_lam_fs = multi_constraint_lagrangian.get_average_lambdas()
                log_data.update({
                    'avg_lambda_dihedral': avg_lam_dih,
                    'avg_lambda_gnn': avg_lam_gnn,
                    'avg_lambda_foldseek': avg_lam_fs,
                })
            wandb.log(log_data)

        if scaler is not None:
            scaler.scale(combined_loss).backward()
        else:
            combined_loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()

            if multi_constraint_lagrangian is not None:
                multi_constraint_lagrangian.update_dual_variables(
                    per_sample_dihedral_losses,
                    per_sample_gnn_losses,
                    per_sample_foldseek_losses,
                    indices
                )

        total_loss += combined_loss.item() * gradient_accumulation_steps
        dihedral_loss_total += dihedral_loss.item()
        mlm_loss_total += mlm_loss.item()
        structure_alignment_loss_total += struct_align_loss.item()
        gnn_loss_total += gnn_loss.item()
        foldseek_loss_total += foldseek_loss.item()

        progress_bar.set_postfix({
            'Loss': f'{combined_loss.item() * gradient_accumulation_steps:.4f}',
            'MLM': f'{mlm_loss.item():.4f}',
            'Dihedral': f'{dihedral_loss.item():.4f}',
        })

    avg_loss = total_loss / len(dataloader)
    avg_dihedral_loss = dihedral_loss_total / len(dataloader)
    avg_mlm_loss = mlm_loss_total / len(dataloader)
    avg_struct_align_loss = structure_alignment_loss_total / len(dataloader)
    avg_gnn_loss = gnn_loss_total / len(dataloader)
    avg_foldseek_loss = foldseek_loss_total / len(dataloader)

    if config.logging.use_wandb:
        wandb.log({
            'epoch_avg_train_loss': avg_loss,
            'epoch_avg_train_mlm_loss': avg_mlm_loss,
            'epoch_avg_train_dihedral_loss': avg_dihedral_loss,
            'epoch_avg_train_gnn_loss': avg_gnn_loss,
            'epoch_avg_train_foldseek_loss': avg_foldseek_loss,
        })

    return avg_loss, avg_mlm_loss, avg_dihedral_loss, avg_struct_align_loss, avg_gnn_loss, avg_foldseek_loss


def validate(model, dataloader, dihedral_constraints, device, config, structure_alignment_loss=None, frozen_gnn=None, multi_constraint_lagrangian=None):
    """Validate the model by calculating individual loss components without the Lagrangian."""
    model.eval()
    total_mlm_loss = 0
    total_dihedral_loss = 0
    total_gnn_loss = 0
    total_foldseek_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            n_coords = batch['n_coords'].to(device)
            ca_coords = batch['ca_coords'].to(device)
            c_coords = batch['c_coords'].to(device)
            
            batch_size = input_ids.size(0)
            total_samples += batch_size

            # Forward pass for MLM loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            seq_output = outputs['sequence_output']
            mask_ratio = 0.15
            mask_positions = (torch.rand(batch_size, seq_output.size(1), device=device) < mask_ratio) & (attention_mask.bool())
            labels = input_ids.clone()
            labels[~mask_positions] = -100
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = 32
            
            masked_outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
            seq_logits = model.lm_head(masked_outputs['sequence_output'])
            mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(seq_logits.view(-1, seq_logits.size(-1)), labels.view(-1))
            total_mlm_loss += mlm_loss.item() * batch_size

            # Dihedral loss
            dihedral_results = dihedral_constraints(masked_outputs['sequence_output'], n_coords, ca_coords, c_coords, attention_mask)
            total_dihedral_loss += dihedral_results['dihedral_loss'].item() * batch_size

            # Structure alignment losses
            if structure_alignment_loss is not None and 'structural_tokens' in batch:
                structure_tokens = batch['structural_tokens'].to(device)
                pGNN_embeddings = frozen_gnn(n_coords, ca_coords, c_coords)
                pLM_embeddings = masked_outputs['sequence_output']
                
                struct_align_results = structure_alignment_loss(pLM_embeddings, pGNN_embeddings, structure_tokens, attention_mask)
                gnn_loss = struct_align_results.get('latent_loss', torch.tensor(0.0, device=device))
                foldseek_loss = struct_align_results.get('physical_loss', torch.tensor(0.0, device=device))
                
                total_gnn_loss += gnn_loss.item() * batch_size
                total_foldseek_loss += foldseek_loss.item() * batch_size

    avg_mlm_loss = total_mlm_loss / total_samples
    avg_dihedral_loss = total_dihedral_loss / total_samples
    avg_gnn_loss = total_gnn_loss / total_samples
    avg_foldseek_loss = total_foldseek_loss / total_samples
    
    # The total validation loss is a simple sum of the components
    avg_val_loss = avg_mlm_loss + avg_dihedral_loss + avg_gnn_loss + avg_foldseek_loss

    if config.logging.use_wandb:
        wandb.log({
            'epoch_avg_val_loss': avg_val_loss,
            'epoch_avg_val_mlm_loss': avg_mlm_loss,
            'epoch_avg_val_dihedral_loss': avg_dihedral_loss,
            'epoch_avg_val_gnn_loss': avg_gnn_loss,
            'epoch_avg_val_foldseek_loss': avg_foldseek_loss,
        })

    return avg_val_loss, avg_mlm_loss, avg_dihedral_loss, (avg_gnn_loss + avg_foldseek_loss), avg_gnn_loss, avg_foldseek_loss


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
        wandb.config.update(vars(args))


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

    # Initialize structure alignment loss module
    esm_hidden_size = model.config.hidden_size
    structure_alignment_loss = StructureAlignmentLoss(
        hidden_dim=esm_hidden_size,
        num_structural_classes=21,
        shared_projection_dim=512,
        latent_weight=0.5,
        physical_weight=0.5
    ).to(device)

    # Initialize frozen pre-trained GNN
    frozen_gnn = PretrainedGNNWrapper(hidden_dim=esm_hidden_size, use_gearnet_stub=True).to(device)
    frozen_gnn.eval()

    # Load dataset
    print("Loading dataset...")
    processed_dataset_path = os.path.join(args.data_path)
    if not os.path.exists(os.path.join(processed_dataset_path, "processed_dataset.pkl")):
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_dataset_path}. "
            f"Please run data processing script first."
        )

    struct_token_path = os.path.join(processed_dataset_path, "structural_tokens.pkl")
    include_structural_tokens = os.path.exists(struct_token_path)
    
    full_dataset = EfficientProteinDataset(
        processed_dataset_path,
        max_seq_len=config.training.max_seq_len,
        include_structural_tokens=include_structural_tokens,
        load_embeddings=False # Embeddings are generated on the fly
    )

    # Create train-validation split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.training.seed)
    )
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")

    # NOW initialize the multi-constraint lagrangian with train_size
    multi_constraint_lagrangian = MultiConstraintLagrangian(
        num_training_samples=train_size,
        dihedral_epsilon=args.dihedral_epsilon,
        gnn_epsilon=args.gnn_epsilon,
        foldseek_epsilon=args.foldseek_epsilon,
        alpha=1.0,
    ).to(device)

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

    # Setup primal optimizer
    primal_params = list(model.parameters()) + [
        multi_constraint_lagrangian.s_dihedral,
        multi_constraint_lagrangian.s_gnn,
        multi_constraint_lagrangian.s_foldseek,
    ]
    primal_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, primal_params),
        lr=config.training.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )

    # Setup scheduler
    gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    total_steps = (len(train_loader) // gradient_accumulation_steps) * config.training.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        primal_optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=total_steps
    )

    # Setup mixed precision scaler
    scaler = None
    if hasattr(config.training, 'mixed_precision') and config.training.mixed_precision:
        if torch.cuda.is_available():
            scaler = torch.amp.GradScaler('cuda')
            print("Mixed precision training enabled")
        else:
            print("Mixed precision requested but CUDA not available, using float32")

    # Training loop
    print(f"Starting training for {config.training.num_epochs} epochs...")
    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.training.num_epochs}")

        train_loss, train_mlm_loss, train_dihedral_loss, train_struct_align_loss, train_gnn_loss, train_foldseek_loss = train_epoch(
            model, train_loader, primal_optimizer, scheduler, dihedral_constraints, device, config,
            structure_alignment_loss=structure_alignment_loss, frozen_gnn=frozen_gnn,
            multi_constraint_lagrangian=multi_constraint_lagrangian, scaler=scaler
        )

        val_loss, val_mlm_loss, val_dihedral_loss, val_struct_align_loss, val_gnn_loss, val_foldseek_loss = validate(
            model, val_loader, dihedral_constraints, device, config,
            structure_alignment_loss=structure_alignment_loss, frozen_gnn=frozen_gnn
        )

        if config.logging.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_mlm_loss': train_mlm_loss,
                'train_dihedral_loss': train_dihedral_loss,
                'train_gnn_loss': train_gnn_loss,
                'train_foldseek_loss': train_foldseek_loss,
                'val_loss': val_loss,
                'val_mlm_loss': val_mlm_loss,
                'val_dihedral_loss': val_dihedral_loss,
                'val_gnn_loss': val_gnn_loss,
                'val_foldseek_loss': val_foldseek_loss,
            })

        print(f"Epoch {epoch+1} completed:")
        print(f"  Train Loss: {train_loss:.4f} (MLM: {train_mlm_loss:.4f}, Dihedral: {train_dihedral_loss:.4f}, GNN: {train_gnn_loss:.4f}, FoldSeek: {train_foldseek_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (MLM: {val_mlm_loss:.4f}, Dihedral: {val_dihedral_loss:.4f}, GNN: {val_gnn_loss:.4f}, FoldSeek: {val_foldseek_loss:.4f})")

        if (epoch + 1) % 2 == 0:
            checkpoint_dir = os.path.join(config.training.output_dir, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_lora_adapters(os.path.join(checkpoint_dir, "lora_adapters"))
            print(f"Checkpoint saved to {checkpoint_dir}")

    # Final save
    final_dir = os.path.join(config.training.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_lora_adapters(os.path.join(final_dir, "lora_adapters"))

    # Log final lambda distributions
    if config.logging.use_wandb and multi_constraint_lagrangian is not None:
        final_lam_dihedral = multi_constraint_lagrangian.lam_dihedral.detach().cpu().numpy()
        final_lam_gnn = multi_constraint_lagrangian.lam_gnn.detach().cpu().numpy()
        final_lam_foldseek = multi_constraint_lagrangian.lam_foldseek.detach().cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        axs[0].hist(final_lam_dihedral, bins=50, color='blue', alpha=0.7)
        axs[0].set_title('Final Dihedral Lambdas')
        axs[0].set_xlabel('Lambda Value')
        axs[0].set_ylabel('Frequency')

        axs[1].hist(final_lam_gnn, bins=50, color='green', alpha=0.7)
        axs[1].set_title('Final GNN Lambdas')
        axs[1].set_xlabel('Lambda Value')

        axs[2].hist(final_lam_foldseek, bins=50, color='red', alpha=0.7)
        axs[2].set_title('Final Foldseek Lambdas')
        axs[2].set_xlabel('Lambda Value')

        plt.tight_layout()
        wandb.log({"final_lambda_histograms": wandb.Image(plt)})
        plt.close(fig)

    print("Training completed!")
    if config.logging.use_wandb:
        wandb.finish()



if __name__ == "__main__":
    main()