"""
Main training script for ESM2 with geometric constraints and LoRA
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
import pickle

from models.geotune_esm_model import load_esm_with_lora
from utils.dihedral_utils import DihedralAngleConstraint
from utils.constrained_dihedral_utils import ConstrainedDihedralAngleConstraint
from utils.data_utils import ProteinStructureDataset, EfficientProteinDataset, collate_fn
from utils.structure_alignment_utils import StructureAlignmentLoss, PretrainedGNNWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Train ESM2 with geometric constraints and LoRA")
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


def train_epoch(model, dataloader, optimizer, scheduler, dihedral_constraints, device, config, structure_alignment_loss=None, frozen_gnn=None, scaler=None):
    """Train for one epoch with dihedral angle constraints and structure alignment"""
    model.train()
    total_loss = 0
    constraint_loss_total = 0
    mlm_loss_total = 0
    struct_align_total_loss = 0
    struct_align_latent_loss = 0
    struct_align_physical_loss = 0
    num_batches = 0  # Track number of batches for averaging

    # Get gradient accumulation steps
    gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(progress_bar):
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

        # Forward pass with automatic mixed precision
        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Calculate MLM loss (masked language modeling)
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

        # Calculate dihedral angle constraint loss
        with torch.amp.autocast('cuda', enabled=use_amp):
            dihedral_losses = dihedral_constraints(
                masked_outputs['sequence_output'],
                n_coords,
                ca_coords,
                c_coords,
                attention_mask
            )
            total_dihedral_loss = dihedral_losses['total_dihedral_loss']

            # Calculate structure alignment loss if structural tokens are available
            struct_align_loss = torch.tensor(0.0, device=device, requires_grad=True)
            latent_loss = torch.tensor(0.0, device=device, requires_grad=True)
            physical_loss = torch.tensor(0.0, device=device, requires_grad=True)

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

                # Debug: Check structural token shape and content
                print(f"Training debug - Structural tokens shape: {structure_tokens.shape}")
                print(f"  Attention mask shape: {attention_mask.shape}")
                print(f"  Valid tokens count: {(structure_tokens >= 0).sum().item()}")

                # Calculate structure alignment loss
                struct_align_results = structure_alignment_loss(
                    pLM_embeddings=pLM_embeddings,
                    pGNN_embeddings=pGNN_embeddings,
                    structure_tokens=structure_tokens,
                    attention_mask=attention_mask
                )
                struct_align_loss = struct_align_results['total_loss']
                latent_loss = struct_align_results.get('latent_loss', torch.tensor(0.0, device=device))
                physical_loss = struct_align_results.get('physical_loss', torch.tensor(0.0, device=device))

            # Combine all losses
            combined_loss = mlm_loss + config.model.constraint_weight * total_dihedral_loss + 0.1 * struct_align_loss

            # Scale loss for gradient accumulation
            combined_loss = combined_loss / gradient_accumulation_steps

        # Log only physical and latent losses at batch level for debugging
        if config.logging.use_wandb:
            wandb.log({
                'train_batch_foldseek_loss': physical_loss.item(),  # Physical corresponds to structural token prediction
                'train_batch_gnn_loss': latent_loss.item(),  # Latent corresponds to contrastive GNN learning
            })

        # Backward pass with gradient accumulation and mixed precision
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

            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        # Accumulate losses for epoch averages
        total_loss += combined_loss.item() * gradient_accumulation_steps
        constraint_loss_total += total_dihedral_loss.item()
        mlm_loss_total += mlm_loss.item()
        struct_align_total_loss += struct_align_loss.item()
        struct_align_latent_loss += latent_loss.item()
        struct_align_physical_loss += physical_loss.item()
        num_batches += 1  # Increment batch counter

        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{combined_loss.item() * gradient_accumulation_steps:.4f}',
            'MLM': f'{mlm_loss.item():.4f}',
            'Dihedral': f'{total_dihedral_loss.item():.4f}',
            'StructAlign': f'{struct_align_loss.item():.4f}',
        })


    # Calculate epoch averages
    avg_loss = total_loss / num_batches
    avg_constraint_loss = constraint_loss_total / num_batches
    avg_mlm_loss = mlm_loss_total / num_batches
    avg_struct_align_loss = struct_align_total_loss / num_batches
    avg_latent_loss = struct_align_latent_loss / num_batches if num_batches > 0 else 0
    avg_physical_loss = struct_align_physical_loss / num_batches if num_batches > 0 else 0

    # Log epoch averages for wandb (these are the important epoch-level metrics)
    if config.logging.use_wandb:
        wandb.log({
            'epoch': len(dataloader),  # Track as total batches in epoch
            'train_loss': avg_loss,
            'train_mlm_loss': avg_mlm_loss,
            'train_dihedral_loss': avg_constraint_loss,
            'train_struct_align_loss': avg_struct_align_loss,
            'train_foldseek_loss': avg_physical_loss,  # Physical loss corresponds to Foldseek-like task
            'train_gnn_loss': avg_latent_loss,  # Latent loss corresponds to GNN-like task
        })

    return avg_loss, avg_mlm_loss, avg_constraint_loss, avg_struct_align_loss, avg_latent_loss, avg_physical_loss


def validate(model, dataloader, dihedral_constraints, device, config, structure_alignment_loss=None, frozen_gnn=None):
    """Validate the model with dihedral angle constraints and structure alignment"""
    model.eval()
    total_loss = 0
    constraint_loss_total = 0
    mlm_loss_total = 0
    structure_alignment_loss_total = 0
    struct_align_latent_loss = 0
    struct_align_physical_loss = 0
    num_batches = 0  # Track number of batches for averaging

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
            dihedral_losses = dihedral_constraints(
                masked_outputs['sequence_output'],
                n_coords,
                ca_coords,
                c_coords,
                attention_mask
            )

            total_dihedral_loss = dihedral_losses['total_dihedral_loss']

            # Calculate structure alignment loss if structural tokens are available
            struct_align_loss = torch.tensor(0.0, device=device)
            latent_loss = torch.tensor(0.0, device=device)
            physical_loss = torch.tensor(0.0, device=device)

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
                latent_loss = struct_align_results.get('latent_loss', torch.tensor(0.0, device=device))
                physical_loss = struct_align_results.get('physical_loss', torch.tensor(0.0, device=device))

            # Calculate combined loss
            combined_loss = mlm_loss + config.model.constraint_weight * total_dihedral_loss + struct_align_loss

            total_loss += combined_loss.item()
            constraint_loss_total += total_dihedral_loss.item()
            mlm_loss_total += mlm_loss.item()
            structure_alignment_loss_total += struct_align_loss.item()
            struct_align_latent_loss += latent_loss.item()
            struct_align_physical_loss += physical_loss.item()
            num_batches += 1  # Increment batch counter

            # Log only physical and latent losses at batch level for debugging
            if config.logging.use_wandb:
                wandb.log({
                    'val_batch_foldseek_loss': physical_loss.item(),  # Physical corresponds to structural token prediction
                    'val_batch_gnn_loss': latent_loss.item(),  # Latent corresponds to contrastive GNN learning
                })

    # Calculate epoch averages
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_constraint_loss = constraint_loss_total / num_batches if num_batches > 0 else 0
    avg_mlm_loss = mlm_loss_total / num_batches if num_batches > 0 else 0
    avg_struct_align_loss = structure_alignment_loss_total / num_batches if num_batches > 0 else 0
    avg_latent_loss = struct_align_latent_loss / num_batches if num_batches > 0 else 0
    avg_physical_loss = struct_align_physical_loss / num_batches if num_batches > 0 else 0

    # Log validation epoch averages for wandb (these are the important epoch-level metrics)
    if config.logging.use_wandb:
        wandb.log({
            'val_loss': avg_loss,
            'val_mlm_loss': avg_mlm_loss,
            'val_dihedral_loss': avg_constraint_loss,
            'val_struct_align_loss': avg_struct_align_loss,
            'val_foldseek_loss': avg_physical_loss,
            'val_gnn_loss': avg_latent_loss,
        })

    return avg_loss, avg_mlm_loss, avg_constraint_loss, avg_struct_align_loss, avg_latent_loss, avg_physical_loss


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

    # Initialize dihedral angle constraints
    dihedral_constraints = DihedralAngleConstraint(
        constraint_weight=config.model.constraint_weight
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
    frozen_gnn = PretrainedGNNWrapper(hidden_dim=esm_hidden_size).to(device)
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

    # IMPORTANT: Check if pre-computed embeddings have the correct dimension
    # Pre-computed embeddings must match ESM model's hidden_dim
    # ESM2-150M uses 640, ESM2-35M uses 480, etc.
    load_embeddings = False  # Default to False
    if embeddings_exist:
        # Check if any embedding files exist and have correct dimensions
        embedding_files = [f for f in os.listdir(embeddings_path) if f.endswith('_gearnet_embeddings.pkl')]
        if embedding_files:
            # Load a sample embedding to check dimensions
            sample_embedding_file = os.path.join(embeddings_path, embedding_files[0])
            try:
                with open(sample_embedding_file, 'rb') as f:
                    sample_data = pickle.load(f)
                    sample_embeddings = sample_data['embeddings']
                    if len(sample_embeddings.shape) >= 2 and sample_embeddings.shape[-1] == esm_hidden_size:
                        load_embeddings = True
                        print(f"Pre-computed embeddings found with correct dimension {esm_hidden_size}, will load them.")
                    else:
                        print(f"Pre-computed embeddings have wrong dimension {sample_embeddings.shape[-1]}, expected {esm_hidden_size}.")
                        print(f"Embeddings will be generated on-the-fly with hidden_dim={esm_hidden_size}")
            except Exception as e:
                print(f"Error checking pre-computed embeddings: {e}")
                print(f"Embeddings will be generated on-the-fly with hidden_dim={esm_hidden_size}")
        else:
            print(f"Pre-computed embeddings directory exists but no embedding files found.")
            print(f"Embeddings will be generated on-the-fly with hidden_dim={esm_hidden_size}")
    else:
        print(f"Pre-computed embeddings not available, will generate on-the-fly with hidden_dim={esm_hidden_size}")

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

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # Only optimize trainable params
        lr=config.training.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.98)  # Standard AdamW values
    )

    # Setup scheduler
    # Adjust total steps for gradient accumulation
    gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    total_steps = (len(train_loader) // gradient_accumulation_steps) * config.training.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
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

        # Train
        train_loss, train_mlm_loss, train_constraint_loss, train_struct_align_loss, train_latent_loss, train_physical_loss = train_epoch(
            model, train_loader, optimizer, scheduler, dihedral_constraints, device, config,
            structure_alignment_loss=structure_alignment_loss, frozen_gnn=frozen_gnn, scaler=scaler
        )

        # Validate
        val_loss, val_mlm_loss, val_constraint_loss, val_struct_align_loss, val_latent_loss, val_physical_loss = validate(
            model, val_loader, dihedral_constraints, device, config,
            structure_alignment_loss=structure_alignment_loss, frozen_gnn=frozen_gnn
        )

        # Log metrics (these are the epoch-level metrics that will be shown in plots)
        if config.logging.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_mlm_loss': train_mlm_loss,
                'train_dihedral_loss': train_constraint_loss,
                'train_struct_align_loss': train_struct_align_loss,
                'train_foldseek_loss': train_physical_loss,  # Physical loss corresponds to Foldseek-like task
                'train_gnn_loss': train_latent_loss,  # Latent loss corresponds to GNN-like task
                'val_loss': val_loss,
                'val_mlm_loss': val_mlm_loss,
                'val_dihedral_loss': val_constraint_loss,
                'val_struct_align_loss': val_struct_align_loss,
                'val_foldseek_loss': val_physical_loss,
                'val_gnn_loss': val_latent_loss,
            })

        print(f"Epoch {epoch+1} completed:")
        print(f"  Train Loss: {train_loss:.4f} (MLM: {train_mlm_loss:.4f}, Dihedral: {train_constraint_loss:.4f}, StructAlign: {train_struct_align_loss:.4f})")
        print(f"  Train Foldseek: {train_physical_loss:.4f}, GNN: {train_latent_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} (MLM: {val_mlm_loss:.4f}, Dihedral: {val_constraint_loss:.4f}, StructAlign: {val_struct_align_loss:.4f})")
        print(f"  Val Foldseek: {val_physical_loss:.4f}, GNN: {val_latent_loss:.4f}")

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