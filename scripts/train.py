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
import numpy as np
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


def train_epoch(model, dataloader, optimizer, scheduler, dihedral_constraints, device, config, structure_alignment_loss=None, frozen_gnn=None, scaler=None, embedding_cache=None):
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
            # Pass geometric features if available
            geom_feats = batch.get('sasa', None)
            if geom_feats is not None:
                geom_feats = geom_feats.to(device)

            # Calculate MLM loss (masked language modeling)
            # Create random masks for the MLM task
            mask_ratio = 0.15
            mask_positions = (torch.rand(input_ids.shape, device=device) < mask_ratio) & attention_mask.bool()
            labels = input_ids.clone()
            labels[~mask_positions] = -100  # Ignore non-masked positions in loss

            # Create masked input
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = 32  # ESM2 mask token ID

            # Forward pass with masked input
            masked_outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                geometric_features=geom_feats
            )

            # Get sequence output and calculate MLM loss
            pLM_embeddings = masked_outputs['sequence_output']
            seq_logits = model.lm_head(pLM_embeddings)
            mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                seq_logits.view(-1, seq_logits.size(-1)),
                labels.view(-1)
            )

            # Calculate dihedral constraint loss
            dihedral_losses = dihedral_constraints(
                pLM_embeddings,
                n_coords,
                ca_coords,
                c_coords,
                attention_mask
            )
            total_dihedral_loss = dihedral_losses['total_dihedral_loss']

            # Calculate structure alignment loss
            struct_align_loss = torch.tensor(0.0, device=device)
            latent_loss = torch.tensor(0.0, device=device)
            physical_loss = torch.tensor(0.0, device=device)

            # Check if structure alignment loss module is available and we have the required data
            if structure_alignment_loss is not None:
                # Check if we have structural tokens
                if has_structural_tokens:
                    # Get structural tokens
                    structure_tokens = batch['structural_tokens'].to(device)

                    # Use pre-computed embeddings if available, otherwise use embedding cache
                    if has_precomputed_embeddings:
                        pGNN_embeddings = batch['precomputed_embeddings'].to(device)
                    else:
                        # Generate embeddings using cache (generates on-the-fly and saves to disk)
                        protein_ids = batch['protein_ids']
                        batch_size_gnn = n_coords.shape[0]
                        pGNN_embeddings_list = []

                        for i in range(batch_size_gnn):
                            # Get embedding from cache (generates and saves if not cached)
                            embedding = embedding_cache.get_embedding(
                                protein_id=protein_ids[i],
                                n_coords=n_coords[i],
                                ca_coords=ca_coords[i],
                                c_coords=c_coords[i]
                            )
                            pGNN_embeddings_list.append(embedding)

                        # Stack results back into batch
                        pGNN_embeddings = torch.stack(pGNN_embeddings_list, dim=0)

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

                    print(f"DEBUG: Structure alignment loss computed - Total: {struct_align_loss.item():.4f}, "
                          f"Latent: {latent_loss.item():.4f}, Physical: {physical_loss.item():.4f}")  # Debug print
                else:
                    print("DEBUG: No structural tokens in batch")  # Debug print
            else:
                print("DEBUG: Structure alignment loss module not available")  # Debug print

            # Combine all losses
            combined_loss = mlm_loss + \
                            config.model.constraint_weight * total_dihedral_loss + \
                            0.1 * struct_align_loss

            # Option 3: Check for NaN values in combined loss
            if torch.isnan(combined_loss).any():
                print("WARNING: NaN detected in combined_loss, skipping this batch")
                continue

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

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


def validate(model, dataloader, dihedral_constraints, device, config, structure_alignment_loss=None, frozen_gnn=None, embedding_cache=None):
    """Validate the model with dihedral angle constraints and structure alignment"""
    model.eval()
    total_loss = 0
    constraint_loss_total = 0
    mlm_loss_total = 0
    structure_alignment_loss_total = 0
    struct_align_latent_loss = 0
    struct_align_physical_loss = 0
    num_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating")
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            n_coords = batch['n_coords'].to(device)
            ca_coords = batch['ca_coords'].to(device)
            c_coords = batch['c_coords'].to(device)

            has_structural_tokens = 'structural_tokens' in batch
            has_precomputed_embeddings = 'precomputed_embeddings' in batch

            # === Forward pass for Geometric and Structural Losses (using unmasked inputs) ===
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pLM_embeddings = outputs['sequence_output']

            # Calculate dihedral constraint loss
            dihedral_losses = dihedral_constraints(
                pLM_embeddings,
                n_coords,
                ca_coords,
                c_coords,
                attention_mask
            )
            total_dihedral_loss = dihedral_losses['total_dihedral_loss']

            # Calculate structure alignment loss
            struct_align_loss = torch.tensor(0.0, device=device)
            latent_loss = torch.tensor(0.0, device=device)
            physical_loss = torch.tensor(0.0, device=device)

            if structure_alignment_loss is not None and has_structural_tokens:
                structure_tokens = batch['structural_tokens'].to(device)
                
                if has_precomputed_embeddings:
                    pGNN_embeddings = batch['precomputed_embeddings'].to(device)
                else:
                    protein_ids = batch['protein_ids']
                    pGNN_embeddings_list = []
                    for i in range(len(protein_ids)):
                        embedding = embedding_cache.get_embedding(
                            protein_id=protein_ids[i],
                            n_coords=n_coords[i],
                            ca_coords=ca_coords[i],
                            c_coords=c_coords[i]
                        )
                        pGNN_embeddings_list.append(embedding)
                    pGNN_embeddings = torch.stack(pGNN_embeddings_list, dim=0)

                struct_align_results = structure_alignment_loss(
                    pLM_embeddings=pLM_embeddings,
                    pGNN_embeddings=pGNN_embeddings,
                    structure_tokens=structure_tokens,
                    attention_mask=attention_mask
                )
                struct_align_loss = struct_align_results['total_loss']
                latent_loss = struct_align_results.get('latent_loss', torch.tensor(0.0, device=device))
                physical_loss = struct_align_results.get('physical_loss', torch.tensor(0.0, device=device))

            # === Forward pass for MLM Loss (using masked inputs) ===
            mask_ratio = 0.15
            mask_positions = (torch.rand(input_ids.shape, device=device) < mask_ratio) & attention_mask.bool()
            labels = input_ids.clone()
            labels[~mask_positions] = -100

            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = 32  # ESM2 mask token ID

            masked_outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask
            )
            seq_logits = model.lm_head(masked_outputs['sequence_output'])
            mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(seq_logits.view(-1, seq_logits.size(-1)), labels.view(-1))

            # Combine all losses
            combined_loss = mlm_loss + \
                            config.model.constraint_weight * total_dihedral_loss + \
                            0.1 * struct_align_loss
            
            if torch.isnan(combined_loss).any() or torch.isinf(combined_loss).any():
                print(f"WARNING: NaN or Inf in combined validation loss for batch {batch_idx}. Skipping.")
                continue

            # Accumulate losses for epoch averages
            total_loss += combined_loss.item()
            constraint_loss_total += total_dihedral_loss.item()
            mlm_loss_total += mlm_loss.item()
            structure_alignment_loss_total += struct_align_loss.item()
            struct_align_latent_loss += latent_loss.item()
            struct_align_physical_loss += physical_loss.item()
            num_batches += 1

            # Log batch-level validation losses for debugging
            if config.logging.use_wandb:
                wandb.log({
                    'val_batch_foldseek_loss': physical_loss.item(),
                    'val_batch_gnn_loss': latent_loss.item(),
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
        'target_modules': list(config.lora.target_modules)  # Convert OmegaConf ListConfig to regular list for JSON serialization
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

    # Get ESM model's hidden dimension (used by dihedral constraints, structure alignment, and GNN)
    esm_hidden_size = model.config.hidden_size

    # Initialize dihedral angle constraints
    dihedral_constraints = DihedralAngleConstraint(
        hidden_dim=esm_hidden_size,
        constraint_weight=config.model.constraint_weight
    ).to(device)

    # Initialize structure alignment loss module and GNN (if enabled)
    use_structure_alignment = getattr(config.constraints, 'use_structure_alignment', True)
    use_precomputed = getattr(config.data, 'use_precomputed_embeddings', False)

    frozen_gnn = None
    structure_alignment_loss = None
    embedding_cache = None

    if use_structure_alignment:
        print("Structure alignment loss ENABLED.")

        # Determine whether to load pre-computed embeddings or generate them
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
            
            # We still need to get the dimension from a sample embedding
            embedding_files = [f for f in os.listdir(embeddings_path) if f.endswith('_gearnet_embeddings.pkl')]
            sample_embedding_file = os.path.join(embeddings_path, embedding_files[0])
            with open(sample_embedding_file, 'rb') as f:
                sample_data = pickle.load(f)
                sample_embeddings = sample_data['embeddings']
                if isinstance(sample_embeddings, list):
                    sample_embeddings = np.array(sample_embeddings)
                pgnn_hidden_dim = sample_embeddings.shape[-1]
                print(f"Inferred GNN embedding dimension from sample: {pgnn_hidden_dim}")

        else:
            print("Generating embeddings on-the-fly.")
            load_embeddings = False
            
            # Initialize frozen pre-trained GNN for on-the-fly generation
            frozen_gnn = PretrainedGNNWrapper(
                hidden_dim=esm_hidden_size,
                use_simple_encoder=False
            ).to(device)
            frozen_gnn.eval()
            pgnn_hidden_dim = frozen_gnn.output_dim

            # Initialize embedding cache for on-the-fly generation
            from utils.embedding_cache import EmbeddingCache
            cache_dir = os.path.join(config.training.output_dir, "embedding_cache")
            embedding_cache = EmbeddingCache(
                cache_dir=cache_dir,
                gnn_model=frozen_gnn,
                device=device,
                verbose=True
            )
            print(f"Embedding cache for on-the-fly generation initialized at: {cache_dir}")

        # Create structure alignment loss module
        structure_alignment_loss = StructureAlignmentLoss(
            hidden_dim=esm_hidden_size,
            pgnn_hidden_dim=pgnn_hidden_dim,
            num_structural_classes=21,
            shared_projection_dim=512,
            latent_weight=0.5,
            physical_weight=0.5
        ).to(device)
    else:
        print("Structure alignment loss DISABLED.")

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

    # Setup optimizer with separate learning rates for primal and dual tasks
    # Primal task: ESM model (LoRA params) + LM head + dihedral constraints
    # Dual task: Structure alignment loss module (projection layers + prediction head)

    # Use single learning rate for all parameters
    print(f"Using single learning rate: {config.training.learning_rate}")
    all_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    # Add dihedral constraint parameters (prediction heads must be trained)
    all_params += list(filter(lambda p: p.requires_grad, dihedral_constraints.parameters()))

    # Add structure alignment loss parameters if enabled
    if structure_alignment_loss is not None:
        all_params += list(filter(lambda p: p.requires_grad, structure_alignment_loss.parameters()))

    optimizer = torch.optim.AdamW(
        all_params,
        lr=config.training.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.98)
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
    use_mixed_precision = getattr(config.training, 'mixed_precision', False)
    if use_mixed_precision:
        if torch.cuda.is_available():
            scaler = torch.amp.GradScaler('cuda')
            print("Mixed precision training enabled.")
        else:
            print("Mixed precision requested but CUDA not available, using float32.")
    else:
        print("Mixed precision training disabled.")

    # Training loop
    print(f"Starting training for {config.training.num_epochs} epochs...")
    print(f"Effective batch size: {config.training.batch_size * gradient_accumulation_steps}")

    best_val_loss = float('inf')

    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.training.num_epochs}")

        # Train
        train_loss, train_mlm_loss, train_constraint_loss, train_struct_align_loss, train_latent_loss, train_physical_loss = train_epoch(
            model, train_loader, optimizer, scheduler, dihedral_constraints, device, config,
            structure_alignment_loss=structure_alignment_loss, frozen_gnn=frozen_gnn, scaler=scaler,
            embedding_cache=embedding_cache
        )

        # Validate
        val_loss, val_mlm_loss, val_constraint_loss, val_struct_align_loss, val_latent_loss, val_physical_loss = validate(
            model, val_loader, dihedral_constraints, device, config,
            structure_alignment_loss=structure_alignment_loss, frozen_gnn=frozen_gnn,
            embedding_cache=embedding_cache
        )

        # Log metrics
        if config.logging.use_wandb:
            wandb.log({
                'epoch': epoch,
                'learning_rate': scheduler.get_last_lr()[0],
                'train_loss': train_loss,
                'train_mlm_loss': train_mlm_loss,
                'train_dihedral_loss': train_constraint_loss,
                'train_struct_align_loss': train_struct_align_loss,
                'train_foldseek_loss': train_physical_loss,
                'train_gnn_loss': train_latent_loss,
                'val_loss': val_loss,
                'val_mlm_loss': val_mlm_loss,
                'val_dihedral_loss': val_constraint_loss,
                'val_struct_align_loss': val_struct_align_loss,
                'val_foldseek_loss': val_physical_loss,
                'val_gnn_loss': val_latent_loss,
            })

        print(f"Epoch {epoch+1} completed:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save model checkpoint if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir = os.path.join(config.training.output_dir, "best_model")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_lora_adapters(os.path.join(checkpoint_dir, "lora_adapters"))
            print(f"New best model saved to {checkpoint_dir} with validation loss: {best_val_loss:.4f}")

    # Final save
    final_dir = os.path.join(config.training.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_lora_adapters(os.path.join(final_dir, "lora_adapters"))

    # Print embedding cache statistics if used
    if embedding_cache is not None:
        embedding_cache.print_statistics()

    print("Training completed!")
    if config.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()