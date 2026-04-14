"""
Main training script for ESM2 with geometric constraints and LoRA
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import transformers
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, EsmForMaskedLM, EsmTokenizer
from tqdm import tqdm
import wandb
import argparse
from omegaconf import OmegaConf
import pickle
import logging
import glob
import re
import pandas as pd
from scipy.stats import spearmanr
import openpyxl

from models.geotune_esm_model import load_esm_with_lora
from utils.dihedral_utils import DihedralAngleConstraint
from utils.data_utils import EfficientProteinDataset, collate_fn
from utils.structure_alignment_utils import StructureAlignmentLoss, PretrainedGNNWrapper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DMS Benchmark Evaluation (Matches ProteinGym exactly)
# ---------------------------------------------------------------------------
MUTATION_RE = re.compile(r"([A-Za-z*])(\d+)([A-Za-z*])")
ESM2_MAX_LEN = 1024


def get_optimal_window(mutation_position_relative, seq_len_wo_special, model_window):
    half_model_window = model_window // 2
    if seq_len_wo_special <= model_window:
        return [0, seq_len_wo_special]
    elif mutation_position_relative < half_model_window:
        return [0, model_window]
    elif mutation_position_relative >= seq_len_wo_special - half_model_window:
        return [seq_len_wo_special - model_window, seq_len_wo_special]
    else:
        return [max(0, mutation_position_relative - half_model_window),
                min(seq_len_wo_special, mutation_position_relative + half_model_window)]


def compute_masked_marginals(model, tokenizer, wt_seq, device):
    seq_len = len(wt_seq)
    full_tokens = tokenizer(wt_seq, return_tensors="pt", add_special_tokens=True)
    full_input_ids = full_tokens["input_ids"].to(device)
    log_probs_map = {}

    for i in range(seq_len):
        absolute_token_pos = i + 1
        seq_len_wo_special = seq_len + 2
        window_start, window_end = get_optimal_window(
            mutation_position_relative=absolute_token_pos,
            seq_len_wo_special=seq_len_wo_special,
            model_window=ESM2_MAX_LEN
        )
        window_input_ids = full_input_ids[:, window_start:window_end].clone()
        pos_in_window = absolute_token_pos - window_start
        window_input_ids[0, pos_in_window] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(input_ids=window_input_ids)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)

        log_probs_map[i] = log_probs[0, pos_in_window].cpu().numpy()

    return log_probs_map


def compute_effect_from_mutant_str(log_probs_map, mutant_str, tokenizer, wt_seq, offset_idx=1):
    effect = 0.0
    for mut in re.split(r"[:,]", mutant_str):
        m = MUTATION_RE.fullmatch(mut.strip())
        if not m:
            return None
        wt_aa, pos_str, mut_aa = m.groups()
        wt_aa, mut_aa = wt_aa.upper(), mut_aa.upper()
        idx = int(pos_str) - offset_idx
        if idx < 0 or idx >= len(wt_seq):
            return None
        if wt_seq[idx].upper() != wt_aa:
            return None
        if wt_aa == mut_aa:
            continue
        if idx not in log_probs_map:
            return None
        wt_id = tokenizer.convert_tokens_to_ids(wt_aa)
        mut_id = tokenizer.convert_tokens_to_ids(mut_aa)
        effect += log_probs_map[idx][mut_id] - log_probs_map[idx][wt_id]
    return effect


def evaluate_dms_file(model, tokenizer, device, dms_file, meta_row, valid_aas):
    wt_seq_full = meta_row["target_seq"]
    try:
        df = pd.read_csv(dms_file)
    except Exception:
        return None, {}, "CSV read error"
    if len(df) == 0:
        return None, {}, "Empty CSV"

    msa_start = int(meta_row["MSA_start"]) if pd.notna(meta_row.get("MSA_start")) else 1
    msa_end = int(meta_row["MSA_end"]) if pd.notna(meta_row.get("MSA_end")) else len(wt_seq_full)
    wt_seq_cropped = wt_seq_full[msa_start - 1:msa_end]
    offset_idx = msa_start

    log_probs_map = compute_masked_marginals(model, tokenizer, wt_seq_cropped, device)

    predicted, experimental = [], []
    for _, row in df.iterrows():
        score = row.get("DMS_score")
        if pd.isna(score):
            continue
        mutant_val = row.get("mutant", "")
        if isinstance(mutant_val, str) and mutant_val.upper() == "WT":
            continue
        effect = compute_effect_from_mutant_str(log_probs_map, mutant_val, tokenizer, wt_seq_cropped, offset_idx)
        if effect is None or np.isnan(effect):
            continue
        predicted.append(effect)
        experimental.append(score)

    if len(predicted) < 2:
        return None, {}, f"Not enough valid predictions ({len(predicted)} < 2)"

    corr, _ = spearmanr(predicted, experimental)
    if np.isnan(corr):
        return None, {}, "Correlation is NaN"

    return corr, {row.get("mutant", ""): compute_effect_from_mutant_str(
        log_probs_map, row.get("mutant", ""), tokenizer, wt_seq_cropped, offset_idx
    ) for _, row in df.iterrows() if pd.notna(row.get("DMS_score"))}, "Success"


def load_dms_metadata(metadata_path):
    df = pd.read_csv(metadata_path)
    meta_dict = {}
    uid_col = next((c for c in df.columns if c.lower() == 'uniprot_id'), None)
    sel_col = next((c for c in df.columns if 'coarse_selection_type' in c.lower()), None)
    for _, row in df.iterrows():
        uid = row[uid_col] if uid_col else "Unknown"
        if pd.isna(uid) or str(uid).strip() == "":
            uid = "Unknown"
        sel_type = row[sel_col] if sel_col else "Unknown"
        if pd.isna(sel_type) or str(sel_type).strip() == "":
            sel_type = "Unknown"
        meta_dict[row["DMS_id"]] = {
            "target_seq": row["target_seq"],
            "MSA_start": row.get("MSA_start", np.nan),
            "MSA_end": row.get("MSA_end", np.nan),
            "uniprot_id": str(uid).strip(),
            "coarse_selection_type": str(sel_type).strip()
        }
    return meta_dict


def evaluate_dms_benchmark(esm_model, tokenizer, device, dms_dir, metadata_path, output_excel_path=None):
    """
    Evaluate the model on all DMS assays and return aggregated metrics.
    
    Returns:
        dict with keys:
            - simple_mean: raw mean across all assays
            - uniprot_mean: mean of per-UniProt averages
            - leaderboard_mean: 3-level hierarchical aggregation (matches benchmark)
            - per_file: dict of {dms_id: spearman_corr}
            - by_category: dict of {selection_type: mean_corr}
    """
    meta_dict = load_dms_metadata(metadata_path)
    vocab = tokenizer.get_vocab()
    valid_aas = {k for k in vocab if len(k) == 1 and k.isalpha()}

    esm_model.eval()
    per_file_results = {}
    csv_records = []

    files = [f for f in os.listdir(dms_dir) if f.endswith(".csv")]

    for f in sorted(files):
        dms_id = f.replace(".csv", "")
        if dms_id not in meta_dict:
            continue

        path = os.path.join(dms_dir, f)
        corr, _, status = evaluate_dms_file(
            esm_model, tokenizer, device, path, meta_dict[dms_id], valid_aas
        )

        uid = meta_dict[dms_id]["uniprot_id"]
        sel_type = meta_dict[dms_id]["coarse_selection_type"]

        if corr is not None:
            per_file_results[dms_id] = corr
            csv_records.append({
                "DMS_id": dms_id,
                "UniProt_ID": uid,
                "coarse_selection_type": sel_type,
                "Spearman": corr,
                "Status": "Success"
            })
        else:
            csv_records.append({
                "DMS_id": dms_id,
                "UniProt_ID": uid,
                "coarse_selection_type": sel_type,
                "Spearman": np.nan,
                "Status": status
            })

    # Build performance DataFrame
    perf_df = pd.DataFrame(csv_records)
    successful = perf_df[perf_df["Status"] == "Success"]

    if len(successful) == 0:
        return None

    # Level 1: Mean within each UniProt_ID
    uniprot_means = successful.groupby("UniProt_ID")["Spearman"].mean()
    uniprot_mean_overall = uniprot_means.mean()

    # Level 2: Mean within each UniProt_ID + coarse_selection_type
    uniprot_function_means = successful.groupby(["UniProt_ID", "coarse_selection_type"])["Spearman"].mean()

    # Level 3: Average within each coarse_selection_type, then average across types
    function_level = uniprot_function_means.groupby("coarse_selection_type").mean()
    leaderboard_mean = function_level.mean()

    # Simple mean
    simple_mean = successful["Spearman"].mean()

    # By category
    by_category = function_level.to_dict()

    # Save to Excel if path provided
    if output_excel_path is not None:
        out_df = pd.DataFrame(csv_records)
        os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
        out_df.to_csv(output_excel_path, index=False)

    return {
        "simple_mean": simple_mean,
        "uniprot_mean": uniprot_mean_overall,
        "leaderboard_mean": leaderboard_mean,
        "by_category": by_category,
        "per_file": per_file_results,
        "num_success": len(successful),
        "num_total": len(csv_records)
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train ESM2 with geometric constraints and LoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_path", type=str, default=None, help="Path to protein data directory (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint directory to resume training from")
    # Debug options (not in config)
    parser.add_argument("--debug_subset", action="store_true",
                        help="Use 1%% of dataset for quick debugging")
    # DMS benchmark evaluation
    parser.add_argument("--dms_dir", type=str, default=None,
                        help="Directory containing DMS CSV files for periodic evaluation")
    parser.add_argument("--dms_metadata", type=str, default=None,
                        help="Metadata CSV file with WT sequences and MSA bounds")
    parser.add_argument("--dms_eval_every", type=int, default=1,
                        help="Run DMS evaluation every N epochs (1 = every epoch)")

    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, scheduler, dihedral_constraints, device, config, epoch, 
                structure_alignment_loss=None, frozen_gnn=None, scaler=None, embedding_cache=None):
    """Train for one epoch with dihedral angle constraints and structure alignment"""
    model.train()
    total_loss = 0
    constraint_loss_total = 0
    mlm_loss_total = 0
    struct_align_total_loss = 0
    struct_align_latent_loss = 0
    struct_align_physical_loss = 0
    num_batches = 0

    gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    mask_ratio = getattr(config.training, 'mask_ratio', 0.15)
    struct_align_weight = getattr(config.training, 'struct_align_weight', 0.1)

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        if 'n_coords' not in batch or 'ca_coords' not in batch or 'c_coords' not in batch:
            raise KeyError("Dataset missing coordinate data. Please re-process with proper format.")

        has_structural_tokens = 'structural_tokens' in batch
        has_precomputed_embeddings = 'precomputed_embeddings' in batch

        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            geom_feats = batch.get('sasa', None)
            if geom_feats is not None:
                geom_feats = geom_feats.to(device)

            # MLM loss
            mask_positions = (torch.rand(input_ids.shape, device=device) < mask_ratio) & attention_mask.bool()
            labels = input_ids.clone()
            labels[~mask_positions] = -100
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = 32

            masked_outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                geometric_features=geom_feats
            )

            pLM_embeddings = masked_outputs['sequence_output']
            seq_logits = model.lm_head(pLM_embeddings)
            mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                seq_logits.view(-1, seq_logits.size(-1)),
                labels.view(-1)
            )

            # Dihedral constraint loss
            dihedral_losses = dihedral_constraints(
                pLM_embeddings,
                n_coords=batch['n_coords'].to(device),
                ca_coords=batch['ca_coords'].to(device),
                c_coords=batch['c_coords'].to(device),
                attention_mask=attention_mask
            )
            per_sample_dihedral_losses = dihedral_losses['per_sample_dihedral_losses']
            total_dihedral_loss = per_sample_dihedral_losses.mean() * dihedral_constraints.constraint_weight

            # Structure alignment loss
            struct_align_loss = torch.tensor(0.0, device=device)
            latent_loss = torch.tensor(0.0, device=device)
            physical_loss = torch.tensor(0.0, device=device)

            if structure_alignment_loss is not None and has_structural_tokens:
                structure_tokens = batch['structural_tokens'].to(device)

                if has_precomputed_embeddings:
                    pGNN_embeddings = batch['precomputed_embeddings'].to(device)
                else:
                    protein_ids = batch['protein_ids']
                    batch_size_gnn = batch['n_coords'].shape[0]
                    pGNN_embeddings_list = []
                    for i in range(batch_size_gnn):
                        embedding = embedding_cache.get_embedding(
                            protein_id=protein_ids[i],
                            n_coords=batch['n_coords'][i],
                            ca_coords=batch['ca_coords'][i],
                            c_coords=batch['c_coords'][i]
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

            # Combine losses
            combined_loss = mlm_loss + \
                            config.model.constraint_weight * total_dihedral_loss + \
                            struct_align_weight * struct_align_loss

            if torch.isnan(combined_loss).any() or torch.isinf(combined_loss).any():
                logger.warning("NaN/Inf detected in combined_loss, skipping batch")
                continue

            combined_loss = combined_loss / gradient_accumulation_steps

        # Log batch metrics
        if batch_idx % 10 == 0 and config.logging.use_wandb:
            wandb.log({
                'train_batch_mlm_loss': mlm_loss.item(),
                'train_batch_dihedral_loss': total_dihedral_loss.item(),
                'train_batch_gnn_loss': latent_loss.item(),
                'train_batch_foldseek_loss': physical_loss.item(),
                'learning_rate': scheduler.get_last_lr()[0]
            })

        # Backward pass
        if scaler is not None:
            scaler.scale(combined_loss).backward()
        else:
            combined_loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        # Accumulate losses
        total_loss += combined_loss.item() * gradient_accumulation_steps
        constraint_loss_total += total_dihedral_loss.item()
        mlm_loss_total += mlm_loss.item()
        struct_align_total_loss += struct_align_loss.item()
        struct_align_latent_loss += latent_loss.item()
        struct_align_physical_loss += physical_loss.item()
        num_batches += 1

        progress_bar.set_postfix({
            'Loss': f'{combined_loss.item() * gradient_accumulation_steps:.4f}',
            'MLM': f'{mlm_loss.item():.4f}',
            'Dihedral': f'{total_dihedral_loss.item():.4f}',
            'StructAlign': f'{struct_align_loss.item():.4f}',
        })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_constraint_loss = constraint_loss_total / num_batches if num_batches > 0 else 0
    avg_mlm_loss = mlm_loss_total / num_batches if num_batches > 0 else 0
    avg_struct_align_loss = struct_align_total_loss / num_batches if num_batches > 0 else 0
    avg_latent_loss = struct_align_latent_loss / num_batches if num_batches > 0 else 0
    avg_physical_loss = struct_align_physical_loss / num_batches if num_batches > 0 else 0

    if config.logging.use_wandb:
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_mlm_loss': avg_mlm_loss,
            'train_dihedral_loss': avg_constraint_loss,
            'train_struct_align_loss': avg_struct_align_loss,
            'train_foldseek_loss': avg_physical_loss,
            'train_gnn_loss': avg_latent_loss,
            'learning_rate': scheduler.get_last_lr()[0]
        })

    return avg_loss, avg_mlm_loss, avg_constraint_loss, avg_struct_align_loss, avg_latent_loss, avg_physical_loss


def validate(model, dataloader, dihedral_constraints, device, config, 
             structure_alignment_loss=None, frozen_gnn=None, embedding_cache=None):
    """Validate the model"""
    import time

    model.eval()
    if frozen_gnn is not None:
        frozen_gnn.eval()
    if embedding_cache is not None and hasattr(embedding_cache, 'gnn_model'):
        embedding_cache.gnn_model.eval()
    dihedral_constraints.eval()
    if structure_alignment_loss is not None:
        structure_alignment_loss.eval()

    total_loss = 0
    constraint_loss_total = 0
    mlm_loss_total = 0
    structure_alignment_loss_total = 0
    struct_align_latent_loss = 0
    struct_align_physical_loss = 0
    num_batches = 0

    precomputed_count = 0
    onthefly_count = 0
    val_start_time = time.time()
    forward_time = 0
    loss_time = 0
    dataloader_time = 0

    mask_ratio = getattr(config.training, 'mask_ratio', 0.15)
    struct_align_weight = getattr(config.training, 'struct_align_weight', 0.1)

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

            has_structural_tokens = 'structural_tokens' in batch
            has_precomputed_embeddings = 'precomputed_embeddings' in batch

            if has_precomputed_embeddings:
                precomputed_count += 1
            else:
                onthefly_count += 1

            mask_positions = (torch.rand(input_ids.shape, device=device) < mask_ratio) & attention_mask.bool()
            labels = input_ids.clone()
            labels[~mask_positions] = -100
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = 32

            forward_start = time.time()
            outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
            pLM_embeddings = outputs['sequence_output']
            forward_time += time.time() - forward_start

            loss_start = time.time()
            seq_logits = model.lm_head(pLM_embeddings)
            mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                seq_logits.view(-1, seq_logits.size(-1)), 
                labels.view(-1)
            )

            dihedral_losses = dihedral_constraints(
                pLM_embeddings,
                n_coords=n_coords,
                ca_coords=ca_coords,
                c_coords=c_coords,
                attention_mask=attention_mask
            )
            per_sample_dihedral_losses = dihedral_losses['per_sample_dihedral_losses']
            total_dihedral_loss = per_sample_dihedral_losses.mean() * dihedral_constraints.constraint_weight

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
            loss_time += time.time() - loss_start

            combined_loss = mlm_loss + \
                            config.model.constraint_weight * total_dihedral_loss + \
                            struct_align_weight * struct_align_loss

            if torch.isnan(combined_loss).any() or torch.isinf(combined_loss).any():
                logger.warning(f"NaN/Inf in validation loss for batch {batch_idx}, skipping")
                continue

            total_loss += combined_loss.item()
            constraint_loss_total += total_dihedral_loss.item()
            mlm_loss_total += mlm_loss.item()
            structure_alignment_loss_total += struct_align_loss.item()
            struct_align_latent_loss += latent_loss.item()
            struct_align_physical_loss += physical_loss.item()
            num_batches += 1

            iter_per_sec = (batch_idx + 1) / (time.time() - val_start_time)
            progress_bar.set_postfix({
                'Loss': f'{combined_loss.item():.4f}',
                'iter/s': f'{iter_per_sec:.1f}',
            })

            batch_start = time.time()

    total_time = time.time() - val_start_time

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_constraint_loss = constraint_loss_total / num_batches if num_batches > 0 else 0
    avg_mlm_loss = mlm_loss_total / num_batches if num_batches > 0 else 0
    avg_struct_align_loss = structure_alignment_loss_total / num_batches if num_batches > 0 else 0
    avg_latent_loss = struct_align_latent_loss / num_batches if num_batches > 0 else 0
    avg_physical_loss = struct_align_physical_loss / num_batches if num_batches > 0 else 0

    logger.info(f"Validation completed in {total_time:.2f}s ({num_batches/total_time:.1f} iter/s)")

    return avg_loss, avg_mlm_loss, avg_constraint_loss, avg_struct_align_loss, avg_latent_loss, avg_physical_loss


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
    
    return checkpoint['epoch'], checkpoint['val_loss']


def main():
    args = parse_args()

    # Load config - all training arguments must come from config file
    config = OmegaConf.load(args.config)

    # Override config with command-line args (only for paths, not training params)
    if args.data_path:
        config.data.data_path = args.data_path
    if args.output_dir:
        config.training.output_dir = args.output_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine output directory
    if args.resume_from:
        # Resume from existing checkpoint - use same output directory
        config.training.output_dir = os.path.dirname(args.resume_from)
        logger.info(f"Resuming training in {config.training.output_dir}")
    else:
        config.training.output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
        os.makedirs(config.training.output_dir, exist_ok=True)
        logger.info(f"Output directory: {config.training.output_dir}")

    # Set random seed
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize wandb
    if config.logging.use_wandb:
        model_name_parts = config.model.model_name.split('/')[-1].split('_')
        model_short = next((part for part in model_name_parts if 'M' in part), 'unknown')
        run_name = f"unconstrained_{model_short}_lr{config.training.learning_rate}_bs{config.training.batch_size}_{timestamp}"
        wandb.init(project=config.logging.project_name, config=OmegaConf.to_container(config), name=run_name)

    # Load model
    logger.info("Loading model...")
    lora_params = {
        'r': config.lora.r,
        'lora_alpha': config.lora.alpha,
        'lora_dropout': config.lora.dropout,
        'target_modules': list(config.lora.target_modules)
    }
    model, tokenizer = load_esm_with_lora(config.model.model_name, lora_params)
    model.to(device)

    # Enable gradient checkpointing
    if getattr(config.training, 'use_gradient_checkpointing', False):
        if hasattr(model.model, 'gradient_checkpointing_enable'):
            model.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

    # Print parameter counts
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    if config.logging.use_wandb:
        wandb.config.update({
            'trainable_params': trainable_params,
            'total_params': total_params
        })

    esm_hidden_size = model.config.hidden_size

    # Initialize modules
    dihedral_constraints = DihedralAngleConstraint(
        hidden_dim=esm_hidden_size,
        constraint_weight=config.model.constraint_weight
    ).to(device)

    use_structure_alignment = getattr(config.constraints, 'use_structure_alignment', True)
    structure_alignment_loss = None
    frozen_gnn = None
    embedding_cache = None
    load_embeddings = False

    if use_structure_alignment:
        logger.info("Structure alignment loss ENABLED")
        
        embeddings_path = os.path.join(config.data.data_path, "embeddings")
        has_precomputed = os.path.exists(embeddings_path) and \
                         any(f.endswith('_gearnet_embeddings.pkl') for f in os.listdir(embeddings_path))
        use_precomputed_config = getattr(config.data, 'use_precomputed_embeddings', True)
        use_precomputed = has_precomputed and use_precomputed_config

        if use_precomputed:
            logger.info(f"Using pre-computed embeddings from {embeddings_path}")
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
            frozen_gnn = PretrainedGNNWrapper(
                hidden_dim=esm_hidden_size,
                use_simple_encoder=False
            ).to(device)
            frozen_gnn.eval()
            pgnn_hidden_dim = frozen_gnn.output_dim
            
            from utils.embedding_cache import EmbeddingCache
            cache_dir = os.path.join(config.training.output_dir, "embedding_cache")
            embedding_cache = EmbeddingCache(
                cache_dir=cache_dir,
                gnn_model=frozen_gnn,
                device=device
            )

        structure_alignment_loss = StructureAlignmentLoss(
            hidden_dim=esm_hidden_size,
            pgnn_hidden_dim=pgnn_hidden_dim,
            num_structural_classes=21,
            shared_projection_dim=512,
            latent_weight=0.5,
            physical_weight=0.5
        ).to(device)

    # Load dataset
    logger.info("Loading dataset...")
    processed_dataset_path = os.path.join(config.data.data_path, "processed_dataset.pkl")
    if not os.path.exists(processed_dataset_path):
        raise FileNotFoundError(f"Processed dataset not found at {processed_dataset_path}")

    struct_token_path = os.path.join(config.data.data_path, "structural_tokens.pkl")
    include_structural_tokens = os.path.exists(struct_token_path)

    # Use config value for subset fraction, override only for debug mode
    subset_fraction = getattr(config.data, 'subset_fraction', 1.0)
    if args.debug_subset:
        subset_fraction = 0.01

    full_dataset = EfficientProteinDataset(
        config.data.data_path,
        max_seq_len=config.training.max_seq_len,
        include_structural_tokens=include_structural_tokens,
        load_embeddings=load_embeddings,
        subset_fraction=subset_fraction
    )

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.training.seed)
    )

    if subset_fraction < 1.0:
        logger.info(f"Using {subset_fraction*100:.1f}% of dataset: {len(train_dataset)} train, {len(val_dataset)} val")

    num_workers = min(8, getattr(config.data, 'num_workers', 4))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )

    # Setup optimizer - use config.training.learning_rate (which may be primal_lr)
    lr = getattr(config.training, 'primal_lr', config.training.learning_rate)
    logger.info(f"Using learning rate: {lr}")
    
    all_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    all_params += list(filter(lambda p: p.requires_grad, dihedral_constraints.parameters()))
    if structure_alignment_loss is not None:
        all_params += list(filter(lambda p: p.requires_grad, structure_alignment_loss.parameters()))

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01, betas=(0.9, 0.98))

    gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    total_steps = (len(train_loader) // gradient_accumulation_steps) * config.training.num_epochs
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
        start_epoch, best_val_loss = load_checkpoint(
            args.resume_from, model, optimizer, scheduler, 
            lagrangian_module=None, device=device
        )
        start_epoch += 1  # Start from next epoch

    # Training loop
    logger.info(f"Starting training for {config.training.num_epochs} epochs from epoch {start_epoch}")
    logger.info(f"Effective batch size: {config.training.batch_size * gradient_accumulation_steps}")

    for epoch in range(start_epoch, config.training.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.training.num_epochs}")

        train_loss, train_mlm_loss, train_constraint_loss, train_struct_align_loss, train_latent_loss, train_physical_loss = train_epoch(
            model, train_loader, optimizer, scheduler, dihedral_constraints, device, config, epoch,
            structure_alignment_loss=structure_alignment_loss, frozen_gnn=frozen_gnn, scaler=scaler,
            embedding_cache=embedding_cache
        )

        val_loss, val_mlm_loss, val_constraint_loss, val_struct_align_loss, val_latent_loss, val_physical_loss = validate(
            model, val_loader, dihedral_constraints, device, config,
            structure_alignment_loss=structure_alignment_loss, frozen_gnn=frozen_gnn,
            embedding_cache=embedding_cache
        )

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

        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        checkpoint_dir = os.path.join(config.training.output_dir, f"checkpoint_epoch_{epoch+1}")
        save_checkpoint(
            model, lagrangian_module=None, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, val_loss=val_loss, config=config,
            checkpoint_dir=checkpoint_dir, is_best=is_best
        )

        if is_best:
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")

        # DMS Benchmark Evaluation
        if args.dms_dir and args.dms_metadata and (epoch + 1) % args.dms_eval_every == 0:
            logger.info(f"\nRunning DMS benchmark evaluation (epoch {epoch+1})...")

            # Determine the ESM model to use for DMS evaluation
            # The model may be a wrapper; access the underlying ESM model
            esm_model_for_dms = model
            if hasattr(model, 'model'):
                esm_model_for_dms = model.model
            # If it's a PeftModel, we need the base model
            if hasattr(esm_model_for_dms, 'base_model'):
                esm_model_for_dms = esm_model_for_dms.base_model

            dms_excel_path = os.path.join(
                config.training.output_dir,
                f"dms_results_epoch_{epoch+1}.csv"
            )

            dms_results = evaluate_dms_benchmark(
                esm_model_for_dms, tokenizer, device,
                args.dms_dir, args.dms_metadata, dms_excel_path
            )

            if dms_results is not None:
                logger.info(f"DMS Benchmark Results (epoch {epoch+1}):")
                logger.info(f"  Simple Average:    {dms_results['simple_mean']:.4f}")
                logger.info(f"  UniProt Macro-Avg: {dms_results['uniprot_mean']:.4f}")
                logger.info(f"  Leaderboard (3-level): {dms_results['leaderboard_mean']:.4f}")
                logger.info(f"  Successful: {dms_results['num_success']}/{dms_results['num_total']}")
                logger.info(f"  By Category:")
                for cat, val in dms_results['by_category'].items():
                    logger.info(f"    {cat}: {val:.4f}")

                if config.logging.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'dms_simple_mean': dms_results['simple_mean'],
                        'dms_uniprot_mean': dms_results['uniprot_mean'],
                        'dms_leaderboard_mean': dms_results['leaderboard_mean'],
                        'dms_num_success': dms_results['num_success'],
                        'dms_num_total': dms_results['num_total'],
                    })
                    # Log per-category
                    for cat, val in dms_results['by_category'].items():
                        wandb.log({f'dms_{cat}': val})
                    # Log per-file correlations
                    for dms_id, corr in dms_results['per_file'].items():
                        wandb.log({f'dms_{dms_id}': corr})
            else:
                logger.warning("DMS evaluation returned no results")

    # Save final model
    final_dir = os.path.join(config.training.output_dir, "final_model")
    save_checkpoint(
        model, lagrangian_module=None, optimizer=optimizer, scheduler=scheduler,
        epoch=config.training.num_epochs - 1, val_loss=val_loss, config=config,
        checkpoint_dir=final_dir
    )

    if embedding_cache is not None:
        embedding_cache.print_statistics()

    # Combine all DMS epoch results into a single summary Excel file
    if args.dms_dir and args.dms_metadata:
        logger.info("\nCombining all DMS results into summary file...")
        dms_csv_files = glob.glob(os.path.join(config.training.output_dir, "dms_results_epoch_*.csv"))
        if dms_csv_files:
            all_dms_results = []
            for csv_file in sorted(dms_csv_files):
                # Extract epoch number from filename
                epoch_num = int(csv_file.split("epoch_")[1].split(".csv")[0])
                df = pd.read_csv(csv_file)
                df["epoch"] = epoch_num
                all_dms_results.append(df)

            combined_df = pd.concat(all_dms_results, ignore_index=True)
            # Reorder columns to put epoch first
            cols = ["epoch"] + [c for c in combined_df.columns if c != "epoch"]
            combined_df = combined_df[cols]

            summary_excel_path = os.path.join(config.training.output_dir, "dms_all_epochs_summary.xlsx")
            combined_df.to_excel(summary_excel_path, index=False)
            logger.info(f"DMS results for all epochs saved to: {summary_excel_path}")

            # Also save as CSV for easy viewing
            summary_csv_path = os.path.join(config.training.output_dir, "dms_all_epochs_summary.csv")
            combined_df.to_csv(summary_csv_path, index=False)
            logger.info(f"DMS results for all epochs saved to: {summary_csv_path}")

    logger.info("Training completed!")
    if config.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
