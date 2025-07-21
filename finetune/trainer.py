import functools
import json
import logging
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import constants
from model import AmplifyClassifier
from protein_dataset import ProteinDataset, collate_fn

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    loss_fn: torch.nn.modules.loss._Loss,
    seq_vocab_size: int,
    struc_vocab_size: int,
    loss_weight: List,
    data_type: torch.dtype,
    batch_size: int,
) -> Tuple[float, Tuple[float, float, float]]:
    """
    Compute validation loss on a single GPU.
    Returns average total loss and tuple of (mlm, intra, inter) losses.
    """
    model.eval()
    val_loss_list: List[torch.Tensor] = []
    mlm_loss_list: List[torch.Tensor] = []
    intra_loss_list: List[torch.Tensor] = []
    inter_loss_list: List[torch.Tensor] = []

    mask_denominator = constants.MEAN_MASK_SEQ_LEN * batch_size
    denom = constants.MEAN_SEQ_LEN * batch_size

    for batch in val_dataloader:
        seq_tokens = batch["seq_tokens"].to(device).to(torch.long)
        attention_mask = batch["attention_mask"].to(device).to(data_type)
        real_residue_mask = batch["real_residue_mask"].to(device).to(torch.bool)
        seq_labels = batch["seq_labels"].to(device).to(torch.long)
        struc_labels = batch["struc_labels"].to(device).to(torch.long)
        struc_embeddings = batch["struc_embeddings"].to(device).to(data_type)
        weights = batch["weights"].to(device).to(data_type)
        cl_weights = batch["cl_weights"].to(device).to(data_type)

        logit_mlm, logit_cls, hidden_state = model.main_forward(
            seq_tokens, attention_mask, frozen_trunk=True
        )

        # MLM loss
        mlm_loss = loss_fn(logit_mlm.view(-1, seq_vocab_size), seq_labels.view(-1))
        mlm_loss = mlm_loss.view(logit_mlm.shape[0], logit_mlm.shape[1])
        mlm_loss = (torch.sum(mlm_loss * weights.view(weights.shape[0], -1))
                    / mask_denominator)

        # Intra loss
        intra_loss = loss_fn(
            logit_cls.view(-1, struc_vocab_size), struc_labels.view(-1)
        )
        intra_loss = intra_loss.view(logit_cls.shape[0], logit_cls.shape[1])
        intra_loss = (torch.sum(intra_loss * weights.view(weights.shape[0], -1))
                      / denom)

        # Contrastive loss
        seq_emb = hidden_state[real_residue_mask]
        loss_seq_to_struct, loss_struct_to_seq = model.cl_forward(
            seq_emb, struc_embeddings, cl_weights
        )
        inter_loss = (
            torch.sum(loss_seq_to_struct) / denom + torch.sum(loss_struct_to_seq) / denom
        ) / 2.0

        total_loss = (
            loss_weight[0] * mlm_loss
            + loss_weight[1] * intra_loss
            + loss_weight[2] * inter_loss
        )

        val_loss_list.append(total_loss)
        mlm_loss_list.append(mlm_loss)
        intra_loss_list.append(intra_loss)
        inter_loss_list.append(inter_loss)

    model.train()

    if not val_loss_list:
        return float('inf'), (float('inf'), float('inf'), float('inf'))

    avg_val_loss = torch.stack(val_loss_list).mean().item()
    avg_mlm_loss = torch.stack(mlm_loss_list).mean().item()
    avg_intra_loss = torch.stack(intra_loss_list).mean().item()
    avg_inter_loss = torch.stack(inter_loss_list).mean().item()

    return avg_val_loss, (avg_mlm_loss, avg_intra_loss, avg_inter_loss)


def train(config: DictConfig):
    """
    Single-GPU training process.
    """
    # Build output directories
    prt_model_safe = config.prt_model_name.split('/')[-1]
    ref_model_safe = config.reference_model.split('/')[1] if config.get('reference_model') else 'None'
    output_dir = os.path.join(
        'checkpoint',
        f"{prt_model_safe}_{ref_model_safe}_{'_'.join(map(str, config.loss_weight))}_"
        f"{config.sample_mode}_{config.ratio}_{config.struc_token_type}_{config.struc_embed_type}_{config.seed}"
    )
    os.makedirs(output_dir, exist_ok=True)
    train_metric_file = os.path.join(output_dir, 'train_metric.json')
    valid_metric_file = os.path.join(output_dir, 'valid_metric.json')

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Configure WandB
    wandb.init(
        project='structure-aware-plm',
        name=os.path.basename(output_dir),
        entity='drug-discovery-amgen',
        config=OmegaConf.to_container(config, resolve=True),
        dir=output_dir,
        resume=config.get('resume', False),
        mode=config.get('wandb_mode', 'offline')
    )

    # Build model
    struc_vocab_size = {
        'foldseek': 26,
        'pst': 4098,
        'protoken': 514,
        'aido': 514,
    }[config.struc_token_type]
    seq_D = {
        'chandar-lab/AMPLIFY_120M': 640,
        'chandar-lab/AMPLIFY_350M': 960,
        'facebook/esm2_t6_8M_UR50D': 320,
        'facebook/esm2_t12_35M_UR50D': 480,
        'facebook/esm2_t30_150M_UR50D': 640,
        'facebook/esm2_t33_650M_UR50D': 1280,
    }[config.prt_model_name]
    struc_D = {'af2': 384, 'gearnet': 512}[config.struc_embed_type]
    output_D = min(seq_D, struc_D)

    model = AmplifyClassifier(
        config.prt_model_name,
        num_labels=struc_vocab_size,
        seq_D=seq_D,
        struc_D=struc_D,
        output_D=output_D,
    ).to(device)
    model = torch.compile(model)

    # Optional reference model
    reference_model = None
    if config.get('reference_model'):
        reference_model = AmplifyClassifier(
            config.reference_prt_model_name,
            num_labels=struc_vocab_size,
            seq_D=seq_D,
            struc_D=struc_D,
            output_D=min(seq_D, struc_D),
        )
        ckpt = torch.load(os.path.join(config.reference_model, 'model_trunk.pt'), map_location='cpu')
        reference_model.trunk.load_state_dict(ckpt)
        reference_model.eval()
        reference_model = torch.compile(reference_model).to(device)
        logger.info('Reference model loaded')

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.prt_model_name, trust_remote_code=True)

    # Data loaders
    collate_fn_with_tokenizer = functools.partial(
        collate_fn,
        tokenizer=tokenizer,
        struc_token_type=config.struc_token_type
    )
    dataset_train = ProteinDataset(
        data_type=config.train_data_type,
        struc_token_type=config.struc_token_type,
        struc_embed_type=config.struc_embed_type,
        prefix_path=config.prefix_path
    )

    total_train = len(dataset_train)
    subset_size = max(1, int(total_train * 0.01))
    indices = list(range(subset_size))
    dataset_train = Subset(dataset_train, indices)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn_with_tokenizer,
        num_workers=4,
    )

    dataset_val = ProteinDataset(
        data_type=config.get('valid_data_type', 'valid'),
        struc_token_type=config.struc_token_type,
        struc_embed_type=config.struc_embed_type,
        prefix_path=config.prefix_path
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn_with_tokenizer,
        num_workers=4,
    )

    # Loss, optimizer, scheduler
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    param_groups = [
        {'params': model.trunk.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3},
        {'params': model.cl_model.parameters(), 'lr': 1e-3},
    ]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)
    updates_per_epoch = len(dataloader_train)
    total_updates = updates_per_epoch * config.n_epochs
    warmup_updates = updates_per_epoch * 2
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates
    )

    # Data type for mixed precision
    data_type = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'no': torch.float32,
    }[config.precision]

    # Training loop
    global_step = 0
    for epoch in range(config.n_epochs):
        model.train()
        epoch_losses = []
        for batch in tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{config.n_epochs}"):
            global_step += 1
            # Move batch to device
            for k in batch:
                batch[k] = batch[k].to(device)
            # Forward
            logit_mlm, logit_cls, hidden_state = model.main_forward(
                batch['seq_tokens'].to(torch.long),
                batch['attention_mask'].to(data_type),
                frozen_trunk=False
            )
            # Compute losses (same as validate but for training)
            mlm_loss = loss_fn(
                logit_mlm.view(-1, tokenizer.vocab_size), batch['seq_labels'].view(-1)
            )
            mlm_loss = mlm_loss.view(logit_mlm.shape[0], logit_mlm.shape[1])
            mlm_loss = torch.sum(mlm_loss) / (constants.MEAN_MASK_SEQ_LEN * config.batch_size)

            intra_loss = loss_fn(
                logit_cls.view(-1, struc_vocab_size), batch['struc_labels'].view(-1)
            )
            intra_loss = intra_loss.view(logit_cls.shape[0], logit_cls.shape[1])
            intra_loss = torch.sum(intra_loss) / (constants.MEAN_SEQ_LEN * config.batch_size * config.ratio)

            seq_emb = hidden_state[batch['real_residue_mask'].to(torch.bool)]
            loss_seq_to_struct, loss_struct_to_seq = model.cl_forward(
                seq_emb, batch['struc_embeddings'].to(data_type), batch['cl_weights'].to(data_type)
            )
            inter_loss = (torch.sum(loss_seq_to_struct) + torch.sum(loss_struct_to_seq)) / 2.0
            inter_loss /= (constants.MEAN_SEQ_LEN * config.batch_size * config.ratio)

            loss = (
                config.loss_weight[0] * mlm_loss
                + config.loss_weight[1] * intra_loss
                + config.loss_weight[2] * inter_loss
            )

            # Backward and step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_losses.append(loss.item())

            # Validation
            if global_step % config.eval_steps == 0:
                avg_train_loss = np.mean(epoch_losses)
                val_loss, (v_mlm, v_intra, v_inter) = validate(
                    model,
                    dataloader_val,
                    loss_fn,
                    tokenizer.vocab_size,
                    struc_vocab_size,
                    config.loss_weight,
                    data_type,
                    config.batch_size,
                )
                logger.info(
                    f"Step {global_step}: train_loss={avg_train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, mlm={v_mlm:.4f}, intra={v_intra:.4f}, inter={v_inter:.4f}"
                )
                wandb.log({
                    'step': global_step,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'val_mlm_loss': v_mlm,
                    'val_intra_loss': v_intra,
                    'val_inter_loss': v_inter,
                })
                epoch_losses = []

    logger.info("Training finished")
    wandb.finish()
