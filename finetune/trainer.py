import functools
import logging
import os
import time
import wandb

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import constants
from model import AmplifyClassifier
from protein_dataset import ProteinDataset, collate_fn

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config: DictConfig):
    """
    Single-GPU training process for smoke test.
    """
    # Build output directories
    prt_model_safe = config.prt_model_name.split('/')[-1]
    output_dir = os.path.join(
        'checkpoint',
        f"{prt_model_safe}_{config.struc_token_type}_{config.struc_embed_type}_{config.seed}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Build model
    struc_vocab_size = {
        'foldseek': 26,
        'pst': 4098,
        'protoken': 514,
        'aido': 514,
    }[config.struc_token_type]
    seq_D = {
        'facebook/esm2_t30_150M_UR50D': 640,
    }[config.prt_model_name]
    struc_D = {'af2': 384, 'gearnet': 256}[config.struc_embed_type]
    output_D = min(seq_D, struc_D)

    model = AmplifyClassifier(
        config.prt_model_name,
        num_labels=struc_vocab_size,
        seq_D=seq_D,
        struc_D=struc_D,
        output_D=output_D,
    ).to(device)
    model = torch.compile(model)

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

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size, # Use the batch size from config
        shuffle=True,
        collate_fn=collate_fn_with_tokenizer,
        num_workers=2, # Reduce num_workers
    )

    # Loss, optimizer, scheduler
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    param_groups = [
        {'params': model.trunk.parameters(), 'lr': 1e-5}, # Lower LR for fine-tuning
        {'params': model.classifier.parameters(), 'lr': 1e-4},
        {'params': model.cl_model.parameters(), 'lr': 1e-4},
    ]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)
    updates_per_epoch = len(dataloader_train)
    total_updates = updates_per_epoch * config.n_epochs
    warmup_updates = updates_per_epoch # 1 epoch warmup
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
    logger.info("Starting training for smoke test...")

    run = wandb.init(
        project="original amplify loss",
        name="AMPLIFYYYYYY"
    )

    for epoch in range(config.n_epochs):
        model.train()
        epoch_losses = []
        for batch in tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{config.n_epochs}"):
            global_step += 1
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Forward
            logit_mlm, logit_cls, hidden_state = model.main_forward(
                batch['seq_tokens'].to(torch.long),
                batch['attention_mask'].to(data_type),
                frozen_trunk=False
            )
            print(batch['seq_tokens'].to(torch.long).shape)
            print(logit_mlm.shape)
            print(logit_cls.shape)

            # Compute losses
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

            run.log({"mlm": mlm_loss, "intra_loss": intra_loss, "inter_loss":inter_loss, "total":loss})

            epoch_losses.append(loss.item())

            if global_step % config.eval_steps == 0:
                avg_train_loss = np.mean(epoch_losses)
                logger.info(
                    f"Step {global_step}: train_loss={avg_train_loss:.4f}"
                )
                epoch_losses = []

    logger.info("Training finished")
    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(output_dir, "constraint_prototype.pt"))
    logger.info(f"Model saved to {os.path.join(output_dir, 'constraint_prototype.pt')}")

