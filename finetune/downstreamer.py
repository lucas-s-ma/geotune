import gc
import logging
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torchmetrics
import wandb
from datasets import load_dataset, load_from_disk
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from torchmetrics import (
    AUROC,
    Accuracy,
    MatthewsCorrCoef,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)

from metric import Fmax, LongRangePrecisionAtL
from model import ContactPredictionModel, PointPredictionModel, PPIModel
from protein_dataset import create_transform_collate, obtain_real_residue_mask


@dataclass
class EvalParams:
    task_name: str
    task_output_type: str
    task_num_labels: int
    tokenizer: AutoTokenizer
    data_type: torch.dtype
    label_type: torch.dtype
    device: torch.device
    metric_fn: torchmetrics.Metric


@torch.no_grad()
def downstream_valid(
    model: torch.nn.Module,
    dataloader_valid: DataLoader,
    eval_params: EvalParams,
) -> torch.Tensor:
    model.eval()
    eval_params.metric_fn.reset()

    labels = []
    outputs = []
    if eval_params.task_name == "Bo1015/contact_prediction_binary":
        effective_Ls = []

    for batch in dataloader_valid:
        input_ids = batch["input_ids"].to(torch.long).to(eval_params.device)
        attention_mask = batch["attention_mask"].to(eval_params.data_type).to(eval_params.device)
        real_mask = obtain_real_residue_mask(input_ids, eval_params.tokenizer)
        label = batch["labels"].to(eval_params.label_type).to(eval_params.device)

        if eval_params.task_name == "saprot_data/HumanPPI":
            input_ids_2 = batch["input_ids_2"].to(torch.long).to(eval_params.device)
            attention_mask_2 = batch["attention_mask_2"].to(eval_params.data_type).to(eval_params.device)
            output = model(input_ids, attention_mask, input_ids_2, attention_mask_2, frozen_trunk=True)
        else:
            output = model(input_ids, attention_mask, frozen_trunk=True)

        if eval_params.task_name == "Bo1015/contact_prediction_binary":
            for i in range(output.shape[0]):
                labels.append(label[i])
                outputs.append(output[i])
                effective_Ls.append(int(real_mask[i].sum().item()))
        else:
            if eval_params.task_output_type == "residue":
                label = label[real_mask]
                output = output[real_mask]
            labels.append(label)
            outputs.append(output)

    if eval_params.task_name != "Bo1015/contact_prediction_binary":
        preds = torch.cat(outputs, dim=0).view(-1, eval_params.task_num_labels).squeeze()
        labels = torch.cat(labels, dim=0)
        val_metric = eval_params.metric_fn(preds, labels)
    else:
        for pred, gt, L in zip(outputs, labels, effective_Ls):
            eval_params.metric_fn.update(pred, gt, L)
        val_metric = eval_params.metric_fn.compute()

    model.train()
    eval_params.metric_fn.reset()
    return val_metric


def downstream(config, task_name):
    logger = logging.getLogger(__name__)
    logger.info(f"config: {config}, task_name: {task_name}")

    # Device
    device = torch.device("cuda" + f":{config.device}" if torch.cuda.is_available() else "cpu")

    # Output dirs
    prt_safe = config.prt_model_name.split("/")[-1]
    output_dir = os.path.join("output", prt_safe, config.ft_model_path, task_name, str(config.seed))
    latest_ckpt = os.path.join(output_dir, "latest")
    best_ckpt = os.path.join(output_dir, "best")
    os.makedirs(latest_ckpt, exist_ok=True)
    os.makedirs(best_ckpt, exist_ok=True)

    # WandB
    wandb_id = "_".join(output_dir.split("/")[-3:])
    wandb.init(project="structure-aware-plm", name=wandb_id, entity="drug-discovery-amgen",
               config=OmegaConf.to_container(config, resolve=True), id=wandb_id,
               dir=output_dir, resume=config.get("resume", True), mode="offline")

    # Tokenizer
    if config.prt_model_name == "ism":
        tokenizer = AutoTokenizer.from_pretrained("checkpoint/ISM/ism_model")
    elif config.prt_model_name == "esm-s":
        tokenizer = AutoTokenizer.from_pretrained("checkpoint/ESM-s/esm_s_model")
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.prt_model_name, trust_remote_code=True)

    # Load dataset
    if task_name.split("/")[0] == "Bo1015":
        all_data = load_dataset(task_name)
    else:
        all_data = load_from_disk(task_name)

    # Small subset test
    train_size = int(0.01 * len(all_data["train"]))
    all_data["train"] = all_data["train"].select(range(train_size))
    all_data = all_data.rename_column("label", "labels")
    if task_name == "Bo1015/fitness_prediction":
        def to_float(s): s["labels"] = float(s["labels"]); return s
        for split in all_data: all_data[split] = all_data[split].map(to_float)

    task_cfg = OmegaConf.load("config/task.yaml")[task_name]
    norm = task_cfg.loss_type == "regression"
    mean, std = (np.mean(all_data["train"]["labels"]), np.std(all_data["train"]["labels"])) if norm else (0,1)

    # Transforms
    t_fn, c_fn = create_transform_collate(task_name, task_cfg.output_type, tokenizer, max_len=config.get("max_len",2048))
    all_data.set_transform(t_fn)

    # DataLoaders (single GPU)
    num_workers = 4
    dataloader_train = DataLoader(all_data["train"], batch_size=config.batch_size,
                                  shuffle=True, collate_fn=c_fn, pin_memory=True, num_workers=num_workers)
    dataloader_valid = DataLoader(all_data["valid"], batch_size=config.batch_size,
                                  shuffle=False, collate_fn=c_fn, pin_memory=True, num_workers=num_workers) if "valid" in all_data else None
    dataloader_test = DataLoader(all_data["test"], batch_size=config.batch_size,
                                 shuffle=False, collate_fn=c_fn, pin_memory=True, num_workers=num_workers)

    # Model
    labels = task_cfg.num_labels
    dt = {"bf16":torch.bfloat16,"fp16":torch.float16,"no":torch.float32}[config.precision]
    if task_name == "Bo1015/contact_prediction_binary":
        model = ContactPredictionModel(prt_model_name=config.prt_model_name, ft_model_path=config.ft_model_path, task_num_labels=labels)
    elif task_name == "saprot_data/HumanPPI":
        model = PPIModel(prt_model_name=config.prt_model_name, ft_model_path=config.ft_model_path, task_num_labels=labels)
    else:
        model = PointPredictionModel(prt_model_name=config.prt_model_name, ft_model_path=config.ft_model_path,
                                     task_num_labels=labels, task_output_type=task_cfg.output_type,
                                     normalization=norm, target_mean=mean, target_std=std)
    model = model.eval().to(device).to(dtype=dt)

    # Loss and optimizer
    loss_map = {"classification":torch.nn.CrossEntropyLoss(),"regression":torch.nn.MSELoss(),"multi_classification":torch.nn.BCEWithLogitsLoss()}
    loss_fn = loss_map[task_cfg.loss_type]
    label_ty = {"classification":torch.long,"regression":dt,"multi_classification":dt}[task_cfg.loss_type]
    params = (model.classifier.parameters() if config.frozen_trunk else
              [{'params':model.trunk.parameters(),'lr':1e-4},{'params':model.classifier.parameters(),'lr':1e-3}])
    optimizer = torch.optim.AdamW(params, betas=(0.9,0.95), weight_decay=0.01)

    # Scheduler
    updates = max(len(dataloader_train)//config.opt_interval,1)
    warmup = updates*2
    total = updates*config.n_epochs
    lr_fn = lambda step: (step/warmup if step<warmup else 0.01+0.99*0.5*(1+math.cos(math.pi*((step-warmup)/(total-warmup)))))
    scheduler = LambdaLR(optimizer, lr_fn)

    # Metric
    metric_cls = {"accuracy":Accuracy(task="multiclass",num_classes=max(labels,2)),"mcc":MatthewsCorrCoef(task="binary"),
                  "spearman":SpearmanCorrCoef(),"auc":AUROC(task="binary"),"pcc":PearsonCorrCoef(),
                  "long_range_precision_at_L":LongRangePrecisionAtL(top_factor=5),"fmax":Fmax()}[task_cfg.metric]
    metric_fn = metric_cls.to(device)
    eval_params = EvalParams(task_name, task_cfg.output_type, labels, tokenizer, dt, label_ty, device, metric_fn)

    # Resume logic (no DistributedSampler)
    start_epoch, best_val = 0, float('-inf')
    state_path = os.path.join(latest_ckpt, "training_state.pt")
    if config.get("resume",True) and os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        start_epoch = state.get("epoch",0)+1
        best_val = state.get("best_val_metric",float('-inf'))
        if "optimizer_state" in state: optimizer.load_state_dict(state["optimizer_state"])  
        if "scheduler_state" in state: scheduler.load_state_dict(state["scheduler_state"])

    # Training
    global_step = 0
    start_time = time.time()
    for epoch in range(start_epoch, config.n_epochs):
        model.train() if not config.frozen_trunk else model.classifier.train(); model.trunk.eval() if config.frozen_trunk else None
        train_losses = []
        for batch in tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{config.n_epochs}"):
            global_step += 1
            for k,v in batch.items():
                batch[k] = v.to(device)

            if task_name == "saprot_data/HumanPPI":
                output = model(batch["input_ids"], batch["attention_mask"], batch["input_ids_2"], batch["attention_mask_2"], frozen_trunk=config.frozen_trunk)
            else:
                output = model(batch["input_ids"], batch["attention_mask"], frozen_trunk=config.frozen_trunk)

            if task_cfg.loss_type == "multi_classification":
                loss = loss_fn(output, batch["labels"].to(label_ty))
            else:
                loss = loss_fn(output.view(-1,labels).squeeze(), batch["labels"].view(-1))

            train_losses.append(loss.item())
            (loss/config.opt_interval).backward()
            if global_step % config.opt_interval == 0:
                optimizer.step(); optimizer.zero_grad(); scheduler.step()
                step = global_step//config.opt_interval
                eta = (time.time()-start_time)/global_step*(config.n_epochs*len(dataloader_train)-global_step)/3600
                metrics = {"epoch":epoch,"step":step,"train_loss":np.mean(train_losses),"lr":scheduler.get_last_lr()[0],"ETA_h":eta}
                wandb.log(metrics); logging.getLogger(__name__).info(metrics)
                train_losses = []

        # Checkpoint latest
        save = {"epoch":epoch,"best_val_metric":best_val,"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict()}
        torch.save(save, os.path.join(latest_ckpt,"training_state.pt"))
        if config.frozen_trunk:
            torch.save(model.classifier.state_dict(), os.path.join(latest_ckpt,"model_classifier.pt"))
        else:
            torch.save(model.state_dict(), os.path.join(latest_ckpt,"model.pt"))

        # Validation
        if dataloader_valid:
            val_metric = downstream_valid(model, dataloader_valid, eval_params)
            wandb.log({"epoch":epoch,"val_metric":val_metric})
            if val_metric > best_val:
                best_val = val_metric
                torch.save({"epoch":epoch,"best_val_metric":best_val}, os.path.join(best_ckpt,"training_state.pt"))
                torch.save((model.classifier.state_dict() if config.frozen_trunk else model.state_dict()),
                           os.path.join(best_ckpt,"model_classifier.pt" if config.frozen_trunk else "model.pt"))
        else:
            test_metric = downstream_valid(model, dataloader_test, eval_params)
            wandb.log({"epoch":epoch,"test_metric":test_metric})

    wandb.finish(); logger.info("Downstream Finished!")
    torch.cuda.empty_cache(); gc.collect()
