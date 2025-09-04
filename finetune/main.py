import copy
import logging
import os
import random

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

from downstreamer import downstream
from trainer import train

# set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
logger.propagate = False


def prepare_task_config(cfg: DictConfig, task_name: str) -> DictConfig:
    """
    Prepares a configuration copy for each downstream task.

    Adjusts frozen_trunk, batch_size, and opt_interval settings based on the task name.
    """
    SPECIAL_TASKS_FROZEN = [
        "Bo1015/ssp_q3",
        "Bo1015/fold_prediction",
        "Bo1015/contact_prediction_binary",
        "saprot_data/HumanPPI",
    ]
    CONTACT_PREDICTION_TASK = "Bo1015/contact_prediction_binary"

    cfg_copy = copy.deepcopy(cfg)
    OmegaConf.set_struct(cfg_copy, False)

    cfg_copy.frozen_trunk = task_name in SPECIAL_TASKS_FROZEN

    if not cfg_copy.frozen_trunk:
        cfg_copy.batch_size = int(cfg_copy.batch_size / 2)
        cfg_copy.opt_interval = int(cfg_copy.opt_interval * 2)

    if task_name == CONTACT_PREDICTION_TASK:
        cfg_copy.opt_interval = 64
        cfg_copy.batch_size = 2

    return cfg_copy


@hydra.main(version_base=None, config_path="config/", config_name="config.yaml")
def pipeline(cfg: DictConfig) -> None:
    # Initialize W&B with full Hydra config
    run_name = f"{cfg.experiments.prt_model_name.replace('/', '_')}_seed{cfg.experiments.seed}_{cfg.experiments.mode}"
    wandb.init(
        project="original_loss_test",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        job_type=cfg.experiments.mode,
        tags=[cfg.experiments.mode]
    )

    # seed everything
    random.seed(cfg.experiments.seed)
    np.random.seed(cfg.experiments.seed)
    torch.manual_seed(cfg.experiments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.experiments.seed)

    # Make original CWD available to other modules
    OmegaConf.set_struct(cfg.experiments, False)
    cfg.experiments.original_cwd = get_original_cwd()

    try:
        if cfg.experiments.mode == "train":
            logger.info("Starting training...")
            train(cfg.experiments)

        elif cfg.experiments.mode == "downstream":
            logger.info("Starting downstream evaluations...")
            prt_model_safe = cfg.experiments.prt_model_name.split("/")[-1]
            output_dir = os.path.join(
                cfg.experiments.original_cwd,
                "output",
                prt_model_safe,
                cfg.experiments.ft_model_path,
                str(cfg.experiments.seed),
            )
            os.makedirs(output_dir, exist_ok=True)

            # load or initialize task list
            task_names_file = os.path.join(output_dir, "task_names_todo.npy")
            if os.path.exists(task_names_file):
                task_names_todo = list(np.load(task_names_file, allow_pickle=True))
            else:
                task_names_todo = cfg.experiments.task_names

            for idx, task_name in enumerate(task_names_todo):
                logger.info("Evaluating: %s", task_name)
                cfg_task = prepare_task_config(cfg.experiments, task_name)
                metrics = downstream(cfg_task, task_name)
                # log downstream metrics
                wandb.log({f"downstream/{task_name}/{k}": v for k, v in metrics.items()})
                np.save(task_names_file, task_names_todo[idx + 1:])

            logger.info("All downstream tasks finished")

        else:
            logger.error("Unknown mode: %s", cfg.experiments.mode)
    except Exception as e:
        logger.exception("Run failed: %s", e)
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    pipeline()