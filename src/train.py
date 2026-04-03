import os
import os.path
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning.pytorch as L
import torch
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    normalize_trainer_devices,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def resolve_exp_name(project_root: str, exp_name: str) -> str:
    match = re.fullmatch(r"exp(\d+)", exp_name)
    if not match:
        return exp_name

    checkpoints_root = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoints_root, exist_ok=True)

    requested_index = int(match.group(1))
    existing_indices = []
    for name in os.listdir(checkpoints_root):
        existing_match = re.fullmatch(r"exp(\d+)", name)
        if existing_match and os.path.isdir(os.path.join(checkpoints_root, name)):
            existing_indices.append(int(existing_match.group(1)))

    if requested_index not in existing_indices:
        return exp_name

    return f"exp{max(existing_indices, default=0) + 1}"


def resolve_requested_exp_name(argv: List[str], default_exp_name: str = "exp1") -> str:
    exp_name = os.environ.get("EVENTT2M_EXP_NAME", default_exp_name)
    for arg in argv:
        if arg.startswith("exp_name="):
            exp_name = arg.split("=", 1)[1]
    return exp_name


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    torch.set_float32_matmul_precision('high')

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=False)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        L.seed_everything(cfg.seed, workers=True)
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    normalize_trainer_devices(cfg)

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    project_root = str(Path(__file__).resolve().parents[1])
    requested_exp_name = resolve_requested_exp_name(sys.argv[1:])
    resolved_exp_name = resolve_exp_name(project_root, requested_exp_name)
    os.environ["EVENTT2M_EXP_NAME"] = resolved_exp_name

    if resolved_exp_name != requested_exp_name:
        log.info(
            f"Experiment directory '{requested_exp_name}' already exists. Using '{resolved_exp_name}' instead."
        )

    main()
