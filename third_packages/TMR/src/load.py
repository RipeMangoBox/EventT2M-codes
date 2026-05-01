import os
from pathlib import Path
from omegaconf import DictConfig
import logging
import hydra

from .config import read_config

logger = logging.getLogger(__name__)


def _summarize_run_dir(run_dir: str, limit: int = 8) -> str:
    run_dir_path = Path(run_dir)
    if not run_dir_path.exists():
        return "run_dir does not exist"

    entries = sorted(run_dir_path.iterdir(), key=lambda path: path.name)
    if not entries:
        return "run_dir is empty"

    summary = [f"{entry.name}{'/' if entry.is_dir() else ''}" for entry in entries[:limit]]
    if len(entries) > limit:
        summary.append("...")
    return ", ".join(summary)


def _raise_missing_weights_error(run_dir: str, ckpt_name: str) -> None:
    ckpt_path = os.path.join(run_dir, f"logs/checkpoints/{ckpt_name}.ckpt")
    pt_path = os.path.join(run_dir, f"{ckpt_name}_weights")
    available_entries = _summarize_run_dir(run_dir)

    raise FileNotFoundError(
        "TMR weights are missing. "
        f"Expected either extracted module weights at '{pt_path}' "
        f"or a Lightning checkpoint at '{ckpt_path}'. "
        f"Available entries under run_dir '{run_dir}': {available_entries}. "
        "Please download/unzip the pretrained TMR model files into this run_dir before starting Event-T2M training."
    )


# split the lightning checkpoint into
# seperate state_dict modules for faster loading
def extract_ckpt(run_dir, ckpt_name="last"):
    import torch

    ckpt_path = os.path.join(run_dir, f"logs/checkpoints/{ckpt_name}.ckpt")
    if not os.path.exists(ckpt_path):
        _raise_missing_weights_error(run_dir, ckpt_name)

    extracted_path = os.path.join(run_dir, f"{ckpt_name}_weights")
    os.makedirs(extracted_path, exist_ok=True)

    new_path_template = os.path.join(extracted_path, "{}.pt")
    ckpt_dict = torch.load(ckpt_path)
    state_dict = ckpt_dict["state_dict"]
    module_names = list(set([x.split(".")[0] for x in state_dict.keys()]))

    for module_name in module_names:
        path = new_path_template.format(module_name)
        sub_state_dict = {
            ".".join(x.split(".")[1:]): y.cpu()
            for x, y in state_dict.items()
            if x.split(".")[0] == module_name
        }
        torch.save(sub_state_dict, path)


def load_model(run_dir, **params):
    cfg = read_config(run_dir)
    cfg.run_dir = run_dir
    return load_model_from_cfg(cfg, **params)


def load_model_from_cfg(cfg, ckpt_name="last", device="cpu", eval_mode=True):
    from . import prepare  # noqa
    import torch

    run_dir = cfg.run_dir
    model = hydra.utils.instantiate(cfg.model)

    pt_path = os.path.join(run_dir, f"{ckpt_name}_weights")
    ckpt_path = os.path.join(run_dir, f"logs/checkpoints/{ckpt_name}.ckpt")

    weights_ready = os.path.exists(pt_path) and len(os.listdir(pt_path)) > 0
    if not weights_ready:
        if os.path.exists(ckpt_path):
            logger.info("The extracted model is not found. Split into submodules..")
            extract_ckpt(run_dir, ckpt_name)
            weights_ready = os.path.exists(pt_path) and len(os.listdir(pt_path)) > 0
        else:
            _raise_missing_weights_error(run_dir, ckpt_name)

    assert weights_ready
    for fname in os.listdir(pt_path):
        module_name, ext = os.path.splitext(fname)

        if ext != ".pt":
            continue

        module = getattr(model, module_name, None)
        if module is None:
            continue

        module_path = os.path.join(pt_path, fname)
        state_dict = torch.load(module_path)
        module.load_state_dict(state_dict)
        logger.info(f"    {module_name} loaded")

    logger.info("Loading previous checkpoint done")
    model = model.to(device)
    logger.info(f"Put the model on {device}")
    if eval_mode:
        model = model.eval()
        logger.info("Put the model in eval mode")
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="load_model")
def hydra_load_model(cfg: DictConfig) -> None:
    run_dir = cfg.run_dir
    ckpt_name = cfg.ckpt
    device = cfg.device
    eval_mode = cfg.eval_mode
    return load_model(run_dir, ckpt_name, device, eval_mode)


if __name__ == "__main__":
    hydra_load_model()
