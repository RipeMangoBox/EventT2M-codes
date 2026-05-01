import copy
import os
import sys
import importlib
from contextlib import contextmanager
from pathlib import Path
from typing import cast
import torch
from torch import Tensor
from torch import nn as nn
import numpy as np
from omegaconf import DictConfig, ListConfig


def rewrite_tmr_targets(node):
    if isinstance(node, (dict, DictConfig)):
        for key, value in node.items():
            if key == "_target_" and isinstance(value, str) and value.startswith("src."):
                node[key] = f"third_packages.TMR.{value}"
            else:
                rewrite_tmr_targets(value)
        return

    if isinstance(node, (list, ListConfig)):
        for item in node:
            rewrite_tmr_targets(item)


def extend_namespace_package(package_name: str, path: Path):
    module = sys.modules.get(package_name)
    if module is None or not hasattr(module, "__path__"):
        return

    path_str = str(path)
    if path_str not in module.__path__:
        module.__path__.append(path_str)

@contextmanager
def temporary_change_cwd(destination):
    current_dir = os.getcwd()
    try:
        os.chdir(destination)
        yield
    finally:
        os.chdir(current_dir)

class TMR_Wrapper(nn.Module):
    def __init__(self, model_dir: Path):
        super().__init__()

        PROJECT_ROOT = Path(__file__).resolve().parent
        TMR_ROOT = PROJECT_ROOT / "third_packages" / "TMR"

        self.concat_clip_dim_layer = nn.Linear(512,512)

        from hydra.utils import instantiate

        if str(TMR_ROOT) not in sys.path:
            sys.path.insert(0, str(TMR_ROOT))

        extend_namespace_package("src", TMR_ROOT / "src")
        extend_namespace_package("src.data", TMR_ROOT / "src" / "data")
        importlib.invalidate_caches()

        from third_packages.TMR.src.config import read_config
        from third_packages.TMR.src.data.collate import collate_x_dict
        from third_packages.TMR.src.data.text import TokenEmbeddings
        from third_packages.TMR.src.load import load_model_from_cfg
        from third_packages.TMR.src.model.tmr import TMR

        self.collate_x_dict = collate_x_dict
        self.model_dir = model_dir
        self.base_cfg = read_config(str(model_dir))

        cfg = copy.deepcopy(self.base_cfg)
        rewrite_tmr_targets(cfg)

        abs_model_dir = (PROJECT_ROOT / model_dir).resolve()

        try:
            cfg.run_dir = abs_model_dir.relative_to(TMR_ROOT.resolve())
        except ValueError:
            cfg.run_dir = abs_model_dir

        cfg.data.text_to_token_emb.path = str((TMR_ROOT / cfg.data.text_to_token_emb.path).resolve())
        if 'text_to_sent_emb' in cfg.data:
            cfg.data.text_to_sent_emb.path = str((TMR_ROOT / cfg.data.text_to_sent_emb.path).resolve())
        cfg.data.path = str((TMR_ROOT / cfg.data.path).resolve())

        with temporary_change_cwd(PROJECT_ROOT):
            cfg.data.text_to_token_emb['_target_'] = 'third_packages.TMR.src.data.text.TokenEmbeddings'
            self.text_model: TokenEmbeddings = instantiate(cfg.data.text_to_token_emb)

        with temporary_change_cwd(TMR_ROOT):
            self.tmr_model: TMR = load_model_from_cfg(cfg)

            if model_dir.parts[-1] == "tmr_humanml3d_guoh3dfeats":
                self.mean = torch.load("stats/humanml3d/guoh3dfeats/mean.pt")
                self.std = torch.load("stats/humanml3d/guoh3dfeats/std.pt")
            elif model_dir.parts[-1] == "tmr_kitml_guoh3dfeats":
                self.mean = torch.load("stats/kitml/guoh3dfeats/mean.pt")
                self.std = torch.load("stats/kitml/guoh3dfeats/std.pt")
            else:
                self.mean = torch.load(f"stats/humanml3d_{model_dir.parts[-1]}/guoh3dfeats/mean.pt")
                self.std = torch.load(f"stats/humanml3d_{model_dir.parts[-1]}/guoh3dfeats/std.pt")

        self.latent_dim = 512

        latent_path = PROJECT_ROOT / "third_packages" / "TMR" / "models" / "tmr_humanml3d_guoh3dfeats" / "latents" / "humanml3d_all.npy"
        humanml3d_all = torch.from_numpy(np.load(str(latent_path))).float()
        self.register_buffer("humanml3d_all", humanml3d_all, persistent=False)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def normalize(self, motion):
        return (motion - self.mean.to(motion.device)) / self.std.to(motion.device)

    
    def encode_motion(self, motions):
        device = next(self.parameters()).device

        motion_x_dicts = [{"x": motion, "length": len(motion)} for motion in motions]

        motion_x_dict = self.collate_x_dict(motion_x_dicts, device=str(device))

        latent = cast(Tensor, self.tmr_model.encode(motion_x_dict, sample_mean=True))

        return latent

    def encode_text(self, texts):
        device = next(self.parameters()).device
        text_x_dict = self.collate_x_dict(self.text_model(texts), device=str(device))
        latent = cast(Tensor, self.tmr_model.encode(text_x_dict, sample_mean=True))

        return latent

    def forward(self, x):
        if isinstance(x[0], Tensor):
            return self.encode_motion(cast(list[Tensor], x))
        else:
            return self.encode_text(cast(list[str], x))

    def encode_text_motion(self, texts):
        device = next(self.parameters()).device
        text_x_dict = self.collate_x_dict(self.text_model(texts), device=str(device))
        latent = cast(Tensor, self.tmr_model.encode(text_x_dict, sample_mean=True))

        humanml3d_all_expanded = self.humanml3d_all.unsqueeze(0)
        latent_expanded = latent.unsqueeze(1)

        cos_text_to_motion = self.cos(latent_expanded, humanml3d_all_expanded)

        top_values, top_indices = torch.topk(cos_text_to_motion, k=3, dim=1)

        random_indices = torch.randint(0, 3, (top_indices.size(0),), device=top_indices.device)
        chosen_indices = top_indices[torch.arange(top_indices.size(0)), random_indices]

        ra_latent = self.humanml3d_all[chosen_indices]

        latent = torch.concat([latent, ra_latent], dim=-1)
        latent = self.concat_clip_dim_layer(latent)

        return latent
