from __future__ import annotations

import random
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


DEFAULT_SAMPLE_IDS = ["004965", "008463", "001969", "003245"]


@dataclass
class EventT2MRuntime:
    model: Any
    cfg: DictConfig
    mean: torch.Tensor
    std: torch.Tensor
    device: torch.device
    data_root: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _hydra_value(value: str | Path) -> str:
    return json.dumps(str(Path(value).resolve()))


def load_eventt2m_runtime(
    ckpt_path: str | Path | None = None,
    device: str = "cpu",
    step_num: int | None = None,
    data_dir: str | Path | None = None,
    extra_overrides: Sequence[str] | None = None,
) -> EventT2MRuntime:
    repo_root = _repo_root()
    config_dir = repo_root / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        overrides = []
        if ckpt_path is not None:
            overrides.append(f"ckpt_path={_hydra_value(ckpt_path)}")
        if step_num is not None:
            overrides.append(f"model.step_num={step_num}")
        if data_dir is not None:
            overrides.append(f"data_dir={_hydra_value(data_dir)}")
        if extra_overrides:
            overrides.extend(extra_overrides)
        cfg = compose(config_name="sample_motion.yaml", overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)

    model = hydra.utils.instantiate(cfg.model)
    state_dict = torch.load(cfg.ckpt_path, map_location="cpu")["state_dict"]
    for key in list(state_dict.keys()):
        if "orig_mod." in key:
            state_dict[key.replace("_orig_mod.", "")] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)

    runtime_device = torch.device(device)
    model = model.to(runtime_device)
    model.eval()

    data_root = Path(cfg.data_dir).resolve()
    mean = torch.from_numpy(np.load(data_root / "Mean.npy")).float().to(runtime_device)
    std = torch.from_numpy(np.load(data_root / "Std.npy")).float().to(runtime_device)

    return EventT2MRuntime(
        model=model,
        cfg=cfg,
        mean=mean,
        std=std,
        device=runtime_device,
        data_root=data_root,
    )


def load_hml3de_test_dict(data_file: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    repo_root = _repo_root()
    if data_file is None:
        data_file = repo_root / "dataset" / "HumanML3D-E" / "data_test.npy"
    return np.load(Path(data_file), allow_pickle=True).item()


def pick_text_entry(sample: Dict[str, Any]) -> Dict[str, Any]:
    entries = sample["text"]
    best = None
    best_len = -1
    for entry in entries:
        decomposed = entry.get("decomposed", [])
        if len(decomposed) > best_len:
            best = entry
            best_len = len(decomposed)
    if best is None:
        raise ValueError("Sample has no text entries")
    return best


def extract_events(text_entry: Dict[str, Any]) -> List[str]:
    events = [item["caption"] for item in text_entry.get("decomposed", []) if item.get("caption", "").strip()]
    return events


def select_sample_ids(
    data_dict: Dict[str, Dict[str, Any]],
    sample_ids: Iterable[str] | None = None,
    max_samples: int | None = None,
    min_events: int = 3,
) -> List[str]:
    if sample_ids:
        chosen = [sample_id for sample_id in sample_ids if sample_id in data_dict]
    else:
        chosen = []
        for sample_id in DEFAULT_SAMPLE_IDS:
            if sample_id in data_dict:
                entry = pick_text_entry(data_dict[sample_id])
                if len(extract_events(entry)) >= min_events:
                    chosen.append(sample_id)
        if not chosen:
            for sample_id, sample in data_dict.items():
                entry = pick_text_entry(sample)
                if len(extract_events(entry)) >= min_events:
                    chosen.append(sample_id)
    if max_samples is not None:
        chosen = chosen[:max_samples]
    return chosen


def build_event_pool(data_dict: Dict[str, Dict[str, Any]]) -> List[str]:
    pool: List[str] = []
    seen = set()
    for sample in data_dict.values():
        for text_entry in sample["text"]:
            for event in extract_events(text_entry):
                if event not in seen:
                    seen.add(event)
                    pool.append(event)
    return pool


def choose_target_index(events: List[str]) -> int:
    if not events:
        raise ValueError("Empty event list")
    if len(events) == 1:
        return 0
    return min(1, len(events) - 1)


def choose_distractor(source_event: str, pool: List[str]) -> str:
    source_len = len(source_event.split())
    candidates = [
        event
        for event in pool
        if event != source_event and abs(len(event.split()) - source_len) <= 2
    ]
    if not candidates:
        candidates = [event for event in pool if event != source_event]
    if not candidates:
        raise ValueError("No distractor candidates available")
    return random.choice(candidates)


def corrupt_events(events: List[str], corruption: str, event_pool: List[str]) -> Dict[str, Any]:
    corrupted = list(events)
    target_idx = choose_target_index(events)

    if corruption == "full":
        return {"events": corrupted, "target_idx": None}
    if corruption == "drop":
        corrupted[target_idx] = ""
        return {"events": corrupted, "target_idx": target_idx}
    if corruption == "swap":
        if len(corrupted) < 2:
            return {"events": corrupted, "target_idx": None}
        swap_idx = min(target_idx, len(corrupted) - 2)
        corrupted[swap_idx], corrupted[swap_idx + 1] = corrupted[swap_idx + 1], corrupted[swap_idx]
        return {"events": corrupted, "target_idx": swap_idx}
    if corruption == "replace":
        corrupted[target_idx] = choose_distractor(events[target_idx], event_pool)
        return {"events": corrupted, "target_idx": target_idx}
    raise ValueError(f"Unknown corruption: {corruption}")


@torch.no_grad()
def generate_raw_motion(
    runtime: EventT2MRuntime,
    caption: str,
    events: List[str],
    length: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    motion = torch.zeros((1, int(length), 263), device=runtime.device)
    lens = torch.tensor([int(length)], dtype=torch.long, device=runtime.device)
    generated = runtime.model.sample_motion(motion, lens, [caption], [events])
    generated_raw = generated * runtime.std + runtime.mean
    return generated_raw[0].detach().cpu().numpy()


def recover_joints(raw_motion: np.ndarray) -> np.ndarray:
    try:
        from src.data.humanml.scripts.motion_process import recover_from_ric
    except ModuleNotFoundError:
        motion_process_path = _repo_root() / "src" / "data" / "humanml" / "scripts" / "motion_process.py"
        spec = importlib.util.spec_from_file_location("eventt2m_motion_process_runtime", motion_process_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        recover_from_ric = module.recover_from_ric

    tensor = torch.from_numpy(raw_motion).float().unsqueeze(0)
    joints = recover_from_ric(tensor, 22)[0].cpu().numpy()
    return joints


def motion_diff(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    diff = candidate - reference
    return {
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "l2_diff": float(np.linalg.norm(diff)),
        "max_abs_diff": float(np.max(np.abs(diff))),
    }
