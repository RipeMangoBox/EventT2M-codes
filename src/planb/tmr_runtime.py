from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


@dataclass
class TMRRuntime:
    model: Any
    text_model: Any
    normalizer: Any
    collate_x_dict: Any
    device: str
    run_dir: Path
    tmr_root: Path


@contextmanager
def temporary_cwd(destination: Path):
    current = Path.cwd()
    os.chdir(destination)
    try:
        yield
    finally:
        os.chdir(current)


def _sanitize_sys_path(repo_root: Path, tmr_root: Path) -> None:
    repo_root = repo_root.resolve()
    tmr_root = tmr_root.resolve()
    cleaned = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            entry_path = Path(entry).resolve()
        except FileNotFoundError:
            cleaned.append(entry)
            continue
        if entry_path == repo_root:
            continue
        cleaned.append(entry)
    sys.path[:] = [str(tmr_root)] + [entry for entry in cleaned if entry != str(tmr_root)]


def load_tmr_runtime(
    repo_root: str | Path,
    tmr_root: str | Path,
    run_dir: str | Path,
    device: str = "cpu",
) -> TMRRuntime:
    repo_root = Path(repo_root).resolve()
    tmr_root = Path(tmr_root).resolve()
    run_dir = Path(run_dir).resolve()

    _sanitize_sys_path(repo_root, tmr_root)

    import src.prepare  # type: ignore # noqa: F401
    from hydra.utils import instantiate  # type: ignore
    from src.config import read_config  # type: ignore
    from src.data.collate import collate_x_dict  # type: ignore
    from src.load import load_model_from_cfg  # type: ignore

    with temporary_cwd(tmr_root):
        cfg = read_config(str(run_dir))
        cfg.run_dir = str(run_dir)
        model = load_model_from_cfg(cfg, device=device, eval_mode=True)
        text_model = instantiate(cfg.data.text_to_token_emb, device=device)
        normalizer = instantiate(cfg.data.motion_loader.normalizer)

    return TMRRuntime(
        model=model,
        text_model=text_model,
        normalizer=normalizer,
        collate_x_dict=collate_x_dict,
        device=device,
        run_dir=run_dir,
        tmr_root=tmr_root,
    )


@torch.no_grad()
def score_motion_text(runtime: TMRRuntime, raw_motion: np.ndarray, text: str) -> float:
    return _score_motion_text_one(runtime, raw_motion, text)


@torch.no_grad()
def _score_motion_text_one(runtime: TMRRuntime, raw_motion: np.ndarray, text: str) -> float:
    from src.model.tmr import get_score_matrix  # type: ignore

    motion = torch.from_numpy(raw_motion).to(torch.float)
    motion = runtime.normalizer(motion)
    motion_x_dict = {"x": motion, "length": len(motion)}
    motion_x_dict = runtime.collate_x_dict([motion_x_dict], device=runtime.device)

    text_x_dict = runtime.collate_x_dict(runtime.text_model([text]), device=runtime.device)

    lat_m = runtime.model.encode(motion_x_dict, sample_mean=True)[0]
    lat_t = runtime.model.encode(text_x_dict, sample_mean=True)[0]
    score = get_score_matrix(lat_t, lat_m).detach().cpu().item()
    return float(score)


@torch.no_grad()
def score_motion_text_batch(runtime: TMRRuntime, raw_motion: np.ndarray, texts: List[str]) -> List[float]:
    from src.model.tmr import get_score_matrix  # type: ignore

    if not texts:
        return []

    motion = torch.from_numpy(raw_motion).to(torch.float)
    motion = runtime.normalizer(motion)
    motion_x_dict = {"x": motion, "length": len(motion)}
    motion_x_dict = runtime.collate_x_dict([motion_x_dict], device=runtime.device)

    text_x_dict = runtime.collate_x_dict(runtime.text_model(texts), device=runtime.device)

    lat_m = runtime.model.encode(motion_x_dict, sample_mean=True)[0]
    lat_t = runtime.model.encode(text_x_dict, sample_mean=True)[0]
    score_matrix = get_score_matrix(lat_t, lat_m).detach().cpu()
    if score_matrix.ndim == 2 and score_matrix.shape[0] == len(texts):
        return [float(score) for score in score_matrix[:, 0].tolist()]
    if score_matrix.ndim == 1 and score_matrix.shape[0] == len(texts):
        return [float(score) for score in score_matrix.tolist()]
    if len(texts) == 1:
        return [float(score_matrix.item())]
    return [_score_motion_text_one(runtime, raw_motion, text) for text in texts]


def score_motion_texts(runtime: TMRRuntime, raw_motion: np.ndarray, texts: List[str]) -> List[float]:
    return score_motion_text_batch(runtime, raw_motion, texts)


def split_motion_into_k_windows(raw_motion: np.ndarray, k: int) -> List[np.ndarray]:
    if k <= 0:
        return []
    length = raw_motion.shape[0]
    boundaries = np.linspace(0, length, num=k + 1, dtype=int)
    windows: List[np.ndarray] = []
    for idx in range(k):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        if end <= start:
            end = min(length, start + 1)
        windows.append(raw_motion[start:end])
    return windows


def windowed_order_bundle(
    runtime: TMRRuntime,
    raw_motion: np.ndarray,
    events: List[str],
    tau: float = 0.1,
) -> Dict[str, Any]:
    valid_events = [event for event in events if event.strip()]
    k = len(valid_events)
    if k < 2:
        return {
            "event_count": k,
            "window_count": k,
            "pair_scores": [],
            "mean_pair_score": None,
            "centers": [],
        }

    windows = split_motion_into_k_windows(raw_motion, k)
    window_positions = np.arange(k, dtype=np.float32)
    per_event_window_scores: List[List[float]] = []
    centers: List[float] = []
    for event in valid_events:
        scores = np.array([score_motion_text(runtime, window, event) for window in windows], dtype=np.float32)
        logits = scores / tau
        logits = logits - logits.max()
        weights = np.exp(logits)
        weights = weights / weights.sum()
        center = float(np.sum(window_positions * weights))
        per_event_window_scores.append(scores.tolist())
        centers.append(center)

    pair_scores: List[float] = []
    for idx in range(k - 1):
        delta = centers[idx + 1] - centers[idx]
        pair_scores.append(float(1.0 / (1.0 + np.exp(-delta))))

    return {
        "event_count": k,
        "window_count": k,
        "window_scores": per_event_window_scores,
        "centers": centers,
        "pair_scores": pair_scores,
        "mean_pair_score": float(np.mean(pair_scores)) if pair_scores else None,
    }


def load_tmr_stats(tmr_root: str | Path) -> Dict[str, np.ndarray]:
    tmr_root = Path(tmr_root).resolve()
    mean = torch.load(tmr_root / "stats" / "humanml3d" / "guoh3dfeats" / "mean.pt", map_location="cpu").numpy()
    std = torch.load(tmr_root / "stats" / "humanml3d" / "guoh3dfeats" / "std.pt", map_location="cpu").numpy()
    return {"mean": mean, "std": std}


def stats_diff(eventt2m_mean: np.ndarray, eventt2m_std: np.ndarray, tmr_stats: Dict[str, np.ndarray]) -> Dict[str, float]:
    mean_diff = eventt2m_mean - tmr_stats["mean"]
    std_diff = eventt2m_std - tmr_stats["std"]
    return {
        "mean_l2": float(np.linalg.norm(mean_diff)),
        "mean_max_abs": float(np.max(np.abs(mean_diff))),
        "std_l2": float(np.linalg.norm(std_diff)),
        "std_max_abs": float(np.max(np.abs(std_diff))),
    }


def read_manifest_rows(manifest_path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
