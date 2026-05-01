from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf


HUMANML3D_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]


@dataclass
class MotionPatchesRuntime:
    model: Any
    tokenizer: Any
    device: str
    checkpoint_dir: Path
    repo_root: Path
    data_root: Path
    mean_raw: np.ndarray
    std_raw: np.ndarray
    patch_size: int
    max_motion_length: int
    kinematic_chain: List[List[int]]


@contextmanager
def temporary_cwd(destination: Path):
    current = Path.cwd()
    os.chdir(destination)
    try:
        yield
    finally:
        os.chdir(current)


def _sanitize_sys_path(repo_root: Path, mp_root: Path) -> None:
    repo_root = repo_root.resolve()
    mp_root = mp_root.resolve()
    cleaned: List[str] = []
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
    sys.path[:] = [str(mp_root)] + [entry for entry in cleaned if entry != str(mp_root)]


def _resolve_data_root(mp_root: Path, data_root_value: str) -> Path:
    data_root = Path(data_root_value)
    if data_root.is_absolute():
        return data_root
    return (mp_root / data_root).resolve()


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY_TEMPORAL_POOLING_MISSING_KEYS = {
    "temporal_pool_norm.weight",
    "temporal_pool_norm.bias",
    "temporal_time_score.weight",
    "temporal_time_score.bias",
}


def _load_checkpoint_state_dict(model, model_path: Path, device: str):
    checkpoint_state = torch.load(model_path, map_location=device)
    incompatible = model.load_state_dict(checkpoint_state, strict=False)

    missing_keys = set(incompatible.missing_keys)
    unexpected_keys = set(incompatible.unexpected_keys)
    model.loaded_checkpoint_missing_keys = sorted(missing_keys)
    model.loaded_checkpoint_unexpected_keys = sorted(unexpected_keys)

    if unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys while loading {model_path}: {sorted(unexpected_keys)}"
        )
    if missing_keys:
        if bool(getattr(model, "enable_temporal_adapter", False)) and missing_keys == LEGACY_TEMPORAL_POOLING_MISSING_KEYS:
            model.legacy_temporal_pooling = True
        else:
            raise RuntimeError(
                f"Missing checkpoint keys while loading {model_path}: {sorted(missing_keys)}"
            )
    return model


def load_motionpatches_runtime(
    repo_root: str | Path,
    mp_root: str | Path,
    checkpoint_dir: str | Path,
    device: str = "cpu",
) -> MotionPatchesRuntime:
    repo_root = Path(repo_root).resolve()
    mp_root = Path(mp_root).resolve()
    checkpoint_dir = Path(checkpoint_dir).resolve()

    _sanitize_sys_path(repo_root, mp_root)

    clip_module = _load_module_from_path("motionpatches_clip_runtime", mp_root / "models" / "clip.py")
    ClipModel = clip_module.ClipModel
    build_text_tokenizer = clip_module.build_text_tokenizer

    cfg = OmegaConf.load(checkpoint_dir / ".hydra" / "config.yaml")
    temporal_adapter_cfg = cfg.train.temporal_adapter
    event_patch_alignment_cfg = cfg.train.event_patch_alignment
    tmr_transfer_cfg = cfg.train.get("tmr_transfer", {})
    temporal_cfg = cfg.train.get("event_temporal", {})
    motion_attention_cfg = cfg.model.get("motion_attention", {})

    tokenizer = build_text_tokenizer(
        str(cfg.model.text_encoder),
        base_dir=str(mp_root),
    )
    model = ClipModel(
        motion_encoder_alias=str(cfg.model.motion_encoder),
        text_encoder_alias=str(cfg.model.text_encoder),
        text_encoder_base_dir=str(mp_root),
        motion_embedding_dims=768,
        text_embedding_dims=768,
        projection_dims=256,
        patch_size=int(cfg.train.patch_size),
        enable_temporal_adapter=bool(temporal_adapter_cfg.enable),
        temporal_adapter_layers=int(temporal_adapter_cfg.layers),
        temporal_adapter_heads=int(temporal_adapter_cfg.num_heads),
        temporal_adapter_dropout=float(temporal_adapter_cfg.dropout),
        temporal_adapter_use_cls_token=bool(temporal_adapter_cfg.use_cls_token),
        use_temporal_adapter_for_inference=bool(temporal_adapter_cfg.use_inference),
        enable_event_patch_alignment=bool(event_patch_alignment_cfg.enable),
        event_patch_alignment_prior_strength=float(event_patch_alignment_cfg.prior_strength),
        event_patch_alignment_prior_sigma=float(event_patch_alignment_cfg.prior_sigma),
        event_patch_alignment_use_content_similarity=bool(
            event_patch_alignment_cfg.use_content_similarity
        ),
        enable_tmr_transfer=bool(tmr_transfer_cfg.get("enable", False)),
        tmr_transfer_proj_dim=int(tmr_transfer_cfg.get("tmr_transfer_proj_dim", 256)),
        tmr_transfer_event_align_tau=float(tmr_transfer_cfg.get("event_align_tau", 0.1)),
        enable_tmr_event_head=bool(temporal_cfg.get("tmr_event_head", {}).get("enable", False)),
        tmr_event_head_proj_dim=int(temporal_cfg.get("tmr_event_head", {}).get("proj_dim", 256)),
        tmr_event_head_tau=float(temporal_cfg.get("tmr_event_head", {}).get("tau", 0.1)),
        motion_attention_mode=str(motion_attention_cfg.get("mode", "disabled")),
    )

    _load_checkpoint_state_dict(model, checkpoint_dir / "best_model.pt", device)
    model = model.to(device)
    model.eval()

    data_root = _resolve_data_root(mp_root, str(cfg.dataset.data_root))
    mean_raw = np.load(data_root / "Mean_raw.npy")
    std_raw = np.load(data_root / "Std_raw.npy")

    return MotionPatchesRuntime(
        model=model,
        tokenizer=tokenizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        repo_root=mp_root,
        data_root=data_root,
        mean_raw=mean_raw,
        std_raw=std_raw,
        patch_size=int(cfg.train.patch_size),
        max_motion_length=int(cfg.dataset.max_motion_length),
        kinematic_chain=HUMANML3D_KINEMATIC_CHAIN,
    )


def joints_to_motionpatches_tensor(
    runtime: MotionPatchesRuntime,
    joints: np.ndarray,
) -> torch.Tensor:
    motion = joints.astype(np.float32)
    motion = (motion - runtime.mean_raw[np.newaxis, ...]) / runtime.std_raw[np.newaxis, ...]

    if runtime.patch_size != 16:
        raise NotImplementedError("Only patch_size=16 is currently supported")

    motion_kin = np.zeros(
        (motion.shape[0], len(runtime.kinematic_chain) * 16, motion.shape[2]),
        dtype=np.float32,
    )
    for frame_idx in range(motion.shape[0]):
        for chain_idx, chain in enumerate(runtime.kinematic_chain):
            joint_parts = motion[frame_idx, chain]
            joint_parts = joint_parts.reshape(1, -1, 3)
            joint_parts = cv2.resize(joint_parts, (16, 1), interpolation=cv2.INTER_LINEAR)
            motion_kin[frame_idx, 16 * chain_idx : 16 * (chain_idx + 1)] = joint_parts[0]

    motion_len = motion_kin.shape[0]
    max_len = runtime.max_motion_length
    if motion_len >= max_len:
        motion_kin = motion_kin[:max_len]
    else:
        pad_len = max_len - motion_len
        pad = np.zeros((pad_len, motion_kin.shape[1], motion_kin.shape[2]), dtype=np.float32)
        motion_kin = np.concatenate([motion_kin, pad], axis=0)

    tensor = torch.from_numpy(motion_kin).float().permute(2, 0, 1).unsqueeze(0)
    return tensor.to(runtime.device)


def encode_event_embeddings(
    runtime: MotionPatchesRuntime,
    events: List[str],
    caption: str,
    mode: str = "independent",
) -> np.ndarray:
    from structured_rerank import encode_events_batch  # type: ignore

    embs_list = encode_events_batch(
        [events],
        [caption],
        runtime.model,
        runtime.tokenizer,
        runtime.device,
        mode=mode,
    )
    return embs_list[0]


def structured_temporal_scores(
    runtime: MotionPatchesRuntime,
    joints: np.ndarray,
    events: List[str],
    caption: str,
    *,
    event_encode_mode: str = "independent",
    dp_mode: str = "strict",
) -> Dict[str, Any]:
    from structured_rerank import classify_sample, compute_structured_score  # type: ignore

    valid_events = [event for event in events if event.strip()]
    if not valid_events:
        return {
            "sample_type": "single",
            "structured_score": 0.0,
            "reverse_order_margin": 0.0,
            "event_count": 0,
        }

    motion_tensor = joints_to_motionpatches_tensor(runtime, joints)
    with torch.no_grad():
        time_tokens = runtime.model.encode_motion_time_tokens(motion_tensor)[0].cpu().numpy()

    event_embs = encode_event_embeddings(
        runtime,
        valid_events,
        caption,
        mode=event_encode_mode,
    )
    sample_type = classify_sample(valid_events, caption)
    structured_score = compute_structured_score(
        event_embs,
        time_tokens,
        sample_type,
        dp_mode=dp_mode,
    )

    reverse_margin = 0.0
    if sample_type == "ordered" and len(valid_events) >= 2:
        reverse_score = compute_structured_score(
            event_embs[::-1],
            time_tokens,
            sample_type,
            dp_mode=dp_mode,
        )
        reverse_margin = float(structured_score - reverse_score)

    return {
        "sample_type": sample_type,
        "structured_score": float(structured_score),
        "reverse_order_margin": float(reverse_margin),
        "event_count": len(valid_events),
    }


def read_manifest_rows(manifest_path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
