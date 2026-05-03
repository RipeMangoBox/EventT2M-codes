from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from planb.eventt2m_runtime import extract_events, pick_text_entry
from planb.tmr_runtime import load_tmr_runtime


CORRUPTIONS = ("drop", "replace", "shuffle")


@dataclass
class PairBatch:
    motion: torch.Tensor
    full: torch.Tensor
    drop: torch.Tensor
    replace: torch.Tensor
    shuffle: torch.Tensor
    masks: torch.Tensor
    mask_valid: torch.Tensor


class RewardHead(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, motion_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([motion_emb, text_emb], dim=-1)
        return self.net(x).squeeze(-1)


class PairDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tensors: Dict[str, torch.Tensor]) -> None:
        self.rows = rows
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        key = row["row_id"]
        return {
            "motion": self.tensors["motion"][key],
            "full": self.tensors["text"][row["text_keys"]["full"]],
            "drop": self.tensors["text"][row["text_keys"]["drop"]],
            "replace": self.tensors["text"][row["text_keys"]["replace"]],
            "shuffle": self.tensors["text"][row["text_keys"]["shuffle"]],
            "masks": torch.stack([self.tensors["text"][text_key] for text_key in row["text_keys"]["masks"]]),
            "mask_valid": torch.ones(len(row["text_keys"]["masks"]), dtype=torch.bool),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoDebug S2a frozen-TMR reward head sanity probe.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--tmr-root", type=str, default=None)
    parser.add_argument("--tmr-run-dir", type=str, default=None)
    parser.add_argument("--train-data", type=str, default="dataset/HumanML3D-E/data_train.npy")
    parser.add_argument("--val-data", type=str, default="dataset/HumanML3D-E/data_val.npy")
    parser.add_argument("--train-events-json", type=str, default="dataset/HumanML3D-E/.tamr_hml3de_gt_events_train.json")
    parser.add_argument("--val-events-json", type=str, default="dataset/HumanML3D-E/.tamr_hml3de_gt_events_val.json")
    parser.add_argument("--output-dir", type=str, default="logs/modebug_reward_s2a_probe")
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-val-samples", type=int, default=1000)
    parser.add_argument("--min-events", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--lambda-event-mask", type=float, default=0.5)
    parser.add_argument("--length-window", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-checkpoint", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_repo_root(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    return Path(__file__).resolve().parents[1]


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def resolve_tmr_root(repo_root: Path, value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    candidates = [repo_root / "third_packages" / "TMR", repo_root.parent / "TMR"]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def events_to_text(events: Sequence[str]) -> str:
    return " ".join(event.strip() for event in events if event.strip())


def bucket_name(event_count: int) -> str:
    return str(event_count) if event_count < 5 else "5plus"


def stable_hash(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_target_index(events: Sequence[str]) -> int:
    if len(events) <= 1:
        return 0
    return min(1, len(events) - 1)


def shuffle_events(events: Sequence[str], seed: int, sample_id: str) -> Tuple[List[str], List[int], bool]:
    indices = list(range(len(events)))
    if len(indices) < 2:
        return list(events), indices, False
    indices.sort(key=lambda idx: stable_hash({"seed": seed, "sample_id": sample_id, "idx": idx, "event": events[idx]}))
    rotated_identity = False
    if indices == list(range(len(events))):
        indices = indices[1:] + indices[:1]
        rotated_identity = True
    return [events[idx] for idx in indices], indices, rotated_identity


def build_event_pool(data_dict: Dict[str, Dict[str, Any]]) -> List[str]:
    pool = set()
    for sample in data_dict.values():
        for text_entry in sample["text"]:
            for event in extract_events(text_entry):
                event = event.strip()
                if event:
                    pool.add(event)
    return sorted(pool, key=lambda event: (event.lower(), event))


def build_event_pool_by_length(event_pool: Sequence[str]) -> Dict[int, List[str]]:
    by_length: Dict[int, List[str]] = {}
    for event in event_pool:
        by_length.setdefault(len(event.split()), []).append(event)
    return by_length


def choose_replacement(
    events: Sequence[str],
    target_idx: int,
    event_pool: Sequence[str],
    event_pool_by_length: Dict[int, List[str]],
    seed: int,
    sample_id: str,
    length_window: int,
) -> Tuple[str, str, int]:
    target_event = events[target_idx]
    source_set = set(events)
    target_len = len(target_event.split())
    length_candidates: List[str] = []
    for length in range(max(0, target_len - length_window), target_len + length_window + 1):
        length_candidates.extend(event_pool_by_length.get(length, []))
    candidates = [event for event in length_candidates if event not in source_set]
    stage = f"global_pool_not_in_source_len_window_{length_window}"
    if not candidates:
        candidates = [event for event in event_pool if event not in source_set]
        stage = "global_pool_not_in_source_fallback_any_length"
    if not candidates:
        raise ValueError(f"No replacement candidate for {sample_id}")
    replacement = min(
        candidates,
        key=lambda event: stable_hash(
            {
                "seed": seed,
                "sample_id": sample_id,
                "target_idx": target_idx,
                "target_event": target_event,
                "candidate": event,
            }
        ),
    )
    return replacement, stage, len(candidates)


def read_data(path: Path) -> Dict[str, Dict[str, Any]]:
    return np.load(path, allow_pickle=True).item()


def select_rows(
    data_dict: Dict[str, Dict[str, Any]],
    *,
    split: str,
    max_samples: int,
    min_events: int,
    seed: int,
    length_window: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    event_pool = build_event_pool(data_dict)
    event_pool_by_length = build_event_pool_by_length(event_pool)
    rows: List[Dict[str, Any]] = []
    skipped = {"too_few_events": 0, "no_replacement": 0}
    for sample_id in sorted(data_dict):
        sample = data_dict[sample_id]
        text_entry = pick_text_entry(sample)
        events = [event.strip() for event in extract_events(text_entry) if event.strip()]
        if len(events) < min_events:
            skipped["too_few_events"] += 1
            continue
        target_idx = choose_target_index(events)
        drop_events = list(events)
        dropped_event = drop_events.pop(target_idx)
        replace_events = list(events)
        try:
            replacement_event, replacement_stage, replacement_candidates = choose_replacement(
                events, target_idx, event_pool, event_pool_by_length, seed, sample_id, length_window
            )
        except ValueError:
            skipped["no_replacement"] += 1
            continue
        replace_events[target_idx] = replacement_event
        shuffled_events, shuffle_permutation, rotated_identity = shuffle_events(events, seed, sample_id)
        mask_texts = []
        for idx in range(len(events)):
            mask_events = list(events)
            mask_events.pop(idx)
            mask_texts.append(events_to_text(mask_events))

        event_count = len(events)
        rows.append(
            {
                "row_id": f"{split}:{sample_id}",
                "split": split,
                "sample_id": sample_id,
                "target_idx": target_idx,
                "event_count": event_count,
                "event_count_bucket": bucket_name(event_count),
                "caption": text_entry.get("caption", events_to_text(events)),
                "source_events": events,
                "full_text": events_to_text(events),
                "drop_text": events_to_text(drop_events),
                "replace_text": events_to_text(replace_events),
                "shuffle_text": events_to_text(shuffled_events),
                "mask_texts": mask_texts,
                "dropped_event": dropped_event,
                "replacement_event": replacement_event,
                "replacement_policy": {
                    "name": "modebug_s2a_replace_v1",
                    "stage": replacement_stage,
                    "candidate_count": replacement_candidates,
                    "length_window": length_window,
                    "selection": "min sha256(seed, sample_id, target_idx, target_event, candidate)",
                },
                "shuffle_policy": {
                    "name": "modebug_s2a_hash_shuffle_v1",
                    "permutation": shuffle_permutation,
                    "identity_rotated": rotated_identity,
                },
            }
        )
        if max_samples > 0 and len(rows) >= max_samples:
            break
    summary = {
        "split": split,
        "data_rows": len(data_dict),
        "selected_rows": len(rows),
        "skipped": skipped,
        "event_count_buckets": bucket_counts(rows),
        "event_pool_size": len(event_pool),
    }
    return rows, summary


def bucket_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        bucket = str(row["event_count_bucket"])
        counts[bucket] = counts.get(bucket, 0) + 1
    return dict(sorted(counts.items()))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def attach_text_keys(rows: List[Dict[str, Any]]) -> List[str]:
    all_keys: List[str] = []
    for row in rows:
        keys = {
            "full": text_key(row["full_text"]),
            "drop": text_key(row["drop_text"]),
            "replace": text_key(row["replace_text"]),
            "shuffle": text_key(row["shuffle_text"]),
            "masks": [text_key(text) for text in row["mask_texts"]],
        }
        row["text_keys"] = keys
        all_keys.extend([keys["full"], keys["drop"], keys["replace"], keys["shuffle"]])
        all_keys.extend(keys["masks"])
    return all_keys


def collect_texts(rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    for row in rows:
        for field in ("full_text", "drop_text", "replace_text", "shuffle_text"):
            texts[text_key(row[field])] = row[field]
        for text in row["mask_texts"]:
            texts[text_key(text)] = text
    return texts


@torch.no_grad()
def encode_latents(model: Any, x_dict: Any) -> torch.Tensor:
    latents = model.encode(x_dict, sample_mean=True)
    if isinstance(latents, tuple):
        latents = latents[0]
    if latents.ndim == 1:
        latents = latents.unsqueeze(0)
    return latents


@torch.no_grad()
def encode_motion(runtime: Any, raw_motion: np.ndarray) -> torch.Tensor:
    motion = torch.from_numpy(raw_motion).to(torch.float)
    motion = runtime.normalizer(motion)
    motion_x_dict = {"x": motion, "length": len(motion)}
    motion_x_dict = runtime.collate_x_dict([motion_x_dict], device=runtime.device)
    latent = encode_latents(runtime.model, motion_x_dict)[0]
    return latent.detach().cpu().float()


@torch.no_grad()
def encode_texts(runtime: Any, texts: Sequence[str], batch_size: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    items = [(text_key(text), text) for text in texts]
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        text_x_dict = runtime.collate_x_dict(runtime.text_model([text for _, text in batch]), device=runtime.device)
        latents = encode_latents(runtime.model, text_x_dict).detach().cpu().float()
        for idx, (key, _text) in enumerate(batch):
            out[key] = latents[idx]
    return out


def precompute_embeddings(
    runtime: Any,
    rows: Sequence[Dict[str, Any]],
    data_by_split: Dict[str, Dict[str, Dict[str, Any]]],
    texts: Dict[str, str],
    batch_size: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    runtime.model.eval()
    motion_tensors: Dict[str, torch.Tensor] = {}
    for idx, row in enumerate(rows, start=1):
        sample = data_by_split[row["split"]][row["sample_id"]]
        motion_tensors[row["row_id"]] = encode_motion(runtime, sample["motion"])
        if idx == 1 or idx % 500 == 0 or idx == len(rows):
            print(f"[S2A] encoded_motion {idx}/{len(rows)}", flush=True)
    text_tensors = encode_texts(runtime, list(texts.values()), batch_size=batch_size)
    print(f"[S2A] encoded_texts {len(text_tensors)}", flush=True)
    return {"motion": motion_tensors, "text": text_tensors}


def score_matrix(motion_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    cos = F.cosine_similarity(motion_emb, text_emb, dim=-1)
    return cos / 2.0 + 0.5


def collate_pairs(batch: List[Dict[str, torch.Tensor]]) -> PairBatch:
    max_masks = max(item["masks"].shape[0] for item in batch)
    emb_dim = batch[0]["motion"].shape[-1]
    masks = torch.zeros(len(batch), max_masks, emb_dim, dtype=batch[0]["masks"].dtype)
    mask_valid = torch.zeros(len(batch), max_masks, dtype=torch.bool)
    for idx, item in enumerate(batch):
        count = item["masks"].shape[0]
        masks[idx, :count] = item["masks"]
        mask_valid[idx, :count] = True
    return PairBatch(
        motion=torch.stack([item["motion"] for item in batch]),
        full=torch.stack([item["full"] for item in batch]),
        drop=torch.stack([item["drop"] for item in batch]),
        replace=torch.stack([item["replace"] for item in batch]),
        shuffle=torch.stack([item["shuffle"] for item in batch]),
        masks=masks,
        mask_valid=mask_valid,
    )


def move_batch(batch: PairBatch, device: torch.device) -> PairBatch:
    return PairBatch(
        motion=batch.motion.to(device),
        full=batch.full.to(device),
        drop=batch.drop.to(device),
        replace=batch.replace.to(device),
        shuffle=batch.shuffle.to(device),
        masks=batch.masks.to(device),
        mask_valid=batch.mask_valid.to(device),
    )


def train_one_epoch(
    model: RewardHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    margin: float,
    lambda_event_mask: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_rank = 0.0
    total_mask = 0.0
    batches = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        full_score = model(batch.motion, batch.full)
        drop_score = model(batch.motion, batch.drop)
        rank_loss = F.relu(margin - (full_score - drop_score)).mean()

        bsz, max_masks, emb_dim = batch.masks.shape
        motion_rep = batch.motion[:, None, :].expand(bsz, max_masks, emb_dim).reshape(bsz * max_masks, emb_dim)
        mask_scores = model(motion_rep, batch.masks.reshape(bsz * max_masks, emb_dim)).reshape(bsz, max_masks)
        mask_losses = F.relu(margin - (full_score[:, None] - mask_scores))
        event_mask_loss = mask_losses[batch.mask_valid].mean()

        loss = rank_loss + lambda_event_mask * event_mask_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_rank += float(rank_loss.detach().cpu())
        total_mask += float(event_mask_loss.detach().cpu())
        batches += 1
    return {
        "loss": total_loss / max(batches, 1),
        "rank_loss": total_rank / max(batches, 1),
        "event_mask_loss": total_mask / max(batches, 1),
    }


@torch.no_grad()
def evaluate_model(model: RewardHead, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    deltas = {"drop": [], "replace": [], "shuffle": [], "mask": []}
    for batch in loader:
        batch = move_batch(batch, device)
        full_score = model(batch.motion, batch.full)
        for name in ("drop", "replace", "shuffle"):
            corrupt = getattr(batch, name)
            corrupt_score = model(batch.motion, corrupt)
            deltas[name].append((full_score - corrupt_score).detach().cpu())
        bsz, max_masks, emb_dim = batch.masks.shape
        motion_rep = batch.motion[:, None, :].expand(bsz, max_masks, emb_dim).reshape(bsz * max_masks, emb_dim)
        mask_scores = model(motion_rep, batch.masks.reshape(bsz * max_masks, emb_dim)).reshape(bsz, max_masks)
        mask_delta = (full_score[:, None] - mask_scores)[batch.mask_valid]
        deltas["mask"].append(mask_delta.detach().cpu())
    return summarize_deltas(deltas)


@torch.no_grad()
def evaluate_cosine(rows: Sequence[Dict[str, Any]], tensors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
    deltas = {"drop": [], "replace": [], "shuffle": [], "mask": []}
    for row in rows:
        motion = tensors["motion"][row["row_id"]].unsqueeze(0)
        full = tensors["text"][row["text_keys"]["full"]].unsqueeze(0)
        full_score = score_matrix(motion, full)
        for name in ("drop", "replace", "shuffle"):
            corrupt = tensors["text"][row["text_keys"][name]].unsqueeze(0)
            deltas[name].append((full_score - score_matrix(motion, corrupt)).cpu())
        for mask_key in row["text_keys"]["masks"]:
            mask = tensors["text"][mask_key].unsqueeze(0)
            deltas["mask"].append((full_score - score_matrix(motion, mask)).cpu())
    return summarize_deltas(deltas)


def summarize_deltas(deltas: Dict[str, List[torch.Tensor]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name, chunks in deltas.items():
        if not chunks:
            continue
        values = torch.cat([chunk.flatten() for chunk in chunks]).float()
        metrics[f"{name}_paired_acc"] = float((values > 0).float().mean().item())
        metrics[f"{name}_mean_delta"] = float(values.mean().item())
        metrics[f"{name}_n"] = int(values.numel())
    return metrics


def canonical_dev_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, Any]:
    mapping = {
        f"{prefix}_gt_pres_full_vs_drop_paired_acc": metrics.get("drop_paired_acc"),
        f"{prefix}_gt_pres_full_vs_replace_paired_acc": metrics.get("replace_paired_acc"),
        f"{prefix}_gt_ord_full_vs_shuffle_paired_acc": metrics.get("shuffle_paired_acc"),
        f"{prefix}_gt_pres_full_vs_event_mask_paired_acc": metrics.get("mask_paired_acc"),
    }
    return {
        name: {
            "value": value,
            "role": "dev_metric",
            "used_for": "selection" if "reward_dev" in prefix else "baseline",
        }
        for name, value in mapping.items()
        if value is not None
    }


def improvement_ci(values: Sequence[float]) -> Dict[str, float | None]:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) < 2:
        return {"mean": mean, "ci95_low": None, "ci95_high": None}
    stderr = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    half = 1.96 * stderr
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    repo_root = resolve_repo_root(args.repo_root)
    output_dir = resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data_path = resolve_path(repo_root, args.train_data)
    val_data_path = resolve_path(repo_root, args.val_data)
    print(f"[S2A] seed={args.seed} device={args.device} output_dir={output_dir}", flush=True)
    print(f"[S2A] loading_data train={train_data_path} val={val_data_path}", flush=True)
    train_data = read_data(train_data_path)
    val_data = read_data(val_data_path)

    print("[S2A] selecting_rows", flush=True)
    train_rows, train_manifest_summary = select_rows(
        train_data,
        split="train",
        max_samples=args.max_train_samples,
        min_events=args.min_events,
        seed=args.seed,
        length_window=args.length_window,
    )
    val_rows, val_manifest_summary = select_rows(
        val_data,
        split="val",
        max_samples=args.max_val_samples,
        min_events=args.min_events,
        seed=args.seed,
        length_window=args.length_window,
    )
    if not train_rows or not val_rows:
        raise RuntimeError("S2a probe requires non-empty train and val rows.")
    print(
        f"[S2A] selected_rows train={len(train_rows)} val={len(val_rows)} "
        f"train_buckets={train_manifest_summary['event_count_buckets']} "
        f"val_buckets={val_manifest_summary['event_count_buckets']}",
        flush=True,
    )

    rows = train_rows + val_rows
    attach_text_keys(rows)
    texts = collect_texts(rows)
    print(f"[S2A] unique_texts={len(texts)}", flush=True)

    tmr_root = resolve_tmr_root(repo_root, args.tmr_root)
    tmr_run_dir = Path(args.tmr_run_dir).resolve() if args.tmr_run_dir else tmr_root / "models" / "tmr_humanml3d_guoh3dfeats"
    print(f"[S2A] loading_tmr run_dir={tmr_run_dir}", flush=True)
    runtime = load_tmr_runtime(repo_root=repo_root, tmr_root=tmr_root, run_dir=tmr_run_dir, device=args.device)

    print("[S2A] precomputing_embeddings", flush=True)
    tensors = precompute_embeddings(
        runtime,
        rows,
        {"train": train_data, "val": val_data},
        texts,
        batch_size=args.embedding_batch_size,
    )
    emb_dim = int(next(iter(tensors["motion"].values())).shape[-1])
    train_tensors = {
        "motion": {row["row_id"]: tensors["motion"][row["row_id"]] for row in train_rows},
        "text": tensors["text"],
    }
    val_tensors = {
        "motion": {row["row_id"]: tensors["motion"][row["row_id"]] for row in val_rows},
        "text": tensors["text"],
    }

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        PairDataset(train_rows, train_tensors),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
        num_workers=args.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        PairDataset(val_rows, val_tensors),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pairs,
        num_workers=args.num_workers,
    )

    train_device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = RewardHead(emb_dim=emb_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(train_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    print(f"[S2A] training epochs={args.epochs} batch_size={args.batch_size}", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            train_device,
            margin=args.margin,
            lambda_event_mask=args.lambda_event_mask,
        )
        val_metrics = evaluate_model(model, val_loader, train_device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)

    cosine_val = evaluate_cosine(val_rows, val_tensors)
    mlp_val = history[-1]["val"] if history else evaluate_model(model, val_loader, train_device)
    drop_improvement = mlp_val["drop_paired_acc"] - cosine_val["drop_paired_acc"]
    go_no_go = {
        "same_protocol_cosine_drop_paired_acc": cosine_val["drop_paired_acc"],
        "mlp_drop_paired_acc": mlp_val["drop_paired_acc"],
        "drop_improvement": drop_improvement,
        "meets_single_seed_2pp_rule": bool(drop_improvement >= 0.02),
        "historical_reference_floor": {
            "metric": "tmr_gt_pres_full_vs_drop_paired_acc",
            "value": 0.7044,
            "role": "side_signal",
            "not_same_protocol_statistical_test": True,
            "review_if_mlp_leq": 0.7244,
            "requires_review": bool(mlp_val["drop_paired_acc"] <= 0.7244),
        },
        "note": "Final S2a go/no-go requires aggregating three seeds and checking 95% CI of improvement > 0.",
    }

    manifest = {
        "task_id": "MDBG-S2A-REWARD-PROBE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "dev_metric",
        "repo_root": str(repo_root),
        "inputs": {
            "train_data": str(train_data_path),
            "val_data": str(val_data_path),
            "train_events_json": str(resolve_path(repo_root, args.train_events_json)),
            "val_events_json": str(resolve_path(repo_root, args.val_events_json)),
            "event_source": "data_*.npy text entry with longest decomposed list",
            "tmr_root": str(tmr_root),
            "tmr_run_dir": str(tmr_run_dir),
        },
        "selection": {
            "seed": args.seed,
            "min_events": args.min_events,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "length_window": args.length_window,
        },
        "row_counts": {
            "train": train_manifest_summary,
            "val": val_manifest_summary,
            "unique_texts": len(texts),
            "embedding_dim": emb_dim,
        },
        "model": {
            "architecture": "frozen TMR embeddings + MLP reward head",
            "input": "concat(motion_embedding, text_embedding)",
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "margin": args.margin,
            "lambda_event_mask": args.lambda_event_mask,
            "rank_loss_corruption": "drop only",
            "replace_shuffle_usage": "eval only",
        },
    }

    train_rows_path = output_dir / "s2a_rows_train.jsonl"
    val_rows_path = output_dir / "s2a_rows_val.jsonl"
    write_json(output_dir / "s2a_manifest.json", manifest)
    write_jsonl(train_rows_path, train_rows)
    write_jsonl(val_rows_path, val_rows)
    summary = {
        "task_id": "MDBG-S2A-REWARD-PROBE-SUMMARY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "dev_metric",
        "manifest": str(output_dir / "s2a_manifest.json"),
        "rows": {
            "train": str(train_rows_path),
            "val": str(val_rows_path),
        },
        "cosine_val": cosine_val,
        "mlp_val": mlp_val,
        "canonical_metrics": {
            "cosine": canonical_dev_metrics("tmr_cosine_dev", cosine_val),
            "reward": canonical_dev_metrics("reward_dev", mlp_val),
        },
        "history": history,
        "go_no_go": go_no_go,
    }
    write_json(output_dir / "s2a_summary.json", summary)
    if not args.no_checkpoint:
        torch.save(
            {
                "model_state_dict": model.cpu().state_dict(),
                "emb_dim": emb_dim,
                "args": vars(args),
                "summary": summary,
            },
            output_dir / "s2a_checkpoint.pt",
        )

    print(json.dumps(summary["go_no_go"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[S2A-ERROR] {exc}", file=sys.stderr)
        raise
