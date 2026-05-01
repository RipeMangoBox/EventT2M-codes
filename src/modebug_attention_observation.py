from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.nets.event_final import EventT2M


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return repo_root() / "logs" / "modebug_generation_observation" / f"g1g2_{stamp}"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def summarize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    for idx, record in enumerate(records):
        weights = record["weights"]
        sums = weights.sum(dim=-1)
        summary = {
            "record_idx": idx,
            "module": record.get("module"),
            "kind": record.get("kind"),
            "sample_step": record.get("sample_step"),
            "timestep": record.get("timestep"),
            "condition": record.get("condition"),
            "shape": record.get("shape"),
            "finite": bool(torch.isfinite(weights).all().item()),
            "last_dim_sum_min": float(sums.min().item()),
            "last_dim_sum_max": float(sums.max().item()),
        }
        summaries.append(summary)
    return summaries


def parse_csv_filter(value: Optional[str]) -> Optional[Set[str]]:
    if not value:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")


def jsonl_existing_keys(path: Path) -> Set[Tuple[str, str]]:
    keys = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((str(row.get("sample_id")), str(row.get("condition"))))
    return keys


def condition_target_index(row: Dict[str, Any]) -> Tuple[Optional[int], bool]:
    condition = row.get("condition")
    target_idx = row.get("target_idx")
    condition_events = row.get("condition_events") or []
    if not isinstance(target_idx, int):
        return None, False

    if condition == "drop":
        return None, False

    if condition == "shuffle":
        permutation = (row.get("condition_detail") or {}).get("shuffle_permutation") or []
        for new_idx, original_idx in enumerate(permutation):
            if original_idx == target_idx and new_idx < len(condition_events):
                return new_idx, True
        return None, False

    if target_idx < len(condition_events):
        return target_idx, True
    return None, False


def seed_for_row(row: Dict[str, Any], fallback_seed: int, row_idx: int) -> int:
    sample_id = str(row.get("sample_id", ""))
    if sample_id.isdigit():
        return int(sample_id)
    return fallback_seed + row_idx


def summarize_attention_metrics(
    records: List[Dict[str, Any]],
    row: Dict[str, Any],
    generated_length: int,
    patch_size: int,
) -> List[Dict[str, Any]]:
    metrics = []
    condition_events = row.get("condition_events") or []
    valid_event_count = min(len(condition_events), 11)
    cond_target_idx, target_available = condition_target_index(row)

    for record_idx, record in enumerate(records):
        weights = record["weights"]
        finite = bool(torch.isfinite(weights).all().item())
        item = {
            "record_idx": record_idx,
            "module": record.get("module"),
            "kind": record.get("kind"),
            "sample_step": record.get("sample_step"),
            "timestep": record.get("timestep"),
            "shape": record.get("shape"),
            "finite": finite,
            "target_event_available": bool(target_available),
            "condition_target_idx": cond_target_idx,
            "target_attn_peak_patch": None,
            "target_attn_peak_t": None,
            "target_attn_entropy_norm": None,
            "target_attn_mean_mass": None,
            "valid_event_count": valid_event_count,
            "event_peak_order": None,
        }

        if weights.ndim == 4 and weights.shape[0] > 0 and valid_event_count > 0 and finite:
            conditional_weights = weights[0]
            event_mass = conditional_weights[:, :, :valid_event_count].mean(dim=(0, 1))
            order = torch.argsort(event_mass, descending=True)
            item["event_peak_order"] = [int(idx.item()) for idx in order]

            if (
                target_available
                and cond_target_idx is not None
                and cond_target_idx < conditional_weights.shape[-1]
            ):
                target_mass_by_patch = conditional_weights[:, :, cond_target_idx].mean(dim=0)
                peak_patch = int(torch.argmax(target_mass_by_patch).item())
                patch_sum = target_mass_by_patch.sum()
                if torch.isfinite(patch_sum).item() and float(patch_sum.item()) > 0.0:
                    prob = target_mass_by_patch / patch_sum
                    entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum()
                    denom = math.log(max(int(prob.numel()), 2))
                    item["target_attn_entropy_norm"] = float((entropy / denom).item())
                item["target_attn_peak_patch"] = peak_patch
                item["target_attn_peak_t"] = min(
                    int(generated_length) - 1,
                    peak_patch * int(patch_size) + int(patch_size) // 2,
                )
                item["target_attn_mean_mass"] = float(target_mass_by_patch.mean().item())

        metrics.append(item)
    return metrics


def run_synthetic_smoke(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    model = EventT2M(
        motion_dim=263,
        max_motion_len=max(196, args.motion_len + 1),
        text_dim=512,
        dropout=0.0,
        stage_dim=args.stage_dim,
        num_groups=16,
        patch_size=args.patch_size,
        conformer_cfg={
            "nhead": args.heads,
            "dim_ff": args.dim_ff,
            "dropout": 0.0,
            "kernel_size": 7,
        },
    ).to(device)
    model.eval()
    model.set_modebug_attention_observation(True, clear=True)
    model.set_modebug_attention_context(
        mode="synthetic_smoke",
        condition="synthetic_full",
        sample_id="synthetic",
        sample_step=0,
        timestep=7,
    )

    motion = torch.randn(args.batch_size, args.motion_len, 263, device=device)
    motion_mask = torch.ones(args.batch_size, args.motion_len, dtype=torch.bool, device=device)
    timestep = torch.full((args.batch_size,), 7, dtype=torch.long, device=device)
    text = {"text_emb": torch.randn(args.batch_size, 512, device=device)}
    decomposed_embed = {
        "text_emb": torch.randn(args.batch_size, args.event_tokens, 512, device=device)
    }
    decomposed_mask = torch.zeros(
        args.batch_size, args.event_tokens, dtype=torch.bool, device=device
    )
    decomposed_mask[:, : args.valid_events] = True

    with torch.no_grad():
        output = model(motion, motion_mask, timestep, text, decomposed_embed, decomposed_mask)

    records = model.get_modebug_attention_records()
    model.set_modebug_attention_observation(False, clear=False)
    record_summaries = summarize_records(records)
    expected_motion_patches = math.ceil(args.motion_len / args.patch_size)
    expected_shapes = [
        [args.batch_size, args.heads, expected_motion_patches, args.event_tokens]
    ] * len(model.layers)
    observed_shapes = [item["shape"] for item in record_summaries]
    passed = (
        list(output.shape) == [args.batch_size, args.motion_len, 263]
        and observed_shapes == expected_shapes
        and all(item["finite"] for item in record_summaries)
        and all(
            abs(item["last_dim_sum_min"] - 1.0) < 1e-5
            and abs(item["last_dim_sum_max"] - 1.0) < 1e-5
            for item in record_summaries
        )
    )

    summary = {
        "task_id": "MDBG-G1G2-INSTRUMENT",
        "mode": "synthetic_smoke",
        "status": "passed" if passed else "failed",
        "seed": args.seed,
        "output_shape": list(output.shape),
        "records_count": len(records),
        "shape_semantics": "[batch, head, motion_patch, event_token]",
        "expected_record_shapes": expected_shapes,
        "observed_record_shapes": observed_shapes,
        "records": record_summaries,
    }
    write_json(output_dir / "smoke_summary.json", summary)
    return summary


def load_manifest(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= max_samples:
                break
    return rows


def load_condition_manifest(
    path: Path,
    max_rows: Optional[int],
    conditions: Optional[Set[str]],
    sample_ids: Optional[Set[str]],
) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if conditions is not None and row.get("condition") not in conditions:
                continue
            if sample_ids is not None and row.get("sample_id") not in sample_ids:
                continue
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def run_real_observation(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = repo_root() / ckpt_path
    if not ckpt_path.exists():
        summary = {
            "task_id": "MDBG-G1G2-INSTRUMENT",
            "mode": "real_observation",
            "status": "blocked",
            "blocker_type": "weights_path",
            "ckpt_path": str(ckpt_path),
            "reason": "Event-T2M checkpoint is not present.",
            "next_command": (
                "conda run -n event-t2m python src/modebug_attention_observation.py "
                "--run-real --ckpt-path /path/to/hml3d.ckpt --max-samples 1"
            ),
        }
        write_json(output_dir / "real_observation_blocked.json", summary)
        return summary

    try:
        from src.planb.eventt2m_runtime import load_eventt2m_runtime

        runtime = load_eventt2m_runtime(
            ckpt_path=ckpt_path,
            device=args.device,
            step_num=args.step_num,
            data_dir=args.data_dir,
        )
        manifest_path = Path(args.condition_manifest or args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = repo_root() / manifest_path
        max_rows = args.max_rows
        if max_rows is None and args.max_samples is not None:
            max_rows = args.max_samples
        rows = load_condition_manifest(
            manifest_path,
            max_rows=max_rows,
            conditions=parse_csv_filter(args.conditions),
            sample_ids=parse_csv_filter(args.sample_ids),
        )
        if args.save_jsonl:
            jsonl_path = Path(args.save_jsonl)
            if not jsonl_path.is_absolute():
                jsonl_path = output_dir / jsonl_path
        else:
            jsonl_path = output_dir / "observations.jsonl"
        existing_keys = jsonl_existing_keys(jsonl_path) if args.continue_existing else set()

        sample_summaries = []
        for row_idx, row in enumerate(rows):
            if (str(row.get("sample_id")), str(row.get("condition"))) in existing_keys:
                continue
            seed = seed_for_row(row, args.seed, row_idx)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            length = int(args.length)
            motion = torch.zeros((1, length, 263), device=runtime.device)
            lens = torch.tensor([length], dtype=torch.long, device=runtime.device)
            generated, records = runtime.model.sample_motion(
                motion,
                lens,
                [row["condition_text"]],
                [row["condition_events"]],
                modebug_attention_observation=True,
                modebug_attention_context={
                    "sample_id": row["sample_id"],
                    "condition": row["condition"],
                    "event_count": len(row["condition_events"]),
                    "target_idx": row.get("target_idx"),
                },
            )
            record_metrics = summarize_attention_metrics(
                records,
                row,
                generated_length=generated.shape[1],
                patch_size=args.patch_size,
            )
            sample_summary = {
                "sample_id": row["sample_id"],
                "condition": row["condition"],
                "target_idx": row.get("target_idx"),
                "event_text": row.get("event_text"),
                "condition_events": row.get("condition_events"),
                "condition_text": row.get("condition_text"),
                "seed": seed,
                "generated_shape": list(generated.shape),
                "records_count": len(records),
                "records": record_metrics,
            }
            sample_summaries.append(sample_summary)
            append_jsonl(jsonl_path, sample_summary)
    except Exception as exc:
        text = "".join(traceback.format_exception_only(type(exc), exc))
        full_trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        blocker_type = (
            "dependency"
            if "numpy" in full_trace.lower() or "scipy" in full_trace.lower()
            else "runtime"
        )
        summary = {
            "task_id": "MDBG-G1G2-INSTRUMENT",
            "mode": "real_observation",
            "status": "blocked",
            "blocker_type": blocker_type,
            "ckpt_path": str(ckpt_path),
            "reason": text.strip(),
            "traceback_tail": full_trace[-4000:],
            "next_command": (
                "conda run -n event-t2m python src/modebug_attention_observation.py "
                "--run-real --ckpt-path /path/to/hml3d.ckpt --max-samples 1"
            ),
        }
        write_json(output_dir / "real_observation_blocked.json", summary)
        return summary

    summary = {
        "task_id": "MDBG-G1G2-INSTRUMENT",
        "mode": "real_observation",
        "status": "completed",
        "manifest_path": str(manifest_path),
        "jsonl_path": str(jsonl_path),
        "rows_selected": len(rows),
        "rows_completed": len(sample_summaries),
        "samples": sample_summaries,
    }
    write_json(output_dir / "real_observation_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoDebug G1/G2 cross-attention observation smoke.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--motion-len", type=int, default=32)
    parser.add_argument("--event-tokens", type=int, default=7)
    parser.add_argument("--valid-events", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=128)
    parser.add_argument("--stage-dim", default="64-64")
    parser.add_argument("--run-real", action="store_true")
    parser.add_argument("--ckpt-path", default="checkpoints/pretrained/HumanML3D/hml3d.ckpt")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--manifest",
        default="logs/modebug_observation_pool/manifest.jsonl",
    )
    parser.add_argument(
        "--condition-manifest",
        default="logs/modebug_generation_observation/condition_manifest.jsonl",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--sample-ids", default=None)
    parser.add_argument("--save-jsonl", type=Path, default=None)
    parser.add_argument("--continue-existing", action="store_true")
    parser.add_argument("--length", type=int, default=196)
    parser.add_argument("--step-num", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    smoke_summary = run_synthetic_smoke(args, output_dir)
    if args.run_real:
        real_summary = run_real_observation(args, output_dir)
        print(json.dumps({"smoke": smoke_summary["status"], "real": real_summary["status"], "output_dir": str(output_dir)}))
    else:
        print(json.dumps({"smoke": smoke_summary["status"], "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
