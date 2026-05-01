from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from planb.eventt2m_runtime import (
    build_event_pool,
    choose_distractor,
    choose_target_index,
    extract_events,
    load_hml3de_test_dict,
    pick_text_entry,
)
from planb.tmr_runtime import load_tmr_runtime, score_motion_text_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoDebug Plan B TMR native omission eval")
    parser.add_argument("--sample-ids", type=str, default="all")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-events", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--tmr-root", type=str, default=None)
    parser.add_argument("--tmr-run-dir", type=str, default=None)
    parser.add_argument("--data-file", type=str, default="dataset/HumanML3D-E/data_test.npy")
    parser.add_argument("--output-dir", type=str, default="logs/planb_tmr_native_omission_dataset_eval")
    return parser.parse_args()


def events_to_text(events: List[str]) -> str:
    return " ".join(event.strip() for event in events if event.strip())


def select_dataset_ids(
    data_dict: Dict[str, Dict[str, Any]],
    sample_ids_arg: str,
    min_events: int,
    max_samples: int,
) -> List[str]:
    requested = [sample_id.strip() for sample_id in sample_ids_arg.split(",") if sample_id.strip()]
    if requested and requested != ["all"]:
        candidates = [sample_id for sample_id in requested if sample_id in data_dict]
    else:
        candidates = list(data_dict.keys())

    selected: List[str] = []
    for sample_id in candidates:
        events = extract_events(pick_text_entry(data_dict[sample_id]))
        if len(events) >= min_events:
            selected.append(sample_id)

    if max_samples > 0:
        selected = selected[:max_samples]
    return selected


def summarize(rows: List[Dict[str, Any]], selected: List[str], args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    drop_deltas = np.array([row["delta_full_minus_drop"] for row in rows], dtype=np.float32)
    replace_deltas = np.array([row["delta_full_minus_replace"] for row in rows], dtype=np.float32)
    event_counts: Dict[str, int] = {}
    for row in rows:
        key = str(row["event_count"] if row["event_count"] < 5 else "5plus")
        event_counts[key] = event_counts.get(key, 0) + 1

    return {
        "dataset": "HumanML3D-E data_test.npy",
        "split": "test",
        "scorer": "native TMR global text-motion score",
        "sample_policy": {
            "sample_ids": args.sample_ids,
            "min_events": args.min_events,
            "max_samples": args.max_samples,
            "seed": args.seed,
        },
        "num_samples": len(selected),
        "event_count_buckets": event_counts,
        "drop": {
            "mean_delta_full_minus_drop": float(drop_deltas.mean()) if len(drop_deltas) else None,
            "median_delta_full_minus_drop": float(np.median(drop_deltas)) if len(drop_deltas) else None,
            "std_delta_full_minus_drop": float(drop_deltas.std()) if len(drop_deltas) else None,
            "paired_accuracy_full_gt_drop": float(np.mean(drop_deltas > 0)) if len(drop_deltas) else None,
        },
        "replace": {
            "mean_delta_full_minus_replace": float(replace_deltas.mean()) if len(replace_deltas) else None,
            "median_delta_full_minus_replace": float(np.median(replace_deltas)) if len(replace_deltas) else None,
            "std_delta_full_minus_replace": float(replace_deltas.std()) if len(replace_deltas) else None,
            "paired_accuracy_full_gt_replace": float(np.mean(replace_deltas > 0)) if len(replace_deltas) else None,
        },
        "rows_path": str(output_dir / "omission_rows.jsonl"),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    tmr_root = Path(args.tmr_root).resolve() if args.tmr_root else repo_root.parent / "TMR"
    tmr_run_dir = (
        Path(args.tmr_run_dir).resolve()
        if args.tmr_run_dir
        else tmr_root / "models" / "tmr_humanml3d_guoh3dfeats"
    )
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dict = load_hml3de_test_dict(repo_root / args.data_file)
    selected = select_dataset_ids(data_dict, args.sample_ids, args.min_events, args.max_samples)
    if not selected:
        raise RuntimeError("No eligible samples selected for TMR native omission eval")

    runtime = load_tmr_runtime(
        repo_root=repo_root,
        tmr_root=tmr_root,
        run_dir=tmr_run_dir,
        device=args.device,
    )
    event_pool = build_event_pool(data_dict)

    rows: List[Dict[str, Any]] = []
    for sample_id in selected:
        sample = data_dict[sample_id]
        text_entry = pick_text_entry(sample)
        events = extract_events(text_entry)
        target_idx = choose_target_index(events)

        drop_events = list(events)
        dropped_event = drop_events.pop(target_idx)

        replace_events = list(events)
        replacement_event = choose_distractor(events[target_idx], event_pool)
        replace_events[target_idx] = replacement_event

        full_text = events_to_text(events)
        drop_text = events_to_text(drop_events)
        replace_text = events_to_text(replace_events)
        full_score, drop_score, replace_score = score_motion_text_batch(
            runtime,
            sample["motion"],
            [full_text, drop_text, replace_text],
        )

        rows.append(
            {
                "sample_id": sample_id,
                "event_count": len(events),
                "target_idx": target_idx,
                "dropped_event": dropped_event,
                "replacement_event": replacement_event,
                "full_text": full_text,
                "drop_text": drop_text,
                "replace_text": replace_text,
                "full_score": full_score,
                "drop_score": drop_score,
                "replace_score": replace_score,
                "delta_full_minus_drop": full_score - drop_score,
                "delta_full_minus_replace": full_score - replace_score,
            }
        )

    rows_path = output_dir / "omission_rows.jsonl"
    with open(rows_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows, selected, args, output_dir)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
