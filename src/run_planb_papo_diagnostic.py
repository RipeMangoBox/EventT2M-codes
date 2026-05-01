from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from planb.eventt2m_runtime import (
    build_event_pool,
    corrupt_events,
    load_hml3de_test_dict,
    pick_text_entry,
    select_sample_ids,
)
from planb.tmr_runtime import (
    load_tmr_runtime,
    read_manifest_rows,
    score_motion_texts,
    windowed_order_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoDebug Plan B Phase 1 PAPO-lite diagnostic")
    parser.add_argument("--sample-ids", type=str, default="004965,008463,001969,003245")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--tmr-root", type=str, default=None)
    parser.add_argument("--tmr-run-dir", type=str, default=None)
    parser.add_argument("--data-file", type=str, default="dataset/HumanML3D-E/data_test.npy")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="logs/planb_phase1_papo")
    parser.add_argument("--tau", type=float, default=0.1)
    return parser.parse_args()


def collect_generated_full_rows(manifest_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for row in read_manifest_rows(manifest_path):
        if row["variant"] == "full":
            mapping[row["sample_id"]] = row["raw_motion_path"]
    return mapping


def non_empty(events: List[str]) -> List[str]:
    return [event for event in events if event.strip()]


def average_pres_score(scores: List[float]) -> float | None:
    if not scores:
        return None
    return float(np.mean(scores))


def build_variant_bundle(
    motion: np.ndarray,
    events: List[str],
    runtime,
    tau: float,
) -> Dict[str, Any]:
    valid_events = non_empty(events)
    slot_scores: List[float] = []
    for event in events:
        if event.strip():
            slot_scores.append(score_motion_texts(runtime, motion, [event])[0])
        else:
            slot_scores.append(0.0)
    ord_bundle = windowed_order_bundle(runtime, motion, valid_events, tau=tau)
    return {
        "events": events,
        "valid_events": valid_events,
        "r_pres_slot_scores": slot_scores,
        "r_pres_mean": average_pres_score(slot_scores),
        "r_ord_mean": ord_bundle["mean_pair_score"],
        "r_ord_pair_scores": ord_bundle["pair_scores"],
        "r_ord_centers": ord_bundle["centers"],
        "event_count": ord_bundle["event_count"],
    }


def subtract_or_none(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)


def main() -> None:
    args = parse_args()
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
    sample_ids = [sample_id.strip() for sample_id in args.sample_ids.split(",") if sample_id.strip()]
    selected = select_sample_ids(data_dict, sample_ids, args.max_samples, min_events=3)
    if not selected:
        raise RuntimeError("No eligible samples selected for PAPO diagnostic")

    generated_full_rows: Dict[str, str] = {}
    if args.manifest:
        generated_full_rows = collect_generated_full_rows((repo_root / args.manifest).resolve())

    runtime = load_tmr_runtime(
        repo_root=repo_root,
        tmr_root=tmr_root,
        run_dir=tmr_run_dir,
        device=args.device,
    )
    event_pool = build_event_pool(data_dict)

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for sample_id in selected:
        sample = data_dict[sample_id]
        text_entry = pick_text_entry(sample)
        events = [item["caption"] for item in text_entry["decomposed"]]
        gt_motion = sample["motion"]

        motion_sources: List[Tuple[str, np.ndarray]] = [("gt", gt_motion)]
        generated_path = generated_full_rows.get(sample_id)
        if generated_path:
            motion_sources.append(("generated_full", np.load(generated_path)))

        for source_name, motion in motion_sources:
            full_bundle = build_variant_bundle(motion, events, runtime, args.tau)
            for corruption_name in ["drop", "swap", "replace"]:
                corruption = corrupt_events(events, corruption_name, event_pool)
                variant_bundle = build_variant_bundle(motion, corruption["events"], runtime, args.tau)
                row = {
                    "sample_id": sample_id,
                    "source": source_name,
                    "corruption": corruption_name,
                    "target_idx": corruption["target_idx"],
                    "full": full_bundle,
                    "corrupted": variant_bundle,
                    "delta_r_pres_mean": subtract_or_none(full_bundle["r_pres_mean"], variant_bundle["r_pres_mean"]),
                    "delta_r_ord_mean": subtract_or_none(full_bundle["r_ord_mean"], variant_bundle["r_ord_mean"]),
                }
                rows.append(row)

                if corruption_name == "replace":
                    delta = row["delta_r_pres_mean"]
                    if delta is None or delta < 0.05:
                        failures.append(
                            {
                                "sample_id": sample_id,
                                "source": source_name,
                                "failure_type": "weak_distractor_delta",
                                "delta_r_pres_mean": delta,
                                "full_events": events,
                                "corrupted_events": corruption["events"],
                            }
                        )

    rows_path = output_dir / "papo_rows.jsonl"
    with open(rows_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary: Dict[str, Any] = {
        "sample_ids": selected,
        "num_rows": len(rows),
        "tmr_run_dir": str(tmr_run_dir),
        "rows_path": str(rows_path),
        "failure_count": len(failures),
        "failures_path": str(output_dir / "failure_cases.json"),
    }

    for source_name in ["gt", "generated_full"]:
        source_rows = [row for row in rows if row["source"] == source_name]
        if not source_rows:
            continue
        source_summary: Dict[str, Any] = {}
        for corruption_name in ["drop", "swap", "replace"]:
            group = [row for row in source_rows if row["corruption"] == corruption_name]
            if not group:
                continue
            pres_deltas = [row["delta_r_pres_mean"] for row in group if row["delta_r_pres_mean"] is not None]
            ord_deltas = [row["delta_r_ord_mean"] for row in group if row["delta_r_ord_mean"] is not None]
            source_summary[corruption_name] = {
                "mean_delta_r_pres": float(np.mean(pres_deltas)) if pres_deltas else None,
                "mean_delta_r_ord": float(np.mean(ord_deltas)) if ord_deltas else None,
                "num_rows": len(group),
            }
        summary[source_name] = source_summary

    with open(output_dir / "failure_cases.json", "w", encoding="utf-8") as handle:
        json.dump(failures, handle, ensure_ascii=False, indent=2)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
