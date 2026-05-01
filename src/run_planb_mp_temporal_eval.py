from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from planb.eventt2m_runtime import build_event_pool, corrupt_events, load_hml3de_test_dict, pick_text_entry, select_sample_ids
from planb.motionpatches_runtime import (
    load_motionpatches_runtime,
    read_manifest_rows,
    structured_temporal_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoDebug Plan B MotionPatches temporal eval-only")
    parser.add_argument("--sample-ids", type=str, default="004965,008463,001969,003245")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--mp-root", type=str, default=None)
    parser.add_argument(
        "--mp-checkpoint-dir",
        type=str,
        default=None,
        help="Default: MotionPatches stage5_s2e_v2/HumanML3D best checkpoint",
    )
    parser.add_argument("--data-file", type=str, default="dataset/HumanML3D-E/data_test.npy")
    parser.add_argument("--manifest", type=str, default="logs/planb_phase_minus1_run/manifest.jsonl")
    parser.add_argument("--output-dir", type=str, default="logs/planb_mp_temporal_eval")
    parser.add_argument("--event-encode-mode", type=str, default="independent")
    parser.add_argument("--dp-mode", type=str, default="strict")
    return parser.parse_args()


def collect_variant_rows(manifest_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    mapping: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in read_manifest_rows(manifest_path):
        mapping[(row["sample_id"], row["variant"])] = row
    return mapping


def subtract(a: float, b: float) -> float:
    return float(a - b)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    mp_root = Path(args.mp_root).resolve() if args.mp_root else repo_root.parent / "MotionPatches-main"
    mp_checkpoint_dir = (
        Path(args.mp_checkpoint_dir).resolve()
        if args.mp_checkpoint_dir
        else mp_root / "checkpoints" / "stage5_s2e_v2" / "HumanML3D"
    )
    manifest_path = (repo_root / args.manifest).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dict = load_hml3de_test_dict(repo_root / args.data_file)
    sample_ids = [sample_id.strip() for sample_id in args.sample_ids.split(",") if sample_id.strip()]
    selected = select_sample_ids(data_dict, sample_ids, args.max_samples, min_events=3)
    if not selected:
        raise RuntimeError("No eligible samples selected for MotionPatches temporal eval")

    runtime = load_motionpatches_runtime(
        repo_root=repo_root,
        mp_root=mp_root,
        checkpoint_dir=mp_checkpoint_dir,
        device=args.device,
    )
    manifest_rows = collect_variant_rows(manifest_path)
    event_pool = build_event_pool(data_dict)

    fixed_motion_rows: List[Dict[str, Any]] = []
    variant_motion_rows: List[Dict[str, Any]] = []

    for sample_id in selected:
        sample = data_dict[sample_id]
        text_entry = pick_text_entry(sample)
        caption = text_entry["caption"]
        events = [item["caption"] for item in text_entry["decomposed"]]
        gt_joints = np.load(runtime.data_root / "new_joints" / f"{sample_id}.npy")

        motion_sources = [("gt", gt_joints)]
        full_variant_joints = np.load(manifest_rows[(sample_id, "full")]["joints_path"])["motion"]
        motion_sources.append(("generated_full", full_variant_joints))

        full_bundle = structured_temporal_scores(
            runtime,
            full_variant_joints,
            events,
            caption,
            event_encode_mode=args.event_encode_mode,
            dp_mode=args.dp_mode,
        )
        variant_motion_rows.append(
            {
                "sample_id": sample_id,
                "motion_variant": "full",
                "score": full_bundle,
            }
        )

        for variant in ["drop", "swap", "replace"]:
            corruption = corrupt_events(events, variant, event_pool)

            for source_name, joints in motion_sources:
                base = structured_temporal_scores(
                    runtime,
                    joints,
                    events,
                    caption,
                    event_encode_mode=args.event_encode_mode,
                    dp_mode=args.dp_mode,
                )
                corr = structured_temporal_scores(
                    runtime,
                    joints,
                    corruption["events"],
                    caption,
                    event_encode_mode=args.event_encode_mode,
                    dp_mode=args.dp_mode,
                )
                fixed_motion_rows.append(
                    {
                        "sample_id": sample_id,
                        "source": source_name,
                        "corruption": variant,
                        "target_idx": corruption["target_idx"],
                        "full_events": events,
                        "corrupted_events": corruption["events"],
                        "base": base,
                        "corrupted": corr,
                        "delta_structured_score": subtract(base["structured_score"], corr["structured_score"]),
                        "delta_reverse_margin": subtract(base["reverse_order_margin"], corr["reverse_order_margin"]),
                    }
                )

            variant_joints = np.load(manifest_rows[(sample_id, variant)]["joints_path"])["motion"]
            variant_score = structured_temporal_scores(
                runtime,
                variant_joints,
                events,
                caption,
                event_encode_mode=args.event_encode_mode,
                dp_mode=args.dp_mode,
            )
            variant_motion_rows.append(
                {
                    "sample_id": sample_id,
                    "motion_variant": variant,
                    "score": variant_score,
                    "delta_vs_full_structured": subtract(
                        full_bundle["structured_score"], variant_score["structured_score"]
                    ),
                    "delta_vs_full_reverse_margin": subtract(
                        full_bundle["reverse_order_margin"], variant_score["reverse_order_margin"]
                    ),
                }
            )

    fixed_rows_path = output_dir / "fixed_motion_rows.jsonl"
    with open(fixed_rows_path, "w", encoding="utf-8") as handle:
        for row in fixed_motion_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    variant_rows_path = output_dir / "variant_motion_rows.jsonl"
    with open(variant_rows_path, "w", encoding="utf-8") as handle:
        for row in variant_motion_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary: Dict[str, Any] = {
        "sample_ids": selected,
        "mp_checkpoint_dir": str(mp_checkpoint_dir),
        "event_encode_mode": args.event_encode_mode,
        "dp_mode": args.dp_mode,
        "fixed_rows_path": str(fixed_rows_path),
        "variant_rows_path": str(variant_rows_path),
    }

    for source_name in ["gt", "generated_full"]:
        source_rows = [row for row in fixed_motion_rows if row["source"] == source_name]
        if source_rows:
            summary[source_name] = {}
            for corruption in ["drop", "swap", "replace"]:
                rows = [row for row in source_rows if row["corruption"] == corruption]
                summary[source_name][corruption] = {
                    "mean_delta_structured_score": float(np.mean([row["delta_structured_score"] for row in rows])) if rows else None,
                    "mean_delta_reverse_margin": float(np.mean([row["delta_reverse_margin"] for row in rows])) if rows else None,
                    "num_rows": len(rows),
                }

    variant_summary: Dict[str, Any] = {}
    for variant in ["full", "drop", "swap", "replace"]:
        rows = [row for row in variant_motion_rows if row["motion_variant"] == variant]
        if not rows:
            continue
        variant_summary[variant] = {
            "mean_structured_score": float(np.mean([row["score"]["structured_score"] for row in rows])),
            "mean_reverse_order_margin": float(np.mean([row["score"]["reverse_order_margin"] for row in rows])),
            "mean_delta_vs_full_structured": float(np.mean([row.get("delta_vs_full_structured", 0.0) for row in rows if "delta_vs_full_structured" in row])) if variant != "full" else 0.0,
        }
    summary["variant_motion_vs_full_events"] = variant_summary

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
