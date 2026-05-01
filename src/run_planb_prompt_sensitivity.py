from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from planb.eventt2m_runtime import (
    build_event_pool,
    corrupt_events,
    generate_raw_motion,
    load_eventt2m_runtime,
    load_hml3de_test_dict,
    motion_diff,
    pick_text_entry,
    recover_joints,
    select_sample_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoDebug Plan B Phase -1 prompt sensitivity")
    parser.add_argument("--sample-ids", type=str, default="004965,008463,001969,003245")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="checkpoints/pretrained/HumanML3D/hml3d.ckpt",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="dataset/HumanML3D-E/data_test.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/planb_phase_minus1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_dict = load_hml3de_test_dict(repo_root / args.data_file)
    sample_ids = [sample_id.strip() for sample_id in args.sample_ids.split(",") if sample_id.strip()]
    selected = select_sample_ids(data_dict, sample_ids, args.max_samples, min_events=3)
    if not selected:
        raise RuntimeError("No eligible samples selected for prompt sensitivity")

    runtime = load_eventt2m_runtime(
        ckpt_path=repo_root / args.ckpt_path,
        device=args.device,
        data_dir=repo_root / "dataset" / "HumanML3D",
    )
    event_pool = build_event_pool(data_dict)

    output_dir = (repo_root / args.output_dir).resolve()
    motions_dir = output_dir / "motions"
    joints_dir = output_dir / "joints"
    output_dir.mkdir(parents=True, exist_ok=True)
    motions_dir.mkdir(parents=True, exist_ok=True)
    joints_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    rows: List[Dict] = []
    for sample_id in selected:
        sample = data_dict[sample_id]
        text_entry = pick_text_entry(sample)
        caption = text_entry["caption"]
        events = [item["caption"] for item in text_entry["decomposed"]]
        length = int(sample["length"])

        full_raw = None
        for variant in ["full", "drop", "swap", "replace"]:
            corruption = corrupt_events(events, variant, event_pool)
            variant_events = corruption["events"]
            raw_motion = generate_raw_motion(runtime, caption, variant_events, length, args.seed)
            joints = recover_joints(raw_motion)

            motion_path = motions_dir / f"{sample_id}_{variant}.npy"
            joints_path = joints_dir / f"{sample_id}_{variant}.npz"
            np.save(motion_path, raw_motion)
            np.savez(joints_path, motion=joints, text=caption)

            diff_metrics = None
            if variant == "full":
                full_raw = raw_motion
            else:
                diff_metrics = motion_diff(full_raw, raw_motion)

            row = {
                "sample_id": sample_id,
                "variant": variant,
                "seed": args.seed,
                "caption": caption,
                "events": events,
                "variant_events": variant_events,
                "target_idx": corruption["target_idx"],
                "length": length,
                "raw_motion_path": str(motion_path),
                "joints_path": str(joints_path),
                "diff_metrics": diff_metrics,
            }
            rows.append(row)

    with open(manifest_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "num_rows": len(rows),
        "sample_ids": selected,
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
