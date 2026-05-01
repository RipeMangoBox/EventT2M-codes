from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from planb.eventt2m_runtime import build_event_pool, load_hml3de_test_dict, pick_text_entry, select_sample_ids
from planb.tmr_runtime import load_tmr_runtime, load_tmr_stats, read_manifest_rows, score_motion_text, stats_diff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoDebug Plan B Phase 0 reward diagnostic")
    parser.add_argument("--sample-ids", type=str, default="004965,008463,001969,003245")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-events-per-sample", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--tmr-root", type=str, default=None)
    parser.add_argument("--tmr-run-dir", type=str, default=None)
    parser.add_argument("--data-file", type=str, default="dataset/HumanML3D-E/data_test.npy")
    parser.add_argument("--eventt2m-stats-dir", type=str, default="dataset/HumanML3D")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="logs/planb_phase0_reward")
    return parser.parse_args()


def choose_negative_event(sample_id: str, positive_event: str, pool: List[str]) -> str:
    positive_len = len(positive_event.split())
    for candidate in pool:
        if candidate == positive_event:
            continue
        if abs(len(candidate.split()) - positive_len) <= 2:
            return candidate
    for candidate in pool:
        if candidate != positive_event:
            return candidate
    raise RuntimeError(f"No negative event available for sample {sample_id}")


def collect_generated_full_rows(manifest_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for row in read_manifest_rows(manifest_path):
        if row["variant"] == "full":
            mapping[row["sample_id"]] = row["raw_motion_path"]
    return mapping


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
        raise RuntimeError("No eligible samples selected for reward diagnostic")

    generated_full_rows: Dict[str, str] = {}
    if args.manifest:
        generated_full_rows = collect_generated_full_rows((repo_root / args.manifest).resolve())

    runtime = load_tmr_runtime(
        repo_root=repo_root,
        tmr_root=tmr_root,
        run_dir=tmr_run_dir,
        device=args.device,
    )

    eventt2m_mean = np.load(repo_root / args.eventt2m_stats_dir / "Mean.npy")
    eventt2m_std = np.load(repo_root / args.eventt2m_stats_dir / "Std.npy")
    tmr_stats = load_tmr_stats(tmr_root)
    stats_report = stats_diff(eventt2m_mean, eventt2m_std, tmr_stats)
    stats_report["stats_aligned"] = bool(
        stats_report["mean_max_abs"] < 1e-4 and stats_report["std_max_abs"] < 1e-4
    )

    event_pool = build_event_pool(data_dict)
    rows: List[Dict] = []
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
            for event_index, event_text in enumerate(events[: args.max_events_per_sample]):
                negative_event = choose_negative_event(sample_id, event_text, event_pool)
                matched_score = score_motion_text(runtime, motion, event_text)
                mismatched_score = score_motion_text(runtime, motion, negative_event)
                masked_score = score_motion_text(runtime, motion, "")

                rows.append(
                    {
                        "sample_id": sample_id,
                        "source": source_name,
                        "event_index": event_index,
                        "event_text": event_text,
                        "negative_event": negative_event,
                        "matched_score": matched_score,
                        "mismatched_score": mismatched_score,
                        "masked_score": masked_score,
                        "delta_matched_minus_mismatched": matched_score - mismatched_score,
                        "delta_matched_minus_masked": matched_score - masked_score,
                    }
                )

    rows_path = output_dir / "reward_rows.jsonl"
    with open(rows_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    matched = np.array([row["matched_score"] for row in rows], dtype=np.float32)
    mismatched = np.array([row["mismatched_score"] for row in rows], dtype=np.float32)
    masked = np.array([row["masked_score"] for row in rows], dtype=np.float32)
    summary = {
        "sample_ids": selected,
        "num_rows": len(rows),
        "tmr_run_dir": str(tmr_run_dir),
        "stats_report": stats_report,
        "mean_matched_score": float(matched.mean()) if len(matched) else None,
        "mean_mismatched_score": float(mismatched.mean()) if len(mismatched) else None,
        "mean_masked_score": float(masked.mean()) if len(masked) else None,
        "mean_delta_matched_minus_mismatched": float((matched - mismatched).mean()) if len(matched) else None,
        "mean_delta_matched_minus_masked": float((matched - masked).mean()) if len(matched) else None,
        "rows_path": str(rows_path),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
