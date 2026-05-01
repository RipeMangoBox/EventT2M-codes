from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from planb.tmr_runtime import load_tmr_runtime, score_motion_text_batch


OLD_ALIGNED_REPLACE_TMR_ACCURACY = 0.835820895522388


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score MoDebug hard replace manifest with native TMR.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--tmr-root", type=str, default=None)
    parser.add_argument("--tmr-run-dir", type=str, default=None)
    parser.add_argument("--data-file", type=str, default="dataset/HumanML3D-E/data_test.npy")
    parser.add_argument(
        "--manifest",
        type=str,
        default="logs/modebug_hard_replace_eval/hard_replace_manifest.jsonl",
    )
    parser.add_argument("--output-dir", type=str, default="logs/modebug_hard_replace_eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_hml3de_test_dict(data_file: Path) -> Dict[str, Dict[str, Any]]:
    return np.load(data_file, allow_pickle=True).item()


def summarize(rows: List[Dict[str, Any]], output_dir: Path, args: argparse.Namespace, manifest_path: Path) -> Dict[str, Any]:
    replace_deltas = np.array([row["delta_full_minus_replace"] for row in rows], dtype=np.float32)
    drop_deltas = np.array([row["delta_full_minus_drop"] for row in rows], dtype=np.float32)
    buckets: Dict[str, Dict[str, Any]] = {}
    for bucket in sorted({row["event_count_bucket"] for row in rows}):
        bucket_rows = [row for row in rows if row["event_count_bucket"] == bucket]
        bucket_replace = np.array([row["delta_full_minus_replace"] for row in bucket_rows], dtype=np.float32)
        buckets[bucket] = {
            "count": len(bucket_rows),
            "replace_paired_accuracy_full_gt_replace": float(np.mean(bucket_replace > 0))
            if len(bucket_replace)
            else None,
            "replace_mean_delta_full_minus_replace": float(bucket_replace.mean()) if len(bucket_replace) else None,
        }

    hard_accuracy = float(np.mean(replace_deltas > 0)) if len(replace_deltas) else None
    return {
        "task_id": "MDBG-P0-TMR-HARD-REPLACE-EVAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "scorer": "native TMR global text-motion score",
            "dataset": "HumanML3D-E",
            "evaluator_side_cross_check": True,
            "not_final_standalone_judge": True,
            "excluded_extensions": ["MotionPatches", "attention_filtering", "ordering", "duration", "judge"],
        },
        "inputs": {
            "manifest": str(manifest_path),
            "data_file": args.data_file,
        },
        "row_counts": {
            "scored_rows": len(rows),
            "max_rows": args.max_rows,
        },
        "drop": {
            "paired_accuracy_full_gt_drop": float(np.mean(drop_deltas > 0)) if len(drop_deltas) else None,
            "mean_delta_full_minus_drop": float(drop_deltas.mean()) if len(drop_deltas) else None,
        },
        "replace": {
            "paired_accuracy_full_gt_replace": hard_accuracy,
            "mean_delta_full_minus_replace": float(replace_deltas.mean()) if len(replace_deltas) else None,
            "median_delta_full_minus_replace": float(np.median(replace_deltas)) if len(replace_deltas) else None,
            "std_delta_full_minus_replace": float(replace_deltas.std()) if len(replace_deltas) else None,
        },
        "comparison": {
            "old_aligned_replace_tmr_full_gt_replace": OLD_ALIGNED_REPLACE_TMR_ACCURACY,
            "hard_minus_old_aligned_accuracy": hard_accuracy - OLD_ALIGNED_REPLACE_TMR_ACCURACY
            if hard_accuracy is not None
            else None,
            "easy_negative_inflation_supported": hard_accuracy is not None
            and hard_accuracy < OLD_ALIGNED_REPLACE_TMR_ACCURACY,
        },
        "buckets": buckets,
        "outputs": {
            "rows_jsonl": str(output_dir / "tmr_hard_replace_rows.jsonl"),
            "summary_json": str(output_dir / "tmr_hard_replace_summary.json"),
            "summary_md": str(output_dir / "tmr_hard_replace_summary.md"),
        },
    }


def make_markdown_summary(summary: Dict[str, Any]) -> str:
    hard = summary["replace"]["paired_accuracy_full_gt_replace"]
    old = summary["comparison"]["old_aligned_replace_tmr_full_gt_replace"]
    diff = summary["comparison"]["hard_minus_old_aligned_accuracy"]
    supported = summary["comparison"]["easy_negative_inflation_supported"]
    lines = [
        "# MoDebug TMR hard-replace summary",
        "",
        f"- Dataset: HumanML3D-E",
        f"- Scored rows: {summary['row_counts']['scored_rows']}",
        f"- TMR full > hard_replace paired accuracy: {hard}",
        f"- Old aligned replace paired accuracy: {old}",
        f"- Delta hard - old: {diff}",
        f"- Easy-negative inflation supported: {supported}",
        f"- Rows: `{summary['outputs']['rows_jsonl']}`",
        f"- JSON: `{summary['outputs']['summary_json']}`",
        "",
        "MotionPatches is excluded; this is an evaluator-side diagnostic artifact.",
    ]
    return "\n".join(lines) + "\n"


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

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest_rows = read_jsonl(manifest_path)
    if args.max_rows > 0:
        manifest_rows = manifest_rows[: args.max_rows]
    if not manifest_rows:
        raise RuntimeError("No manifest rows to score.")

    data_dict = load_hml3de_test_dict(repo_root / args.data_file)
    runtime = load_tmr_runtime(repo_root=repo_root, tmr_root=tmr_root, run_dir=tmr_run_dir, device=args.device)

    scored_rows: List[Dict[str, Any]] = []
    for row in manifest_rows:
        sample_id = row["sample_id"]
        sample = data_dict[sample_id]
        full_score, drop_score, replace_score = score_motion_text_batch(
            runtime,
            sample["motion"],
            [row["full_text"], row["drop_text"], row["replace_text"]],
        )
        scored_rows.append(
            {
                **row,
                "scorer": "native_tmr",
                "full_score": full_score,
                "drop_score": drop_score,
                "replace_score": replace_score,
                "delta_full_minus_drop": full_score - drop_score,
                "delta_full_minus_replace": full_score - replace_score,
                "positive_full_gt_replace": bool(full_score > replace_score),
            }
        )

    rows_path = output_dir / "tmr_hard_replace_rows.jsonl"
    write_jsonl(rows_path, scored_rows)
    summary = summarize(scored_rows, output_dir, args, manifest_path)
    with open(output_dir / "tmr_hard_replace_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "tmr_hard_replace_summary.md", "w", encoding="utf-8") as handle:
        handle.write(make_markdown_summary(summary))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
