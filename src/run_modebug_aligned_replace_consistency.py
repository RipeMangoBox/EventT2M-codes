from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


JoinKey = Tuple[str, int, int, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MoDebug aligned replace consistency.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="logs/modebug_aligned_replace_eval")
    parser.add_argument(
        "--manifest",
        type=str,
        default="logs/modebug_aligned_replace_eval/aligned_replace_manifest.jsonl",
    )
    parser.add_argument(
        "--tmr-rows",
        type=str,
        default="logs/modebug_aligned_replace_eval/tmr_aligned_replace_rows.jsonl",
    )
    parser.add_argument(
        "--chron-rows",
        type=str,
        default="../ChronAccRet/output/bert_orig/aligned_replace_eval/chronaccret_aligned_replace_rows.jsonl",
    )
    return parser.parse_args()


def resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repo_path = (repo_root / path).resolve()
    if repo_path.exists():
        return repo_path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    parts = path.parts
    if "ChronAccRet" in parts:
        suffix = Path(*parts[parts.index("ChronAccRet") + 1 :])
        vault_root = Path("/data/Life Me/ResearchWY Vault/linkedCodebases/ChronAccRet")
        vault_chron = vault_root / suffix
        if vault_root.exists():
            return vault_chron.resolve()
    return repo_path


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


def row_id(row: Dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("keyid"))


def aligned_key(row: Dict[str, Any]) -> JoinKey:
    return (
        row_id(row),
        int(row["target_idx"]),
        int(row["event_count"]),
        str(row["replacement_event"]),
        str(row["replace_text"]),
    )


def bucket_name(event_count: int) -> str:
    return str(event_count) if event_count < 5 else "5plus"


def summarize_agreement(joined_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    agreement_rows = sum(1 for row in joined_rows if row["agreement_full_gt_replace"])
    tmr_positive = sum(1 for row in joined_rows if row["tmr_positive_full_gt_replace"])
    chron_positive = sum(1 for row in joined_rows if row["chronaccret_positive_full_gt_replace"])
    buckets: Dict[str, Dict[str, Any]] = {}
    for bucket in sorted({row["event_count_bucket"] for row in joined_rows}):
        rows = [row for row in joined_rows if row["event_count_bucket"] == bucket]
        bucket_agree = sum(1 for row in rows if row["agreement_full_gt_replace"])
        buckets[bucket] = {
            "count": len(rows),
            "agreement_rows": bucket_agree,
            "agreement_rate": float(bucket_agree / len(rows)) if rows else None,
            "tmr_positive_rows": sum(1 for row in rows if row["tmr_positive_full_gt_replace"]),
            "chronaccret_positive_rows": sum(1 for row in rows if row["chronaccret_positive_full_gt_replace"]),
        }
    return {
        "agreement_rows": agreement_rows,
        "disagreement_rows": len(joined_rows) - agreement_rows,
        "agreement_rate": float(agreement_rows / len(joined_rows)) if joined_rows else None,
        "tmr_positive_rows": tmr_positive,
        "tmr_positive_rate": float(tmr_positive / len(joined_rows)) if joined_rows else None,
        "chronaccret_positive_rows": chron_positive,
        "chronaccret_positive_rate": float(chron_positive / len(joined_rows)) if joined_rows else None,
        "buckets": buckets,
    }


def pending_summary(
    *,
    output_dir: Path,
    repo_root: Path,
    manifest_path: Path,
    manifest_rows: List[Dict[str, Any]],
    tmr_rows_path: Path,
    chron_rows_path: Path,
    missing_inputs: List[str],
) -> Dict[str, Any]:
    bucket_counts: Dict[str, int] = {}
    for row in manifest_rows:
        bucket = row.get("event_count_bucket") or bucket_name(int(row["event_count"]))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    return {
        "task_id": "MDBG-E4-ALIGNED-REPLACE-CONSISTENCY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "dataset": "HumanML3D-E",
            "evaluator_side_cross_check": True,
            "not_final_standalone_judge": True,
            "excluded_extensions": ["MotionPatches", "ordering", "duration", "judge"],
        },
        "replace_scoring_status": "pending",
        "pending_reason": "Both TMR and ChronAccRet aligned-replace scored rows are required before reporting full>replace agreement.",
        "missing_inputs": missing_inputs,
        "manifest": {
            "path": str(manifest_path),
            "rows": len(manifest_rows),
            "event_count_buckets": dict(sorted(bucket_counts.items())),
        },
        "commands": {
            "build_manifest": (
                "conda run -n event-t2m python src/run_modebug_aligned_replace_manifest.py "
                "--output-dir logs/modebug_aligned_replace_eval"
            ),
            "score_tmr": (
                "conda run -n event-t2m python src/run_modebug_tmr_aligned_replace_eval.py "
                "--device cuda --output-dir logs/modebug_aligned_replace_eval"
            ),
            "score_chronaccret": (
                "cd '/data/Life Me/ResearchWY Vault/linkedCodebases/ChronAccRet' && "
                "python retrieval_omission_aligned_replace.py "
                "+aligned_replace_manifest='/home/ripemangobox/Coding/Github/Motion/EventT2M-codes-main/logs/modebug_aligned_replace_eval/aligned_replace_manifest.jsonl' "
                "+aligned_replace_output_dir=./output/bert_orig/aligned_replace_eval"
            ),
            "rerun_consistency": (
                "conda run -n event-t2m python src/run_modebug_aligned_replace_consistency.py "
                "--output-dir logs/modebug_aligned_replace_eval"
            ),
        },
        "outputs": {
            "summary_json": str(output_dir / "aligned_replace_consistency_summary.json"),
            "tmr_rows_expected": str(tmr_rows_path),
            "chronaccret_rows_expected": str(chron_rows_path),
            "repo_root": str(repo_root),
        },
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    output_dir = resolve(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = resolve(repo_root, args.manifest)
    tmr_rows_path = resolve(repo_root, args.tmr_rows)
    chron_rows_path = resolve(repo_root, args.chron_rows)
    manifest_rows = read_jsonl(manifest_path)

    missing_inputs = [
        name
        for name, path in [("tmr_rows", tmr_rows_path), ("chronaccret_rows", chron_rows_path)]
        if not path.exists()
    ]
    if missing_inputs:
        summary = pending_summary(
            output_dir=output_dir,
            repo_root=repo_root,
            manifest_path=manifest_path,
            manifest_rows=manifest_rows,
            tmr_rows_path=tmr_rows_path,
            chron_rows_path=chron_rows_path,
            missing_inputs=missing_inputs,
        )
        with open(output_dir / "aligned_replace_consistency_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    tmr_rows = read_jsonl(tmr_rows_path)
    chron_rows = read_jsonl(chron_rows_path)
    chron_by_key = {aligned_key(row): row for row in chron_rows}
    joined_rows: List[Dict[str, Any]] = []
    for tmr_row in tmr_rows:
        chron_row = chron_by_key.get(aligned_key(tmr_row))
        if chron_row is None:
            continue
        tmr_positive = tmr_row["delta_full_minus_replace"] > 0
        chron_positive = chron_row["delta_full_minus_replace"] > 0
        joined_rows.append(
            {
                "manifest_id": tmr_row.get("manifest_id"),
                "sample_id": row_id(tmr_row),
                "keyid": row_id(chron_row),
                "target_idx": int(tmr_row["target_idx"]),
                "event_count": int(tmr_row["event_count"]),
                "event_count_bucket": tmr_row.get("event_count_bucket") or bucket_name(int(tmr_row["event_count"])),
                "replacement_event": tmr_row["replacement_event"],
                "replace_text": tmr_row["replace_text"],
                "tmr_delta_full_minus_replace": tmr_row["delta_full_minus_replace"],
                "chronaccret_delta_full_minus_replace": chron_row["delta_full_minus_replace"],
                "tmr_positive_full_gt_replace": bool(tmr_positive),
                "chronaccret_positive_full_gt_replace": bool(chron_positive),
                "agreement_full_gt_replace": bool(tmr_positive == chron_positive),
            }
        )

    agreement = summarize_agreement(joined_rows)
    summary = {
        "task_id": "MDBG-E4-ALIGNED-REPLACE-CONSISTENCY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "dataset": "HumanML3D-E",
            "reported_metric": "sample-level aligned-replace consistency for full > replace",
            "evaluator_side_cross_check": True,
            "not_final_standalone_judge": True,
            "excluded_extensions": ["MotionPatches", "ordering", "duration", "judge"],
        },
        "replace_scoring_status": "complete",
        "join_policy": {
            "name": "aligned_replace_join",
            "fields": ["sample_id/keyid", "target_idx", "event_count", "replacement_event", "replace_text"],
            "positive_definition": "delta_full_minus_replace > 0",
        },
        "inputs": {
            "manifest": str(manifest_path),
            "tmr_rows": str(tmr_rows_path),
            "chronaccret_rows": str(chron_rows_path),
        },
        "row_counts": {
            "manifest_rows": len(manifest_rows),
            "tmr_rows": len(tmr_rows),
            "chronaccret_rows": len(chron_rows),
            "joined_rows": len(joined_rows),
        },
        "coverage": {
            "joined_vs_manifest": float(len(joined_rows) / len(manifest_rows)) if manifest_rows else None,
            "joined_vs_tmr": float(len(joined_rows) / len(tmr_rows)) if tmr_rows else None,
            "joined_vs_chronaccret": float(len(joined_rows) / len(chron_rows)) if chron_rows else None,
        },
        "overall": {key: value for key, value in agreement.items() if key != "buckets"},
        "buckets": agreement["buckets"],
        "outputs": {
            "summary_json": str(output_dir / "aligned_replace_consistency_summary.json"),
            "joined_rows_jsonl": str(output_dir / "aligned_replace_consistency_rows.jsonl"),
        },
    }
    write_jsonl(output_dir / "aligned_replace_consistency_rows.jsonl", joined_rows)
    with open(output_dir / "aligned_replace_consistency_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
