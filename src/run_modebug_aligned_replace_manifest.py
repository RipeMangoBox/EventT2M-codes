from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


JoinKey = Tuple[str, int, int, str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic MoDebug aligned replace manifest.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--data-file", type=str, default="dataset/HumanML3D-E/data_test.npy")
    parser.add_argument(
        "--tmr-rows",
        type=str,
        default="logs/planb_tmr_native_omission_dataset_eval/omission_rows.jsonl",
    )
    parser.add_argument(
        "--chron-rows",
        type=str,
        default="../ChronAccRet/output/bert_orig/omission_eval/omission_rows.jsonl",
    )
    parser.add_argument("--output-dir", type=str, default="logs/modebug_aligned_replace_eval")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--length-window", type=int, default=2)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_input(repo_root: Path, value: str) -> Path:
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


def load_hml3de_test_dict(data_file: Path) -> Dict[str, Dict[str, Any]]:
    return np.load(data_file, allow_pickle=True).item()


def pick_text_entry(sample: Dict[str, Any]) -> Dict[str, Any]:
    best = None
    best_len = -1
    for entry in sample["text"]:
        decomposed = entry.get("decomposed", [])
        if len(decomposed) > best_len:
            best = entry
            best_len = len(decomposed)
    if best is None:
        raise ValueError("Sample has no text entries")
    return best


def extract_events(text_entry: Dict[str, Any]) -> List[str]:
    return [item["caption"] for item in text_entry.get("decomposed", []) if item.get("caption", "").strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def events_to_text(events: List[str]) -> str:
    return " ".join(event.strip() for event in events if event.strip())


def row_id(row: Dict[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("keyid"))


def safe_drop_key(row: Dict[str, Any]) -> JoinKey:
    return (
        row_id(row),
        int(row["target_idx"]),
        int(row["event_count"]),
        str(row["dropped_event"]),
        str(row["full_text"]),
        str(row["drop_text"]),
    )


def count_duplicates(rows: List[Dict[str, Any]]) -> int:
    seen = set()
    duplicates = 0
    for row in rows:
        key = safe_drop_key(row)
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def unique_event_pool(data_dict: Dict[str, Dict[str, Any]]) -> List[str]:
    pool = set()
    for sample in data_dict.values():
        for text_entry in sample["text"]:
            pool.update(event.strip() for event in extract_events(text_entry) if event.strip())
    return sorted(pool, key=lambda event: (event.lower(), event))


def stable_rank(seed: int, sample_id: str, target_idx: int, target_event: str, candidate: str) -> str:
    payload = json.dumps(
        {
            "seed": seed,
            "sample_id": sample_id,
            "target_idx": target_idx,
            "target_event": target_event,
            "candidate": candidate,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def choose_replacement(
    *,
    sample_id: str,
    target_idx: int,
    target_event: str,
    source_events: List[str],
    event_pool: List[str],
    seed: int,
    length_window: int,
) -> Tuple[str, str, int]:
    source_set = set(source_events)
    target_len = len(target_event.split())
    length_matched = [
        event
        for event in event_pool
        if event not in source_set and event != target_event and abs(len(event.split()) - target_len) <= length_window
    ]
    candidates = length_matched
    policy_stage = f"global_pool_not_in_source_len_window_{length_window}"
    if not candidates:
        candidates = [event for event in event_pool if event not in source_set and event != target_event]
        policy_stage = "global_pool_not_in_source_fallback_any_length"
    if not candidates:
        raise ValueError(f"No replacement candidate for {sample_id} target_idx={target_idx}")

    replacement = min(
        candidates,
        key=lambda event: (stable_rank(seed, sample_id, target_idx, target_event, event), event),
    )
    return replacement, policy_stage, len(candidates)


def source_events_for_sample(data_dict: Dict[str, Dict[str, Any]], sample_id: str) -> List[str]:
    sample = data_dict[sample_id]
    return extract_events(pick_text_entry(sample))


def bucket_name(event_count: int) -> str:
    return str(event_count) if event_count < 5 else "5plus"


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_file = resolve_input(repo_root, args.data_file)
    data_dict = load_hml3de_test_dict(data_file)
    event_pool = unique_event_pool(data_dict)

    tmr_rows_path = resolve_input(repo_root, args.tmr_rows)
    chron_rows_path = resolve_input(repo_root, args.chron_rows)
    tmr_rows = read_jsonl(tmr_rows_path)
    chron_rows = read_jsonl(chron_rows_path)

    chron_by_key = {safe_drop_key(row): row for row in chron_rows}
    manifest_rows: List[Dict[str, Any]] = []
    skipped_missing_source = []
    skipped_source_mismatch = []
    skipped_no_replacement = []

    for tmr_row in tmr_rows:
        key = safe_drop_key(tmr_row)
        chron_row = chron_by_key.get(key)
        if chron_row is None:
            continue

        sample_id = row_id(tmr_row)
        if sample_id not in data_dict:
            skipped_missing_source.append(sample_id)
            continue

        source_events = source_events_for_sample(data_dict, sample_id)
        target_idx = int(tmr_row["target_idx"])
        if target_idx >= len(source_events):
            skipped_source_mismatch.append(sample_id)
            continue

        drop_events = list(source_events)
        dropped_event = drop_events.pop(target_idx)
        full_text = events_to_text(source_events)
        drop_text = events_to_text(drop_events)
        if (
            len(source_events) != int(tmr_row["event_count"])
            or dropped_event != tmr_row["dropped_event"]
            or full_text != tmr_row["full_text"]
            or drop_text != tmr_row["drop_text"]
        ):
            skipped_source_mismatch.append(sample_id)
            continue

        try:
            replacement_event, policy_stage, candidate_count = choose_replacement(
                sample_id=sample_id,
                target_idx=target_idx,
                target_event=dropped_event,
                source_events=source_events,
                event_pool=event_pool,
                seed=args.seed,
                length_window=args.length_window,
            )
        except ValueError:
            skipped_no_replacement.append(sample_id)
            continue

        replace_events = list(source_events)
        replace_events[target_idx] = replacement_event
        event_count = len(source_events)
        manifest_rows.append(
            {
                "manifest_id": f"{sample_id}__target{target_idx}",
                "sample_id": sample_id,
                "keyid": sample_id,
                "target_idx": target_idx,
                "event_count": event_count,
                "event_count_bucket": bucket_name(event_count),
                "full_text": full_text,
                "drop_text": drop_text,
                "dropped_event": dropped_event,
                "replacement_event": replacement_event,
                "replace_text": events_to_text(replace_events),
                "source_events": source_events,
                "deterministic_seed": args.seed,
                "replacement_policy": {
                    "name": "modebug_aligned_replace_v1",
                    "dataset": "HumanML3D-E data_test.npy",
                    "event_pool": "sorted unique HumanML3D-E decomposed events",
                    "selection": "min sha256(seed, sample_id, target_idx, target_event, candidate)",
                    "stage": policy_stage,
                    "candidate_count": candidate_count,
                    "length_window": args.length_window,
                    "excludes_original_source_events": True,
                },
                "alignment_source": {
                    "join": "safe_drop",
                    "fields": [
                        "sample_id/keyid",
                        "target_idx",
                        "event_count",
                        "dropped_event",
                        "full_text",
                        "drop_text",
                    ],
                    "tmr_rows": str(tmr_rows_path),
                    "chronaccret_rows": str(chron_rows_path),
                },
            }
        )

    manifest_rows.sort(key=lambda row: (row["sample_id"], row["target_idx"]))
    manifest_path = output_dir / "aligned_replace_manifest.jsonl"
    write_jsonl(manifest_path, manifest_rows)

    bucket_counts: Dict[str, int] = {}
    for row in manifest_rows:
        bucket = row["event_count_bucket"]
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    safe_drop_join_rows = len({safe_drop_key(row) for row in tmr_rows} & set(chron_by_key.keys()))
    summary = {
        "task_id": "MDBG-E4-ALIGNED-REPLACE-MANIFEST",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "artifact": "deterministic aligned replace corruption manifest",
            "dataset": "HumanML3D-E",
            "evaluator_side_cross_check": True,
            "not_final_standalone_judge": True,
            "excluded_extensions": ["MotionPatches", "ordering", "duration", "judge"],
        },
        "inputs": {
            "tmr_rows": str(tmr_rows_path),
            "chronaccret_rows": str(chron_rows_path),
            "data_file": str(data_file),
        },
        "policy": {
            "name": "modebug_aligned_replace_v1",
            "seed": args.seed,
            "length_window": args.length_window,
            "event_pool_size": len(event_pool),
            "selection": "stable sha256 rank over global HumanML3D-E event pool",
            "constraints": [
                "replacement_event != dropped_event",
                "replacement_event not in original source_events",
                "prefer absolute token-length difference <= length_window",
            ],
        },
        "row_counts": {
            "tmr_rows": len(tmr_rows),
            "chronaccret_rows": len(chron_rows),
            "safe_drop_join_rows": safe_drop_join_rows,
            "manifest_rows": len(manifest_rows),
            "skipped_missing_source": len(skipped_missing_source),
            "skipped_source_mismatch": len(skipped_source_mismatch),
            "skipped_no_replacement": len(skipped_no_replacement),
            "tmr_duplicate_safe_drop_keys": count_duplicates(tmr_rows),
            "chronaccret_duplicate_safe_drop_keys": count_duplicates(chron_rows),
        },
        "coverage": {
            "manifest_vs_tmr_rows": float(len(manifest_rows) / len(tmr_rows)) if tmr_rows else None,
            "manifest_vs_chronaccret_rows": float(len(manifest_rows) / len(chron_rows)) if chron_rows else None,
            "manifest_vs_safe_drop_join_rows": float(len(manifest_rows) / safe_drop_join_rows)
            if safe_drop_join_rows
            else None,
        },
        "event_count_buckets": dict(sorted(bucket_counts.items())),
        "outputs": {
            "manifest_jsonl": str(manifest_path),
            "summary_json": str(output_dir / "aligned_replace_manifest_summary.json"),
        },
    }
    with open(output_dir / "aligned_replace_manifest_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
