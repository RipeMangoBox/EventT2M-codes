from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONDITIONS = ("full", "drop", "replace", "shuffle")
FIXED_SEED_IDS = ("004965", "008463", "001969", "003245")
TASK_ID = "MDBG-GCOND-MANIFEST"
SCHEMA_VERSION = "0.1.0-condition-manifest"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    linked_root = repo_root.parent
    parser = argparse.ArgumentParser(
        description="Build MoDebug G1/G2 generation observation condition manifest"
    )
    parser.add_argument(
        "--pool-manifest",
        type=Path,
        default=repo_root / "logs/modebug_observation_pool/manifest.jsonl",
    )
    parser.add_argument(
        "--hml3de-events-test",
        type=Path,
        default=linked_root / "datasets/HumanML3D-E/.tamr_hml3de_gt_events_test.json",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=repo_root / "logs/modebug_generation_observation/condition_manifest.jsonl",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=repo_root / "logs/modebug_generation_observation/condition_manifest_summary.json",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_global_event_pool(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    pool: list[str] = []
    for sample_entries in data.values():
        for events in sample_entries.values():
            for event in events:
                event = event.strip()
                if event and event not in seen:
                    seen.add(event)
                    pool.append(event)
    return pool


def events_to_text(events: list[str]) -> str:
    return " ".join(event.strip() for event in events if event.strip())


def stable_int(*parts: object) -> int:
    text = "||".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def choose_target_idx(sample: dict[str, Any]) -> tuple[int, str]:
    events = sample["events"]
    probe_idx = sample.get("target_idx_from_tmr_omission_probe")
    if isinstance(probe_idx, int) and 0 <= probe_idx < len(events):
        return probe_idx, "tmr_omission_probe_target_if_valid_else_middle_event"
    return len(events) // 2, "middle_event_fallback"


def choose_replacement(sample_id: str, events: list[str], target_idx: int, event_pool: list[str]) -> str:
    target = events[target_idx]
    original_events = set(events)
    candidates = [
        event
        for event in event_pool
        if event != target and event not in original_events
    ]
    if not candidates:
        candidates = [event for event in event_pool if event != target]
    if not candidates:
        raise RuntimeError(f"No replacement candidate available for {sample_id}")
    return min(candidates, key=lambda event: stable_int("replace", sample_id, target_idx, target, event))


def shuffle_permutation(sample_id: str, events: list[str]) -> list[int]:
    perm = sorted(
        range(len(events)),
        key=lambda idx: stable_int("shuffle", sample_id, idx, events[idx]),
    )
    if perm == list(range(len(events))):
        perm = perm[1:] + perm[:1]
    return perm


def build_rows(pool_rows: list[dict[str, Any]], event_pool: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in pool_rows:
        sample_id = sample["sample_id"]
        events = [event.strip() for event in sample["events"]]
        target_idx, target_policy = choose_target_idx(sample)
        target_event = events[target_idx]
        replacement = choose_replacement(sample_id, events, target_idx, event_pool)
        perm = shuffle_permutation(sample_id, events)

        condition_events = {
            "full": list(events),
            "drop": events[:target_idx] + events[target_idx + 1 :],
            "replace": events[:target_idx] + [replacement] + events[target_idx + 1 :],
            "shuffle": [events[idx] for idx in perm],
        }
        details = {
            "full": {
                "reference_condition": "full",
                "target_policy": target_policy,
                "corruption": "none",
            },
            "drop": {
                "reference_condition": "full",
                "target_policy": target_policy,
                "corruption": "drop_target_event",
                "corrupted_event_idx": target_idx,
                "corrupted_event_text": target_event,
            },
            "replace": {
                "reference_condition": "full",
                "target_policy": target_policy,
                "corruption": "replace_target_event_with_global_hml3de_test_distractor",
                "corrupted_event_idx": target_idx,
                "corrupted_event_text": target_event,
                "replacement_event_text": replacement,
                "replacement_policy": (
                    "deterministic_sha256_min_from_HumanML3D-E_test_global_event_pool_"
                    "excluding_target_and_original_events"
                ),
            },
            "shuffle": {
                "reference_condition": "full",
                "target_policy": target_policy,
                "corruption": "deterministic_event_order_permutation",
                "shuffle_permutation": perm,
                "shuffle_policy": "deterministic_sha256_sort_new_position_to_original_event_idx",
            },
        }

        for condition in CONDITIONS:
            row = {
                "schema_version": SCHEMA_VERSION,
                "task_id": TASK_ID,
                "source_dataset": "HumanML3D-E",
                "split": "test",
                "purpose": "generation_observation_input_only_not_evaluator_or_judge",
                "sample_id": sample_id,
                "selection_rank": sample.get("selection_rank"),
                "event_count": len(events),
                "event_bucket": sample["event_bucket"],
                "event_idx": target_idx,
                "target_idx": target_idx,
                "event_text": target_event,
                "full_text": sample["full_text"],
                "condition": condition,
                "condition_text": events_to_text(condition_events[condition]),
                "events": events,
                "condition_events": condition_events[condition],
                "condition_detail": details[condition],
            }
            rows.append(row)
    return rows


def validate(rows: list[dict[str, Any]], pool_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)

    conditions_by_sample = {
        sample_id: sorted(row["condition"] for row in sample_rows)
        for sample_id, sample_rows in by_sample.items()
    }
    per_sample_four_conditions = all(
        conditions == sorted(CONDITIONS) for conditions in conditions_by_sample.values()
    )
    all_event_count_gte_3 = all(row["event_count"] >= 3 for row in rows)
    drop_valid = all(
        row["condition_events"] == row["events"][: row["target_idx"]] + row["events"][row["target_idx"] + 1 :]
        for row in rows
        if row["condition"] == "drop"
    )
    replace_valid = all(
        len(row["condition_events"]) == row["event_count"]
        and row["condition_events"][row["target_idx"]] == row["condition_detail"]["replacement_event_text"]
        and row["condition_detail"]["replacement_event_text"] != row["event_text"]
        and row["condition_detail"]["replacement_event_text"] not in row["events"]
        for row in rows
        if row["condition"] == "replace"
    )
    shuffle_valid = all(
        sorted(row["condition_detail"]["shuffle_permutation"]) == list(range(row["event_count"]))
        and row["condition_events"] != row["events"]
        and sorted(row["condition_events"]) == sorted(row["events"])
        for row in rows
        if row["condition"] == "shuffle"
    )
    fixed_seed_presence = {
        sample_id: any(row["sample_id"] == sample_id for row in pool_rows)
        for sample_id in FIXED_SEED_IDS
    }

    return {
        "per_sample_four_conditions": per_sample_four_conditions,
        "all_event_count_gte_3": all_event_count_gte_3,
        "drop_valid": drop_valid,
        "replace_valid": replace_valid,
        "shuffle_valid": shuffle_valid,
        "fixed_seed_presence": fixed_seed_presence,
        "fixed_seed_complete": all(fixed_seed_presence.values()),
    }


def summarize(rows: list[dict[str, Any]], pool_rows: list[dict[str, Any]], event_pool: list[str]) -> dict[str, Any]:
    validation = validate(rows, pool_rows)
    condition_counts = Counter(row["condition"] for row in rows)
    bucket_counts = Counter(row["event_bucket"] for row in pool_rows)
    row_bucket_counts = Counter(row["event_bucket"] for row in rows)
    target_policy_counts = Counter(
        row["condition_detail"]["target_policy"]
        for row in rows
        if row["condition"] == "full"
    )

    summary = {
        "task_id": TASK_ID,
        "purpose": "Generation observation condition manifest for G1/G2 runners only; not evaluator or judge.",
        "source_dataset": "HumanML3D-E",
        "split": "test",
        "motionpatches_used": False,
        "model_run": False,
        "model_code_modified": False,
        "conditions": list(CONDITIONS),
        "sample_count": len(pool_rows),
        "condition_rows": len(rows),
        "condition_counts": dict(sorted(condition_counts.items())),
        "event_bucket_distribution": dict(sorted(bucket_counts.items())),
        "condition_row_event_bucket_distribution": dict(sorted(row_bucket_counts.items())),
        "event_count_min": min(row["event_count"] for row in rows) if rows else None,
        "event_count_max": max(row["event_count"] for row in rows) if rows else None,
        "global_hml3de_test_event_pool_size": len(event_pool),
        "target_policy_counts": dict(sorted(target_policy_counts.items())),
        "fixed_seed_presence": validation["fixed_seed_presence"],
        "fixed_seed_complete": validation["fixed_seed_complete"],
        "validation": {
            "per_sample_four_conditions": validation["per_sample_four_conditions"],
            "all_event_count_gte_3": validation["all_event_count_gte_3"],
            "drop_valid": validation["drop_valid"],
            "replace_valid": validation["replace_valid"],
            "shuffle_valid": validation["shuffle_valid"],
        },
    }
    if not all(summary["validation"].values()) or not summary["fixed_seed_complete"]:
        raise RuntimeError(f"Condition manifest validation failed: {summary['validation']}")
    return summary


def main() -> None:
    args = parse_args()
    pool_rows = read_jsonl(args.pool_manifest)
    event_pool = load_global_event_pool(args.hml3de_events_test)
    rows = build_rows(pool_rows, event_pool)
    summary = summarize(rows, pool_rows, event_pool)

    write_jsonl(args.output_jsonl, rows)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
