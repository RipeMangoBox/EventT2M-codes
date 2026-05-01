from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


JoinKey = Tuple[str, int, int, str, str, str]
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "back",
    "be",
    "being",
    "both",
    "down",
    "for",
    "from",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "left",
    "man",
    "of",
    "on",
    "one",
    "person",
    "right",
    "she",
    "someone",
    "the",
    "their",
    "then",
    "to",
    "up",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MoDebug hard-negative replace manifest.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--tmr-root", type=str, default=None)
    parser.add_argument("--tmr-run-dir", type=str, default=None)
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
    parser.add_argument("--output-dir", type=str, default="logs/modebug_hard_replace_eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument(
        "--candidate-backend",
        choices=["auto", "tmr", "lexical"],
        default="auto",
        help="auto tries TMR text cosine and falls back to lexical overlap.",
    )
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


def events_to_text(events: Sequence[str]) -> str:
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


def bucket_name(event_count: int) -> str:
    return str(event_count) if event_count < 5 else "5plus"


def stable_rank(seed: int, payload: Dict[str, Any]) -> str:
    text = json.dumps({"seed": seed, **payload}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize(text: str) -> List[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def lexical_score(target: str, candidate: str) -> float:
    target_tokens = tokenize(target)
    candidate_tokens = tokenize(candidate)
    return lexical_score_from_tokens(target_tokens, candidate_tokens)


def lexical_score_from_tokens(target_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> float:
    target_set = set(target_tokens)
    candidate_set = set(candidate_tokens)
    if not target_set or not candidate_set:
        jaccard = 0.0
    else:
        jaccard = len(target_set & candidate_set) / len(target_set | candidate_set)
    verb_overlap = 1.0 if target_tokens and candidate_tokens and target_tokens[0] == candidate_tokens[0] else 0.0
    length_gap = abs(len(target_tokens) - len(candidate_tokens))
    length_score = 1.0 / (1.0 + length_gap)
    return 0.75 * jaccard + 0.15 * verb_overlap + 0.10 * length_score


def build_event_entries(data_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for sample_id in sorted(data_dict):
        for text_entry_idx, text_entry in enumerate(data_dict[sample_id]["text"]):
            for event_idx, event in enumerate(extract_events(text_entry)):
                entries.append(
                    {
                        "text": event.strip(),
                        "sample_id": sample_id,
                        "text_entry_idx": text_entry_idx,
                        "event_idx": event_idx,
                        "tokens": tokenize(event),
                    }
                )
    entries.sort(key=lambda item: (item["text"].lower(), item["sample_id"], item["text_entry_idx"], item["event_idx"]))
    return entries


def unique_texts(entries: Sequence[Dict[str, Any]]) -> List[str]:
    return sorted({entry["text"] for entry in entries}, key=lambda text: (text.lower(), text))


def source_events_for_sample(data_dict: Dict[str, Dict[str, Any]], sample_id: str) -> List[str]:
    return extract_events(pick_text_entry(data_dict[sample_id]))


def collect_base_rows(
    data_dict: Dict[str, Dict[str, Any]],
    tmr_rows: List[Dict[str, Any]],
    chron_rows: List[Dict[str, Any]],
    max_rows: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    chron_by_key = {safe_drop_key(row): row for row in chron_rows}
    base_rows: List[Dict[str, Any]] = []
    skipped_missing_source = 0
    skipped_source_mismatch = 0

    for tmr_row in tmr_rows:
        if max_rows > 0 and len(base_rows) >= max_rows:
            break
        key = safe_drop_key(tmr_row)
        if key not in chron_by_key:
            continue

        sample_id = row_id(tmr_row)
        if sample_id not in data_dict:
            skipped_missing_source += 1
            continue

        source_events = source_events_for_sample(data_dict, sample_id)
        target_idx = int(tmr_row["target_idx"])
        if target_idx >= len(source_events):
            skipped_source_mismatch += 1
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
            skipped_source_mismatch += 1
            continue

        base_rows.append(
            {
                "sample_id": sample_id,
                "target_idx": target_idx,
                "event_count": len(source_events),
                "event_count_bucket": bucket_name(len(source_events)),
                "full_text": full_text,
                "drop_text": drop_text,
                "dropped_event": dropped_event,
                "source_events": source_events,
            }
        )

    safe_drop_join_rows = len({safe_drop_key(row) for row in tmr_rows} & set(chron_by_key.keys()))
    counts = {
        "safe_drop_join_rows": safe_drop_join_rows,
        "base_rows": len(base_rows),
        "skipped_missing_source": skipped_missing_source,
        "skipped_source_mismatch": skipped_source_mismatch,
        "tmr_duplicate_safe_drop_keys": count_duplicates(tmr_rows),
        "chronaccret_duplicate_safe_drop_keys": count_duplicates(chron_rows),
    }
    return base_rows, counts


def encode_tmr_texts(
    *,
    repo_root: Path,
    tmr_root: Path,
    tmr_run_dir: Path,
    device: str,
    texts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    import torch

    from planb.tmr_runtime import load_tmr_runtime

    runtime = load_tmr_runtime(repo_root=repo_root, tmr_root=tmr_root, run_dir=tmr_run_dir, device=device)
    latents: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            text_x_dict = runtime.collate_x_dict(runtime.text_model(batch), device=runtime.device)
            latent = runtime.model.encode(text_x_dict, sample_mean=True)[0]
            latent = torch.nn.functional.normalize(latent, dim=-1)
            latents.append(latent.detach().cpu().numpy())
    return np.concatenate(latents, axis=0)


def hard_replacement_for_row(
    *,
    row: Dict[str, Any],
    event_entries: Sequence[Dict[str, Any]],
    entry_scores: np.ndarray,
    seed: int,
    top_k: int,
) -> Tuple[Dict[str, Any], int]:
    source_set = set(row["source_events"])
    source_sample_id = row["sample_id"]
    candidates: List[Tuple[float, int, Dict[str, Any]]] = []
    for idx, (score, entry) in enumerate(zip(entry_scores.tolist(), event_entries)):
        event = entry["text"]
        if entry["sample_id"] == source_sample_id:
            continue
        if event == row["dropped_event"] or event in source_set:
            continue
        if not event.strip():
            continue
        candidates.append((float(score), idx, entry))
    if not candidates:
        raise ValueError(f"No hard replacement candidate for {source_sample_id} target_idx={row['target_idx']}")

    candidate_scores = np.array([candidate[0] for candidate in candidates], dtype=np.float32)
    shortlist_size = min(max(1, top_k), len(candidates))
    if len(candidates) > shortlist_size:
        shortlist_indices = np.argpartition(-candidate_scores, shortlist_size - 1)[:shortlist_size]
        shortlist = [candidates[int(idx)] for idx in shortlist_indices]
    else:
        shortlist = candidates
    shortlist.sort(
        key=lambda item: (
            -item[0],
            stable_rank(
                seed,
                {
                    "sample_id": source_sample_id,
                    "target_idx": row["target_idx"],
                    "target_event": row["dropped_event"],
                    "candidate": item[2],
                },
            ),
        )
    )
    score, _, entry = min(
        shortlist,
        key=lambda item: stable_rank(
            seed,
            {
                "sample_id": source_sample_id,
                "target_idx": row["target_idx"],
                "target_event": row["dropped_event"],
                "candidate": item[2],
                "stage": "hard_top_k_tiebreak",
            },
        ),
    )
    return {**entry, "similarity_score": float(score)}, len(candidates)


def make_markdown_summary(summary: Dict[str, Any]) -> str:
    backend = summary["policy"]["candidate_backend"]
    rows = summary["row_counts"]["manifest_rows"]
    old_acc = summary["comparison"]["old_aligned_replace_tmr_full_gt_replace"]
    lines = [
        "# MoDebug hard-replace manifest summary",
        "",
        f"- Dataset: HumanML3D-E",
        f"- Candidate backend: `{backend}`",
        f"- Manifest rows: {rows}",
        f"- Old aligned-replace TMR full>replace accuracy: {old_acc}",
        f"- Manifest: `{summary['outputs']['manifest_jsonl']}`",
        "",
        "This manifest excludes MotionPatches and does not enter the formal Event-T2M chain.",
    ]
    if backend == "lexical":
        lines.append("")
        lines.append("Note: candidates are lexical hard negatives, not TMR text-embedding hard negatives.")
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

    data_file = resolve_input(repo_root, args.data_file)
    data_dict = load_hml3de_test_dict(data_file)
    tmr_rows_path = resolve_input(repo_root, args.tmr_rows)
    chron_rows_path = resolve_input(repo_root, args.chron_rows)
    tmr_rows = read_jsonl(tmr_rows_path)
    chron_rows = read_jsonl(chron_rows_path)
    base_rows, counts = collect_base_rows(data_dict, tmr_rows, chron_rows, args.max_rows)
    if not base_rows:
        raise RuntimeError("No eligible base rows for hard replace manifest.")

    event_entries = build_event_entries(data_dict)
    backend = args.candidate_backend
    backend_error = None
    text_to_embedding: Dict[str, np.ndarray] = {}

    if backend in {"auto", "tmr"}:
        try:
            texts = unique_texts(event_entries)
            text_embeddings = encode_tmr_texts(
                repo_root=repo_root,
                tmr_root=tmr_root,
                tmr_run_dir=tmr_run_dir,
                device=args.device,
                texts=texts,
                batch_size=args.embedding_batch_size,
            )
            text_to_embedding = {text: emb for text, emb in zip(texts, text_embeddings)}
            backend = "tmr"
        except Exception as exc:
            if args.candidate_backend == "tmr":
                raise
            backend = "lexical"
            backend_error = repr(exc)

    manifest_rows: List[Dict[str, Any]] = []
    skipped_no_replacement = 0
    for row in base_rows:
        if backend == "tmr":
            target_embedding = text_to_embedding[row["dropped_event"]]
            entry_scores = np.array([float(np.dot(target_embedding, text_to_embedding[entry["text"]])) for entry in event_entries])
            similarity_name = "tmr_text_cosine"
        else:
            target_tokens = tokenize(row["dropped_event"])
            entry_scores = np.array(
                [lexical_score_from_tokens(target_tokens, entry["tokens"]) for entry in event_entries],
                dtype=np.float32,
            )
            similarity_name = "lexical_jaccard_verb_length"

        try:
            replacement, candidate_count = hard_replacement_for_row(
                row=row,
                event_entries=event_entries,
                entry_scores=entry_scores,
                seed=args.seed,
                top_k=args.top_k,
            )
        except ValueError:
            skipped_no_replacement += 1
            continue

        replace_events = list(row["source_events"])
        replace_events[row["target_idx"]] = replacement["text"]
        manifest_rows.append(
            {
                "manifest_id": f"{row['sample_id']}__target{row['target_idx']}",
                "sample_id": row["sample_id"],
                "keyid": row["sample_id"],
                "target_idx": row["target_idx"],
                "target_event_text": row["dropped_event"],
                "event_count": row["event_count"],
                "event_count_bucket": row["event_count_bucket"],
                "full_text": row["full_text"],
                "drop_text": row["drop_text"],
                "dropped_event": row["dropped_event"],
                "replacement_event": replacement["text"],
                "replacement_event_text": replacement["text"],
                "replacement_event_source": {
                    "sample_id": replacement["sample_id"],
                    "text_entry_idx": replacement["text_entry_idx"],
                    "event_idx": replacement["event_idx"],
                },
                "replace_text": events_to_text(replace_events),
                "source_events": row["source_events"],
                "deterministic_seed": args.seed,
                "hard_negative_score": replacement["similarity_score"],
                "similarity_score": replacement["similarity_score"],
                "similarity_backend": similarity_name,
                "replacement_policy": {
                    "name": "modebug_hard_replace_v1",
                    "dataset": "HumanML3D-E data_test.npy",
                    "candidate_backend": backend,
                    "similarity": similarity_name,
                    "selection": "sample deterministic item from top-k similar non-source event candidates",
                    "top_k": args.top_k,
                    "candidate_count": candidate_count,
                    "excludes_original_source_events": True,
                    "excludes_same_source_sample": True,
                    "excludes_identical_target_event_text": True,
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

    manifest_rows.sort(key=lambda item: (item["sample_id"], item["target_idx"]))
    manifest_path = output_dir / "hard_replace_manifest.jsonl"
    write_jsonl(manifest_path, manifest_rows)

    bucket_counts: Dict[str, int] = {}
    scores = [row["similarity_score"] for row in manifest_rows]
    for row in manifest_rows:
        bucket_counts[row["event_count_bucket"]] = bucket_counts.get(row["event_count_bucket"], 0) + 1
    selection_description = (
        "TMR text cosine top-k from event pool"
        if backend == "tmr"
        else "lexical Jaccard/verb/length overlap top-k from event pool"
    )

    summary = {
        "task_id": "MDBG-P0-HARD-REPLACE-MANIFEST",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "artifact": "hard-negative replace corruption manifest",
            "dataset": "HumanML3D-E",
            "evaluator_side_cross_check": True,
            "not_final_standalone_judge": True,
            "excluded_extensions": ["MotionPatches", "attention_filtering", "ordering", "duration", "judge"],
        },
        "inputs": {
            "tmr_rows": str(tmr_rows_path),
            "chronaccret_rows": str(chron_rows_path),
            "data_file": str(data_file),
        },
        "policy": {
            "name": "modebug_hard_replace_v1",
            "candidate_backend": backend,
            "backend_error": backend_error,
            "seed": args.seed,
            "top_k": args.top_k,
            "max_rows": args.max_rows,
            "event_entry_count": len(event_entries),
            "unique_event_text_count": len(unique_texts(event_entries)),
            "selection": selection_description,
        },
        "row_counts": {
            "tmr_rows": len(tmr_rows),
            "chronaccret_rows": len(chron_rows),
            **counts,
            "manifest_rows": len(manifest_rows),
            "skipped_no_replacement": skipped_no_replacement,
        },
        "coverage": {
            "manifest_vs_tmr_rows": float(len(manifest_rows) / len(tmr_rows)) if tmr_rows else None,
            "manifest_vs_chronaccret_rows": float(len(manifest_rows) / len(chron_rows)) if chron_rows else None,
            "manifest_vs_safe_drop_join_rows": float(len(manifest_rows) / counts["safe_drop_join_rows"])
            if counts["safe_drop_join_rows"]
            else None,
        },
        "event_count_buckets": dict(sorted(bucket_counts.items())),
        "similarity": {
            "backend": backend,
            "mean": float(np.mean(scores)) if scores else None,
            "median": float(np.median(scores)) if scores else None,
            "min": float(np.min(scores)) if scores else None,
            "max": float(np.max(scores)) if scores else None,
        },
        "comparison": {
            "old_aligned_replace_tmr_full_gt_replace": 0.835820895522388,
            "old_aligned_replace_summary": str(repo_root / "logs/modebug_aligned_replace_eval/tmr_aligned_replace_summary.json"),
        },
        "outputs": {
            "manifest_jsonl": str(manifest_path),
            "summary_json": str(output_dir / "hard_replace_manifest_summary.json"),
            "summary_md": str(output_dir / "hard_replace_manifest_summary.md"),
        },
    }
    summary_path = output_dir / "hard_replace_manifest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "hard_replace_manifest_summary.md", "w", encoding="utf-8") as handle:
        handle.write(make_markdown_summary(summary))

    if not math.isfinite(float(summary["similarity"]["mean"] or 0.0)):
        raise RuntimeError("Invalid similarity summary.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
