from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "logs"
    / "modebug_generation_observation"
    / "g1g2_condition_probe_64samples_step10"
)

LAYER_RE = re.compile(r"layers\.(\d+)")


class Stats:
    def __init__(self) -> None:
        self.records = 0
        self.target_available = 0
        self.finite_failures = 0
        self.entropy_values: List[float] = []
        self.mass_values: List[float] = []
        self.peak_t_values: List[float] = []
        self.peak_order_checked = 0
        self.peak_order_matches = 0
        self.condition_order_checked = 0
        self.condition_order_matches = 0
        self.target_rank_values: List[int] = []
        self.event_peak_first = Counter()

    def add(self, rec: Dict[str, Any]) -> None:
        self.records += 1
        if rec.get("finite") is False:
            self.finite_failures += 1

        order = rec.get("event_peak_order")
        if isinstance(order, list) and order:
            self.event_peak_first[str(order[0])] += 1
            valid_event_count = rec.get("valid_event_count")
            if isinstance(valid_event_count, int):
                self.condition_order_checked += 1
                if order == list(range(valid_event_count)):
                    self.condition_order_matches += 1

        target_idx = rec.get("condition_target_idx")
        target_available = bool(rec.get("target_event_available"))
        if target_available:
            self.target_available += 1
            entropy = rec.get("target_attn_entropy_norm")
            if isinstance(entropy, (int, float)):
                self.entropy_values.append(float(entropy))
            mass = rec.get("target_attn_mean_mass")
            if isinstance(mass, (int, float)):
                self.mass_values.append(float(mass))
            peak_t = rec.get("target_attn_peak_t")
            if isinstance(peak_t, (int, float)):
                self.peak_t_values.append(float(peak_t))
            if isinstance(target_idx, int) and isinstance(order, list) and order:
                self.peak_order_checked += 1
                if order[0] == target_idx:
                    self.peak_order_matches += 1
                if target_idx in order:
                    self.target_rank_values.append(order.index(target_idx) + 1)

    @staticmethod
    def _mean(values: List[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    def summary(self) -> Dict[str, Any]:
        entropy_mean = self._mean(self.entropy_values)
        mass_mean = self._mean(self.mass_values)
        peak_t_mean = self._mean(self.peak_t_values)
        rank_mean = self._mean([float(v) for v in self.target_rank_values])
        return {
            "records": self.records,
            "target_available_records": self.target_available,
            "finite_failures": self.finite_failures,
            "entropy_norm_mean": entropy_mean,
            "entropy_norm_min": min(self.entropy_values) if self.entropy_values else None,
            "target_mean_mass_mean": mass_mean,
            "target_peak_t_mean": peak_t_mean,
            "target_peak_t_median": median(self.peak_t_values) if self.peak_t_values else None,
            "peak_order_checked": self.peak_order_checked,
            "peak_order_top1_match_rate": (
                self.peak_order_matches / self.peak_order_checked
                if self.peak_order_checked
                else None
            ),
            "condition_order_match_rate": (
                self.condition_order_matches / self.condition_order_checked
                if self.condition_order_checked
                else None
            ),
            "condition_order_checked": self.condition_order_checked,
            "target_rank_mean": rank_mean,
            "event_peak_first_top": self.event_peak_first.most_common(5),
        }


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_layer(module: Any) -> str:
    if not isinstance(module, str):
        return "unknown"
    match = LAYER_RE.search(module)
    return match.group(1) if match else "unknown"


def has_head_dimension(keys: set, shapes: Counter) -> Tuple[bool, Dict[str, Any]]:
    candidate_keys = {"head", "head_idx", "head_index", "num_heads"}
    found_head_keys = candidate_keys.intersection(keys)
    return bool(found_head_keys), {
        "record_keys": sorted(keys),
        "head_keys_found": sorted(found_head_keys),
        "shape_counts": shapes.most_common(),
        "shape_semantics_from_instrumentation": "[batch, head, motion_patch, event_token]",
        "head_axis_note": (
            "The saved records keep the original attention tensor shape, but all saved "
            "metrics were already averaged across batch/head. No head id or per-head "
            "metric is present in observations.jsonl."
        ),
    }


def group_keys(row: Dict[str, Any], rec: Dict[str, Any]) -> Dict[str, Tuple[str, ...]]:
    condition = str(row.get("condition"))
    module = str(rec.get("module"))
    layer = extract_layer(module)
    step = str(rec.get("sample_step"))
    return {
        "condition": (condition,),
        "module": (module,),
        "layer": (layer,),
        "step": (step,),
        "module_step_condition": (module, step, condition),
        "layer_step_condition": (layer, step, condition),
        "module_condition": (module, condition),
        "layer_condition": (layer, condition),
        "step_condition": (step, condition),
    }


def add_pair(
    pairs: Dict[Tuple[str, str, str, str], Dict[str, Any]],
    row: Dict[str, Any],
    rec: Dict[str, Any],
) -> None:
    key = (
        str(row.get("sample_id")),
        str(rec.get("module")),
        str(rec.get("sample_step")),
        str(rec.get("record_idx")),
    )
    condition = str(row.get("condition"))
    pairs[key][condition] = {
        "entropy": rec.get("target_attn_entropy_norm"),
        "mass": rec.get("target_attn_mean_mass"),
        "peak_t": rec.get("target_attn_peak_t"),
        "order_top1": (
            rec.get("event_peak_order")[0]
            if isinstance(rec.get("event_peak_order"), list) and rec.get("event_peak_order")
            else None
        ),
    }


def mean_or_none(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def pair_summary(pairs: Dict[Tuple[str, str, str, str], Dict[str, Any]], condition: str) -> Dict[str, Any]:
    entropy_deltas: List[float] = []
    mass_deltas: List[float] = []
    peak_shifts: List[float] = []
    top1_changes = 0
    top1_checked = 0
    paired = 0
    for bundle in pairs.values():
        full = bundle.get("full")
        other = bundle.get(condition)
        if not full or not other:
            continue
        paired += 1
        if isinstance(full.get("entropy"), (int, float)) and isinstance(other.get("entropy"), (int, float)):
            entropy_deltas.append(float(other["entropy"]) - float(full["entropy"]))
        if isinstance(full.get("mass"), (int, float)) and isinstance(other.get("mass"), (int, float)):
            mass_deltas.append(float(other["mass"]) - float(full["mass"]))
        if isinstance(full.get("peak_t"), (int, float)) and isinstance(other.get("peak_t"), (int, float)):
            peak_shifts.append(abs(float(other["peak_t"]) - float(full["peak_t"])))
        if full.get("order_top1") is not None and other.get("order_top1") is not None:
            top1_checked += 1
            if full.get("order_top1") != other.get("order_top1"):
                top1_changes += 1
    return {
        "paired_records": paired,
        "entropy_delta_mean_vs_full": mean_or_none(entropy_deltas),
        "target_mean_mass_delta_mean_vs_full": mean_or_none(mass_deltas),
        "abs_peak_shift_mean_vs_full": mean_or_none(peak_shifts),
        "abs_peak_shift_median_vs_full": median(peak_shifts) if peak_shifts else None,
        "event_peak_top1_change_rate_vs_full": (
            top1_changes / top1_checked if top1_checked else None
        ),
        "event_peak_top1_checked": top1_checked,
    }


def summarize_named_groups(groups: Dict[str, Dict[Tuple[str, ...], Stats]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for name, table in groups.items():
        items = []
        for key, stats in table.items():
            item = stats.summary()
            item["key"] = list(key)
            items.append(item)
        output[name] = sorted(
            items,
            key=lambda x: (
                x["entropy_norm_mean"] is None,
                x["entropy_norm_mean"] if x["entropy_norm_mean"] is not None else 9.0,
                -float(x["peak_order_top1_match_rate"] or 0.0),
            ),
        )
    return output


def gate_verdict(
    head_available: bool,
    lowest_groups: List[Dict[str, Any]],
    best_order_groups: List[Dict[str, Any]],
) -> Dict[str, Any]:
    filtered_entropy_lt_095 = any(
        item.get("entropy_norm_mean") is not None and item["entropy_norm_mean"] < 0.95
        for item in lowest_groups
    )
    peak_match_gt_015 = any(
        item.get("condition_order_match_rate") is not None
        and item["condition_order_match_rate"] > 0.15
        for item in best_order_groups
    )
    supports_attention_path = bool(head_available and filtered_entropy_lt_095 and peak_match_gt_015)
    if not head_available:
        status = "blocked_no_per_head_artifact"
        rationale = (
            "observations.jsonl does not contain head ids or per-head metrics, so a "
            "minority usable head subset cannot be identified from this artifact."
        )
    elif supports_attention_path:
        status = "pass_filtered_attention_candidate"
        rationale = "Filtered groups pass both entropy and peak/order thresholds."
    else:
        status = "fail_raw_attention_filtering_gate"
        rationale = "Filtered groups do not pass both entropy and peak/order thresholds."
    return {
        "status": status,
        "supports_continuing_attention_path": supports_attention_path,
        "filtered_entropy_lt_0_95": filtered_entropy_lt_095,
        "peak_order_match_gt_0_15": peak_match_gt_015,
        "rationale": rationale,
    }


def write_markdown(path: Path, result: Dict[str, Any]) -> None:
    verdict = result["gate_verdict"]
    schema = result["schema"]
    lines = [
        "# MoDebug Attention Head Filtering Analysis",
        "",
        "## Schema Verdict",
        "",
        f"- Head dimension available for per-head analysis: `{schema['has_per_head_metrics']}`",
        f"- Record keys: `{', '.join(schema['record_schema']['record_keys'])}`",
        f"- Shape counts: `{schema['record_schema']['shape_counts']}`",
        f"- Note: {schema['record_schema']['head_axis_note']}",
        "",
        "## Gate Checks",
        "",
        f"- Filtered entropy < 0.95: `{verdict['filtered_entropy_lt_0_95']}`",
        f"- Peak/order match > 0.15: `{verdict['peak_order_match_gt_0_15']}`",
        f"- Final verdict: `{verdict['status']}`",
        f"- Supports continuing attention path: `{verdict['supports_continuing_attention_path']}`",
        f"- Rationale: {verdict['rationale']}",
        "",
        "## Lowest Entropy Top Groups",
        "",
        "| rank | module | step | condition | records | entropy_mean | entropy_min | peak_match | mass_mean |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, item in enumerate(result["top_groups"]["lowest_entropy_module_step_condition"], 1):
        module, step, condition = item["key"]
        lines.append(
            f"| {idx} | {module} | {step} | {condition} | {item['records']} | "
            f"{fmt(item['entropy_norm_mean'])} | {fmt(item['entropy_norm_min'])} | "
            f"{fmt(item['condition_order_match_rate'])} | {fmt(item['target_mean_mass_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Best Condition Order Match Groups",
            "",
            "| rank | module | step | condition | records | entropy_mean | condition_order_match | target_top1_match |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, item in enumerate(result["top_groups"]["best_condition_order_match_module_step_condition"], 1):
        module, step, condition = item["key"]
        lines.append(
            f"| {idx} | {module} | {step} | {condition} | {item['records']} | "
            f"{fmt(item['entropy_norm_mean'])} | {fmt(item['condition_order_match_rate'])} | "
            f"{fmt(item['peak_order_top1_match_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Condition Comparison",
            "",
            "| condition | records | target_records | entropy_mean | entropy_min | condition_order_match | target_top1_match | mass_mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition in ["full", "drop", "replace", "shuffle"]:
        item = result["groups"]["condition"].get(condition)
        if not item:
            continue
        lines.append(
            f"| {condition} | {item['records']} | {item['target_available_records']} | "
            f"{fmt(item['entropy_norm_mean'])} | {fmt(item['entropy_norm_min'])} | "
            f"{fmt(item['condition_order_match_rate'])} | "
            f"{fmt(item['peak_order_top1_match_rate'])} | {fmt(item['target_mean_mass_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Corruption Sensitivity vs Full",
            "",
            "| condition | paired | entropy_delta | mass_delta | abs_peak_shift_mean | abs_peak_shift_median | top1_change |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition in ["drop", "replace", "shuffle"]:
        item = result["counterfactual_pairs_vs_full"].get(condition)
        if not item:
            continue
        lines.append(
            f"| {condition} | {item['paired_records']} | {fmt(item['entropy_delta_mean_vs_full'])} | "
            f"{fmt(item['target_mean_mass_delta_mean_vs_full'])} | "
            f"{fmt(item['abs_peak_shift_mean_vs_full'])} | "
            f"{fmt(item['abs_peak_shift_median_vs_full'])} | "
            f"{fmt(item['event_peak_top1_change_rate_vs_full'])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    artifact_dir = args.artifact_dir
    observations_path = args.observations_jsonl or artifact_dir / "observations.jsonl"
    real_summary_path = artifact_dir / "real_observation_summary.json"
    prior_summary_path = artifact_dir / "g1g2_observation_analysis_summary.json"

    groups: Dict[str, Dict[Tuple[str, ...], Stats]] = defaultdict(lambda: defaultdict(Stats))
    pairs: Dict[Tuple[str, str, str, str], Dict[str, Any]] = defaultdict(dict)
    sample_ids = set()
    condition_rows = Counter()
    records_seen = 0
    record_keys = set()
    shape_counts = Counter()

    for row in iter_rows(observations_path):
        sample_ids.add(str(row.get("sample_id")))
        condition_rows[str(row.get("condition"))] += 1
        records = row.get("records") or []
        for rec in records:
            record_keys.update(rec.keys())
            shape = rec.get("shape")
            if isinstance(shape, list):
                shape_counts[str(shape)] += 1
            records_seen += 1
            for name, key in group_keys(row, rec).items():
                groups[name][key].add(rec)
            add_pair(pairs, row, rec)

    has_heads, record_schema = has_head_dimension(record_keys, shape_counts)
    summarized = summarize_named_groups(groups)

    condition_summary = {
        key[0]: value.summary() for key, value in groups["condition"].items()
    }
    compact_groups = {
        "condition": condition_summary,
        "layer": {key[0]: value.summary() for key, value in groups["layer"].items()},
        "step": {key[0]: value.summary() for key, value in groups["step"].items()},
        "layer_condition": {
            "|".join(key): value.summary() for key, value in groups["layer_condition"].items()
        },
    }
    lowest = summarized["module_step_condition"][: args.top_k]
    best_peak = sorted(
        summarized["module_step_condition"],
        key=lambda x: (
            x["peak_order_top1_match_rate"] is None,
            -(x["peak_order_top1_match_rate"] or 0.0),
            x["entropy_norm_mean"] if x["entropy_norm_mean"] is not None else 9.0,
        ),
    )[: args.top_k]
    best_order = sorted(
        summarized["module_step_condition"],
        key=lambda x: (
            x["condition_order_match_rate"] is None,
            -(x["condition_order_match_rate"] or 0.0),
            x["entropy_norm_mean"] if x["entropy_norm_mean"] is not None else 9.0,
        ),
    )[: args.top_k]
    verdict = gate_verdict(has_heads, summarized["module_step_condition"], best_order)

    result = {
        "task_id": "MDBG-P0-ATTENTION-HEAD-FILTERING",
        "inputs": {
            "observations_jsonl": repo_relative(observations_path),
            "real_observation_summary_json": repo_relative(real_summary_path),
            "prior_observation_analysis_summary_json": repo_relative(prior_summary_path),
        },
        "schema": {
            "has_per_head_metrics": has_heads,
            "per_head_analysis_status": "available" if has_heads else "not_available",
            "record_schema": record_schema,
        },
        "row_counts": {
            "condition_rows": sum(condition_rows.values()),
            "samples": len(sample_ids),
            "conditions": dict(sorted(condition_rows.items())),
            "attention_records": records_seen,
        },
        "groups": compact_groups,
        "top_groups": {
            "lowest_entropy_module_step_condition": lowest,
            "best_condition_order_match_module_step_condition": best_order,
            "best_peak_match_module_step_condition": best_peak,
            "lowest_entropy_layer_step_condition": summarized["layer_step_condition"][: args.top_k],
        },
        "counterfactual_pairs_vs_full": {
            condition: pair_summary(pairs, condition)
            for condition in ["drop", "replace", "shuffle"]
        },
        "gate_verdict": verdict,
        "source_summaries_loaded": {
            "real_observation_status": load_json(real_summary_path).get("status"),
            "prior_verdict": load_json(prior_summary_path).get("verdict"),
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MoDebug attention filtering from saved observation artifacts."
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--observations-jsonl", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.output_prefix or args.artifact_dir / "head_filtering_analysis"
    result = analyze(args)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(md_path, result)
    print(
        json.dumps(
            {
                "status": result["gate_verdict"]["status"],
                "json": repo_relative(json_path),
                "markdown": repo_relative(md_path),
                "has_per_head_metrics": result["schema"]["has_per_head_metrics"],
                "supports_continuing_attention_path": result["gate_verdict"][
                    "supports_continuing_attention_path"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
