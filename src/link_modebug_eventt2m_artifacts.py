from __future__ import annotations

import argparse
import os
from pathlib import Path


LINKS = {
    "eval/geometry_audit.csv": "results/eventt2m/geometry_audit.csv",
    "eval/geometry_trace_audit_summary.md": "results/eventt2m/geometry_trace_audit_summary.md",
    "eval/run_manifest.json": "results/eventt2m/run_manifest.json",
    "network_activation/trace_summary.jsonl": "results/eventt2m/trace_summary.jsonl",
    "network_activation/traces": "results/eventt2m/traces",
    "vis/static_plots": "results/eventt2m/static_plots",
    "vis/native_animations": "results/eventt2m/videos",
    "native_outputs/generation_root": "results/eventt2m/native_outputs",
    "native_outputs/joints": "results/eventt2m/native_outputs/joints",
    "native_outputs/raw_263": "results/eventt2m/native_outputs/raw_263",
}


def relative_target(link_path: Path, target_path: Path) -> Path:
    return Path(os.path.relpath(target_path, start=link_path.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the MoDebug by_model/eventt2m symlink index.")
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--model", default="eventt2m")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    model_root = root / "by_model" / args.model
    for rel_link, rel_target in LINKS.items():
        link_path = model_root / rel_link
        target_path = root / rel_target
        if not target_path.exists():
            continue
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.is_symlink() or link_path.exists():
            if not args.overwrite:
                continue
            if link_path.is_dir() and not link_path.is_symlink():
                raise IsADirectoryError(f"refusing to replace directory: {link_path}")
            link_path.unlink()
        link_path.symlink_to(relative_target(link_path, target_path))
    print(model_root)


if __name__ == "__main__":
    main()
