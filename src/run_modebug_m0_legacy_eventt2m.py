from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rootutils
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.humanml.scripts.paramUtil import t2m_kinematic_chain
from src.planb.eventt2m_runtime import generate_raw_motion, load_eventt2m_runtime, recover_joints


MODEL_NAME = "eventt2m"
DEFAULT_OVERRIDES = [
    "model.noise_scheduler.prediction_type=sample",
    "model.denoiser.stage_dim=256*4",
]
CSV_FIELDS = [
    "index",
    "model",
    "prompt_id",
    "base_id",
    "category",
    "condition",
    "trace_tier",
    "prompt",
    "expected_length",
    "main_npy",
    "ik_npy",
    "mp4",
    "bvh",
    "status",
    "motion_length",
    "finite_rate",
    "mean_joint_step",
    "max_joint_step",
    "root_x_span",
    "root_z_span",
    "static_plot",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "item"


def split_prompt_events(prompt: str, max_events: int = 11) -> list[str]:
    parts = [
        part.strip()
        for part in re.split(r"\s*(?:,|;|\band then\b|\bthen\b)\s*", prompt)
        if part.strip()
    ]
    if not parts:
        return [prompt.strip()]
    events = [part if part.endswith(".") else f"{part}." for part in parts]
    if len(events) <= max_events:
        return events
    return [*events[: max_events - 1], " ".join(events[max_events - 1 :])]


def geometry_summary(joints: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(joints)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    finite = np.isfinite(arr)
    frame_delta = np.linalg.norm(np.diff(arr, axis=0), axis=-1) if len(arr) > 1 else np.zeros((0, arr.shape[1]))
    root_xz = arr[:, 0, [0, 2]]
    root_span = root_xz.max(axis=0) - root_xz.min(axis=0)
    return {
        "shape": list(arr.shape),
        "length": int(arr.shape[0]),
        "finite_rate": float(finite.mean()),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "abs_mean": float(np.abs(arr).mean()),
        "mean_joint_step": float(frame_delta.mean()) if frame_delta.size else 0.0,
        "max_joint_step": float(frame_delta.max()) if frame_delta.size else 0.0,
        "root_x_span": float(root_span[0]),
        "root_z_span": float(root_span[1]),
    }


def save_static_plot(joints: np.ndarray, out_path: Path, title: str) -> None:
    arr = np.asarray(joints)
    frame = arr[min(len(arr) // 2, len(arr) - 1)]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for chain in t2m_kinematic_chain:
        pts = frame[chain]
        ax.plot(pts[:, 0], pts[:, 2], pts[:, 1], linewidth=2)
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=10)
    flat = frame.reshape(-1, 3)
    center = flat.mean(axis=0)
    radius = max(float(np.abs(flat - center).max()), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(center[1] - radius, center[1] + radius)
    ax.set_title(title[:80])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_xz_plane(ax, minx: float, maxx: float, miny: float, minz: float, maxz: float) -> None:
    verts = [
        [minx, miny, minz],
        [minx, miny, maxz],
        [maxx, miny, maxz],
        [maxx, miny, minz],
    ]
    xz_plane = Poly3DCollection([verts])
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
    ax.add_collection3d(xz_plane)


def render_humanml3d_style(save_path: Path, joints: np.ndarray, title: str, fps: float = 20.0, radius: float = 4.0) -> None:
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    colors = [
        "red",
        "blue",
        "black",
        "red",
        "blue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkred",
        "darkred",
        "darkred",
        "darkred",
        "darkred",
    ]

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        fig.suptitle(title, fontsize=20)
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.axis("off")

    init()
    mins = data.min(axis=0).min(axis=0)
    maxs = data.max(axis=0).max(axis=0)

    height_offset = mins[1]
    data[:, :, 1] -= height_offset
    traj = data[:, 0, [0, 2]]
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index: int):
        ax.cla()
        init()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xz_plane(ax, mins[0] - traj[index, 0], maxs[0] - traj[index, 0], 0, mins[2] - traj[index, 1], maxs[2] - traj[index, 1])
        if index > 1:
            ax.plot3D(
                traj[:index, 0] - traj[index, 0],
                np.zeros_like(traj[:index, 0]),
                traj[:index, 1] - traj[index, 1],
                linewidth=1.0,
                color="blue",
            )
        for i, (chain, color) in enumerate(zip(t2m_kinematic_chain, colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)

    ani = FuncAnimation(fig, update, frames=data.shape[0], interval=1000 / fps, repeat=False)
    ani.save(str(save_path), fps=fps)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_md(path: Path, rows: list[dict[str, Any]], output_root: Path) -> None:
    generated = sum(1 for row in rows if row["status"] == "ok")
    raw = sum(1 for row in rows if row["trace_tier"] == "raw")
    lines = [
        "# M0 v2 EventT2M Geometry And Trace Audit",
        "",
        f"- date: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- model: `{MODEL_NAME}`",
        f"- output_root: `{output_root}`",
        f"- prompts: `{len(rows)}`",
        f"- generated_ok: `{generated}`",
        f"- raw_trace_policy_rows: `{raw}`",
        "- role: `diagnostic`",
        "- used_for: `observation`",
        "- evaluator: `modebug_m0_v2_geometry_trace_audit`",
        "- limitations: synthetic M0 v2 battery only; not a final HumanML3D metric evaluator",
        "",
        "| prompt_id | category | condition | status | length | finite_rate | mean_joint_step | root_x_span | root_z_span | static_plot |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['prompt_id']}` | `{row['category']}` | `{row['condition']}` | `{row['status']}` | "
            f"{row.get('motion_length', '')} | {row.get('finite_rate', '')} | {row.get('mean_joint_step', '')} | "
            f"{row.get('root_x_span', '')} | {row.get('root_z_span', '')} | `{row.get('static_plot', '')}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EventT2M on the MoDebug archived legacy M0 v2 battery.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path, default=Path("checkpoints/pretrained/HumanML3D/hml3d.ckpt"))
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/HumanML3D"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--step-num", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--render-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--extra-override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest = args.manifest.resolve()
    result_dir = args.result_dir.resolve()
    native_dir = result_dir / "native_outputs"
    raw_dir = native_dir / "raw_263"
    joints_dir = native_dir / "joints"
    video_dir = result_dir / "videos"
    plot_dir = result_dir / "static_plots"
    trace_dir = result_dir / "traces"

    rows = read_manifest(manifest)
    if args.limit > 0:
        rows = rows[: args.limit]

    overrides = [*DEFAULT_OVERRIDES, *args.extra_override]
    runtime = load_eventt2m_runtime(
        ckpt_path=(repo_root / args.ckpt_path).resolve() if not args.ckpt_path.is_absolute() else args.ckpt_path,
        device=args.device,
        step_num=args.step_num,
        data_dir=(repo_root / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir,
        extra_overrides=overrides,
    )

    audit_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prompt_id = row["prompt_id"]
        item_id = safe_id(prompt_id)
        length = int(row["length"])
        raw_path = raw_dir / f"{index:03d}_{item_id}.npy"
        joints_path = joints_dir / f"{index:03d}_{item_id}.npy"
        static_plot = plot_dir / f"{item_id}_static.png"
        video_path = video_dir / f"{index:03d}_{item_id}.mp4"
        out_row: dict[str, Any] = {
            "index": index,
            "model": MODEL_NAME,
            "prompt_id": prompt_id,
            "base_id": row["base_id"],
            "category": row["category"],
            "condition": row["condition"],
            "trace_tier": row["trace_tier"],
            "prompt": row["prompt"],
            "expected_length": row["length"],
            "main_npy": str(joints_path),
            "ik_npy": "",
            "mp4": str(video_path) if args.render_videos else "",
            "bvh": "",
            "status": "missing",
            "motion_length": "",
            "finite_rate": "",
            "mean_joint_step": "",
            "max_joint_step": "",
            "root_x_span": "",
            "root_z_span": "",
            "static_plot": "",
        }
        trace_row: dict[str, Any] = {
            "index": index,
            "model": MODEL_NAME,
            "prompt_id": prompt_id,
            "trace_tier": row["trace_tier"],
            "internal_attention_available": False,
            "internal_activation_available": False,
            "internal_logits_available": False,
            "trace_source": "eventt2m_fixed_sampling_output_audit",
            "event_decomposition_source": "rule_split_prompt_text",
            "eventt2m_sampling_overrides": overrides,
            "limitations": "Native EventT2M generation output audited; model-internal hooks were not used in this pass.",
        }
        try:
            if not joints_path.exists() or args.overwrite:
                events = split_prompt_events(row["prompt"])
                raw = generate_raw_motion(runtime, row["prompt"], events, length, seed=args.seed + index)
                joints = recover_joints(raw)
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                joints_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(raw_path, raw.astype(np.float32))
                np.save(joints_path, joints.astype(np.float32))
            else:
                joints = np.load(joints_path)

            geom = geometry_summary(joints)
            save_static_plot(joints, static_plot, prompt_id)
            if args.render_videos and (args.overwrite or not video_path.exists()):
                video_path.parent.mkdir(parents=True, exist_ok=True)
                render_humanml3d_style(video_path, joints, title=prompt_id, fps=20.0)

            out_row.update(
                {
                    "status": "ok",
                    "motion_length": geom["length"],
                    "finite_rate": f"{geom['finite_rate']:.6f}",
                    "mean_joint_step": f"{geom['mean_joint_step']:.6f}",
                    "max_joint_step": f"{geom['max_joint_step']:.6f}",
                    "root_x_span": f"{geom['root_x_span']:.6f}",
                    "root_z_span": f"{geom['root_z_span']:.6f}",
                    "static_plot": str(static_plot),
                }
            )
            trace_row.update({"motion_geometry": geom, "static_plot": str(static_plot), "raw_263_npy": str(raw_path)})
            if row["trace_tier"] == "raw":
                prompt_trace_dir = trace_dir / item_id
                prompt_trace_dir.mkdir(parents=True, exist_ok=True)
                (prompt_trace_dir / "summary.json").write_text(json.dumps(trace_row, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            out_row.update({"status": "error"})
            trace_row.update({"error": repr(exc)})
        audit_rows.append(out_row)
        trace_rows.append(trace_row)

        if (index + 1) % 10 == 0 or index == len(rows) - 1:
            print(f"[eventt2m] processed {index + 1}/{len(rows)}", flush=True)

    result_dir.mkdir(parents=True, exist_ok=True)
    write_csv(result_dir / "geometry_audit.csv", audit_rows)
    write_jsonl(result_dir / "trace_summary.jsonl", trace_rows)
    write_summary_md(result_dir / "geometry_trace_audit_summary.md", audit_rows, native_dir)
    run_manifest = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "manifest": str(manifest),
        "output_root": str(native_dir),
        "result_dir": str(result_dir),
        "prompts": len(rows),
        "generated_ok": sum(1 for row in audit_rows if row["status"] == "ok"),
        "missing_or_error": sum(1 for row in audit_rows if row["status"] != "ok"),
        "eventt2m_git_head": git_head(repo_root),
        "eventt2m_ckpt": str((repo_root / args.ckpt_path).resolve() if not args.ckpt_path.is_absolute() else args.ckpt_path),
        "eventt2m_ckpt_sha256": sha256_file((repo_root / args.ckpt_path).resolve() if not args.ckpt_path.is_absolute() else args.ckpt_path),
        "data_dir": str((repo_root / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir),
        "device": args.device,
        "step_num": args.step_num,
        "seed": args.seed,
        "sampling_overrides": overrides,
        "evaluator": "modebug_m0_v2_geometry_trace_audit",
        "protocol": "archived_legacy_m0_v2_20260510 synthetic event-temporal battery",
        "motion_source": "EventT2M released HumanML3D checkpoint with fixed x0 sampling config",
        "condition_pair": "full/drop, full/replace, full/shuffle, full/repeat",
        "n/evaluable": f"{sum(1 for row in audit_rows if row['status'] == 'ok')}/{len(audit_rows)}",
        "coverage": "18 base cases x 5 conditions from archived legacy M0 v2 manifest",
        "role": "diagnostic",
        "used_for": "observation",
        "limitations": "Synthetic battery has no paired GT motion distribution; do not use for FID/R-Precision final evaluation.",
    }
    (result_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: run_manifest[k] for k in ["result_dir", "prompts", "generated_ok", "missing_or_error"]}, indent=2))


if __name__ == "__main__":
    main()
