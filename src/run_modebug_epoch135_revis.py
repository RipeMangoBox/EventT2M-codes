from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.humanml.scripts.paramUtil import t2m_kinematic_chain
from src.planb.eventt2m_runtime import extract_events, load_eventt2m_runtime, load_hml3de_test_dict, pick_text_entry, recover_joints


def geometry_summary(joints: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(joints)
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
    frame = joints[min(len(joints) // 2, len(joints) - 1)]
    fig = plt.figure(figsize=(7, 7))
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
    ax.set_title(title[:100])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-visualize EventT2M epoch-135 HumanML3D-E scale sanity samples.")
    parser.add_argument("--data-file", type=Path, default=Path("dataset/HumanML3D-E/data_test.npy"))
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/HumanML3D"))
    parser.add_argument("--ckpt-path", type=Path, default=Path("logs/event/runs/eventt2m_clean_hml3d_retrain_seed1/checkpoints/epoch=135.ckpt"))
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--sample-ids", default="003245")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--step-num", type=int, default=10)
    parser.add_argument("--all-blank", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    data_file = args.data_file if args.data_file.is_absolute() else repo_root / args.data_file
    data_dir = args.data_dir if args.data_dir.is_absolute() else repo_root / args.data_dir
    ckpt_path = args.ckpt_path if args.ckpt_path.is_absolute() else repo_root / args.ckpt_path
    data = load_hml3de_test_dict(data_file)
    runtime = load_eventt2m_runtime(
        ckpt_path=ckpt_path,
        device=args.device,
        step_num=args.step_num,
        data_dir=data_dir,
    )

    records = []
    for offset, sample_id in enumerate([item.strip() for item in args.sample_ids.split(",") if item.strip()]):
        sample = data[sample_id]
        text_entry = pick_text_entry(sample)
        caption = text_entry["caption"]
        events = extract_events(text_entry)
        if args.all_blank:
            events = [""] * max(len(events), 1)
        length = int(sample["length"])
        gt_raw = sample["motion"][:length]
        gt_joints = recover_joints(gt_raw)

        seed = args.seed + offset
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        motion = torch.zeros((1, length, 263), device=runtime.device)
        lens = torch.tensor([length], dtype=torch.long, device=runtime.device)
        generated = runtime.model.sample_motion(motion, lens, [caption], [events])
        raw = (generated * runtime.std + runtime.mean)[0].detach().cpu().numpy()
        gen_joints = recover_joints(raw)

        gt_static = result_dir / f"{sample_id}_gt_static.png"
        gen_static = result_dir / f"{sample_id}_epoch135_static.png"
        save_static_plot(gt_joints, gt_static, f"{sample_id} GT")
        save_static_plot(gen_joints, gen_static, f"{sample_id} epoch135")
        np.save(result_dir / f"{sample_id}_gt_joints.npy", gt_joints.astype(np.float32))
        np.save(result_dir / f"{sample_id}_epoch135_joints.npy", gen_joints.astype(np.float32))
        records.append(
            {
                "date": datetime.now().isoformat(timespec="seconds"),
                "sample_id": sample_id,
                "caption": caption,
                "events": events,
                "length": length,
                "seed": seed,
                "ckpt_path": str(ckpt_path),
                "data_file": str(data_file),
                "data_dir": str(data_dir),
                "device": args.device,
                "step_num": args.step_num,
                "gt_geometry": geometry_summary(gt_joints),
                "generated_geometry": geometry_summary(gen_joints),
                "gt_static": str(gt_static),
                "generated_static": str(gen_static),
                "role": "diagnostic",
                "used_for": "observation",
                "limitations": "Single-sample static skeleton scale sanity; not full-level safety, not held-out final evaluator evidence.",
            }
        )

    (result_dir / "revis_summary.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result_dir": str(result_dir), "samples": len(records)}, indent=2))


if __name__ == "__main__":
    main()
