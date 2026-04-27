"""Offline diagnostics for SmolVLA grape grasping.

The script keeps the robot out of the loop and answers one question first:
are the predicted action chunks already noisy before execution?

It loads a SmolVLA policy and a LeRobot dataset, runs synchronous chunk
prediction over selected episodes, then writes CSV/PNG artifacts for action
MAE, velocity, acceleration, jerk, chunk-boundary jumps, latency, and gripper
timing. It can also repeat inference on one fixed observation to estimate
sampling/model variance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from copy import copy
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is on the path when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.utils import prepare_observation_for_inference
except ImportError as exc:
    print(f"Error importing lerobot: {exc}")
    sys.exit(1)

from important_code.utils import DEVICE, resolve_checkpoint_path


DEFAULT_POLICY = "kaixiyao/smolvla_widowx_grape_grasping_V3"
DEFAULT_DATASET = "kaixiyao/widowxai_grape_grasping_V3"
DEFAULT_TASK = "Grab the grape"
DEFAULT_OUTPUT_DIR = "outputs/diagnostics/smolvla_grape"

JOINT_NAMES = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "left_carriage_joint",
]


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_episode_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def frame_to_obs_numpy(frame: dict) -> dict:
    """Convert dataset frame tensors to the policy's raw observation dict."""
    obs = {}
    for key in ("observation.images.wrist", "observation.images.right"):
        img = frame[key]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)
            img_np = img.numpy()
        else:
            img_np = np.asarray(img)

        if img_np.dtype != np.uint8:
            scale = 255.0 if float(np.max(img_np)) <= 1.0 else 1.0
            img_np = np.clip(img_np * scale, 0, 255).astype(np.uint8)
        obs[key] = img_np

    obs["observation.state"] = _to_numpy(frame["observation.state"]).astype(np.float32)
    return obs


def get_episode_frames(dataset: LeRobotDataset, episode_idx: int) -> list[dict]:
    indices = [
        idx
        for idx, ep in enumerate(dataset.hf_dataset["episode_index"])
        if _as_episode_int(ep) == episode_idx
    ]
    if not indices:
        raise ValueError(f"Episode {episode_idx} not found in dataset.")
    return [dataset[idx] for idx in indices]


def parse_episode_selector(selector: str, dataset: LeRobotDataset) -> list[int]:
    if selector.strip().lower() == "all":
        return list(range(int(dataset.num_episodes)))

    episodes: set[int] = set()
    for part in selector.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            episodes.update(range(start, end + 1))
        else:
            episodes.add(int(part))
    return sorted(episodes)


def load_policy_and_processors(policy_path_arg: str):
    policy_path = resolve_checkpoint_path(policy_path_arg)
    print(f"Loading policy from: {policy_path}")
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.to(DEVICE).eval()

    device_overrides = {"device_processor": {"device": str(DEVICE)}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides=device_overrides,
        postprocessor_overrides=device_overrides,
    )
    return policy, preprocessor, postprocessor, policy_path


def predict_chunk(policy, preprocessor, postprocessor, frame: dict, task: str):
    obs_numpy = frame_to_obs_numpy(frame)
    with torch.no_grad():
        obs_tensors = prepare_observation_for_inference(copy(obs_numpy), DEVICE, task=task)
        obs_processed = preprocessor(obs_tensors)

    start = time.perf_counter()
    actions = policy.predict_action_chunk(obs_processed)
    latency_s = time.perf_counter() - start

    with torch.no_grad():
        postprocessed = postprocessor(actions).squeeze(0)
    return postprocessed.detach().cpu().numpy(), latency_s


def derivative_metrics(actions: np.ndarray, fps: float) -> dict[str, np.ndarray | float]:
    metrics: dict[str, np.ndarray | float] = {}
    specs = (
        ("vel", 1, fps),
        ("accel", 2, fps**2),
        ("jerk", 3, fps**3),
    )
    for name, order, scale in specs:
        if len(actions) <= order:
            diff = np.zeros((0, actions.shape[1]), dtype=np.float64)
        else:
            diff = np.diff(actions, n=order, axis=0) * scale
        abs_diff = np.abs(diff)
        metrics[f"{name}_mean_abs_joint"] = (
            abs_diff.mean(axis=0) if len(abs_diff) else np.zeros(actions.shape[1])
        )
        metrics[f"{name}_max_abs_joint"] = (
            abs_diff.max(axis=0) if len(abs_diff) else np.zeros(actions.shape[1])
        )
        metrics[f"{name}_mean_abs"] = float(abs_diff.mean()) if len(abs_diff) else 0.0
        metrics[f"{name}_max_abs"] = float(abs_diff.max()) if len(abs_diff) else 0.0
    return metrics


def gripper_change_frame(actions: np.ndarray) -> int | None:
    gripper = actions[:, -1]
    span = float(np.max(gripper) - np.min(gripper))
    if span < 1e-6:
        return None
    deltas = np.abs(gripper - gripper[0])
    hits = np.flatnonzero(deltas >= max(0.25 * span, 1e-4))
    return int(hits[0]) if len(hits) else None


def boundary_jump(prev_tail: np.ndarray | None, chunk: np.ndarray) -> tuple[float, float]:
    if prev_tail is None or len(chunk) == 0:
        return 0.0, 0.0
    jump = chunk[0] - prev_tail
    return float(np.linalg.norm(jump)), float(np.max(np.abs(jump)))


def episode_prediction(
    policy,
    preprocessor,
    postprocessor,
    frames: list[dict],
    task: str,
    fps: float,
    chunk_stride: int,
    max_frames: int | None,
) -> dict:
    if max_frames is not None:
        frames = frames[:max_frames]
    gt = np.stack([_to_numpy(frame["action"]).astype(np.float64) for frame in frames])

    executed_segments: list[np.ndarray] = []
    chunk_rows: list[dict] = []
    chunk_starts: list[int] = []
    latencies: list[float] = []
    prev_tail = None

    n_frames = len(frames)
    for chunk_start in range(0, n_frames, chunk_stride):
        chunk, latency_s = predict_chunk(policy, preprocessor, postprocessor, frames[chunk_start], task)
        steps = min(chunk_stride, len(chunk), n_frames - chunk_start)
        pred_segment = chunk[:steps].astype(np.float64)
        gt_segment = gt[chunk_start : chunk_start + steps]
        mae_joint = np.abs(pred_segment - gt_segment).mean(axis=0)
        jump_l2, jump_max = boundary_jump(prev_tail, chunk)
        metrics = derivative_metrics(chunk, fps)

        row = {
            "chunk_start": chunk_start,
            "steps_used": steps,
            "latency_ms": latency_s * 1000.0,
            "mae_mean": float(mae_joint.mean()),
            "boundary_jump_l2": jump_l2,
            "boundary_jump_max_abs": jump_max,
            "vel_max_abs": metrics["vel_max_abs"],
            "accel_max_abs": metrics["accel_max_abs"],
            "jerk_max_abs": metrics["jerk_max_abs"],
        }
        for idx, name in enumerate(JOINT_NAMES):
            row[f"mae_{name}"] = float(mae_joint[idx])
            row[f"chunk_vel_max_{name}"] = float(metrics["vel_max_abs_joint"][idx])
            row[f"chunk_accel_max_{name}"] = float(metrics["accel_max_abs_joint"][idx])
            row[f"chunk_jerk_max_{name}"] = float(metrics["jerk_max_abs_joint"][idx])
        chunk_rows.append(row)

        executed_segments.append(pred_segment)
        chunk_starts.append(chunk_start)
        latencies.append(latency_s)
        prev_tail = chunk[-1].copy()

        print(
            f"  chunk @ {chunk_start:5d}: {latency_s*1000:6.1f} ms, "
            f"MAE={row['mae_mean']:.4f}, jerk_max={row['jerk_max_abs']:.4f}, "
            f"jump={jump_l2:.4f}"
        )

    pred = np.concatenate(executed_segments, axis=0)
    gt_used = gt[: len(pred)]
    episode_metrics = derivative_metrics(pred, fps)
    gt_close = gripper_change_frame(gt_used)
    pred_close = gripper_change_frame(pred)
    gripper_error = (
        None if gt_close is None or pred_close is None else int(pred_close - gt_close)
    )

    return {
        "pred": pred,
        "gt": gt_used,
        "chunk_rows": chunk_rows,
        "chunk_starts": chunk_starts,
        "latencies": latencies,
        "episode_metrics": episode_metrics,
        "gt_gripper_change_frame": gt_close,
        "pred_gripper_change_frame": pred_close,
        "gripper_change_frame_error": gripper_error,
    }


def write_per_step_csv(path: Path, episode_idx: int, pred: np.ndarray, gt: np.ndarray) -> None:
    fieldnames = ["episode", "frame_index"]
    for prefix in ("pred", "gt", "abs_error"):
        fieldnames.extend(f"{prefix}_{name}" for name in JOINT_NAMES)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_idx, (pred_row, gt_row) in enumerate(zip(pred, gt)):
            row = {"episode": episode_idx, "frame_index": frame_idx}
            abs_error = np.abs(pred_row - gt_row)
            for idx, name in enumerate(JOINT_NAMES):
                row[f"pred_{name}"] = float(pred_row[idx])
                row[f"gt_{name}"] = float(gt_row[idx])
                row[f"abs_error_{name}"] = float(abs_error[idx])
            writer.writerow(row)


def write_chunk_csv(path: Path, rows: Iterable[dict], episode_idx: int) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = ["episode"] + list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"episode": episode_idx, **row})


def append_summary_row(summary_path: Path, row: dict) -> None:
    exists = summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_episode(path: Path, episode_idx: int, pred: np.ndarray, gt: np.ndarray, chunk_starts: list[int]) -> None:
    t = np.arange(len(pred))
    fig, axes = plt.subplots(len(JOINT_NAMES), 1, figsize=(14, 2.3 * len(JOINT_NAMES)), sharex=True)
    fig.suptitle(f"SmolVLA grape diagnostic - episode {episode_idx}", fontsize=13)
    for idx, (axis, joint_name) in enumerate(zip(axes, JOINT_NAMES)):
        axis.plot(t, gt[:, idx], label="dataset action", color="steelblue", linewidth=1.3)
        axis.plot(t, pred[:, idx], label="predicted action", color="tomato", linewidth=1.2, linestyle="--")
        for chunk_start in chunk_starts:
            axis.axvline(chunk_start, color="gray", linewidth=0.5, alpha=0.35, linestyle=":")
        axis.set_ylabel(joint_name, fontsize=8)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("frame")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_fixed_observation_variance(
    policy,
    preprocessor,
    postprocessor,
    frame: dict,
    task: str,
    repeats: int,
) -> dict:
    chunks = []
    latencies = []
    for repeat_idx in range(repeats):
        chunk, latency_s = predict_chunk(policy, preprocessor, postprocessor, frame, task)
        chunks.append(chunk)
        latencies.append(latency_s)
        print(f"  fixed repeat {repeat_idx + 1:02d}/{repeats}: {latency_s*1000:.1f} ms")

    stacked = np.stack(chunks, axis=0)
    std = stacked.std(axis=0)
    result = {
        "repeats": repeats,
        "latency_ms_mean": float(np.mean(latencies) * 1000.0),
        "latency_ms_max": float(np.max(latencies) * 1000.0),
        "std_mean": float(std.mean()),
        "std_max": float(std.max()),
        "std_mean_per_joint": {
            name: float(std[:, idx].mean()) for idx, name in enumerate(JOINT_NAMES)
        },
        "std_max_per_joint": {
            name: float(std[:, idx].max()) for idx, name in enumerate(JOINT_NAMES)
        },
    }
    return result


def build_summary_row(episode_idx: int, result: dict, fps: float, n_chunks: int) -> dict:
    pred = result["pred"]
    gt = result["gt"]
    mae_joint = np.abs(pred - gt).mean(axis=0)
    metrics = result["episode_metrics"]
    row = {
        "episode": episode_idx,
        "frames": len(pred),
        "chunks": n_chunks,
        "fps": fps,
        "mae_mean": float(mae_joint.mean()),
        "latency_ms_mean": float(np.mean(result["latencies"]) * 1000.0),
        "latency_ms_max": float(np.max(result["latencies"]) * 1000.0),
        "vel_max_abs": float(metrics["vel_max_abs"]),
        "accel_max_abs": float(metrics["accel_max_abs"]),
        "jerk_max_abs": float(metrics["jerk_max_abs"]),
        "gt_gripper_change_frame": result["gt_gripper_change_frame"],
        "pred_gripper_change_frame": result["pred_gripper_change_frame"],
        "gripper_change_frame_error": result["gripper_change_frame_error"],
    }
    for idx, name in enumerate(JOINT_NAMES):
        row[f"mae_{name}"] = float(mae_joint[idx])
        row[f"vel_max_{name}"] = float(metrics["vel_max_abs_joint"][idx])
        row[f"accel_max_{name}"] = float(metrics["accel_max_abs_joint"][idx])
        row[f"jerk_max_{name}"] = float(metrics["jerk_max_abs_joint"][idx])
    return row


def write_notes(path: Path, policy_path: str, dataset_id: str, task: str, episodes: list[int], chunk_stride: int) -> None:
    notes = {
        "policy": policy_path,
        "dataset": dataset_id,
        "task": task,
        "episodes": episodes,
        "chunk_stride": chunk_stride,
        "interpretation": [
            "High offline jerk or large chunk-boundary jumps points to model/data/pipeline, before robot execution.",
            "Low offline jerk but high real-robot jitter points to control timing, goal_time, velocity limits, or execution smoothing.",
            "High fixed-observation variance points to sampling/model uncertainty; compare seeds or deterministic settings if available.",
            "Late or missing gripper change points to gripper labeling/timing or insufficient close examples.",
        ],
    }
    path.write_text(json.dumps(notes, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose SmolVLA grape grasping chunks offline.")
    parser.add_argument("--policy-path", "--train-dir", dest="policy_path", default=DEFAULT_POLICY)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--episodes", default="all", help="all, a comma list, or ranges like 0,3-5")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--chunk-stride", type=int, default=None, help="Defaults to policy n_action_steps/chunk_size.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--fixed-repeats", type=int, default=10)
    parser.add_argument("--fixed-episode", type=int, default=None)
    parser.add_argument("--fixed-frame", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task!r}")
    print("RTC: disabled (synchronous offline diagnostic)")

    policy, preprocessor, postprocessor, policy_path = load_policy_and_processors(args.policy_path)
    dataset = LeRobotDataset(args.dataset)
    fps = float(getattr(dataset, "fps", 30.0) or 30.0)
    chunk_stride = args.chunk_stride or int(
        getattr(policy.config, "n_action_steps", None)
        or getattr(policy.config, "chunk_size", None)
        or 50
    )
    episodes = parse_episode_selector(args.episodes, dataset)
    summary_path = output_dir / "episode_summary.csv"

    write_notes(output_dir / "diagnosis_notes.json", policy_path, args.dataset, args.task, episodes, chunk_stride)

    print(f"Evaluating episodes: {episodes}")
    print(f"FPS={fps:g}, chunk_stride={chunk_stride}")

    for episode_idx in episodes:
        print(f"\nEpisode {episode_idx}")
        frames = get_episode_frames(dataset, episode_idx)
        result = episode_prediction(
            policy,
            preprocessor,
            postprocessor,
            frames,
            args.task,
            fps,
            chunk_stride,
            args.max_frames,
        )
        episode_dir = output_dir / f"episode_{episode_idx:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        write_per_step_csv(episode_dir / "per_step_actions.csv", episode_idx, result["pred"], result["gt"])
        write_chunk_csv(episode_dir / "chunk_metrics.csv", result["chunk_rows"], episode_idx)
        if args.save_plots:
            plot_episode(
                episode_dir / "actions.png",
                episode_idx,
                result["pred"],
                result["gt"],
                result["chunk_starts"],
            )
        summary_row = build_summary_row(episode_idx, result, fps, len(result["chunk_rows"]))
        append_summary_row(summary_path, summary_row)
        print(
            f"Episode {episode_idx} summary: MAE={summary_row['mae_mean']:.4f}, "
            f"jerk_max={summary_row['jerk_max_abs']:.4f}, "
            f"gripper_dt={summary_row['gripper_change_frame_error']}"
        )

    fixed_episode = args.fixed_episode if args.fixed_episode is not None else episodes[0]
    if args.fixed_repeats > 1:
        print(f"\nFixed-observation variance: episode={fixed_episode}, frame={args.fixed_frame}")
        fixed_frames = get_episode_frames(dataset, fixed_episode)
        if args.fixed_frame < 0 or args.fixed_frame >= len(fixed_frames):
            raise ValueError(f"--fixed-frame {args.fixed_frame} outside episode length {len(fixed_frames)}")
        variance = run_fixed_observation_variance(
            policy,
            preprocessor,
            postprocessor,
            fixed_frames[args.fixed_frame],
            args.task,
            args.fixed_repeats,
        )
        variance_path = output_dir / "fixed_observation_variance.json"
        variance_path.write_text(json.dumps(variance, indent=2), encoding="utf-8")
        print(f"Fixed-observation std_mean={variance['std_mean']:.6f}, std_max={variance['std_max']:.6f}")

    print(f"\nDiagnostics written to: {output_dir}")


if __name__ == "__main__":
    main()
