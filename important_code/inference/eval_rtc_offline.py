"""Offline SmolVLA chunk evaluation.

Default mode is non-RTC synchronous chunking for grape-grasp diagnosis.
Use --rtc-enabled only when you explicitly want to compare the RTC path.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is on the path when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.latency_tracker import LatencyTracker
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.utils import prepare_observation_for_inference
except ImportError as exc:
    print(f"Error importing lerobot: {exc}")
    sys.exit(1)

from important_code.utils import DEVICE, resolve_checkpoint_path


DEFAULT_TRAIN_DIR = "kaixiyao/smolvla_widowx_grape_grasping_V3"
DATASET_REPO_ID = "kaixiyao/widowxai_grape_grasping_V3"
TASK = "Grab the grape"
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


def frame_to_obs_numpy(frame: dict) -> dict:
    obs = {}
    for key in ("observation.images.wrist", "observation.images.right"):
        img = _to_numpy(frame[key])
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            scale = 255.0 if float(np.max(img)) <= 1.0 else 1.0
            img = np.clip(img * scale, 0, 255).astype(np.uint8)
        obs[key] = img
    obs["observation.state"] = _to_numpy(frame["observation.state"]).astype(np.float32)
    return obs


def get_episode_frames(dataset: LeRobotDataset, episode_idx: int) -> list[dict]:
    indices = [
        idx
        for idx, ep in enumerate(dataset.hf_dataset["episode_index"])
        if int(ep.item() if hasattr(ep, "item") else ep) == episode_idx
    ]
    if not indices:
        raise ValueError(f"Episode {episode_idx} not found in dataset.")
    return [dataset[idx] for idx in indices]


def run_chunk_simulation(
    policy,
    preprocessor,
    postprocessor,
    frames: list[dict],
    rtc_config: RTCConfig,
    task: str,
    queue_threshold: int,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, list[int], list[float]]:
    """Simulate policy inference and queue consumption on recorded frames."""
    time_per_step = 1.0 / fps
    action_queue = ActionQueue(rtc_config)
    latency_tracker = LatencyTracker()

    executed_actions: list[np.ndarray] = []
    gt_actions: list[np.ndarray] = []
    chunk_frames: list[int] = []
    latencies: list[float] = []

    for frame_idx, frame in enumerate(frames):
        gt_actions.append(_to_numpy(frame["action"]))

        if action_queue.qsize() <= queue_threshold:
            action_index_before = action_queue.get_action_index()
            prev_actions = action_queue.get_left_over()
            inference_latency = latency_tracker.max() or 0.0
            inference_delay = math.ceil(inference_latency / time_per_step)

            obs_numpy = frame_to_obs_numpy(frame)
            with torch.no_grad():
                obs_tensors = prepare_observation_for_inference(
                    copy(obs_numpy), DEVICE, task=task
                )
                obs_processed = preprocessor(obs_tensors)

            start = time.perf_counter()
            actions = policy.predict_action_chunk(
                obs_processed,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_actions,
            )
            latency_s = time.perf_counter() - start
            latency_tracker.add(latency_s)

            actual_delay = math.ceil(latency_s / time_per_step)
            postprocessed = postprocessor(actions).squeeze(0)
            original = actions.squeeze(0).clone()
            action_queue.merge(original, postprocessed, actual_delay, action_index_before)

            chunk_frames.append(frame_idx)
            latencies.append(latency_s)
            print(
                f"  [t={frame_idx:4d}/{len(frames)}] chunk {latency_s*1000:.0f} ms "
                f"(delay={actual_delay}, queue={action_queue.qsize()})"
            )

        action = action_queue.get()
        if action is None:
            executed_actions.append(np.zeros(len(JOINT_NAMES), dtype=np.float32))
        else:
            executed_actions.append(action.detach().cpu().numpy())

    return np.asarray(executed_actions), np.asarray(gt_actions), chunk_frames, latencies


def derivative_metrics(actions: np.ndarray, fps: float) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, order, scale in (("vel", 1, fps), ("accel", 2, fps**2), ("jerk", 3, fps**3)):
        if len(actions) <= order:
            output[f"{name}_max_abs"] = 0.0
            output[f"{name}_mean_abs"] = 0.0
            continue
        diff = np.diff(actions, n=order, axis=0) * scale
        abs_diff = np.abs(diff)
        output[f"{name}_max_abs"] = float(abs_diff.max())
        output[f"{name}_mean_abs"] = float(abs_diff.mean())
    return output


def boundary_metrics(executed: np.ndarray, chunk_frames: list[int]) -> dict[str, float]:
    jumps = []
    for frame_idx in chunk_frames[1:]:
        if 0 < frame_idx < len(executed):
            jumps.append(executed[frame_idx] - executed[frame_idx - 1])
    if not jumps:
        return {"boundary_jump_l2_mean": 0.0, "boundary_jump_max_abs": 0.0}
    jumps_np = np.stack(jumps)
    return {
        "boundary_jump_l2_mean": float(np.linalg.norm(jumps_np, axis=1).mean()),
        "boundary_jump_max_abs": float(np.abs(jumps_np).max()),
    }


def write_csv_outputs(
    output_dir: Path,
    episode_idx: int,
    executed: np.ndarray,
    gt: np.ndarray,
    chunk_frames: list[int],
    latencies: list[float],
    fps: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    per_step_path = output_dir / f"episode_{episode_idx:03d}_per_step.csv"
    with per_step_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["frame_index"]
        for prefix in ("pred", "gt", "abs_error"):
            fieldnames.extend(f"{prefix}_{name}" for name in JOINT_NAMES)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_idx, (pred_row, gt_row) in enumerate(zip(executed, gt)):
            row = {"frame_index": frame_idx}
            err = np.abs(pred_row - gt_row)
            for joint_idx, joint_name in enumerate(JOINT_NAMES):
                row[f"pred_{joint_name}"] = float(pred_row[joint_idx])
                row[f"gt_{joint_name}"] = float(gt_row[joint_idx])
                row[f"abs_error_{joint_name}"] = float(err[joint_idx])
            writer.writerow(row)

    mae_joint = np.abs(executed - gt).mean(axis=0)
    summary = {
        "episode": episode_idx,
        "frames": len(executed),
        "chunks": len(chunk_frames),
        "mae_mean": float(mae_joint.mean()),
        "latency_ms_mean": float(np.mean(latencies) * 1000.0) if latencies else 0.0,
        "latency_ms_max": float(np.max(latencies) * 1000.0) if latencies else 0.0,
        **derivative_metrics(executed, fps),
        **boundary_metrics(executed, chunk_frames),
    }
    for joint_idx, joint_name in enumerate(JOINT_NAMES):
        summary[f"mae_{joint_name}"] = float(mae_joint[joint_idx])

    summary_path = output_dir / "summary.csv"
    exists = summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(summary)

    print(f"CSV written: {per_step_path}")
    print(f"Summary updated: {summary_path}")


def plot_results(
    executed: np.ndarray,
    gt: np.ndarray,
    chunk_frames: list[int],
    latencies: list[float],
    rtc_enabled: bool,
    episode_idx: int,
    save_path: Path | None,
) -> None:
    t = np.arange(len(executed))
    fig, axes = plt.subplots(len(JOINT_NAMES), 1, figsize=(14, 2.4 * len(JOINT_NAMES)), sharex=True)
    mode = "RTC enabled" if rtc_enabled else "Non-RTC synchronous"
    fig.suptitle(f"Offline SmolVLA evaluation - {mode} - episode {episode_idx}", fontsize=13)

    for joint_idx, (axis, joint_name) in enumerate(zip(axes, JOINT_NAMES)):
        axis.plot(t, gt[:, joint_idx], label="dataset action", color="steelblue", linewidth=1.4)
        axis.plot(t, executed[:, joint_idx], label="predicted action", color="tomato", linewidth=1.2, linestyle="--")
        for frame_idx in chunk_frames:
            axis.axvline(frame_idx, color="gray", linewidth=0.6, alpha=0.35, linestyle=":")
        axis.set_ylabel(joint_name, fontsize=8)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("frame")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)

    mae = np.abs(executed - gt).mean(axis=0)
    print("\n--- Action MAE per joint ---")
    for name, err in zip(JOINT_NAMES, mae):
        print(f"  {name}: {err:.4f}")
    print(f"  Overall MAE: {mae.mean():.4f}")

    if latencies:
        print("\n--- Inference latency ---")
        print(f"  Mean:   {np.mean(latencies)*1000:.1f} ms")
        print(f"  Max:    {np.max(latencies)*1000:.1f} ms")
        print(f"  Chunks: {len(latencies)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline SmolVLA evaluation on one dataset episode.")
    parser.add_argument("--train-dir", "--policy-path", dest="train_dir", type=str, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--dataset", type=str, default=DATASET_REPO_ID)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--task", type=str, default=TASK)
    parser.add_argument("--rtc-enabled", action="store_true", default=False)
    parser.add_argument("--no-rtc", dest="rtc_enabled", action="store_false")
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--guidance-weight", type=float, default=10.0)
    parser.add_argument("--attention-schedule", type=str, default="EXP", choices=["LINEAR", "EXP", "ONES", "ZEROS"])
    parser.add_argument("--queue-threshold", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics/eval_offline"))
    parser.add_argument("--save-plot", type=Path, default=None)
    parser.add_argument("--no-csv", dest="save_csv", action="store_false")
    parser.set_defaults(save_csv=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Device: {DEVICE}")
    print(f"RTC: {'enabled' if args.rtc_enabled else 'disabled'}")
    print(f"Task: {args.task!r}")
    if args.rtc_enabled:
        print("Warning: RTC is enabled. Use the default non-RTC mode for first-pass grape diagnostics.")

    policy_path = resolve_checkpoint_path(args.train_dir)
    print(f"Loading policy from: {policy_path}")
    policy = SmolVLAPolicy.from_pretrained(policy_path)

    rtc_config = RTCConfig(
        enabled=args.rtc_enabled,
        execution_horizon=args.execution_horizon,
        max_guidance_weight=args.guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule[args.attention_schedule],
    )
    policy.config.rtc_config = rtc_config
    policy.init_rtc_processor()
    policy.to(DEVICE).eval()

    device_overrides = {"device_processor": {"device": str(DEVICE)}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides=device_overrides,
        postprocessor_overrides=device_overrides,
    )

    print(f"Loading dataset: {args.dataset}")
    dataset = LeRobotDataset(args.dataset)
    fps = float(getattr(dataset, "fps", 30.0) or 30.0)
    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames, fps={fps:g}")

    frames = get_episode_frames(dataset, args.episode)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    print(f"Episode {args.episode}: evaluating {len(frames)} frames")

    executed, gt, chunk_frames, latencies = run_chunk_simulation(
        policy,
        preprocessor,
        postprocessor,
        frames,
        rtc_config,
        args.task,
        args.queue_threshold,
        fps,
    )

    if args.save_csv:
        write_csv_outputs(args.output_dir, args.episode, executed, gt, chunk_frames, latencies, fps)

    plot_path = args.save_plot
    if plot_path is None and args.output_dir is not None:
        plot_path = args.output_dir / f"episode_{args.episode:03d}.png"
    plot_results(executed, gt, chunk_frames, latencies, args.rtc_enabled, args.episode, plot_path)


if __name__ == "__main__":
    main()
