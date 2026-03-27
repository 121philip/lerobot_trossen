"""
Offline RTC evaluation script.

Simulates the full RTC inference loop on a recorded dataset episode,
comparing predicted actions vs ground truth. Use this to verify RTC
works correctly before running on real hardware.

Usage:
  # Evaluate episode 0 with RTC (default):
  python important_code/eval_rtc_offline.py

  # Evaluate episode 2 and save the plot:
  python important_code/eval_rtc_offline.py --episode 2 --save-plot eval_ep2.png

  # Compare RTC vs standard chunking:
  python important_code/eval_rtc_offline.py --no-rtc --save-plot eval_no_rtc.png

  # Only evaluate first 100 frames (faster):
  python important_code/eval_rtc_offline.py --max-frames 100
"""

import argparse
import math
import sys
import time
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.rtc.latency_tracker import LatencyTracker
except ImportError as e:
    print(f"Error importing lerobot: {e}")
    sys.exit(1)

DEFAULT_TRAIN_DIR = "outputs/train/smolvla_widowx_grape_grasping"
DATASET_REPO_ID = "kaixiyao/widowxai_grape_grasping"
TASK = "The robot grasps a grape"
JOINT_NAMES = ["joint_0", "joint_1", "joint_2",
               "joint_3", "joint_4", "joint_5", "gripper"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint_path(train_dir: str) -> str:
    """Return path to the pretrained_model dir inside the last checkpoint."""
    last_ptr = Path(train_dir) / "checkpoints" / "last"
    if not last_ptr.exists():
        raise FileNotFoundError(
            f"No checkpoint 'last' pointer found at: {last_ptr}")
    if last_ptr.is_dir():
        checkpoint_dir = last_ptr
    elif last_ptr.is_file():
        checkpoint_dir = last_ptr.parent / last_ptr.read_text().strip()
    else:
        raise ValueError(
            f"'last' at {last_ptr} is neither a file nor a directory.")
    pretrained = checkpoint_dir / "pretrained_model"
    if not pretrained.exists():
        raise FileNotFoundError(f"pretrained_model not found at: {pretrained}")
    return str(pretrained)


def frame_to_obs_numpy(frame: dict) -> dict:
    """Convert a dataset frame to numpy dict for prepare_observation_for_inference.

    Dataset images are (C, H, W) tensors (float [0,1] or uint8 [0,255]).
    prepare_observation_for_inference expects (H, W, C) uint8 numpy arrays.
    """
    obs = {}
    for key in ["observation.images.wrist", "observation.images.right"]:
        img = frame[key]  # (C, H, W)
        if img.dtype == torch.uint8:
            obs[key] = img.permute(1, 2, 0).numpy()  # (H, W, C) uint8
        else:
            obs[key] = (img.permute(1, 2, 0).numpy() *
                        255).astype(np.uint8)  # float→uint8
    obs["observation.state"] = frame["observation.state"].numpy().astype(np.float32)
    return obs


def get_episode_frames(dataset: LeRobotDataset, episode_idx: int) -> list:
    """Return all frames for a given episode index."""
    indices = [
        i for i, ep in enumerate(dataset.hf_dataset["episode_index"])
        if ep.item() == episode_idx
    ]
    if not indices:
        raise ValueError(f"Episode {episode_idx} not found in dataset.")
    return [dataset[i] for i in indices]


def run_rtc_simulation(
    policy,
    preprocessor,
    postprocessor,
    frames: list,
    rtc_config: RTCConfig,
    task: str,
    queue_threshold: int,
) -> tuple:
    """Simulate the RTC inference loop on dataset frames.

    Returns:
        executed_actions: (N, 7) array of actions the RTC system would send
        gt_actions:       (N, 7) array of ground truth actions from dataset
        chunk_frames:     list of timesteps where a new chunk was requested
        latencies:        list of inference latencies (seconds) per chunk
    """
    fps = 30.0
    time_per_step = 1.0 / fps

    action_queue = ActionQueue(rtc_config)
    latency_tracker = LatencyTracker()

    executed_actions = []
    gt_actions = []
    chunk_frames = []
    latencies = []

    n_frames = len(frames)
    print(
        f"\nSimulating {n_frames} frames (queue_threshold={queue_threshold})...")

    for t, frame in enumerate(frames):
        gt_actions.append(frame["action"].numpy())

        if action_queue.qsize() <= queue_threshold:
            action_index_before = action_queue.get_action_index()
            prev_actions = action_queue.get_left_over()

            # Use tracked latency to compute expected delay for this chunk
            inference_latency = latency_tracker.max() or 0.0
            inference_delay = math.ceil(inference_latency / time_per_step)

            obs_numpy = frame_to_obs_numpy(frame)

            # NOTE: Use torch.no_grad() (not inference_mode) so RTC's internal
            # enable_grad() can still work inside predict_action_chunk.
            with torch.no_grad():
                obs_tensors = prepare_observation_for_inference(
                    copy(obs_numpy), DEVICE, task=task
                )
                obs_processed = preprocessor(obs_tensors)

            # predict_action_chunk must be called OUTSIDE torch.no_grad()
            t_start = time.perf_counter()
            actions = policy.predict_action_chunk(
                obs_processed,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_actions,
            )
            latency = time.perf_counter() - t_start
            latency_tracker.add(latency)

            actual_delay = math.ceil(latency / time_per_step)

            postprocessed = postprocessor(actions).squeeze(0)
            original = actions.squeeze(0).clone()

            action_queue.merge(original, postprocessed,
                               actual_delay, action_index_before)

            chunk_frames.append(t)
            latencies.append(latency)
            print(
                f"  [t={t:4d}/{n_frames}] chunk in {latency*1000:.0f}ms "
                f"(delay={actual_delay}, queue={action_queue.qsize()})"
            )

        action = action_queue.get()
        executed_actions.append(
            action.numpy() if action is not None else np.zeros(7))

    return np.array(executed_actions), np.array(gt_actions), chunk_frames, latencies


def plot_results(
    executed: np.ndarray,
    gt: np.ndarray,
    chunk_frames: list,
    latencies: list,
    joint_names: list,
    rtc_enabled: bool,
    episode_idx: int,
    save_path: str | None = None,
) -> None:
    """Plot executed vs ground truth actions for each joint."""
    n_joints = len(joint_names)
    t = np.arange(len(executed))

    fig, axes = plt.subplots(n_joints, 1, figsize=(
        14, 2.5 * n_joints), sharex=True)
    mode = "RTC Enabled" if rtc_enabled else "Standard Chunking"
    fig.suptitle(
        f"Offline RTC Evaluation — {mode} — Episode {episode_idx}", fontsize=13)

    for i, (ax, name) in enumerate(zip(axes, joint_names)):
        ax.plot(t, gt[:, i], label="Ground Truth",
                color="steelblue", linewidth=1.5, alpha=0.8)
        ax.plot(t, executed[:, i], label="Predicted", color="tomato",
                linewidth=1.5, linestyle="--", alpha=0.9)
        for cf in chunk_frames:
            ax.axvline(x=cf, color="gray", linewidth=0.6,
                       alpha=0.4, linestyle=":")
        ax.set_ylabel(name, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()

    # Print stats
    mae = np.abs(executed - gt).mean(axis=0)
    print("\n--- Action MAE per joint ---")
    for name, err in zip(joint_names, mae):
        print(f"  {name}: {err:.4f}")
    print(f"  Overall MAE: {np.abs(executed - gt).mean():.4f}")

    if latencies:
        print("\n--- Inference Latency ---")
        print(f"  Mean:   {np.mean(latencies)*1000:.1f} ms")
        print(f"  Max:    {np.max(latencies)*1000:.1f} ms")
        print(f"  Chunks: {len(latencies)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline RTC evaluation on a dataset episode")
    parser.add_argument("--train-dir", type=str, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--dataset", type=str, default=DATASET_REPO_ID)
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to evaluate")
    parser.add_argument("--task", type=str, default=TASK)
    parser.add_argument("--rtc-enabled", action="store_true", default=True)
    parser.add_argument("--no-rtc", dest="rtc_enabled", action="store_false",
                        help="Disable RTC (standard chunking)")
    parser.add_argument("--execution-horizon", type=int, default=10,
                        help="RTC: steps to blend between chunks (default: 10)")
    parser.add_argument("--guidance-weight", type=float, default=10.0,
                        help="RTC: consistency guidance strength (default: 10.0)")
    parser.add_argument("--attention-schedule", type=str, default="EXP",
                        choices=["LINEAR", "EXP", "ONES", "ZEROS"])
    parser.add_argument("--queue-threshold", type=int, default=5,
                        help="Request new chunk when queue drops below this (default: 5)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit to first N frames (default: full episode)")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Path to save plot image (e.g. eval_rtc.png)")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Device: {DEVICE}")
    print(f"RTC: {'enabled' if args.rtc_enabled else 'disabled'}")

    # Load policy
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
    policy.to(DEVICE)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=policy_path
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = LeRobotDataset(args.dataset)
    print(
        f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames total")

    frames = get_episode_frames(dataset, args.episode)
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
    print(
        f"Episode {args.episode}: evaluating {len(frames)} frames at {dataset.fps} FPS")

    # Run simulation
    executed, gt, chunk_frames, latencies = run_rtc_simulation(
        policy, preprocessor, postprocessor,
        frames, rtc_config, args.task, args.queue_threshold,
    )

    # Plot and print stats
    plot_results(
        executed, gt, chunk_frames, latencies,
        JOINT_NAMES, args.rtc_enabled, args.episode,
        save_path=args.save_plot,
    )


if __name__ == "__main__":
    main()
