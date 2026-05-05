#!/usr/bin/env python
"""Offline train/validation evaluation for SmolVLA V4 checkpoints.

This does not estimate real robot success. It compares predicted action chunks
against recorded LeRobot actions on held-out validation episodes, then writes a
checkpoint ranking that can be used before robot rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on the path when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from important_code.utils import DEVICE


DEFAULT_TRAIN_DIR = "outputs/train/smolvla_widowx_grape_grasping_V4_pos234_lora"
DEFAULT_TRAIN_DATASET = "kaixiyao/widowxai_grape_grasping_V4_pos234_train"
DEFAULT_VAL_DATASET = "kaixiyao/widowxai_grape_grasping_V4_pos234_val"
DEFAULT_OUTPUT_DIR = "outputs/validation/smolvla_widowx_grape_grasping_V4_pos234_lora"
DEFAULT_STEPS = "40000,60000,80000,100000,120000"
DEFAULT_TASK = "Grab the grape"
DEFAULT_RENAME_MAP = {
    "observation.images.right": "observation.images.cam_main",
    "observation.images.wrist": "observation.images.cam_wrist",
}

JOINT_NAMES = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "left_carriage_joint",
]


@dataclass(frozen=True)
class CheckpointPath:
    step: int
    path: Path


@dataclass
class MetricAccumulator:
    abs_sum: np.ndarray
    sq_sum: np.ndarray
    max_abs: np.ndarray
    count: int = 0
    chunks: int = 0
    latency_s: list[float] | None = None

    @classmethod
    def create(cls) -> "MetricAccumulator":
        return cls(
            abs_sum=np.zeros(len(JOINT_NAMES), dtype=np.float64),
            sq_sum=np.zeros(len(JOINT_NAMES), dtype=np.float64),
            max_abs=np.zeros(len(JOINT_NAMES), dtype=np.float64),
            latency_s=[],
        )

    def update(self, pred: np.ndarray, gt: np.ndarray, latency_s: float) -> None:
        err = pred - gt
        abs_err = np.abs(err)
        self.abs_sum += abs_err.sum(axis=0)
        self.sq_sum += np.square(err).sum(axis=0)
        self.max_abs = np.maximum(self.max_abs, abs_err.max(axis=0))
        self.count += len(pred)
        self.chunks += 1
        assert self.latency_s is not None
        self.latency_s.append(latency_s)

    def to_row(self, *, split: str, checkpoint: CheckpointPath, dataset_repo: str, episodes: list[int]) -> dict:
        if self.count == 0:
            raise ValueError("No frames were evaluated; check episode selection and max frame settings.")

        mae_joint = self.abs_sum / self.count
        rmse_joint = np.sqrt(self.sq_sum / self.count)
        assert self.latency_s is not None
        row = {
            "split": split,
            "checkpoint_step": checkpoint.step,
            "checkpoint_path": str(checkpoint.path),
            "dataset_repo": dataset_repo,
            "episodes": ",".join(str(ep) for ep in episodes),
            "num_episodes": len(episodes),
            "num_chunks": self.chunks,
            "num_action_frames": self.count,
            "mae_mean": float(mae_joint.mean()),
            "rmse_mean": float(rmse_joint.mean()),
            "max_abs_mean": float(self.max_abs.mean()),
            "latency_ms_mean": float(np.mean(self.latency_s) * 1000.0) if self.latency_s else 0.0,
            "latency_ms_max": float(np.max(self.latency_s) * 1000.0) if self.latency_s else 0.0,
        }
        for joint_idx, joint_name in enumerate(JOINT_NAMES):
            row[f"mae_{joint_name}"] = float(mae_joint[joint_idx])
            row[f"rmse_{joint_name}"] = float(rmse_joint[joint_idx])
            row[f"max_abs_{joint_name}"] = float(self.max_abs[joint_idx])
        return row


def parse_episode_selector(selector: str, *, total_episodes: int) -> list[int]:
    if selector.strip().lower() == "all":
        return list(range(total_episodes))

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

    selected = sorted(episodes)
    invalid = [ep for ep in selected if ep < 0 or ep >= total_episodes]
    if invalid:
        raise ValueError(f"Invalid episode indices for dataset with {total_episodes} episodes: {invalid}")
    return selected


def parse_step_selector(selector: str) -> list[int] | None:
    if selector.strip().lower() == "all":
        return None
    return sorted({int(part.strip()) for part in selector.split(",") if part.strip()})


def find_checkpoint_paths(train_dir: Path, requested_steps: list[int] | None) -> list[CheckpointPath]:
    checkpoints_dir = train_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoints_dir}")

    if requested_steps is None:
        candidates = sorted(checkpoints_dir.glob("*/pretrained_model"))
    else:
        candidates = [checkpoints_dir / f"{step:06d}" / "pretrained_model" for step in requested_steps]

    checkpoints: list[CheckpointPath] = []
    for path in candidates:
        if not (path / "config.json").exists():
            print(f"Skipping missing checkpoint: {path}", file=sys.stderr)
            continue
        try:
            step = int(path.parent.name)
        except ValueError:
            print(f"Skipping non-step checkpoint directory: {path}", file=sys.stderr)
            continue
        checkpoints.append(CheckpointPath(step=step, path=path))

    if not checkpoints:
        raise FileNotFoundError(f"No matching checkpoints found under {checkpoints_dir}")
    return sorted(checkpoints, key=lambda item: item.step)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _episode_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


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


def episode_indices(dataset, episode_idx: int) -> list[int]:
    return [
        idx
        for idx, ep in enumerate(dataset.hf_dataset["episode_index"])
        if _episode_int(ep) == episode_idx
    ]


def load_policy_config(checkpoint_path: Path):
    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    config.pretrained_path = checkpoint_path
    return config


def load_policy_bundle(checkpoint_path: Path, ds_meta, rename_map: dict[str, str] | None = None):
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    print(f"Loading checkpoint: {checkpoint_path}")
    policy_config = load_policy_config(checkpoint_path)
    policy = make_policy(policy_config, ds_meta=ds_meta, rename_map=rename_map)
    policy.to(DEVICE).eval()

    device_overrides = {"device_processor": {"device": str(DEVICE)}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides=device_overrides,
        postprocessor_overrides=device_overrides,
    )
    return policy, preprocessor, postprocessor


def predict_chunk(policy, preprocessor, postprocessor, frame: dict, task: str) -> tuple[np.ndarray, float]:
    from lerobot.policies.utils import prepare_observation_for_inference

    obs_numpy = frame_to_obs_numpy(frame)
    with torch.no_grad():
        obs_tensors = prepare_observation_for_inference(copy(obs_numpy), DEVICE, task=task)
        obs_processed = preprocessor(obs_tensors)

    start = time.perf_counter()
    with torch.no_grad():
        actions = policy.predict_action_chunk(obs_processed)
        postprocessed = postprocessor(actions).squeeze(0)
    latency_s = time.perf_counter() - start
    return postprocessed.detach().cpu().numpy(), latency_s


def evaluate_split(
    *,
    split_name: str,
    checkpoint: CheckpointPath,
    policy,
    preprocessor,
    postprocessor,
    dataset,
    dataset_repo: str,
    episodes: list[int],
    task: str,
    frame_stride: int,
    max_frames_per_episode: int | None,
) -> dict:
    metrics = MetricAccumulator.create()
    chunk_size = int(getattr(policy.config, "chunk_size", 50))

    for episode_idx in episodes:
        indices = episode_indices(dataset, episode_idx)
        if max_frames_per_episode is not None:
            indices = indices[:max_frames_per_episode]
        print(f"  {split_name} episode {episode_idx}: {len(indices)} frames")

        for offset in range(0, len(indices), frame_stride):
            window_indices = indices[offset : offset + chunk_size]
            if not window_indices:
                continue
            frame = dataset[window_indices[0]]
            pred_chunk, latency_s = predict_chunk(policy, preprocessor, postprocessor, frame, task)

            horizon = min(len(pred_chunk), len(window_indices))
            gt = np.stack([_to_numpy(dataset[index]["action"]) for index in window_indices[:horizon]])
            metrics.update(pred_chunk[:horizon], gt, latency_s)

    return metrics.to_row(split=split_name, checkpoint=checkpoint, dataset_repo=dataset_repo, episodes=episodes)


def _rows_by_split(rows: list[dict], metric: str) -> dict[str, tuple[list[int], list[float]]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if metric not in row:
            continue
        grouped.setdefault(str(row["split"]), []).append(row)

    output: dict[str, tuple[list[int], list[float]]] = {}
    for split, split_rows in grouped.items():
        ordered = sorted(split_rows, key=lambda item: int(item["checkpoint_step"]))
        output[split] = (
            [int(row["checkpoint_step"]) for row in ordered],
            [float(row[metric]) for row in ordered],
        )
    return output


def write_metric_plots(rows: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    overview_metrics = [
        ("mae_mean", "Mean Absolute Error"),
        ("rmse_mean", "Root Mean Squared Error"),
        ("max_abs_mean", "Max Absolute Error Mean"),
        ("latency_ms_mean", "Mean Latency (ms)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle("SmolVLA Offline Checkpoint Metrics")
    for axis, (metric, title) in zip(axes.ravel(), overview_metrics, strict=True):
        has_series = False
        for split, (steps, values) in _rows_by_split(rows, metric).items():
            axis.plot(steps, values, marker="o", linewidth=1.8, label=split)
            has_series = True
        axis.set_title(title)
        axis.set_xlabel("checkpoint step")
        axis.grid(True, alpha=0.3)
        if has_series:
            axis.legend()
    fig.tight_layout()
    overview_path = output_dir / "checkpoint_metrics_overview.png"
    fig.savefig(overview_path, dpi=150)
    plt.close(fig)
    print(f"Metric plot written: {overview_path}")

    for prefix, title, filename in (
        ("mae", "SmolVLA Joint MAE by Checkpoint", "checkpoint_joint_mae.png"),
        ("rmse", "SmolVLA Joint RMSE by Checkpoint", "checkpoint_joint_rmse.png"),
    ):
        fig, axes = plt.subplots(len(JOINT_NAMES), 1, figsize=(13, 2.2 * len(JOINT_NAMES)), sharex=True)
        fig.suptitle(title)
        for joint_name, axis in zip(JOINT_NAMES, axes, strict=True):
            metric = f"{prefix}_{joint_name}"
            has_series = False
            for split, (steps, values) in _rows_by_split(rows, metric).items():
                axis.plot(steps, values, marker="o", linewidth=1.5, label=split)
                has_series = True
            axis.set_ylabel(joint_name)
            axis.grid(True, alpha=0.3)
            if has_series:
                axis.legend(loc="upper right")
        axes[-1].set_xlabel("checkpoint step")
        fig.tight_layout()
        plot_path = output_dir / filename
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Joint plot written: {plot_path}")


def write_summary(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "offline_checkpoint_eval.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary CSV written: {summary_csv}")

    val_rows = [row for row in rows if row["split"] == "val"]
    train_rows = {row["checkpoint_step"]: row for row in rows if row["split"] == "train"}
    if val_rows:
        best_val = min(val_rows, key=lambda row: row["mae_mean"])
        train_match = train_rows.get(best_val["checkpoint_step"])
        recommendation = {
            "selection_metric": "lowest offline validation MAE",
            "note": "This is an offline proxy. NXP recommends choosing checkpoints by train/validation success; use robot rollout success when available.",
            "best_checkpoint_step": best_val["checkpoint_step"],
            "best_checkpoint_path": best_val["checkpoint_path"],
            "val_mae_mean": best_val["mae_mean"],
            "train_mae_mean": train_match["mae_mean"] if train_match else None,
            "train_val_mae_gap": (best_val["mae_mean"] - train_match["mae_mean"]) if train_match else None,
        }
        recommendation_path = output_dir / "offline_checkpoint_recommendation.json"
        recommendation_path.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
        print(f"Recommendation written: {recommendation_path}")
        print(
            f"Best offline validation checkpoint: step {best_val['checkpoint_step']} "
            f"(val_mae_mean={best_val['mae_mean']:.6f})"
        )

    write_metric_plots(rows, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SmolVLA V4 checkpoints on train and validation LeRobot datasets without a robot."
    )
    parser.add_argument("--train-dir", type=Path, default=Path(DEFAULT_TRAIN_DIR))
    parser.add_argument("--train-dataset", default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--val-dataset", default=DEFAULT_VAL_DATASET)
    parser.add_argument("--steps", default=DEFAULT_STEPS, help="Comma-separated steps, or 'all'.")
    parser.add_argument("--train-episodes", default="all")
    parser.add_argument("--val-episodes", default="all")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--frame-stride", type=int, default=50)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_train and args.skip_val:
        raise ValueError("At least one of train or validation split must be evaluated.")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    requested_steps = parse_step_selector(args.steps)
    checkpoints = find_checkpoint_paths(args.train_dir, requested_steps)
    print(f"Device: {DEVICE}")
    print(f"Checkpoints: {[item.step for item in checkpoints]}")
    print(f"Frame stride: {args.frame_stride}")

    datasets: dict[str, tuple[object, str, list[int]]] = {}
    if not args.skip_train:
        train_dataset = LeRobotDataset(args.train_dataset)
        train_episodes = parse_episode_selector(args.train_episodes, total_episodes=int(train_dataset.num_episodes))
        datasets["train"] = (train_dataset, args.train_dataset, train_episodes)
    if not args.skip_val:
        val_dataset = LeRobotDataset(args.val_dataset)
        val_episodes = parse_episode_selector(args.val_episodes, total_episodes=int(val_dataset.num_episodes))
        datasets["val"] = (val_dataset, args.val_dataset, val_episodes)

    policy_ds_meta = next(iter(datasets.values()))[0].meta
    rows: list[dict] = []
    for checkpoint in checkpoints:
        policy, preprocessor, postprocessor = load_policy_bundle(
            checkpoint.path,
            policy_ds_meta,
            rename_map=DEFAULT_RENAME_MAP,
        )
        for split_name, (dataset, dataset_repo, episodes) in datasets.items():
            print(f"Evaluating {split_name} on checkpoint step {checkpoint.step}")
            row = evaluate_split(
                split_name=split_name,
                checkpoint=checkpoint,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                dataset_repo=dataset_repo,
                episodes=episodes,
                task=args.task,
                frame_stride=args.frame_stride,
                max_frames_per_episode=args.max_frames_per_episode,
            )
            rows.append(row)
            print(
                f"  {split_name} step {checkpoint.step}: "
                f"mae_mean={row['mae_mean']:.6f}, rmse_mean={row['rmse_mean']:.6f}"
            )
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary(rows, args.output_dir)


if __name__ == "__main__":
    main()
