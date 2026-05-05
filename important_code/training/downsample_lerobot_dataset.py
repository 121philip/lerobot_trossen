#!/usr/bin/env python
"""Downsample a LeRobot video dataset by rebuilding it at a lower FPS."""

from __future__ import annotations

import argparse
import copy
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


DEFAULT_SOURCE_REPO_ID = "kaixiyao/widowxai_grape_grasping_V4_pos234_val"
DEFAULT_TARGET_FPS = 10
INDEX_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def compute_stride(source_fps: int, target_fps: int) -> int:
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if source_fps % target_fps != 0:
        raise ValueError(
            f"source_fps={source_fps} must be an integer multiple of target_fps={target_fps}"
        )
    return source_fps // target_fps


def iter_downsampled_relative_indices(length: int, *, stride: int) -> Iterable[int]:
    if length <= 0:
        raise ValueError("episode length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    return range(0, length, stride)


def to_hwc_uint8(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D image array, got shape {array.shape}")

    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)

    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0)
        array = np.rint(array * 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)

    return np.ascontiguousarray(array)


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def build_downsampled_frame(item: dict, features: dict) -> dict:
    frame = {"task": item["task"]}

    for key, feature in features.items():
        if key in INDEX_FEATURES:
            continue

        value = item[key]
        if feature["dtype"] in ("image", "video"):
            frame[key] = to_hwc_uint8(value)
        else:
            frame[key] = _to_numpy(value)

    return frame


def target_repo_id_for(source_repo_id: str, target_fps: int) -> str:
    namespace, name = source_repo_id.split("/", maxsplit=1)
    return f"{namespace}/{name}_{target_fps}hz"


def features_for_target_fps(features: dict, target_fps: int) -> dict:
    target_features = copy.deepcopy(features)
    for feature in target_features.values():
        if feature.get("dtype") == "video":
            feature.setdefault("info", {})["video.fps"] = target_fps
    return target_features


def _episode_bounds(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    episode = dataset.meta.episodes[episode_index]
    return int(episode["dataset_from_index"]), int(episode["dataset_to_index"])


def downsample_dataset(
    *,
    source_repo_id: str,
    target_repo_id: str,
    target_fps: int,
    root: Path | None,
    episodes: list[int] | None,
    overwrite: bool,
    push_to_hub: bool,
    vcodec: str,
    streaming_encoding: bool,
    encoder_queue_maxsize: int,
    encoder_threads: int | None,
) -> LeRobotDataset:
    source = LeRobotDataset(source_repo_id, episodes=episodes, download_videos=True)
    source_fps = int(source.fps)
    stride = compute_stride(source_fps, target_fps)

    output_root = root if root is not None else HF_LEROBOT_HOME / target_repo_id
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)

    target = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=target_fps,
        features=features_for_target_fps(source.meta.features, target_fps),
        root=output_root,
        robot_type=source.meta.robot_type,
        use_videos=len(source.meta.video_keys) > 0,
        vcodec=vcodec,
        streaming_encoding=streaming_encoding,
        encoder_queue_maxsize=encoder_queue_maxsize,
        encoder_threads=encoder_threads,
    )

    episode_indices = episodes if episodes is not None else list(range(source.meta.total_episodes))
    for new_episode_index, source_episode_index in enumerate(episode_indices, start=1):
        start, stop = _episode_bounds(source, source_episode_index)
        length = stop - start
        kept = 0
        print(
            f"[{new_episode_index}/{len(episode_indices)}] episode {source_episode_index}: "
            f"{length} frames -> {(length + stride - 1) // stride} frames",
            flush=True,
        )
        for relative_index in iter_downsampled_relative_indices(length, stride=stride):
            item = source[start + relative_index]
            target.add_frame(build_downsampled_frame(item, source.meta.features))
            kept += 1
        if kept == 0:
            raise RuntimeError(f"episode {source_episode_index} produced no frames")
        target.save_episode()

    target.finalize()
    if push_to_hub:
        target.push_to_hub()

    return target


def parse_episode_indices(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild a LeRobot dataset at a lower FPS.")
    parser.add_argument("--source-repo-id", default=DEFAULT_SOURCE_REPO_ID)
    parser.add_argument("--target-repo-id", default=None)
    parser.add_argument("--target-fps", type=int, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--root", type=Path, default=None, help="Output root directory.")
    parser.add_argument("--episodes", default=None, help="Comma-separated source episode indices.")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing local output root.")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the rebuilt dataset to Hub.")
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--no-streaming-encoding", action="store_true")
    parser.add_argument("--encoder-queue-maxsize", type=int, default=30)
    parser.add_argument("--encoder-threads", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_repo_id = args.target_repo_id or target_repo_id_for(args.source_repo_id, args.target_fps)

    dataset = downsample_dataset(
        source_repo_id=args.source_repo_id,
        target_repo_id=target_repo_id,
        target_fps=args.target_fps,
        root=args.root,
        episodes=parse_episode_indices(args.episodes),
        overwrite=args.overwrite,
        push_to_hub=args.push_to_hub,
        vcodec=args.vcodec,
        streaming_encoding=not args.no_streaming_encoding,
        encoder_queue_maxsize=args.encoder_queue_maxsize,
        encoder_threads=args.encoder_threads,
    )
    print(
        f"Created {dataset.repo_id} at {dataset.root} with "
        f"{dataset.meta.total_episodes} episodes and {dataset.meta.total_frames} frames."
    )


if __name__ == "__main__":
    main()
