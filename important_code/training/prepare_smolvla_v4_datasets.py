#!/usr/bin/env python
"""Prepare merged train/validation LeRobot datasets for SmolVLA V4 fine-tuning."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass


MERGED_TRAIN_REPO = "kaixiyao/widowxai_grape_grasping_V4_pos234_train"
MERGED_VAL_REPO = "kaixiyao/widowxai_grape_grasping_V4_pos234_val"


@dataclass(frozen=True)
class DatasetSplit:
    repo_id: str
    target_positions: int
    episodes_per_position: int = 10
    val_episodes_per_position: int = 2

    @property
    def train_repo_id(self) -> str:
        return f"{self.repo_id}_train"

    @property
    def val_repo_id(self) -> str:
        return f"{self.repo_id}_val"

    @property
    def train_episodes(self) -> list[int]:
        episodes: list[int] = []
        for position_idx in range(self.target_positions):
            start = position_idx * self.episodes_per_position
            train_end = start + self.episodes_per_position - self.val_episodes_per_position
            episodes.extend(range(start, train_end))
        return episodes

    @property
    def val_episodes(self) -> list[int]:
        episodes: list[int] = []
        for position_idx in range(self.target_positions):
            start = position_idx * self.episodes_per_position
            val_start = start + self.episodes_per_position - self.val_episodes_per_position
            stop = start + self.episodes_per_position
            episodes.extend(range(val_start, stop))
        return episodes


DATASET_SPLITS = [
    DatasetSplit(
        repo_id="kaixiyao/widowxai_grape_grasping_V4_position2",
        target_positions=5,
    ),
    DatasetSplit(
        repo_id="kaixiyao/widowxai_grape_grasping_V4_position3",
        target_positions=5,
    ),
    DatasetSplit(
        repo_id="kaixiyao/widowxai_grape_grasping_V4_position4",
        target_positions=4,
    ),
]


def _append_push_flag(command: list[str], push_to_hub: bool) -> list[str]:
    if push_to_hub:
        return [*command, "--push_to_hub", "true"]
    return command


def _compact_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def build_commands(*, push_to_hub: bool, include_info: bool) -> list[list[str]]:
    commands: list[list[str]] = []

    for split in DATASET_SPLITS:
        split_spec = {
            "train": split.train_episodes,
            "val": split.val_episodes,
        }
        commands.append(
            _append_push_flag(
                [
                    "lerobot-edit-dataset",
                    "--repo_id",
                    split.repo_id,
                    "--operation.type",
                    "split",
                    "--operation.splits",
                    _compact_json(split_spec),
                ],
                push_to_hub,
            )
        )

    commands.append(
        _append_push_flag(
            [
                "lerobot-edit-dataset",
                "--new_repo_id",
                MERGED_TRAIN_REPO,
                "--operation.type",
                "merge",
                "--operation.repo_ids",
                _compact_json([split.train_repo_id for split in DATASET_SPLITS]).replace('"', "'"),
            ],
            push_to_hub,
        )
    )
    commands.append(
        _append_push_flag(
            [
                "lerobot-edit-dataset",
                "--new_repo_id",
                MERGED_VAL_REPO,
                "--operation.type",
                "merge",
                "--operation.repo_ids",
                _compact_json([split.val_repo_id for split in DATASET_SPLITS]).replace('"', "'"),
            ],
            push_to_hub,
        )
    )

    if include_info:
        for repo_id in (MERGED_TRAIN_REPO, MERGED_VAL_REPO):
            commands.append(
                [
                    "lerobot-edit-dataset",
                    "--repo_id",
                    repo_id,
                    "--operation.type",
                    "info",
                    "--operation.show_features",
                    "true",
                ]
            )

    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split and merge SmolVLA V4 LeRobot datasets for train/validation."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push split and merged datasets to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--skip-info",
        action="store_true",
        help="Skip final lerobot-edit-dataset info checks for merged datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = build_commands(push_to_hub=args.push_to_hub, include_info=not args.skip_info)

    for index, command in enumerate(commands, start=1):
        rendered = shlex.join(command)
        print(f"[{index}/{len(commands)}] {rendered}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
