#!/usr/bin/env python
"""Merge widowxai_pipe_bomb_transfer p1/p2/p11 datasets into one."""

from __future__ import annotations

import argparse
import shlex
import subprocess

OWNER = "kaixiyao"

SOURCE_REPOS = [
    f"{OWNER}/widowxai_pipe_bomb_transfer_p1_V1",
    f"{OWNER}/widowxai_pipe_bomb_transfer_p2_V1",
    f"{OWNER}/widowxai_pipe_bomb_transfer_p11_V1",
]

MERGED_REPO = f"{OWNER}/widowxai_pipe_bomb_transfer_merged"


def build_commands(*, push_to_hub: bool, include_info: bool) -> list[list[str]]:
    repo_ids_arg = "['" + "','".join(SOURCE_REPOS) + "']"

    merge_cmd = [
        "lerobot-edit-dataset",
        "--new_repo_id", MERGED_REPO,
        "--operation.type", "merge",
        "--operation.repo_ids", repo_ids_arg,
    ]
    if push_to_hub:
        merge_cmd += ["--push_to_hub", "true"]

    commands = [merge_cmd]

    if include_info:
        commands.append([
            "lerobot-edit-dataset",
            "--repo_id", MERGED_REPO,
            "--operation.type", "info",
            "--operation.show_features", "false",
        ])

    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge pipe_bomb_transfer p1/p2/p11 datasets into one."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--push-to-hub", action="store_true", help="Push merged dataset to Hugging Face Hub.")
    parser.add_argument("--skip-info", action="store_true", help="Skip final info check.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = build_commands(push_to_hub=args.push_to_hub, include_info=not args.skip_info)

    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
