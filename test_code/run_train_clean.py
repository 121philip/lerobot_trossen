#!/usr/bin/env python

import sys
import draccus
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import train


from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig


def main():
    print("DEBUG: Script started", flush=True)

    # 1. Capture original args
    args = sys.argv[1:]

    # 2. Filter for draccus (like wrap does)
    # We know policy has path arg, so we filter it out for draccus
    filtered_args = parser.filter_path_args(["policy"], args)
    cfg = draccus.parse(TrainPipelineConfig, args=filtered_args)
    print("DEBUG: Config parsed", flush=True)

    # 3. Manually load policy (mimic validate logic)
    policy_path = parser.get_path_arg("policy", args)
    if policy_path:
        print(f"DEBUG: Loading policy from {policy_path}", flush=True)
        cli_overrides = parser.get_cli_overrides("policy", args)
        cfg.policy = PreTrainedConfig.from_pretrained(
            policy_path, cli_overrides=cli_overrides)
        from pathlib import Path
        cfg.policy.pretrained_path = Path(policy_path)

    # 4. Modify policy
    if cfg.policy and "observation.images.camera3" in cfg.policy.input_features:
        print("WARNING: Detected 'observation.images.camera3' in config. Removing it...", flush=True)
        del cfg.policy.input_features["observation.images.camera3"]
        print("INFO: Successfully removed 'observation.images.camera3'.", flush=True)
    else:
        print("DEBUG: 'observation.images.camera3' not found in config.", flush=True)

    # 5. Remove policy.path from sys.argv to prevent re-loading in validate()
    # We need to find the arg that starts with --policy.path and remove it
    # 原因：train() 内部会调用 validate() 函数，validate() 会重新读取 sys.argv 并再次解析 --policy.path，
    # 这会覆盖掉我们刚才修改过的配置。通过清理 sys.argv，阻止重新加载。
    new_argv = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if not arg.startswith("--policy.path"):
            new_argv.append(arg)
    sys.argv = new_argv
    print("DEBUG: sys.argv modified to prevent reload", flush=True)

    print("DEBUG: Starting training...", flush=True)
    # Run the training loop with the modified configuration
    train(cfg)


if __name__ == "__main__":
    main()
