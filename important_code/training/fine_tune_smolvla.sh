#!/usr/bin/env bash
set -euo pipefail
 
# Fine-tune SmolVLA on merged V4 grape-grasping datasets.
#
# Cameras in dataset:
#   observation.images.right -> mapped to observation.images.cam_main
#   observation.images.wrist -> mapped to observation.images.cam_wrist
#
# Dataset preparation:
#   uv run python important_code/training/prepare_smolvla_v4_datasets.py --push-to-hub
#
# Usage:
#   bash important_code/training/fine_tune_smolvla.sh
#
# Prepare datasets, train, and evaluate checkpoints automatically:
#   bash important_code/training/prepare_then_fine_tune_smolvla_v4.sh
 
uv run python important_code/training/run_train_clean.py \
  --policy.path=TrossenRoboticsCommunity/smolvla_solo_red_block \
  --policy.push_to_hub=true \
  --policy.repo_id=kaixiyao/smolvla_pipe_bomb_transfer_V1 \
  \
  --policy.input_features='{
      "observation.state": {"type": "STATE", "shape": [7]},
      "observation.images.cam_main": {"type": "VISUAL", "shape": [3, 480, 640]},
      "observation.images.cam_wrist": {"type": "VISUAL", "shape": [3, 480, 640]}
  }' \
  --policy.output_features='{
      "action": {"type": "ACTION", "shape": [7]}
  }' \
  \
  --dataset.repo_id=kaixiyao/widowxai_pipe_bomb_transfer_merged \
  --output_dir=outputs/train/smolvla_pipe_bomb_transfer_V1 \
  --job_name=smolvla_pipe_bomb_transfer_V1 \
  \
  --batch_size=64 \
  --num_workers=4 \
  --steps=20000 \
  --save_freq=5000 \
  --policy.scheduler_decay_steps=20000 \
  --policy.optimizer_lr=2e-4 \
  --policy.n_action_steps=10 \
  --policy.chunk_size=25 \
  --policy.resize_imgs_with_padding='[384, 384]' \
  --policy.device=cuda \
  --policy.use_amp=true \
  \
  --wandb.enable=true \
  --wandb.project=smolvla_pipe_bomb_transfer \
  \
  --rename_map='{
      "observation.images.right": "observation.images.cam_main",
      "observation.images.wrist": "observation.images.cam_wrist"
  }'
