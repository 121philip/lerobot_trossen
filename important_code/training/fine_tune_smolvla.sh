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
  --policy.repo_id=kaixiyao/smolvla_widowx_grape_grasping_V4_pos234_10hz \
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
  --dataset.repo_id=kaixiyao/widowxai_grape_grasping_V4_pos234_train_10hz \
  --output_dir=outputs/train/smolvla_widowx_grape_grasping_V4_pos234_10hz \
  --job_name=smolvla_widowx_grape_grasping_V4_pos234_10hz \
  \
  --batch_size=32 \
  --num_workers=4 \
  --steps=40000 \
  --save_freq=5000 \
  --policy.scheduler_decay_steps=40000 \
  --policy.optimizer_lr=2e-4 \
  --policy.n_action_steps=10 \
  --policy.chunk_size=25 \
  --policy.resize_imgs_with_padding='[512, 512]' \
  --policy.device=cuda \
  --policy.use_amp=false \
  \
  --wandb.enable=true \
  --wandb.project=smolvla_widowx_grape_grasping \
  \
  --rename_map='{
      "observation.images.right": "observation.images.cam_main",
      "observation.images.wrist": "observation.images.cam_wrist"
  }'
