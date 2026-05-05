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

# RTX A2000 6GB low-VRAM defaults. The desktop session can leave only ~2-3GB
# free, so avoid allocator fragmentation and let Accelerate use FP16 autocast.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-fp16}"

uv run python important_code/training/run_train_clean.py \
  --policy.path=TrossenRoboticsCommunity/smolvla_solo_red_block \
  --policy.push_to_hub=true \
  --policy.repo_id=kaixiyao/smolvla_widowx_grape_grasping_V4_pos234_lora \
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
  --dataset.repo_id=kaixiyao/widowxai_grape_grasping_V4_pos234_train \
  --output_dir=outputs/train/smolvla_widowx_grape_grasping_V4_pos234_lora \
  --job_name=smolvla_widowx_grape_grasping_V4_pos234_lora \
  \
  --peft.method_type=LORA \
  --peft.r=16 \
  \
  --batch_size=8 \
  --num_workers=4 \
  --steps=40000 \
  --save_freq=5000 \
  --policy.scheduler_decay_steps=40000 \
  --policy.n_action_steps=50 \
  --policy.chunk_size=50 \
  --policy.resize_imgs_with_padding='[512, 512]' \
  --policy.device=cuda \
  --policy.use_amp=true \
  \
  --wandb.enable=true \
  --wandb.project=smolvla_widowx_grape_grasping \
  \
  --rename_map='{
      "observation.images.right": "observation.images.cam_main",
      "observation.images.wrist": "observation.images.cam_wrist"
  }'
