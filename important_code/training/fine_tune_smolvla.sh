#!/usr/bin/env bash
# Fine-tune SmolVLA on kaixiyao/widowxai_grape_grasping
#
# Cameras in dataset:
#   observation.images.wrist  -> mapped to observation.images.camera1
#   observation.images.right  -> mapped to observation.images.camera2
#
# NOTE: routes through run_train_clean.py so that observation.images.camera3
# (inherited from smolvla_base's pretrained config) is automatically removed
# before training starts.
#
# Usage:
#   chmod +x fine_tune_smolvla.sh
#   ./important_code/fine_tune_smolvla.sh

python important_code/run_train_clean.py \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=true \
  --policy.repo_id=kaixiyao/smolvla_widowx_grape_grasping_V3 \
  \
  --policy.input_features='{
      "observation.state": {"type": "STATE", "shape": [7]},
      "observation.images.camera1": {"type": "VISUAL", "shape": [3, 480, 640]},
      "observation.images.camera2": {"type": "VISUAL", "shape": [3, 480, 640]}
  }' \
  --policy.output_features='{
      "action": {"type": "ACTION", "shape": [7]}
  }' \
  \
  --dataset.repo_id=kaixiyao/widowxai_grape_grasping_V3 \
  --output_dir=outputs/train/smolvla_widowx_grape_grasping_V3 \
  --job_name=smolvla_widowx_grape_grasping_V3 \
  \
  --batch_size=8 \
  --steps=30000 \
  --save_freq=5000 \
  --policy.n_action_steps=50 \
  --policy.chunk_size=50 \
  --policy.device=cuda \
  \
  --wandb.enable=true \
  --wandb.project=smolvla_widowx_grape_grasping \
  \
  --rename_map='{
      "observation.images.wrist": "observation.images.camera1",
      "observation.images.right": "observation.images.camera2"
  }'
