#!/usr/bin/env bash
set -euo pipefail

# Prepare the merged V4 train/validation datasets, fine-tune, then run offline
# train/validation checkpoint evaluation.
#
# Preview dataset commands without starting training:
#   bash important_code/training/prepare_then_fine_tune_smolvla_v4.sh --dry-run
#
# Prepare datasets, train, then evaluate checkpoints:
#   bash important_code/training/prepare_then_fine_tune_smolvla_v4.sh
#
# If datasets already exist, skip split/merge and only train/evaluate:
#   bash important_code/training/prepare_then_fine_tune_smolvla_v4.sh --skip-data-prep

dry_run=false
skip_data_prep=false
skip_eval=false
data_prep_args=(--push-to-hub)
eval_args=()

while (($#)); do
  case "$1" in
    --dry-run)
      dry_run=true
      data_prep_args+=(--dry-run)
      ;;
    --skip-info)
      data_prep_args+=(--skip-info)
      ;;
    --skip-data-prep)
      skip_data_prep=true
      ;;
    --skip-eval)
      skip_eval=true
      ;;
    --eval-steps)
      shift
      if (($# == 0)); then
        echo "Missing value for --eval-steps" >&2
        exit 2
      fi
      eval_args+=(--steps "$1")
      ;;
    --eval-frame-stride)
      shift
      if (($# == 0)); then
        echo "Missing value for --eval-frame-stride" >&2
        exit 2
      fi
      eval_args+=(--frame-stride "$1")
      ;;
    --eval-max-frames-per-episode)
      shift
      if (($# == 0)); then
        echo "Missing value for --eval-max-frames-per-episode" >&2
        exit 2
      fi
      eval_args+=(--max-frames-per-episode "$1")
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash important_code/training/prepare_then_fine_tune_smolvla_v4.sh [--dry-run] [--skip-data-prep] [--skip-info] [--skip-eval] [--eval-steps STEPS] [--eval-frame-stride N] [--eval-max-frames-per-episode N]" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$skip_data_prep" == true ]]; then
  echo "Skipping dataset preparation; assuming merged train/validation datasets already exist."
else
  echo "Preparing SmolVLA V4 datasets..."
  uv run python important_code/training/prepare_smolvla_v4_datasets.py "${data_prep_args[@]}"
fi

if [[ "$dry_run" == true ]]; then
  echo "Dry run complete. Training was not started."
  exit 0
fi

echo "Starting SmolVLA V4 fine-tuning..."
bash important_code/training/fine_tune_smolvla.sh

if [[ "$skip_eval" == true ]]; then
  echo "Training completed. Offline checkpoint evaluation was skipped."
  exit 0
fi

echo "Training completed. Starting offline train/validation checkpoint evaluation..."
uv run python important_code/training/evaluate_smolvla_v4_checkpoints.py "${eval_args[@]}"
