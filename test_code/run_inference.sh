#!/bin/bash
# Wrapper to run SmolVLA inference on WidowX robot
#
# Usage:
#   bash test_code/run_inference.sh                     # Default: connect to robot
#   bash test_code/run_inference.sh --dry-run            # No robot hardware needed
#   bash test_code/run_inference.sh --ip 192.168.2.5     # Override robot IP
#   bash test_code/run_inference.sh --cam-high 10 --cam-wrist 2  # Override camera IDs

# Get the directory where this script lives, to find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default arguments
IP="192.168.2.3"
DRY_RUN=""
EXTRA_ARGS=""

# Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ip) IP="$2"; shift ;;
        --dry-run) DRY_RUN="--dry-run" ;;
        --cam-high) EXTRA_ARGS="$EXTRA_ARGS --cam-high $2"; shift ;;
        --cam-wrist) EXTRA_ARGS="$EXTRA_ARGS --cam-wrist $2"; shift ;;
        --task) EXTRA_ARGS="$EXTRA_ARGS --task \"$2\""; shift ;;
        --fps) EXTRA_ARGS="$EXTRA_ARGS --fps $2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "=== SmolVLA Inference ==="
echo "Robot IP: $IP"
if [ -n "$DRY_RUN" ]; then
    echo "Mode: DRY RUN (no robot)"
else
    echo "Mode: LIVE"
fi

cd "$PROJECT_ROOT"
exec ./.venv/bin/python test_code/run_inference.py \
    --robot-ip "$IP" \
    $DRY_RUN \
    $EXTRA_ARGS
