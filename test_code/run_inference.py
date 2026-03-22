"""
Script to run inference with a fine-tuned SmolVLA policy on a WidowX robot.
Handles 2-camera setup (cam_high, cam_wrist) mapped to model inputs (camera1, camera2)
via the saved preprocessor pipeline.

Usage:
  # Dry run (no robot hardware needed):
  python test_code/run_inference.py --dry-run

  # With robot:
  python test_code/run_inference.py --robot-ip 192.168.2.3

  # Override camera IDs:
  python test_code/run_inference.py --cam-high 10 --cam-wrist 2
"""

import time
import torch
import numpy as np
import logging
import argparse
import sys
from pathlib import Path
from copy import copy

# === LeRobot Imports ===
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.cameras.opencv import OpenCVCameraConfig
except ImportError as e:
    print(f"Error importing lerobot: {e}")
    print("Please make sure 'lerobot' is installed and in your PYTHONPATH.")
    sys.exit(1)

# === Configuration Defaults ===
DEFAULT_ROBOT_IP = "192.168.2.3"
DEFAULT_CAM_HIGH_ID = 10
DEFAULT_CAM_WRIST_ID = 2

# Path to the training output directory
TRAIN_OUTPUT_DIR = "outputs/train/smolvla_widowx_aluminum"
TASK_DESCRIPTION = "The robot grabs an aluminum profile and puts it into the box"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Joint names matching WidowXAIFollowerConfig
JOINT_NAMES = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "left_carriage_joint",
]


def resolve_checkpoint_path(train_dir):
    """
    Resolves the actual path to the latest checkpoint.
    Reads 'checkpoints/last' file if it exists to find the step number.
    """
    base_path = Path(train_dir)
    checkpoints_dir = base_path / "checkpoints"

    last_ptr = checkpoints_dir / "last"
    if not last_ptr.exists():
        raise FileNotFoundError(f"Could not find 'last' pointer at: {last_ptr}")

    # Case A: 'last' is a symlink to a directory or a directory itself (LeRobot default)
    if last_ptr.is_dir():
        checkpoint_dir = last_ptr

    # Case B: 'last' is a text file containing the directory name (Legacy/Alternative)
    elif last_ptr.is_file():
        with open(last_ptr, "r") as f:
            step_name = f.read().strip()
        checkpoint_dir = checkpoints_dir / step_name

    else:
        raise ValueError(f"'last' at {last_ptr} is neither a directory nor a file.")

    checkpoint_path = checkpoint_dir / "pretrained_model"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint path pointed to by 'last' does not exist: {checkpoint_path}"
        )

    return str(checkpoint_path)


def robot_obs_to_policy_obs(raw_obs, joint_names):
    """
    Convert the robot's raw observation dict into the format expected by
    prepare_observation_for_inference / the preprocessor pipeline.

    Robot returns:
        {
            "joint_0.pos": float, "joint_1.pos": float, ...,
            "cam_high": ndarray (H,W,3), "cam_wrist": ndarray (H,W,3),
        }

    Policy expects (before preprocessing):
        {
            "observation.state": ndarray (7,),
            "observation.images.cam_high": ndarray (H,W,3),
            "observation.images.cam_wrist": ndarray (H,W,3),
        }

    Note: The preprocessor pipeline handles cam_high -> camera1 renaming,
    image normalization, batching, tokenization, and device transfer.
    """
    obs = {}

    # 1. Assemble joint positions into observation.state
    joint_positions = []
    for name in joint_names:
        key = f"{name}.pos"
        if key in raw_obs:
            joint_positions.append(raw_obs[key])
        else:
            logging.warning(f"Missing joint {key} in observation, using 0.0")
            joint_positions.append(0.0)
    obs["observation.state"] = np.array(joint_positions, dtype=np.float32)

    # 2. Copy camera images with the expected key format
    # The preprocessor will rename cam_high -> camera1 and cam_wrist -> camera2
    if "cam_high" in raw_obs:
        obs["observation.images.cam_high"] = raw_obs["cam_high"]
    if "cam_wrist" in raw_obs:
        obs["observation.images.cam_wrist"] = raw_obs["cam_wrist"]

    return obs


def create_mock_observation(joint_names):
    """Create a mock observation for dry-run mode without robot hardware."""
    obs = {}
    for name in joint_names:
        obs[f"{name}.pos"] = 0.0
    obs["cam_high"] = np.zeros((480, 640, 3), dtype=np.uint8)
    obs["cam_wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
    return obs


def policy_action_to_robot_action(action_tensor, joint_names):
    """
    Convert the policy's output action tensor to a robot action dict.

    The policy outputs a tensor of shape (action_dim,) = (7,) after squeeze.
    The robot expects a dict: {"joint_0.pos": float, "joint_1.pos": float, ...}
    """
    action_np = action_tensor.squeeze(0).cpu().numpy()
    action_dict = {}
    for i, name in enumerate(joint_names):
        if i < len(action_np):
            action_dict[f"{name}.pos"] = float(action_np[i])
    return action_dict


def main():
    parser = argparse.ArgumentParser(description="Run SmolVLA Inference on WidowX Robot")
    parser.add_argument(
        "--robot-ip", type=str, default=DEFAULT_ROBOT_IP, help="Robot IP address"
    )
    parser.add_argument(
        "--cam-high",
        type=int,
        default=DEFAULT_CAM_HIGH_ID,
        help="Camera ID for cam_high (overhead camera)",
    )
    parser.add_argument(
        "--cam-wrist",
        type=int,
        default=DEFAULT_CAM_WRIST_ID,
        help="Camera ID for cam_wrist (wrist camera)",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=TRAIN_OUTPUT_DIR,
        help="Path to training output directory",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=TASK_DESCRIPTION,
        help="Task description for the policy",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Control loop frequency in Hz",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without connecting to real robot hardware",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print(f"Device: {DEVICE}")

    # ============================================================
    # 1. Resolve and Load Policy
    # ============================================================
    try:
        policy_path = resolve_checkpoint_path(args.train_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    print(f"Loading policy from: {policy_path}")
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.to(DEVICE)
    policy.eval()
    print("Policy loaded successfully.")

    # ============================================================
    # 2. Load Preprocessor & Postprocessor Pipelines
    # ============================================================
    print("Loading processors...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=policy_path
    )
    print("Processors loaded successfully.")

    # ============================================================
    # 3. Connect Robot (skip in dry-run)
    # ============================================================
    robot = None
    if not args.dry_run:
        try:
            from lerobot_robot_trossen.config_widowxai_follower import (
                WidowXAIFollowerConfig,
            )
            from lerobot.robots.utils import make_robot_from_config

            camera_config_map = {
                "cam_high": OpenCVCameraConfig(
                    index_or_path=args.cam_high, fps=args.fps, width=640, height=480
                ),
                "cam_wrist": OpenCVCameraConfig(
                    index_or_path=args.cam_wrist, fps=args.fps, width=640, height=480
                ),
            }

            robot_config = WidowXAIFollowerConfig(
                ip_address=args.robot_ip,
                cameras=camera_config_map,
                max_relative_target=5.0,
            )
            print(f"Robot config: {robot_config}")

            robot = make_robot_from_config(robot_config)
            print("Connecting to robot...")
            robot.connect()
            print("Robot connected successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            raise
    else:
        print("=== DRY RUN MODE — Robot will NOT be connected ===")

    # ============================================================
    # 4. Control Loop
    # ============================================================
    print(f"\nTask: {args.task}")
    print(f"Control rate: {args.fps} Hz")
    print("Press Ctrl+C to stop.\n")

    policy.reset()
    period = 1.0 / args.fps

    try:
        while True:
            loop_start = time.time()

            # A. Get Observation
            if robot:
                raw_obs = robot.get_observation()
            else:
                raw_obs = create_mock_observation(JOINT_NAMES)

            # B. Convert robot observation to policy format
            observation = robot_obs_to_policy_obs(raw_obs, JOINT_NAMES)

            # C. Run inference pipeline (prepare → preprocess → select_action → postprocess)
            with torch.inference_mode():
                # Convert numpy → tensors, normalize images, add batch dim, move to device
                obs_tensors = prepare_observation_for_inference(
                    copy(observation), DEVICE, task=args.task
                )

                # Preprocessor: rename cameras, tokenize task, normalize state
                obs_processed = preprocessor(obs_tensors)

                # Policy inference
                action = policy.select_action(obs_processed)

                # Postprocessor: unnormalize action, move to CPU
                action = postprocessor(action)

            # D. Convert action tensor to robot command dict
            action_dict = policy_action_to_robot_action(action, JOINT_NAMES)

            # E. Send Action
            if robot:
                robot.send_action(action_dict)
            else:
                print(f"[Dry Run] Action: {action_dict}")

            # F. Frequency Control
            dt = time.time() - loop_start
            sleep_time = max(0, period - dt)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        logger.error(f"Error during control loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if robot:
            print("Disconnecting robot...")
            robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
