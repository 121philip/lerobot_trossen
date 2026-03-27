"""
Script to run SmolVLA inference on WidowX robot WITH Real-Time Chunking (RTC).

RTC generates action chunks asynchronously and blends overlapping chunks during
the flow-matching denoising process, preventing jerky transitions and pauses.

Architecture:
  - Inference thread: Continuously generates new action chunks via predict_action_chunk()
  - Actor thread: Sends actions from the queue to the robot at constant FPS
  - Main thread: Monitors duration and handles shutdown

Usage:
  # Dry run (no robot hardware needed):
  python important_code/run_inference_rtc.py --dry-run

  # With robot (default IP):
  python important_code/run_inference_rtc.py --robot-ip 192.168.2.3

  # Use a different local checkpoint:
  python important_code/run_inference_rtc.py --train-dir outputs/train/my_run

  # Tune RTC parameters:
  python important_code/run_inference_rtc.py --execution-horizon 10 --guidance-weight 10.0

  # Set duration limit:
  python important_code/run_inference_rtc.py --duration 120
  python important_code/run_inference_rtc.py --dry-run --debug
  python important_code/run_inference_rtc.py --dry-run --debug --debug-maxlen 200

"""

import logging
import math
import sys
import time
import traceback
from copy import copy
from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np
import torch
from torch import Tensor

# === LeRobot Imports ===
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.policies.rtc.latency_tracker import LatencyTracker
except ImportError as e:
    print(f"Error importing lerobot: {e}")
    print("Requires lerobot >= 0.4.3 with RTC support.")
    sys.exit(1)

try:
    from lerobot.policies.rtc.debug_visualizer import RTCDebugVisualizer
    _HAS_DEBUG_VISUALIZER = True
except ImportError:
    _HAS_DEBUG_VISUALIZER = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Configuration Defaults ===
DEFAULT_ROBOT_IP = "192.168.2.3"
DEFAULT_CAM1_ID = 2   # wrist camera
DEFAULT_CAM2_ID = 10  # right camera
DEFAULT_TRAIN_DIR = "outputs/train/smolvla_widowx_grape_grasping"
TASK_DESCRIPTION = "The robot grasps a grape"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

JOINT_NAMES = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "left_carriage_joint",
]


# ======================================================================
# Utility Functions (shared with run_inference.py)
# ======================================================================

def resolve_checkpoint_path(train_dir):
    """Resolves the path to the latest checkpoint's pretrained_model directory."""
    base_path = Path(train_dir)
    last_ptr = base_path / "checkpoints" / "last"

    if not last_ptr.exists():
        raise FileNotFoundError(f"Could not find 'last' pointer at: {last_ptr}")

    if last_ptr.is_dir():
        checkpoint_dir = last_ptr
    elif last_ptr.is_file():
        with open(last_ptr, "r") as f:
            step_name = f.read().strip()
        checkpoint_dir = base_path / "checkpoints" / step_name
    else:
        raise ValueError(f"'last' at {last_ptr} is neither a directory nor a file.")

    checkpoint_path = checkpoint_dir / "pretrained_model"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return str(checkpoint_path)


def robot_obs_to_policy_obs(raw_obs, joint_names):
    """
    Convert the robot's raw observation dict to the format expected by
    prepare_observation_for_inference / the preprocessor pipeline.

    Robot returns:
        {"joint_0.pos": float, ..., "wrist": ndarray, "right": ndarray}

    Policy expects:
        {"observation.state": ndarray(7,), "observation.images.wrist": ndarray, ...}
    """
    obs = {}

    # Assemble joint positions into observation.state
    joint_positions = []
    for name in joint_names:
        key = f"{name}.pos"
        if key in raw_obs:
            joint_positions.append(raw_obs[key])
        else:
            logging.warning(f"Missing joint {key} in observation, using 0.0")
            joint_positions.append(0.0)
    obs["observation.state"] = np.array(joint_positions, dtype=np.float32)

    # Copy camera images (preprocessor rename_map: wrist→camera1, right→camera2)
    if "wrist" in raw_obs:
        obs["observation.images.wrist"] = raw_obs["wrist"]
    if "right" in raw_obs:
        obs["observation.images.right"] = raw_obs["right"]

    return obs


def create_mock_observation(joint_names):
    """Create a mock observation for dry-run mode."""
    obs = {}
    for name in joint_names:
        obs[f"{name}.pos"] = 0.0
    obs["wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
    obs["right"] = np.zeros((480, 640, 3), dtype=np.uint8)
    return obs


def policy_action_to_robot_action(action_tensor, joint_names):
    """Convert policy action tensor (action_dim,) to robot dict {"joint.pos": float}."""
    action_np = action_tensor.cpu().numpy()
    action_dict = {}
    for i, name in enumerate(joint_names):
        if i < len(action_np):
            action_dict[f"{name}.pos"] = float(action_np[i])
    return action_dict


# ======================================================================
# Thread-safe Robot Wrapper
# ======================================================================

class RobotWrapper:
    """Thread-safe wrapper around the robot for multi-threaded access."""

    def __init__(self, robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self):
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action_dict):
        with self.lock:
            self.robot.send_action(action_dict)


# ======================================================================
# Inference Thread (asynchronous action chunk generation)
# ======================================================================

def inference_thread_fn(
    policy,
    robot_wrapper,
    preprocessor,
    postprocessor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    args,
):
    """Background thread: generates action chunks with RTC."""
    try:
        logger.info("[INFERENCE] Starting inference thread")

        latency_tracker = LatencyTracker()
        fps = args.fps
        time_per_chunk = 1.0 / fps

        # How many actions in the queue before we request a new chunk
        get_actions_threshold = args.queue_threshold if args.rtc_enabled else 0

        while not shutdown_event.is_set():
            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                inference_latency = latency_tracker.max() or 0.0
                inference_delay = math.ceil(inference_latency / time_per_chunk)

                # Get observation
                if robot_wrapper is not None:
                    raw_obs = robot_wrapper.get_observation()
                else:
                    raw_obs = create_mock_observation(JOINT_NAMES)

                # Convert to policy format
                observation = robot_obs_to_policy_obs(raw_obs, JOINT_NAMES)

                # NOTE: Use torch.no_grad() instead of torch.inference_mode()!
                # RTC's denoise_step uses torch.enable_grad() + torch.autograd.grad()
                # internally, and enable_grad() cannot override inference_mode().
                with torch.no_grad():
                    # Prepare observation tensors
                    obs_tensors = prepare_observation_for_inference(
                        copy(observation), DEVICE, task=args.task
                    )

                    # Preprocessor: rename cameras, tokenize task, normalize state
                    obs_processed = preprocessor(obs_tensors)

                # predict_action_chunk must run OUTSIDE torch.no_grad()
                # because RTC guidance needs autograd for correction computation
                actions = policy.predict_action_chunk(
                    obs_processed,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )

                # Inspect the outputs!
                print("Raw chunk shape: ", actions.shape)
                # prints e.g. torch.Size([1, 50, 7]) -> [Batch window, Next N Time Steps, Joint Values]

                # Print the predicted trajectory of the base joint ('joint_0') over the entire chunk
                print("Future base joint trajectory:", actions[0, :, 0].cpu().numpy())

                # Store original actions (before postprocessing) for RTC
                original_actions = actions.squeeze(0).clone()

                # Postprocess (unnormalize + to CPU)
                postprocessed_actions = postprocessor(actions)
                postprocessed_actions = postprocessed_actions.squeeze(0)

                # Track latency
                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)

                if args.rtc_enabled and args.queue_threshold < args.execution_horizon + new_delay:
                    logger.warning(
                        f"[INFERENCE] queue_threshold ({args.queue_threshold}) too small, "
                        f"should be > execution_horizon + delay ({args.execution_horizon + new_delay})"
                    )

                # Merge into the action queue
                action_queue.merge(
                    original_actions, postprocessed_actions, new_delay, action_index_before_inference
                )

                logger.info(
                    f"[INFERENCE] Chunk generated in {new_latency:.3f}s "
                    f"(delay={new_delay}, queue_size={action_queue.qsize()})"
                )
            else:
                time.sleep(0.05)  # Small sleep to prevent busy waiting

        logger.info("[INFERENCE] Thread shutting down")
    except Exception as e:
        logger.error(f"[INFERENCE] Fatal error: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()


# ======================================================================
# Actor Thread (real-time action execution)
# ======================================================================

def actor_thread_fn(
    robot_wrapper,
    action_queue: ActionQueue,
    shutdown_event: Event,
    args,
):
    """Main actor thread: sends actions to robot at fixed FPS."""
    try:
        logger.info("[ACTOR] Starting actor thread")

        action_count = 0
        action_interval = 1.0 / args.fps

        while not shutdown_event.is_set():
            start_time = time.perf_counter()

            # Get next action from queue
            action = action_queue.get()

            if action is not None:
                action_dict = policy_action_to_robot_action(action, JOINT_NAMES)

                if robot_wrapper is not None:
                    robot_wrapper.send_action(action_dict)
                else:
                    if action_count % 10 == 0:  # Print every 10th action in dry-run
                        logger.info(f"[DRY-RUN] Step {action_count}: {action_dict}")

                action_count += 1

            # Maintain constant FPS
            dt_s = time.perf_counter() - start_time
            sleep_time = max(0, action_interval - dt_s - 0.001)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info(f"[ACTOR] Thread shutting down. Total actions: {action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Fatal error: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()


# ======================================================================
# Main
# ======================================================================

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SmolVLA + RTC Inference on WidowX Robot"
    )

    # Robot
    parser.add_argument("--robot-ip", type=str, default=DEFAULT_ROBOT_IP)
    parser.add_argument("--cam1", type=int, default=DEFAULT_CAM1_ID,
                        help="Camera index for camera1 (default: 10)")
    parser.add_argument("--cam2", type=int, default=DEFAULT_CAM2_ID,
                        help="Camera index for camera2 (default: 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without robot hardware")

    # Policy
    parser.add_argument("--train-dir", type=str, default=DEFAULT_TRAIN_DIR,
                        help="Path to local training output directory")
    parser.add_argument("--task", type=str, default=TASK_DESCRIPTION)

    # Control
    parser.add_argument("--fps", type=int, default=30,
                        help="Robot control frequency (Hz)")
    parser.add_argument("--duration", type=float, default=120.0,
                        help="Max execution duration (seconds)")

    # RTC parameters
    parser.add_argument("--rtc-enabled", action="store_true", default=True,
                        help="Enable Real-Time Chunking (default: True)")
    parser.add_argument("--no-rtc", dest="rtc_enabled", action="store_false",
                        help="Disable RTC (fallback to standard chunking)")
    parser.add_argument("--execution-horizon", type=int, default=10,
                        help="RTC: steps to blend between chunks (default: 10)")
    parser.add_argument("--guidance-weight", type=float, default=10.0,
                        help="RTC: consistency guidance strength (default: 10.0)")
    parser.add_argument("--attention-schedule", type=str, default="EXP",
                        choices=["LINEAR", "EXP", "ONES", "ZEROS"],
                        help="RTC: prefix attention schedule (default: EXP)")
    parser.add_argument("--queue-threshold", type=int, default=30,
                        help="Request new chunk when queue drops below this (default: 30)")

    # Debug tracking
    parser.add_argument("--debug", action="store_true",
                        help="Enable RTC built-in debug tracking")
    parser.add_argument("--debug-maxlen", type=int, default=100,
                        help="Max entries stored in RTC debug buffer (default: 100)")

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Device: {DEVICE}")
    logger.info(f"RTC Enabled: {args.rtc_enabled}")

    # ============================================================
    # 1. Load Policy with RTC Config
    # ============================================================
    policy_path = resolve_checkpoint_path(args.train_dir)
    logger.info(f"Loading policy from: {policy_path}")
    policy = SmolVLAPolicy.from_pretrained(policy_path)

    # Configure RTC
    rtc_config = RTCConfig(
        enabled=args.rtc_enabled,
        execution_horizon=args.execution_horizon,
        max_guidance_weight=args.guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule[args.attention_schedule],
    )
    if args.debug:
        rtc_config.debug = True
        rtc_config.debug_maxlen = args.debug_maxlen
        logger.info(f"RTC debug tracking enabled (maxlen={args.debug_maxlen})")

    policy.config.rtc_config = rtc_config
    policy.init_rtc_processor()

    policy.to(DEVICE)
    policy.eval()
    logger.info(f"Policy loaded with RTC config: {rtc_config}")

    # ============================================================
    # 2. Load Preprocessor & Postprocessor
    # ============================================================
    logger.info("Loading processors...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=policy_path,
    )
    logger.info("Processors loaded.")

    # ============================================================
    # 3. Connect Robot (skip in dry-run)
    # ============================================================
    robot = None
    robot_wrapper = None

    if not args.dry_run:
        try:
            from lerobot_robot_trossen.config_widowxai_follower import (
                WidowXAIFollowerConfig,
            )
            from lerobot.robots.utils import make_robot_from_config

            camera_config_map = {
                "wrist": OpenCVCameraConfig(
                    index_or_path=args.cam1, fps=args.fps, width=640, height=480
                ),
                "right": OpenCVCameraConfig(
                    index_or_path=args.cam2, fps=args.fps, width=640, height=480
                ),
            }

            robot_config = WidowXAIFollowerConfig(
                ip_address=args.robot_ip,
                cameras=camera_config_map,
                max_relative_target=5.0,
            )

            robot = make_robot_from_config(robot_config)
            logger.info("Connecting to robot...")
            robot.connect()
            logger.info("Robot connected.")
            robot_wrapper = RobotWrapper(robot)
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            raise
    else:
        logger.info("=== DRY RUN MODE — No robot hardware ===")

    # ============================================================
    # 4. Create Action Queue and Start Threads
    # ============================================================
    shutdown_event = Event()
    action_queue = ActionQueue(rtc_config)

    logger.info(f"Task: {args.task}")
    logger.info(f"FPS: {args.fps}, Duration: {args.duration}s")
    logger.info(f"RTC: execution_horizon={args.execution_horizon}, "
                f"guidance_weight={args.guidance_weight}, "
                f"schedule={args.attention_schedule}")

    # Start inference thread
    inf_thread = Thread(
        target=inference_thread_fn,
        args=(policy, robot_wrapper, preprocessor, postprocessor,
              action_queue, shutdown_event, args),
        daemon=True,
        name="InferenceThread",
    )
    inf_thread.start()
    logger.info("Inference thread started.")

    # Start actor thread
    act_thread = Thread(
        target=actor_thread_fn,
        args=(robot_wrapper, action_queue, shutdown_event, args),
        daemon=True,
        name="ActorThread",
    )
    act_thread.start()
    logger.info("Actor thread started.")

    # ============================================================
    # 5. Main Thread — Monitor duration + handle shutdown
    # ============================================================
    start_time = time.time()

    try:
        while not shutdown_event.is_set():
            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                logger.info(f"Duration limit ({args.duration}s) reached.")
                break

            # Log status periodically
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                logger.info(f"[MAIN] Elapsed: {elapsed:.0f}s, Queue size: {action_queue.qsize()}")

            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("Ctrl+C received, shutting down...")

    # ============================================================
    # 6. Cleanup
    # ============================================================
    shutdown_event.set()

    if inf_thread.is_alive():
        logger.info("Waiting for inference thread...")
        inf_thread.join(timeout=5.0)

    if act_thread.is_alive():
        logger.info("Waiting for actor thread...")
        act_thread.join(timeout=5.0)

    if robot:
        robot.disconnect()
        logger.info("Robot disconnected.")

    # ============================================================
    # 7. Debug Data (if enabled)
    # ============================================================
    if args.debug and hasattr(policy, "rtc_processor") and policy.rtc_processor is not None:
        debug_data = policy.rtc_processor.get_debug_data()
        logger.info(f"[DEBUG] RTC debug data keys: {list(debug_data.keys())}")
        for key, val in debug_data.items():
            logger.info(f"[DEBUG]   {key}: {val}")

        if _HAS_DEBUG_VISUALIZER:
            visualizer = RTCDebugVisualizer()
            visualizer.plot(debug_data)
        else:
            logger.info("[DEBUG] RTCDebugVisualizer not available — skipping plots.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
