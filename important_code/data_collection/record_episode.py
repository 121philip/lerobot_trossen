from lerobot.scripts.lerobot_record import record
import sys
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.import_utils import register_third_party_plugins as register_third_party_devices

from lerobot_robot_trossen import WidowXAIFollowerConfig
from lerobot_teleoperator_trossen import WidowXAILeaderTeleopConfig

# Must be called before instantiating any third-party configs/robots
register_third_party_devices()

# ── Robot & Teleop Config ──────────────────────────────────────────────────────

robot_cfg = WidowXAIFollowerConfig(
    # IP address of the follower (WidowX) arm controller on the local network
    ip_address="192.168.2.3",
    # Logical name used to identify this robot in the dataset and logs
    id="follower",
    cameras={
        # Each entry is a named camera stream. The key becomes the observation key in the dataset.
        # index_or_path: USB device index (from `ls /dev/video*`) or a file path
        # width/height:  capture resolution in pixels
        # fps:           target capture frame rate (should match the top-level FPS)
        
        "wrist":      OpenCVCameraConfig(index_or_path=Path("/dev/video10"), width=640, height=480, fps=30, fourcc="YUYV"),
        "right":       OpenCVCameraConfig(index_or_path=Path("/dev/video4"),  width=640, height=480, fps=30, fourcc="YUYV"),
    },
)

teleop_cfg = WidowXAILeaderTeleopConfig(
    # IP address of the leader (WidowX) arm controller on the local network
    ip_address="192.168.1.2",
    # Logical name used to identify this teleoperator in logs
    id="leader",
)

# ── Dataset Config ─────────────────────────────────────────────────────────────

# Your Hugging Face username (run `huggingface-cli whoami` to confirm)
HF_USER = "kaixiyao"

# Dataset repository name on Hugging Face Hub.
# The full repo will be: {HF_USER}/{DATASET_NAME}
DATASET_NAME = "widowxai_grape_grasping_V4_position1"

# Short description of the task being demonstrated in every episode.
# This label is stored per-frame and is used for language-conditioned training.
TASK_DESCRIPTION = "pick the grape"

# Total number of episodes to record in this session.
# Each episode = one full demonstration from start to end.
NUM_EPISODES = 50

# Duration of the active recording phase per episode (in seconds).
# The robot records observations and actions for this long.
EPISODE_TIME_S = 30

# Duration of the environment reset phase between episodes (in seconds).
# The robot keeps receiving teleop commands but nothing is saved to the dataset.
# Use this time to move objects back to their start position.
RESET_TIME_S = 15

# Target control and recording frequency (frames per second).
# Camera fps (set above) should match this value.
FPS = 30

# If True, opens a Rerun viewer window showing live camera feeds and joint states.
# Disable if you experience FPS drops caused by the display overhead.
DISPLAY_DATA = True

# If True, uploads the finished dataset to Hugging Face Hub automatically
# after all episodes are recorded.
PUSH_TO_HUB = True

# If True, the dataset repository on Hugging Face Hub will be private.
# Requires a write-access token (see `huggingface-cli login`).
PRIVATE = False

# ── Entry point ────────────────────────────────────────────────────────────────
# lerobot-record uses @parser.wrap() which reads from sys.argv.
# We populate sys.argv here so the script can be run directly with `python record_episode.py`.

sys.argv = [
    "lerobot-record",

    # Robot type registered by the Trossen plugin
    f"--robot.type=widowxai_follower_robot",
    # Network address of the follower arm
    f"--robot.ip_address={robot_cfg.ip_address}",
    # Logical robot identifier (used in dataset metadata)
    f"--robot.id={robot_cfg.id}",
    # Camera definitions in YAML-like inline dict format.
    # Keys must match those used in robot_cfg.cameras above.
    "--robot.cameras={"
        "wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}, "
        "right: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}"
    "}",

    # Teleoperator type registered by the Trossen plugin
    f"--teleop.type=widowxai_leader_teleop",
    # Network address of the leader arm
    f"--teleop.ip_address={teleop_cfg.ip_address}",
    # Logical teleoperator identifier
    f"--teleop.id={teleop_cfg.id}",

    # Show live camera + joint data in the Rerun viewer
    f"--display_data={str(DISPLAY_DATA).lower()}",

    # Full HuggingFace dataset repo path: {username}/{dataset_name}
    f"--dataset.repo_id={HF_USER}/{DATASET_NAME}",
    # Task label stored with every frame (used for language-conditioned policies)
    f"--dataset.single_task={TASK_DESCRIPTION}",
    # Recording frequency; must match camera fps
    f"--dataset.fps={FPS}",
    # Number of demonstrations to record
    f"--dataset.num_episodes={NUM_EPISODES}",
    # Active recording duration per episode (seconds)
    f"--dataset.episode_time_s={EPISODE_TIME_S}",
    # Environment reset duration between episodes (seconds); not saved to dataset
    f"--dataset.reset_time_s={RESET_TIME_S}",
    # Push the completed dataset to Hugging Face Hub when done
    f"--dataset.push_to_hub={str(PUSH_TO_HUB).lower()}",
    # Make the Hub repository private
    f"--dataset.private={str(PRIVATE).lower()}",
    # Number of background threads writing camera frames to disk, per camera.
    # Increase to 8 if you observe dropped frames or unstable FPS.
    "--dataset.num_image_writer_threads_per_camera=4",
    # Enable voice narration of episode progress and status updates
    "--play_sounds=true",
]


record()
