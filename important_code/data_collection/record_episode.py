from pathlib import Path

import trossen_arm
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
)
from lerobot.utils.import_utils import register_third_party_plugins as register_third_party_devices
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun

from lerobot_robot_trossen import WidowXAIFollowerConfig
from lerobot_robot_trossen.widowxai_follower import WidowXAIFollower
from lerobot_teleoperator_trossen import WidowXAILeaderTeleopConfig
from lerobot_teleoperator_trossen.widowxai_leader import WidowXAILeaderTeleop

# Must be called before instantiating any third-party configs/robots
register_third_party_devices()

# ── Robot & Teleop Config ──────────────────────────────────────────────────────

robot_cfg = WidowXAIFollowerConfig(
    ip_address="192.168.2.3",
    id="follower",
    cameras={
        "wrist": OpenCVCameraConfig(index_or_path=Path("/dev/video4"),  width=640, height=480, fps=30, fourcc="YUYV"),
        "right": OpenCVCameraConfig(index_or_path=Path("/dev/video10"), width=640, height=480, fps=30, fourcc="YUYV"),
    },
)

teleop_cfg = WidowXAILeaderTeleopConfig(
    ip_address="192.168.1.2",
    id="leader",
)

# ── Dataset Config ─────────────────────────────────────────────────────────────

HF_USER = "kaixiyao"
DATASET_NAME = "widowxai_grape_grasping_V4_position4"
TASK_DESCRIPTION = "pick the grape"
NUM_EPISODES = 40
EPISODE_TIME_S = 20
RESET_TIME_S = 5
FPS = 30
DISPLAY_DATA = True
PUSH_TO_HUB = True
PRIVATE = False

# ── Positioning Config ─────────────────────────────────────────────────────────
# If True, before the first episode the user can teleoperate the robot to a
# custom starting position.  Press Right Arrow (→) to start recording.
# Press Escape to stop all recording early.
ENABLE_POSITIONING = True

# Maximum seconds to spend in the positioning phase.  If the user doesn't press
# Right Arrow before this time elapses, recording starts automatically.
MAX_POSITIONING_TIME_S = 120

# Gripper (left_carriage_joint) position in metres used as the "open" state
# when returning to the home position after each episode.
# Adjust if the gripper doesn't open fully.
GRIPPER_OPEN_POS = 0.035


# ── Helpers ────────────────────────────────────────────────────────────────────

def _return_to_home(robot: WidowXAIFollower, teleop: WidowXAILeaderTeleop, home_follower: list[float], home_leader: list[float]) -> None:
    """Open gripper, wait 2 s, then move both arms back to the home position."""
    import time
    n_joints = len(robot_cfg.joint_names)

    # ── Step 1: open gripper only (arm joints stay put) ───────────────────────
    gripper_only = list(robot.driver.get_all_positions())
    gripper_only[-1] = GRIPPER_OPEN_POS
    robot.driver.set_all_positions(
        trossen_arm.VectorDouble(gripper_only),
        goal_time=2.0,
        blocking=True,
    )
    time.sleep(2.0)

    # ── Step 2: move follower arm to home position (gripper stays open) ────────
    follower_goal = list(home_follower)
    follower_goal[-1] = GRIPPER_OPEN_POS
    robot.driver.set_all_positions(
        trossen_arm.VectorDouble(follower_goal),
        goal_time=3.0,
        blocking=True,
    )

    # ── Step 3: move leader to home position (gripper open) ───────────────────
    leader_goal = list(home_leader)
    leader_goal[-1] = GRIPPER_OPEN_POS
    teleop.driver.set_all_modes(trossen_arm.Mode.position)
    teleop.driver.set_all_positions(
        trossen_arm.VectorDouble(leader_goal),
        goal_time=3.0,
        blocking=True,
    )
    # Restore gravity-compensation (external_effort) mode for the next teleop
    teleop.driver.set_all_modes(trossen_arm.Mode.external_effort)
    teleop.driver.set_all_external_efforts(
        trossen_arm.VectorDouble([0.0] * n_joints),
        goal_time=0.0,
        blocking=True,
    )


# ── Custom record function ─────────────────────────────────────────────────────

def record_with_positioning():
    init_logging()
    if DISPLAY_DATA:
        init_rerun(session_name="recording")

    robot = make_robot_from_config(robot_cfg)
    teleop = make_teleoperator_from_config(teleop_cfg)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    sanity_check_dataset_name(f"{HF_USER}/{DATASET_NAME}", policy_cfg=None)
    dataset = LeRobotDataset.create(
        f"{HF_USER}/{DATASET_NAME}",
        FPS,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_threads=4 * len(robot_cfg.cameras),
    )

    listener = None
    home_follower: list[float] | None = None
    home_leader: list[float] | None = None

    try:
        robot.connect()
        teleop.connect()
        listener, events = init_keyboard_listener()

        with VideoEncodingManager(dataset):
            # ── One-time positioning phase before the first episode ────────────
            if ENABLE_POSITIONING:
                log_say(
                    "Teleoperate to starting position, then press Right Arrow to begin recording.",
                    play_sounds=True,
                )
                print(
                    "\n[POSITIONING]  Move the robot to the desired starting position.\n"
                    "               Press  →  (Right Arrow) to START recording.\n"
                    "               Press  Esc              to STOP all recording.\n"
                )
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=None,  # no data saved during positioning
                    control_time_s=MAX_POSITIONING_TIME_S,
                    single_task=TASK_DESCRIPTION,
                    display_data=DISPLAY_DATA,
                )
                # Capture the home position once the user is satisfied
                home_follower = list(robot.driver.get_all_positions())
                home_leader = list(teleop.driver.get_all_positions())

                # Reset so the first recording loop isn't immediately aborted.
                events["exit_early"] = False

            recorded_episodes = 0
            while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:

                # ── Recording phase ────────────────────────────────────────────
                log_say(f"Recording episode {dataset.num_episodes}", play_sounds=True)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=EPISODE_TIME_S,
                    single_task=TASK_DESCRIPTION,
                    display_data=DISPLAY_DATA,
                )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", play_sounds=True)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1

                # ── Return to home + open gripper (skip after last episode) ────
                if (
                    not events["stop_recording"]
                    and recorded_episodes < NUM_EPISODES
                    and home_follower is not None
                    and home_leader is not None
                ):
                    log_say("Returning to home position", play_sounds=True)
                    _return_to_home(robot, teleop, home_follower, home_leader)

                # ── Reset phase ────────────────────────────────────────────────
                if not events["stop_recording"] and recorded_episodes < NUM_EPISODES:
                    log_say("Reset the environment", play_sounds=True)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=FPS,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=RESET_TIME_S,
                        single_task=TASK_DESCRIPTION,
                        display_data=DISPLAY_DATA,
                    )

    finally:
        log_say("Stop recording", play_sounds=True, blocking=True)

        if dataset:
            dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()
        if not is_headless() and listener:
            listener.stop()

        if PUSH_TO_HUB:
            dataset.push_to_hub(private=PRIVATE)

        log_say("Exiting", play_sounds=True)

    return dataset


if __name__ == "__main__":
    record_with_positioning()
