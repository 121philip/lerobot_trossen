import logging
import time

import trossen_arm
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.control_utils import init_keyboard_listener, is_headless, sanity_check_dataset_name
from lerobot.utils.import_utils import register_third_party_plugins as register_third_party_devices
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.processor import make_default_processors

from lerobot_robot_trossen import WidowXAIFollowerConfig
from lerobot_teleoperator_trossen import WidowXAILeaderTeleopConfig

# Must be called before instantiating any third-party configs/robots
register_third_party_devices()

# ── Robot & Teleop Config ──────────────────────────────────────────────────────

robot_cfg = WidowXAIFollowerConfig(
    ip_address="192.168.2.3",
    id="follower",
    cameras={
        "wrist": OpenCVCameraConfig(index_or_path=2,  width=640, height=480, fps=30),
        "side":  OpenCVCameraConfig(index_or_path=10, width=640, height=480, fps=30),
    },
)

teleop_cfg = WidowXAILeaderTeleopConfig(
    ip_address="192.168.1.2",
    id="leader",
)

# ── Dataset Config ─────────────────────────────────────────────────────────────

HF_USER          = "kaixiyao"
DATASET_NAME     = "widowxai-test"
TASK_DESCRIPTION = "Grab the grape"
NUM_EPISODES     = 5
EPISODE_TIME_S   = 30
RESET_TIME_S     = 15
FPS              = 30
DISPLAY_DATA     = True
PUSH_TO_HUB      = True
PRIVATE          = False


def go_to_home(robot, teleop):
    """Move both arms to their staged (home) positions between episodes."""
    log_say("Going to home position", play_sounds=True)

    # Follower: already in position mode, just move
    robot.driver.set_all_positions(
        robot.config.staged_positions,
        goal_time=2.0,
        blocking=True,
    )

    # Leader: switch from external_effort → position → move → back to external_effort
    teleop.driver.set_all_modes(trossen_arm.Mode.position)
    teleop.driver.set_all_positions(
        teleop.config.staged_positions,
        goal_time=2.0,
        blocking=True,
    )
    teleop.driver.set_all_modes(trossen_arm.Mode.external_effort)
    teleop.driver.set_all_external_efforts(
        [0.0] * len(teleop.config.joint_names),
        goal_time=0.0,
        blocking=True,
    )


def record_with_home():
    init_logging()

    if DISPLAY_DATA:
        init_rerun(session_name="recording")

    robot  = make_robot_from_config(robot_cfg)
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

    repo_id = f"{HF_USER}/{DATASET_NAME}"
    sanity_check_dataset_name(repo_id, policy=None)

    dataset = LeRobotDataset.create(
        repo_id,
        FPS,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_threads=4 * len(robot_cfg.cameras),
    )

    robot.connect()
    teleop.connect()

    listener, events = init_keyboard_listener()

    try:
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
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
                    go_to_home(robot, teleop)
                    continue

                dataset.save_episode()
                recorded_episodes += 1

                # Skip reset after the final episode
                if events["stop_recording"] or recorded_episodes >= NUM_EPISODES:
                    break

                # Move both arms to home, then give time to reset the scene
                go_to_home(robot, teleop)

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

        dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if PUSH_TO_HUB:
            dataset.push_to_hub(private=PRIVATE)

        log_say("Exiting", play_sounds=True)

    return dataset


if __name__ == "__main__":
    record_with_home()
