from lerobot.scripts.lerobot_teleoperate import teleop_loop
import rerun as rr
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.import_utils import register_third_party_plugins as register_third_party_devices
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun

from lerobot_robot_trossen import WidowXAIFollowerConfig
from lerobot_teleoperator_trossen import WidowXAILeaderTeleopConfig

# Must be called before instantiating any third-party configs/robots
register_third_party_devices()

robot_cfg = WidowXAIFollowerConfig(
    ip_address="192.168.2.3",
    id="follower",
    cameras={
        "wrist":      OpenCVCameraConfig(index_or_path=2,  width=640, height=480, fps=30),
        "side":       OpenCVCameraConfig(index_or_path=10, width=640, height=480, fps=30),
    },
)

teleop_cfg = WidowXAILeaderTeleopConfig(
    ip_address="192.168.1.2",
    id="leader",
)

display_data = True
fps = 60

# ── Setup ────────────────────────────────────────────────────────────────────
init_logging()
if display_data:
    init_rerun(session_name="teleoperation")

teleop = make_teleoperator_from_config(teleop_cfg)
robot = make_robot_from_config(robot_cfg)
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

teleop.connect()
robot.connect()

# ── Teleoperation loop ────────────────────────────────────────────────────────

try:
    teleop_loop(
        teleop=teleop,
        robot=robot,
        fps=fps,
        display_data=display_data,
        duration=None,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )
except KeyboardInterrupt:
    pass
finally:
    if display_data:
        rr.rerun_shutdown()
    teleop.disconnect()
    robot.disconnect()
