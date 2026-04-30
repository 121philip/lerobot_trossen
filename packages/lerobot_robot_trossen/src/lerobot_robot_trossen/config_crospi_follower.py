from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("crospi_follower_robot")
@dataclass
class CroSPIFollowerConfig(RobotConfig):
    # IP address of the arm (read-only access; CroSPI/eTaSL handles all commands)
    ip_address: str = "192.168.2.3"

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Troubleshooting: If one of your IntelRealSense cameras freeze during
    # data recording due to bandwidth limit, you might need to plug the camera
    # on another USB hub or PCIe card.

    # Joint names for the WidowX AI follower arm
    joint_names: list[str] = field(
        default_factory=lambda: [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "left_carriage_joint",
        ]
    )
