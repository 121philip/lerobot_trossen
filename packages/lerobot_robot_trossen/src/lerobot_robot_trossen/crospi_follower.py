import logging
import time
from functools import cached_property
from typing import Any

import trossen_arm
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots.robot import Robot

from lerobot_robot_trossen.config_crospi_follower import CroSPIFollowerConfig

logger = logging.getLogger(__name__)


class CroSPIFollower(Robot):
    """
    WidowX AI follower arm in CroSPI shared-control mode.

    Observations are read directly from the arm via the Trossen SDK.
    All joint commands are handled by CroSPI/eTaSL; send_action() is a no-op.
    """

    config_class = CroSPIFollowerConfig
    name = "crospi_follower_robot"

    def __init__(self, config: CroSPIFollowerConfig):
        super().__init__(config)
        self.config = config

        self.driver = trossen_arm.TrossenArmDriver()
        self.cameras = make_cameras_from_configs(config.cameras)
        self._arm_connected = False  # True only when driver.configure() has been called

    @property
    def _joint_ft(self) -> dict[str, type]:
        return {f"{joint_name}.pos": float for joint_name in self.config.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._joint_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._joint_ft

    @property
    def is_connected(self) -> bool:
        return all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True, connect_arm: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if connect_arm:
            # Establish SDK connection for observation reading only.
            # CroSPI/eTaSL is already controlling the arm; we must NOT move it here.
            self.driver.configure(
                model=trossen_arm.Model.wxai_v0,
                end_effector=trossen_arm.StandardEndEffector.wxai_v0_follower,
                serv_ip=self.config.ip_address,
                clear_error=True,
            )
            self._arm_connected = True
            self.configure()
        else:
            logger.info(f"{self} skipping arm SDK connection (bridge provides joint states).")

        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected (arm={'SDK' if connect_arm else 'bridge'}).")

    @property
    def is_calibrated(self) -> bool:
        # Trossen Arm robots do not require calibration
        return True

    def calibrate(self) -> None:
        # Trossen Arm robots do not require calibration
        pass

    def configure(self) -> None:
        # CroSPI/eTaSL owns all joint commands. Read current state only.
        positions = list(self.driver.get_all_positions())
        logger.info(f"{self} initial joint positions: {[f'{p:.3f}' for p in positions]}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        if self._arm_connected:
            # Read joint state directly from arm SDK.
            start = time.perf_counter()
            robot_all_joint_outputs = self.driver.get_robot_output().joint.all
            obs_dict.update(
                {
                    f"{joint_name}.pos": pos
                    for joint_name, pos in zip(
                        self.config.joint_names,
                        robot_all_joint_outputs.positions,
                        strict=True,
                    )
                }
            )
            obs_dict.update(
                {
                    f"{joint_name}.vel": vel
                    for joint_name, vel in zip(
                        self.config.joint_names,
                        robot_all_joint_outputs.velocities,
                        strict=True,
                    )
                }
            )
            obs_dict.update(
                {
                    f"{joint_name}.eff": eff
                    for joint_name, eff in zip(
                        self.config.joint_names,
                        robot_all_joint_outputs.efforts,
                        strict=True,
                    )
                }
            )
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        else:
            # Arm not SDK-connected; inference_thread will overwrite these with bridge values.
            obs_dict.update({f"{n}.pos": 0.0 for n in self.config.joint_names})
            obs_dict.update({f"{n}.vel": 0.0 for n in self.config.joint_names})
            obs_dict.update({f"{n}.eff": 0.0 for n in self.config.joint_names})

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """No-op: CroSPI/eTaSL executes all joint commands.

        VLA predictions reach eTaSL via rviz_publisher → vla_ros_bridge_node →
        /joint_states_VLA. This function returns the action unchanged so that
        actor_thread can still forward it to rviz_publisher.put_actual().
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._arm_connected:
            # Do NOT move the arm — CroSPI/eTaSL owns the joint commands.
            self.driver.cleanup()
            self._arm_connected = False

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
