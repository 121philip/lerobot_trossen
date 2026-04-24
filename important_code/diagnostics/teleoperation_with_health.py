"""
遥操作 + 电机健康监控

在 leader→follower 遥操作的同时，后台线程定期读取 follower 各关节的
温度和力矩，发现异常时在终端打印警告。

用法:
    python important_code/diagnostics/teleoperation_with_health.py
    python important_code/diagnostics/teleoperation_with_health.py --health-interval 3
"""

import argparse
import math
import time
from threading import Event, Thread

import rerun as rr
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_teleoperate import teleop_loop
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.import_utils import register_third_party_plugins as register_third_party_devices
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun

from lerobot_robot_trossen import WidowXAIFollowerConfig
from lerobot_teleoperator_trossen import WidowXAILeaderTeleopConfig

register_third_party_devices()

# ── 健康监控阈值 ──────────────────────────────────────────────────────────────
TEMP_WARN = 55.0
TEMP_CRIT = 70.0
EFFORT_WARN_PCT = 40.0
EFFORT_CRIT_PCT = 70.0

JOINT_NAMES = ["j0-底座", "j1-肩部", "j2-肘部", "j3-前臂", "j4-腕俯", "j5-腕滚", "gripper"]


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def _health_thread(driver, stop_event: Event, interval: float) -> None:
    """后台线程：每隔 interval 秒读取一次电机状态并打印摘要。"""
    try:
        joint_limits = driver.get_joint_limits()
        effort_limits = [lim.effort_max for lim in joint_limits]
    except Exception:
        return

    while not stop_event.wait(timeout=interval):
        try:
            out = driver.get_robot_output()
            joints = out.joint.all
            drv_temps  = list(joints.driver_temperatures)
            rot_temps  = list(joints.rotor_temperatures)
            efforts    = list(joints.efforts)
        except Exception:
            continue

        warnings = []
        for i, name in enumerate(JOINT_NAMES):
            dt  = drv_temps[i]
            rt  = rot_temps[i]
            eff = efforts[i]
            lim = effort_limits[i] if i < len(effort_limits) else 1.0
            pct = abs(eff) / lim * 100 if lim > 0 else 0.0

            issues = []
            if dt >= TEMP_CRIT:
                issues.append(_c(f"驱动温度 {dt:.0f}°C!!!", "91"))
            elif dt >= TEMP_WARN:
                issues.append(_c(f"驱动温度 {dt:.0f}°C!", "93"))

            if rt >= TEMP_CRIT:
                issues.append(_c(f"转子温度 {rt:.0f}°C!!!", "91"))
            elif rt >= TEMP_WARN:
                issues.append(_c(f"转子温度 {rt:.0f}°C!", "93"))

            if pct >= EFFORT_CRIT_PCT:
                issues.append(_c(f"力矩 {pct:.0f}%!!!", "91"))
            elif pct >= EFFORT_WARN_PCT:
                issues.append(_c(f"力矩 {pct:.0f}%!", "93"))

            if issues:
                warnings.append(f"  {name}: {', '.join(issues)}")

        now = time.strftime("%H:%M:%S")
        if warnings:
            print(
                f"\n\033[93m[HEALTH {now}] ⚠ 发现异常:\033[0m",
                flush=True,
            )
            for w in warnings:
                print(w, flush=True)
        else:
            # 正常时只打印一行简洁摘要（最高温度 + 最高力矩占比）
            max_temp = max(max(drv_temps), max(rot_temps))
            max_pct  = max(
                (abs(efforts[i]) / effort_limits[i] * 100)
                for i in range(len(effort_limits))
                if effort_limits[i] > 0
            ) if effort_limits else 0.0
            print(
                f"[HEALTH {now}] "
                + _c("✓ 正常", "92")
                + f"  最高温度 {max_temp:.1f}°C  最高力矩占比 {max_pct:.1f}%",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="遥操作 + 电机健康监控")
    parser.add_argument("--health-interval", type=float, default=5.0,
                        help="健康数据打印间隔（秒，默认 5）")
    parser.add_argument("--no-display",      action="store_true",
                        help="关闭 Rerun 可视化")
    parser.add_argument("--fps",             type=int, default=60,
                        help="遥操作控制频率（默认 60 Hz）")
    args = parser.parse_args()

    display_data = not args.no_display

    init_logging()
    if display_data:
        init_rerun(session_name="teleoperation_health")

    robot_cfg = WidowXAIFollowerConfig(
        ip_address="192.168.2.3",
        id="follower",
        cameras={
            "right": OpenCVCameraConfig(index_or_path=2,  width=640, height=480, fps=30),
            "wrist": OpenCVCameraConfig(index_or_path=10, width=640, height=480, fps=30),
        },
    )
    teleop_cfg = WidowXAILeaderTeleopConfig(
        ip_address="192.168.1.2",
        id="leader",
    )

    teleop = make_teleoperator_from_config(teleop_cfg)
    robot  = make_robot_from_config(robot_cfg)
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    teleop.connect()
    robot.connect()

    # 启动健康监控后台线程
    stop_health = Event()
    health_thread = Thread(
        target=_health_thread,
        args=(robot.driver, stop_health, args.health_interval),
        daemon=True,
        name="HealthMonitor",
    )
    health_thread.start()
    print(
        f"\n\033[1m[HEALTH] 健康监控已启动（间隔 {args.health_interval}s）。"
        f"Ctrl+C 停止遥操作。\033[0m\n",
        flush=True,
    )

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=args.fps,
            display_data=display_data,
            duration=None,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
    except KeyboardInterrupt:
        pass
    finally:
        stop_health.set()
        health_thread.join(timeout=2.0)
        if display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
