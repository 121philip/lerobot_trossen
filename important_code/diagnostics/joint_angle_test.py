"""
交互式关节角度测试工具

连接 follower 机械臂后，循环接收用户指令，移动到指定关节角度，
供用户目视确认各关节方向、零位和运动范围。

用法:
    python important_code/diagnostics/joint_angle_test.py
    python important_code/diagnostics/joint_angle_test.py --ip 192.168.2.3

命令格式（进入交互模式后）:
    <关节序号> <角度(度)>      移动单个关节，其余保持不动
    all <j0> <j1> ... <j6>   同时设置所有 7 个关节（度）
    home                     回到 staged 安全位置
    zero                     所有关节归零（使用前请确认安全）
    show                     显示当前所有关节位置
    q / quit                 安全退出（自动执行回正流程）
"""

import argparse
import math
import sys

import trossen_arm
from lerobot.utils.import_utils import register_third_party_plugins as register_third_party_devices
from lerobot.utils.utils import init_logging

from lerobot_robot_trossen import WidowXAIFollowerConfig
from lerobot_robot_trossen.widowxai_follower import WidowXAIFollower

register_third_party_devices()

JOINT_LABELS = [
    "joint_0  底座旋转       ",
    "joint_1  肩部俯仰       ",
    "joint_2  肘部俯仰       ",
    "joint_3  前臂滚转       ",
    "joint_4  腕部俯仰       ",
    "joint_5  腕部滚转       ",
    "left_carriage  夹爪(m) ",
]
NUM_JOINTS = len(JOINT_LABELS)


def print_positions(positions: list[float]) -> None:
    print()
    for i, (label, pos) in enumerate(zip(JOINT_LABELS, positions)):
        if i < NUM_JOINTS - 1:
            print(f"  [{i}] {label}  {math.degrees(pos):+8.2f}°   ({pos:+.4f} rad)")
        else:
            # 夹爪是线性关节，单位是 m
            print(f"  [{i}] {label}  {pos:+8.4f} m")
    print()


def move_all(driver, positions: list[float], goal_time: float = 3.0) -> None:
    driver.set_all_positions(
        trossen_arm.VectorDouble(positions),
        goal_time=goal_time,
        blocking=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="交互式关节角度测试工具")
    parser.add_argument("--ip", default="192.168.2.3", help="follower IP 地址")
    args = parser.parse_args()

    init_logging()

    cfg = WidowXAIFollowerConfig(ip_address=args.ip, id="follower")
    robot = WidowXAIFollower(cfg)

    print(f"\n正在连接到机器人 ({args.ip})...")
    robot.connect()
    print("连接成功。机械臂已移动到 staged 初始位置。")

    print("\n──────────────────────────────────────────────")
    print(" 命令参考")
    print("──────────────────────────────────────────────")
    print("  <序号> <角度°>            移动单个关节")
    print("  all <j0> <j1>...<j6>     设置所有关节（度）")
    print("  home                     回到 staged 位置")
    print("  zero                     所有关节归零")
    print("  show                     显示当前位置")
    print("  q / quit                 安全退出")
    print("──────────────────────────────────────────────")

    current = list(robot.driver.get_all_positions())
    print_positions(current)

    try:
        while True:
            try:
                line = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line:
                continue

            tokens = line.split()
            cmd = tokens[0].lower()

            if cmd in ("q", "quit", "exit"):
                break

            elif cmd == "show":
                current = list(robot.driver.get_all_positions())
                print_positions(current)

            elif cmd == "home":
                move_all(robot.driver, list(robot.config.staged_positions), goal_time=2.0)
                current = list(robot.driver.get_all_positions())
                print("已回到 staged 位置。")
                print_positions(current)

            elif cmd == "zero":
                confirm = input("  所有关节将归零，请确认安全后输入 yes: ").strip().lower()
                if confirm == "yes":
                    move_all(robot.driver, [0.0] * NUM_JOINTS, goal_time=3.0)
                    current = list(robot.driver.get_all_positions())
                    print("已归零。")
                    print_positions(current)
                else:
                    print("已取消。")

            elif cmd == "all":
                if len(tokens) != NUM_JOINTS + 1:
                    print(f"  需要 {NUM_JOINTS} 个角度值，收到 {len(tokens) - 1} 个。")
                    continue
                try:
                    angles_deg = [float(t) for t in tokens[1:]]
                    # 最后一个关节（夹爪）单位是 m，不做 deg→rad 转换
                    angles_rad = [math.radians(d) for d in angles_deg[:-1]]
                    angles_rad.append(angles_deg[-1])
                    move_all(robot.driver, angles_rad, goal_time=3.0)
                    current = list(robot.driver.get_all_positions())
                    print("移动完成。")
                    print_positions(current)
                except ValueError:
                    print("  角度值无效，请输入数字。")

            else:
                # 单关节命令: <index> <degrees>
                try:
                    idx = int(cmd)
                except ValueError:
                    print(f"  无法识别命令 '{line}'，请重新输入。")
                    continue

                if idx < 0 or idx >= NUM_JOINTS:
                    print(f"  关节序号须在 0~{NUM_JOINTS - 1} 之间。")
                    continue
                if len(tokens) < 2:
                    print("  请同时提供目标角度（度）。")
                    continue

                try:
                    value = float(tokens[1])
                except ValueError:
                    print("  角度值无效，请输入数字。")
                    continue

                current = list(robot.driver.get_all_positions())
                if idx < NUM_JOINTS - 1:
                    current[idx] = math.radians(value)
                    label = f"{value:+.1f}°"
                else:
                    # 夹爪：直接用米
                    current[idx] = value
                    label = f"{value:+.4f} m"

                move_all(robot.driver, current, goal_time=3.0)
                current = list(robot.driver.get_all_positions())
                print(f"  关节 {idx} 已移动到 {label}。")
                print_positions(current)

    finally:
        print("安全断开中（先 staged、再 sleep）...")
        robot.disconnect()
        print("完成。")


if __name__ == "__main__":
    main()
