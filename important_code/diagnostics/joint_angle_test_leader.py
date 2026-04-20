"""
Leader 交互式关节角度测试工具

连接 leader 机械臂后，循环接收用户指令，移动到指定关节角度，
供用户目视确认各关节方向、零位和运动范围。

Leader 有两种模式：
  position  模式 —— 臂锁定在目标位置（测试关节角度时使用）
  gc        模式 —— 重力补偿，可自由拖动（手动摆位后读数时使用）

用法:
    python important_code/diagnostics/joint_angle_test_leader.py
    python important_code/diagnostics/joint_angle_test_leader.py --ip 192.168.1.2

命令格式（进入交互模式后）:
    <关节序号> <角度(度)>      移动单个关节，切换到 position 模式
    all <j0> <j1> ... <j6>   同时设置所有 7 个关节（度）
    home                     回到 staged 安全位置（position 模式）
    zero                     所有关节归零（使用前请确认安全）
    gc                       切换到重力补偿模式（可自由拖动）
    pos                      切换到位置锁定模式
    show                     显示当前所有关节位置
    q / quit                 安全退出
"""

import argparse
import math

import trossen_arm
from lerobot.utils.import_utils import register_third_party_plugins as register_third_party_devices
from lerobot.utils.utils import init_logging

from lerobot_teleoperator_trossen.config_widowxai_leader import WidowXAILeaderTeleopConfig
from lerobot_teleoperator_trossen.widowxai_leader import WidowXAILeaderTeleop

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
            print(f"  [{i}] {label}  {pos:+8.4f} m")
    print()


def set_position_mode(driver) -> None:
    driver.set_all_modes(trossen_arm.Mode.position)


def set_gc_mode(driver) -> None:
    """重力补偿模式：切换到 external_effort 并将所有力矩设为 0。"""
    driver.set_all_modes(trossen_arm.Mode.external_effort)
    driver.set_all_external_efforts(
        [0.0] * NUM_JOINTS,
        goal_time=0.0,
        blocking=True,
    )


def move_all(driver, positions: list[float], goal_time: float = 3.0) -> None:
    set_position_mode(driver)
    driver.set_all_positions(
        trossen_arm.VectorDouble(positions),
        goal_time=goal_time,
        blocking=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Leader 交互式关节角度测试工具")
    parser.add_argument("--ip", default="192.168.1.2", help="leader IP 地址")
    args = parser.parse_args()

    init_logging()

    cfg = WidowXAILeaderTeleopConfig(ip_address=args.ip, id="leader")
    leader = WidowXAILeaderTeleop(cfg)

    print(f"\n正在连接到 leader ({args.ip})...")
    leader.connect()
    # connect() 结束后已处于重力补偿模式（external_effort）
    print("连接成功。机械臂已移动到 staged 位置，当前处于重力补偿模式（可自由拖动）。")

    print("\n──────────────────────────────────────────────────")
    print(" 命令参考")
    print("──────────────────────────────────────────────────")
    print("  <序号> <角度°>            移动单个关节（切换到 position 模式）")
    print("  all <j0> <j1>...<j6>     设置所有关节（度）")
    print("  home                     回到 staged 位置")
    print("  zero                     所有关节归零")
    print("  gc                       切换到重力补偿模式（可拖动）")
    print("  pos                      切换到位置锁定模式（保持当前位置）")
    print("  show                     显示当前位置")
    print("  q / quit                 安全退出")
    print("──────────────────────────────────────────────────")

    current = list(leader.driver.get_all_positions())
    print_positions(current)

    in_gc_mode = True  # connect() 结束时处于 gc 模式

    try:
        while True:
            mode_label = "[gc]" if in_gc_mode else "[pos]"
            try:
                line = input(f"{mode_label} >>> ").strip()
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
                current = list(leader.driver.get_all_positions())
                print_positions(current)

            elif cmd == "gc":
                set_gc_mode(leader.driver)
                in_gc_mode = True
                print("  已切换到重力补偿模式，可自由拖动。")

            elif cmd == "pos":
                current = list(leader.driver.get_all_positions())
                set_position_mode(leader.driver)
                # 锁定在当前位置
                leader.driver.set_all_positions(
                    trossen_arm.VectorDouble(current),
                    goal_time=0.5,
                    blocking=True,
                )
                in_gc_mode = False
                print("  已切换到位置锁定模式。")

            elif cmd == "home":
                move_all(leader.driver, list(leader.config.staged_positions), goal_time=2.0)
                in_gc_mode = False
                current = list(leader.driver.get_all_positions())
                print("  已回到 staged 位置（position 模式）。")
                print_positions(current)

            elif cmd == "zero":
                confirm = input("  所有关节将归零，请确认安全后输入 yes: ").strip().lower()
                if confirm == "yes":
                    move_all(leader.driver, [0.0] * NUM_JOINTS, goal_time=3.0)
                    in_gc_mode = False
                    current = list(leader.driver.get_all_positions())
                    print("  已归零。")
                    print_positions(current)
                else:
                    print("  已取消。")

            elif cmd == "all":
                if len(tokens) != NUM_JOINTS + 1:
                    print(f"  需要 {NUM_JOINTS} 个角度值，收到 {len(tokens) - 1} 个。")
                    continue
                try:
                    angles_deg = [float(t) for t in tokens[1:]]
                    angles_rad = [math.radians(d) for d in angles_deg[:-1]]
                    angles_rad.append(angles_deg[-1])  # 夹爪单位 m
                    move_all(leader.driver, angles_rad, goal_time=3.0)
                    in_gc_mode = False
                    current = list(leader.driver.get_all_positions())
                    print("  移动完成。")
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

                current = list(leader.driver.get_all_positions())
                if idx < NUM_JOINTS - 1:
                    current[idx] = math.radians(value)
                    label = f"{value:+.1f}°"
                else:
                    current[idx] = value
                    label = f"{value:+.4f} m"

                move_all(leader.driver, current, goal_time=3.0)
                in_gc_mode = False
                current = list(leader.driver.get_all_positions())
                print(f"  关节 {idx} 已移动到 {label}（position 模式）。")
                print_positions(current)

    finally:
        print("安全断开中...")
        leader.disconnect()
        print("完成。")


if __name__ == "__main__":
    main()
