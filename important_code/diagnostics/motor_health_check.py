"""
电机健康状态检测脚本

连接 follower 机械臂，读取各关节的温度、力矩、误差信息，
帮助判断碰撞后电机是否受损。

用法:
    python important_code/diagnostics/motor_health_check.py           # 单次快照
    python important_code/diagnostics/motor_health_check.py --watch   # 持续监控（1秒刷新）
    python important_code/diagnostics/motor_health_check.py --ip 192.168.2.3
    
检测内容：

指标	正常	警告（黄）	危险（红）
驱动温度	< 55°C	55–70°C	> 70°C
转子温度	< 55°C	55–70°C	> 70°C
力矩（占额定）	< 40%	40–70%	> 70%
错误信息	无	—	有内容

诊断建议：
连接后机械臂会先移到 staged 位置，然后保持静止，此时力矩偏高说明电机在用力维持姿态（可能内部受损）
用 --watch 监控 2–3 分钟，温度若持续攀升而非平稳，说明有异常发热
若出现错误信息字符串，建议联系 Trossen 厂商
"""

import argparse
import math
import time

import trossen_arm
from lerobot.utils.import_utils import register_third_party_plugins as register_third_party_devices
from lerobot.utils.utils import init_logging

from lerobot_robot_trossen import WidowXAIFollowerConfig
from lerobot_robot_trossen.widowxai_follower import WidowXAIFollower

register_third_party_devices()

JOINT_NAMES = [
    "joint_0  底座",
    "joint_1  肩部",
    "joint_2  肘部",
    "joint_3  前臂",
    "joint_4  腕俯",
    "joint_5  腕滚",
    "gripper  夹爪",
]

# 温度警告阈值（°C）
TEMP_WARN = 55.0
TEMP_CRIT = 70.0

# 力矩占额定值百分比警告阈值
EFFORT_WARN_PCT = 40.0
EFFORT_CRIT_PCT = 70.0


def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def red(t):    return color(t, "91")
def yellow(t): return color(t, "93")
def green(t):  return color(t, "92")
def bold(t):   return color(t, "1")


def temp_label(val: float) -> str:
    s = f"{val:5.1f}°C"
    if val >= TEMP_CRIT:  return red(s + " !!!")
    if val >= TEMP_WARN:  return yellow(s + " ! ")
    return green(s + "   ")


def effort_label(val: float, limit: float) -> str:
    pct = abs(val) / limit * 100 if limit > 0 else 0.0
    s = f"{val:+7.3f} Nm ({pct:4.1f}%)"
    if pct >= EFFORT_CRIT_PCT: return red(s + " !!!")
    if pct >= EFFORT_WARN_PCT: return yellow(s + " ! ")
    return s


def print_snapshot(driver, joint_limits) -> bool:
    """打印一次健康快照，返回是否发现异常。"""
    out = driver.get_robot_output()
    joints = out.joint.all

    positions    = list(joints.positions)
    velocities   = list(joints.velocities)
    efforts      = list(joints.efforts)
    drv_temps    = list(joints.driver_temperatures)
    rot_temps    = list(joints.rotor_temperatures)

    effort_limits = [lim.effort_max for lim in joint_limits]

    now = time.strftime("%H:%M:%S")
    print(f"\n{bold('── 电机健康快照')}  {now}  ─────────────────────────────────────────")
    print(f"  {'关节':<14} {'位置':>8}  {'速度':>9}  {'驱动温度':>13}  {'转子温度':>13}  {'力矩':>24}")
    print(f"  {'─'*14} {'─'*8}  {'─'*9}  {'─'*13}  {'─'*13}  {'─'*24}")

    has_warning = False
    for i, name in enumerate(JOINT_NAMES):
        pos = positions[i]
        vel = velocities[i]
        eff = efforts[i]
        dt  = drv_temps[i]
        rt  = rot_temps[i]
        lim = effort_limits[i] if i < len(effort_limits) else 1.0

        pos_str = f"{math.degrees(pos):+7.1f}°" if i < 6 else f"{pos:+7.4f}m"
        vel_str = f"{vel:+7.3f}"

        dt_str = temp_label(dt)
        rt_str = temp_label(rt)
        eff_str = effort_label(eff, lim)

        if dt >= TEMP_WARN or rt >= TEMP_WARN or (lim > 0 and abs(eff)/lim*100 >= EFFORT_WARN_PCT):
            has_warning = True

        print(f"  {name:<14} {pos_str}  {vel_str}  {dt_str}  {rt_str}  {eff_str}")

    # 误差信息
    err_info = driver.get_error_information()
    print(f"\n  {bold('错误信息:')} {red(err_info) if err_info.strip() else green('无')}")

    # 固件版本（仅首次）
    return has_warning


def print_firmware(driver) -> None:
    try:
        ctrl = driver.get_controller_version()
        drv  = driver.get_driver_version()
        print(f"  {bold('控制器版本:')} {ctrl}   {bold('驱动版本:')} {drv}")
    except Exception:
        pass


def print_limits(joint_limits) -> None:
    print(f"\n  {bold('关节额定限位（供参考）:')}")
    print(f"  {'关节':<14} {'位置范围':>24}  {'速度上限':>10}  {'力矩上限':>10}")
    for i, (name, lim) in enumerate(zip(JOINT_NAMES, joint_limits)):
        pos_max = f"{math.degrees(lim.position_max):+.1f}°" if i < 6 else f"{lim.position_max:.4f}m"
        pos_min = f"{math.degrees(lim.position_min):+.1f}°" if i < 6 else f"{lim.position_min:.4f}m"
        print(f"  {name:<14} {pos_min:>10} ~ {pos_max:<10}  {lim.velocity_max:>9.2f}  {lim.effort_max:>9.2f} Nm")


def main() -> None:
    parser = argparse.ArgumentParser(description="电机健康状态检测")
    parser.add_argument("--ip",    default="192.168.2.3", help="follower IP 地址")
    parser.add_argument("--watch", action="store_true",   help="持续监控模式（每秒刷新）")
    parser.add_argument("--interval", type=float, default=1.0, help="监控刷新间隔（秒，默认1.0）")
    args = parser.parse_args()

    init_logging()

    cfg = WidowXAIFollowerConfig(ip_address=args.ip, id="follower")
    robot = WidowXAIFollower(cfg)

    print(f"\n正在连接到机器人 ({args.ip})...")
    robot.connect()
    print("连接成功。机械臂已移动到 staged 初始位置。\n")

    driver = robot.driver
    print_firmware(driver)

    joint_limits = driver.get_joint_limits()
    print_limits(joint_limits)

    try:
        if args.watch:
            print(f"\n持续监控模式（间隔 {args.interval}s，Ctrl+C 退出）...")
            while True:
                has_warn = print_snapshot(driver, joint_limits)
                if has_warn:
                    print(f"\n  {yellow('⚠  发现异常，请检查上方标记的关节。')}")
                time.sleep(args.interval)
        else:
            has_warn = print_snapshot(driver, joint_limits)
            if has_warn:
                print(f"\n  {yellow('⚠  发现异常，建议用 --watch 持续监控，或联系厂商检修。')}")
            else:
                print(f"\n  {green('✓  未发现明显异常。')}")
    except KeyboardInterrupt:
        print("\n\n监控结束。")
    finally:
        print("安全断开中...")
        robot.disconnect()
        print("完成。")


if __name__ == "__main__":
    main()
