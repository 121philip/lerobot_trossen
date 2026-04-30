"""
SpaceMouse Compact 检测程序

用途：
  检测 SpaceMouse 是否连接，并实时打印 6-DOF 轴数据和按钮状态。
  可用于硬件调试或校验 pyspacemouse 安装是否正常。

依赖安装：
  pip install pyspacemouse

udev 权限（Linux，首次配置）：
  sudo tee /etc/udev/rules.d/50-3dconnexion.rules <<'EOF'
  SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0666"
  EOF
  sudo udevadm control --reload-rules && sudo udevadm trigger

用法：
  python detect_spacemouse.py
  python detect_spacemouse.py --scale 350 --deadzone 0.05
"""

import argparse
import sys
import time

# 3DConnexion SpaceMouse Compact USB ID
VENDOR_ID = 0x256F
PRODUCT_ID_COMPACT = 0xC635  # SpaceMouse Compact


def _check_usb_present() -> bool:
    """快速检查设备是否在 USB 总线上（不依赖 pyspacemouse）。"""
    try:
        import hid
        for dev in hid.enumerate(VENDOR_ID, 0):
            return True
        return False
    except ImportError:
        pass
    try:
        import subprocess
        out = subprocess.check_output(["lsusb"], text=True)
        return "256f" in out.lower() or "3dconnexion" in out.lower()
    except Exception:
        return False


def open_spacemouse(scale: float, deadzone: float):
    """
    打开 SpaceMouse，返回 (state_fn, close_fn)。
    state_fn() → (axes: list[6], buttons: list[2]) 或 None（无新事件）。
    """
    try:
        import pyspacemouse
    except ImportError:
        print("[错误] pyspacemouse 未安装，请运行：pip install pyspacemouse")
        sys.exit(1)

    success = pyspacemouse.open(
        dof_callback=None,
        button_callback=None,
        button_callback_arr=None,
    )
    if not success:
        return None, None

    def read_state():
        state = pyspacemouse.read()
        if state is None:
            return None
        axes = [
            state.x   * scale,
            state.y   * scale,
            state.z   * scale,
            state.roll  * scale,
            state.pitch * scale,
            state.yaw   * scale,
        ]
        buttons = list(state.buttons)
        return axes, buttons

    def close():
        pyspacemouse.close()

    return read_state, close


def _bar(value: float, width: int = 20) -> str:
    """将 [-1,1] 范围的值渲染成 ASCII 进度条。"""
    clamped = max(-1.0, min(1.0, value))
    mid = width // 2
    pos = int(mid + clamped * mid)
    bar = ["-"] * width
    bar[mid] = "|"
    if pos != mid:
        bar[min(pos, width - 1)] = "#"
    return "".join(bar)


def run(scale: float, deadzone: float, hz: float) -> None:
    print("=" * 60)
    print("  SpaceMouse Compact 检测工具")
    print(f"  VID=0x{VENDOR_ID:04X}  PID=0x{PRODUCT_ID_COMPACT:04X}")
    print("=" * 60)

    if _check_usb_present():
        print("[USB] 检测到 3DConnexion 设备")
    else:
        print("[USB] 未检测到设备（可能未插入或权限不足）")
        print("      Linux 请确认 udev 规则，见文件顶部注释")

    print("\n正在打开 pyspacemouse …")
    read_state, close = open_spacemouse(scale, deadzone)

    if read_state is None:
        print("[错误] 无法打开 SpaceMouse（设备未找到或驱动问题）")
        sys.exit(1)

    print("[OK] SpaceMouse 已连接，按 Ctrl-C 退出\n")

    labels = ["X  ", "Y  ", "Z  ", "Rx ", "Ry ", "Rz "]
    dt = 1.0 / hz

    try:
        while True:
            result = read_state()
            if result is not None:
                axes, buttons = result
                lines = []
                for label, val in zip(labels, axes):
                    lines.append(f"  {label} {_bar(val / scale)} {val:+7.3f}")
                btn_str = "  Btn: " + "  ".join(
                    f"[{'#' if b else ' '}] {i}" for i, b in enumerate(buttons)
                )
                # 清行并打印（同一屏幕区域）
                block = "\n".join(lines) + "\n" + btn_str
                print("\033[{}A".format(len(lines) + 1) + block, end="\n", flush=True)
            else:
                # 首次渲染时占位
                print("\n" * 7, end="", flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n\n[退出]")
    finally:
        close()


def main():
    parser = argparse.ArgumentParser(
        description="实时检测 SpaceMouse Compact 轴数据和按钮状态"
    )
    parser.add_argument(
        "--scale", type=float, default=350.0,
        help="原始值缩放因子（默认 350）"
    )
    parser.add_argument(
        "--deadzone", type=float, default=0.05,
        help="轴死区阈值（默认 0.05，暂未使用，备用）"
    )
    parser.add_argument(
        "--hz", type=float, default=50.0,
        help="轮询频率 Hz（默认 50）"
    )
    args = parser.parse_args()
    run(args.scale, args.deadzone, args.hz)


if __name__ == "__main__":
    main()
