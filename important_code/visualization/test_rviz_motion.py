"""
RViz 运动测试脚本：向 rviz_node 发送正弦轨迹，验证可视化是否正常工作。

使用方式：
  1. 先启动: bash important_code/visualization/launch_viz.sh
  2. 再运行: python important_code/visualization/test_rviz_motion.py

RViz 中应看到：
  - 蓝色实体机器人          ：joint_0 正弦旋转（实际位置）
  - 绿色球                  ：当前实际末端执行器位置
  - 橙红色线段 + 小点       ：预测末端轨迹（未来50步，从当前 EE 向前延伸）
三者均可见说明可视化管线完全正常。
"""

import math
import pickle
import socket
import time

import numpy as np

_UDP_HOST = "127.0.0.1"
_UDP_PORT = 9788
# 固定的初始关节位置（基于训练数据的典型姿态）
# [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, gripper]
BASE_POS = [0.0, -0.3, 0.5, 0.0, 0.2, 0.0, 0.02]


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"[TEST] Sending sine trajectory to {_UDP_HOST}:{_UDP_PORT} ...")
    print("[TEST] Blue robot moves, green sphere=current EE, orange line=predicted 50-step EE trajectory.")
    print("[TEST] Press Ctrl+C to stop.")

    try:
        t = 0
        while True:
            # 实际位置
            actual = BASE_POS.copy()
            actual[0] = math.sin(t * 0.1) * 0.5

            # 预测位置：超前 ~8 步（模拟 policy 的前瞻预测）
            predicted = BASE_POS.copy()
            predicted[0] = math.sin((t + 8) * 0.1) * 0.5

            # 发送实际关节状态（msg_type=0x00）
            payload_actual = b"\x00" + pickle.dumps(np.array(actual, dtype=np.float32), protocol=4)
            sock.sendto(payload_actual, (_UDP_HOST, _UDP_PORT))

            # 发送预测块（msg_type=0x01），shape=[50,7]，模拟50步预测
            chunk = np.array([
                [math.sin((t + i) * 0.1) * 0.5,
                 BASE_POS[1], BASE_POS[2], BASE_POS[3], BASE_POS[4], BASE_POS[5], BASE_POS[6]]
                for i in range(50)
            ], dtype=np.float32)
            payload_pred = b"\x01" + pickle.dumps(chunk, protocol=4)
            sock.sendto(payload_pred, (_UDP_HOST, _UDP_PORT))

            if t % 30 == 0:
                print(f"[TEST] t={t:4d}  actual j0={actual[0]:.3f}  predicted j0={predicted[0]:.3f}")

            t += 1
            time.sleep(0.033)   # ~30 Hz

    except KeyboardInterrupt:
        print("\n[TEST] Stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
