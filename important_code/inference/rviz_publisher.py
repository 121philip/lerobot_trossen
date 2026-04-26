"""
RViz 可视化发布器（UDP 发送端）。

架构：venv 中的 Python 3.12 无法直接 import rclpy（rclpy 编译目标为系统 Python 3.10）。
解决方案：
  - rviz_node.py 由 launch_viz.sh 在已 source ROS2 环境的 bash 中启动，负责发布 ROS2 话题
  - 本模块（rviz_publisher.py）在 Python 3.12 主进程中运行，通过 UDP socket 向 rviz_node.py 发送数据

通信协议：
  每条消息 = 1字节类型标志 + pickle 序列化的 numpy 数组
  类型 0 (ACTUAL)    = shape [7,]   实际执行的关节位置（来自 actor_thread）
  类型 1 (PREDICTED) = shape [N,7] 预测动作块（来自 inference_thread）

发布的 ROS2 话题（由 rviz_node.py 发布）：
  /actual/joint_states    → 蓝色实体机器人（实际执行位置）
  /predicted_ee_marker    → 绿色球（当前 EE 位置）+ 橙红色线（50步预测轨迹）

使用方式（由 run_inference_rtc.py 通过 --rviz 标志自动创建）：
  1. 先启动：bash important_code/visualization/launch_viz.sh  （启动 rviz_node.py + RViz）
  2. 再启动：python important_code/inference/run_inference_rtc.py --rviz
"""

import pickle
import queue
import socket
import threading

import numpy as np

_UDP_HOST = "127.0.0.1"
_UDP_PORT = 9788
_MSG_ACTUAL    = b"\x00"
_MSG_PREDICTED = b"\x01"
_MSG_ALPHA     = b"\x02"


class RVizPublisher:
    def __init__(self):
        self._send_queue = queue.Queue(maxsize=120)
        self._send_thread = threading.Thread(
            target=self._send_loop, daemon=True, name="RVizSender"
        )

    def start(self):
        """启动 UDP 发送线程。rviz_node.py 应已由 launch_viz.sh 启动。"""
        self._send_thread.start()

    def put_actual(self, joint_positions: np.ndarray):
        """actor_thread 每帧调用，传入 shape=[7,] 的关节位置（弧度/米）。"""
        self._enqueue(_MSG_ACTUAL, joint_positions)

    def put_predicted(self, chunk: np.ndarray):
        """inference_thread 每块调用，传入 shape=[N, 7] 的预测动作数组。"""
        self._enqueue(_MSG_PREDICTED, chunk)

    def put_alpha(self, alpha: float):
        """Send the final shared-control alpha to the CroSPI bridge."""
        alpha_array = np.array([float(np.clip(alpha, 0.0, 1.0))], dtype=np.float64)
        self._enqueue(_MSG_ALPHA, alpha_array)

    def _enqueue(self, msg_type: bytes, array: np.ndarray):
        try:
            self._send_queue.put_nowait((msg_type, array))
        except queue.Full:
            # 丢掉最旧的一条，放入最新的，绝不阻塞调用方
            try:
                self._send_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._send_queue.put_nowait((msg_type, array))
            except queue.Full:
                pass

    def _send_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            while True:
                try:
                    msg_type, array = self._send_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                try:
                    payload = msg_type + pickle.dumps(array, protocol=4)
                    sock.sendto(payload, (_UDP_HOST, _UDP_PORT))
                except OSError:
                    pass
        finally:
            sock.close()
