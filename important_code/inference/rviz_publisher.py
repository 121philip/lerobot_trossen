"""
RViz 可视化发布器（UDP 发送端）+ CroSPI 关节状态接收端。

架构：venv 中的 Python 3.12 无法直接 import rclpy（rclpy 编译目标为系统 Python 3.10）。
解决方案：
  - vla_ros_bridge_node.py 在已 source ROS2 环境的系统 Python 3.10 中运行，负责发布 ROS2 话题
  - 本模块（rviz_publisher.py）在 Python 3.12 主进程中运行：
      发送侧：通过 UDP :9788 向 bridge_node 发送 VLA 输出（ACTUAL、PREDICTED、ALPHA）
      接收侧：通过 UDP :9789 接收 bridge_node 转发的 /joint_states（从机械臂真实状态）

通信协议（双向 UDP）：
  发送 → bridge :9788
    类型 0 (ACTUAL)    = shape [7,]   实际执行的关节位置（来自 actor_thread）
    类型 1 (PREDICTED) = shape [N,7]  预测动作块（来自 inference_thread）
    类型 2 (ALPHA)     = shape [1,]   共享控制 alpha 值
  接收 ← bridge :9789
    无类型字节：直接 pickle 序列化的 shape [7,] 关节位置数组（弧度）

verbose=True 模式（--print-publish）：
  单独测试 lerobot_trossen 而不启动 CroSPI 时，在 stdout 打印每次发布的内容，
  方便验证推理输出是否正常。
"""

import pickle
import queue
import socket
import threading
from typing import Optional

import numpy as np

_UDP_HOST    = "127.0.0.1"
_UDP_PORT    = 9788
_UDP_PORT_JS = 9789   # 反向：bridge → lerobot 关节状态
_MSG_ACTUAL    = b"\x00"
_MSG_PREDICTED = b"\x01"
_MSG_ALPHA     = b"\x02"


class RVizPublisher:
    def __init__(self, verbose: bool = False):
        self._verbose = verbose
        self._send_queue = queue.Queue(maxsize=120)
        self._send_thread = threading.Thread(
            target=self._send_loop, daemon=True, name="RVizSender"
        )
        # 接收侧：bridge 转发的 /joint_states
        self._latest_joints: Optional[np.ndarray] = None
        self._joints_lock = threading.Lock()
        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="JointStateRecv"
        )

    def start(self):
        """启动 UDP 发送线程和关节状态接收线程。"""
        self._send_thread.start()
        self._recv_thread.start()

    def put_actual(self, joint_positions: np.ndarray):
        """actor_thread 每帧调用，传入 shape=[7,] 的关节位置（弧度/米）。"""
        if self._verbose:
            np.set_printoptions(precision=4, suppress=True)
            print(f"[PUBLISH] ACTUAL  joints={np.asarray(joint_positions).tolist()}", flush=True)
        self._enqueue(_MSG_ACTUAL, joint_positions)

    def put_predicted(self, chunk: np.ndarray):
        """inference_thread 每块调用，传入 shape=[N, 7] 的预测动作数组。"""
        if self._verbose:
            print(f"[PUBLISH] PREDICTED shape={np.asarray(chunk).shape}", flush=True)
        self._enqueue(_MSG_PREDICTED, chunk)

    def put_alpha(self, alpha: float):
        """Send the final shared-control alpha to the CroSPI bridge."""
        alpha_array = np.array([float(np.clip(alpha, 0.0, 1.0))], dtype=np.float64)
        if self._verbose:
            print(f"[PUBLISH] ALPHA   alpha={alpha_array[0]:.4f}", flush=True)
        self._enqueue(_MSG_ALPHA, alpha_array)

    def get_latest_joints(self) -> Optional[np.ndarray]:
        """返回 bridge 最近转发的关节状态 [7,]，若尚未收到则返回 None。"""
        with self._joints_lock:
            return self._latest_joints.copy() if self._latest_joints is not None else None

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

    def _recv_loop(self):
        """接收 bridge_node 通过 UDP :9789 转发的 /joint_states。"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((_UDP_HOST, _UDP_PORT_JS))
        try:
            while True:
                try:
                    data, _ = sock.recvfrom(4096)
                except OSError:
                    break
                try:
                    joints = pickle.loads(data)
                    joints = np.asarray(joints, dtype=np.float64).reshape(-1)
                    if joints.size == 7:
                        with self._joints_lock:
                            self._latest_joints = joints
                except Exception:
                    pass
        finally:
            sock.close()
