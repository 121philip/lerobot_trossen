"""
RViz 可视化发布器（UDP 发送端）+ CroSPI 关节状态接收端。

架构：venv 中的 Python 3.12 无法直接 import rclpy（rclpy 编译目标为系统 Python 3.10）。
解决方案：
  - vla_ros_bridge_node.py 在已 source ROS2 环境的系统 Python 3.10 中运行，负责发布 ROS2 话题
  - 本模块（rviz_publisher.py）在 Python 3.12 主进程中运行：
      发送侧：通过 UDP :9788 向 bridge_node 发送 VLA 输出（ACTUAL、PREDICTED、WEIGHTS）
      接收侧：通过 UDP :9789 接收 bridge_node 转发的 /joint_states（从机械臂真实状态）

通信协议（双向 UDP）：
  发送 → bridge :9788
    类型 0 (ACTUAL)    = shape [7,]   实际执行的关节位置（来自 actor_thread）
    类型 1 (PREDICTED) = shape [N,7]  预测动作块（来自 inference_thread）
    类型 2 (WEIGHTS)   = shape [2,]   Sentinel 直接权重 [w_vla, w_human]
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
_MSG_WEIGHTS   = b"\x02"
_JOINT_COUNT = 7
_WEIGHT_COUNT = 2


def _normalize_actual_joints(joint_positions: np.ndarray) -> np.ndarray:
    """Return a flat 7-joint vector with the gripper as the final element."""
    joints = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
    if joints.size != _JOINT_COUNT:
        raise ValueError(
            f"Expected {_JOINT_COUNT} joints including gripper, got {joints.size}"
        )
    return joints


def _normalize_predicted_chunk(chunk: np.ndarray) -> np.ndarray:
    """Return an Nx7 predicted chunk with the gripper in column 6."""
    joints = np.asarray(chunk, dtype=np.float64)
    if joints.ndim == 1:
        return _normalize_actual_joints(joints).reshape(1, _JOINT_COUNT)
    if joints.ndim != 2 or joints.shape[1] != _JOINT_COUNT:
        raise ValueError(
            f"Expected predicted chunk shape (N, {_JOINT_COUNT}) including gripper, "
            f"got {joints.shape}"
        )
    return joints


def _normalize_weights(w_vla: float, w_human: float) -> np.ndarray:
    """Return non-negative [w_vla, w_human] for Sentinel eTaSL weighting."""
    weights = np.array([w_vla, w_human], dtype=np.float64).reshape(-1)
    if weights.size != _WEIGHT_COUNT:
        raise ValueError(f"Expected {_WEIGHT_COUNT} Sentinel weights, got {weights.size}")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Sentinel weights must be finite")
    return np.maximum(weights, 0.0)


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
        joint_positions = _normalize_actual_joints(joint_positions)
        if self._verbose:
            np.set_printoptions(precision=4, suppress=True)
            print(f"[PUBLISH] ACTUAL  joints={np.asarray(joint_positions).tolist()}", flush=True)
        self._enqueue(_MSG_ACTUAL, joint_positions)

    def put_predicted(self, chunk: np.ndarray):
        """inference_thread 每块调用，传入 shape=[N, 7] 的预测动作数组。"""
        chunk = _normalize_predicted_chunk(chunk)
        if self._verbose:
            print(f"[PUBLISH] PREDICTED shape={np.asarray(chunk).shape}", flush=True)
        self._enqueue(_MSG_PREDICTED, chunk)

    def put_weights(self, w_vla: float, w_human: float):
        """Send Sentinel direct eTaSL weights to the CroSPI bridge."""
        weights = _normalize_weights(w_vla, w_human)
        if self._verbose:
            print(
                f"[PUBLISH] WEIGHTS w_vla={weights[0]:.4f} w_human={weights[1]:.4f}",
                flush=True,
            )
        self._enqueue(_MSG_WEIGHTS, weights)

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
