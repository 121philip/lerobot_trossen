#!/usr/bin/env python3
"""
ROS2 可视化节点 —— 必须用系统 Python 3.10 运行（含 rclpy）。

由 launch_viz.sh 自动启动，也可手动运行：
  /usr/bin/python3 important_code/Inference/rviz_node.py

监听 UDP 127.0.0.1:9788，接收来自主推理进程的关节数据，并发布：
  /actual/joint_states    — 蓝色机器人模型用
  /predicted_ee_marker    — 末端执行器预测轨迹线（红色线段 + 绿色当前点）

配合 launch_viz.sh 使用：
  终端1: bash important_code/rviz_config/launch_viz.sh
  终端2: python important_code/Inference/run_inference_rtc.py --rviz [其他参数]
"""

import pickle
import socket
import sys

sys.path.insert(0, "/opt/ros/humble/local/lib/python3.10/dist-packages")
sys.path.insert(0, "/opt/ros/humble/lib/python3.10/site-packages")

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, TransformStamped
from builtin_interfaces.msg import Duration
from tf2_ros import StaticTransformBroadcaster

_UDP_HOST = "127.0.0.1"
_UDP_PORT = 9788
_MSG_ACTUAL    = 0
_MSG_PREDICTED = 1
_JOINT_NAMES = [
    "joint_0", "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "left_carriage_joint",
]

# ── 正向运动学（从 URDF 提取的关节参数）────────────────────────────────────
# 关节链: base_link → joint_0..5 → link_6 → ee_gripper(fixed) → ee_gripper_link
_FK_JOINTS = [
    {"xyz": [0.0,      0.0,     0.05725],  "axis": [ 0,  0,  1]},  # joint_0
    {"xyz": [0.02,     0.0,     0.04625],  "axis": [ 0,  1,  0]},  # joint_1
    {"xyz": [-0.264,   0.0,     0.0    ],  "axis": [ 0, -1,  0]},  # joint_2
    {"xyz": [0.245,    0.0,     0.06   ],  "axis": [ 0, -1,  0]},  # joint_3
    {"xyz": [0.06775,  0.0,     0.0455 ],  "axis": [ 0,  0, -1]},  # joint_4
    {"xyz": [0.02895,  0.0,    -0.0455 ],  "axis": [ 1,  0,  0]},  # joint_5
]
_EE_OFFSET = np.array([0.156062, 0.0, 0.0])  # ee_gripper fixed joint xyz


def _rot3(axis, angle: float) -> np.ndarray:
    """3×3 旋转矩阵（Rodrigues 公式），axis 可以不是单位向量。"""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,    t*x*y - z*s,  t*x*z + y*s],
        [t*x*y + z*s,  t*y*y + c,    t*y*z - x*s],
        [t*x*z - y*s,  t*y*z + x*s,  t*z*z + c  ],
    ])


def _fk_ee(q7: np.ndarray) -> np.ndarray:
    """
    正向运动学：7个关节角（第7个为夹爪，不影响末端位置）
    → 末端执行器在 base_link 坐标系下的 [x, y, z]。
    """
    T = np.eye(4)
    for j, qi in zip(_FK_JOINTS, q7[:6]):
        Tji = np.eye(4)
        Tji[:3, :3] = _rot3(j["axis"], float(qi))
        Tji[:3, 3] = j["xyz"]
        T = T @ Tji
    # 固定末端偏移
    ee = T[:3, :3] @ _EE_OFFSET + T[:3, 3]
    return ee


# ─────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = Node("trajectory_visualizer")

    actual_pub  = node.create_publisher(JointState, "/actual/joint_states",  10)
    marker_pub  = node.create_publisher(Marker,     "/predicted_ee_marker",  10)

    # 静态变换：将 predicted/base_link 挂载到 actual/base_link
    # （保留兼容性，即使不再渲染预测机器人模型）
    static_br = StaticTransformBroadcaster(node)
    tf_static = TransformStamped()
    tf_static.header.stamp = node.get_clock().now().to_msg()
    tf_static.header.frame_id = "actual/base_link"
    tf_static.child_frame_id  = "predicted/base_link"
    tf_static.transform.rotation.w = 1.0
    static_br.sendTransform(tf_static)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((_UDP_HOST, _UDP_PORT))
    sock.setblocking(False)

    node.get_logger().info(
        f"[rviz_node] Listening on UDP {_UDP_HOST}:{_UDP_PORT}"
    )

    latest_chunk  = None   # shape [N, 7]，最新预测动作块
    latest_actual = None   # shape [7]，最新实际关节角

    # Marker 有效期：200ms（30Hz 刷新时不会闪烁，停止发送后自动消失）
    _LIFETIME = Duration(sec=0, nanosec=200_000_000)

    def _make_point(xyz) -> Point:
        return Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))

    def timer_cb():
        nonlocal latest_chunk, latest_actual

        # 收取所有待处理 UDP 包
        while True:
            try:
                data, _ = sock.recvfrom(65535)
            except BlockingIOError:
                break
            if len(data) < 2:
                continue
            msg_type = data[0]
            array = pickle.loads(data[1:])
            now = node.get_clock().now().to_msg()

            if msg_type == _MSG_ACTUAL:
                latest_actual = array
                msg = JointState()
                msg.header.stamp = now
                msg.name = _JOINT_NAMES
                msg.position = array.tolist()
                actual_pub.publish(msg)

            elif msg_type == _MSG_PREDICTED:
                latest_chunk = array

        now = node.get_clock().now().to_msg()

        # ── 绿色球：当前实际末端位置 ──────────────────────────────────
        if latest_actual is not None:
            ee_now = _fk_ee(latest_actual)
            sphere = Marker()
            sphere.header.frame_id = "actual/base_link"
            sphere.header.stamp    = now
            sphere.ns              = "ee_actual"
            sphere.id              = 0
            sphere.type            = Marker.SPHERE
            sphere.action          = Marker.ADD
            sphere.pose.position   = _make_point(ee_now)
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.02  # 20mm 球
            sphere.color.r = 0.0
            sphere.color.g = 1.0
            sphere.color.b = 0.2
            sphere.color.a = 1.0
            sphere.lifetime = _LIFETIME
            marker_pub.publish(sphere)

        # ── 橙红色线：预测末端轨迹（50步） ───────────────────────────
        if latest_chunk is not None:
            ee_points = [_fk_ee(q) for q in latest_chunk]

            line = Marker()
            line.header.frame_id = "actual/base_link"
            line.header.stamp    = now
            line.ns              = "ee_predicted"
            line.id              = 1
            line.type            = Marker.LINE_STRIP
            line.action          = Marker.ADD
            line.scale.x         = 0.004   # 4mm 线宽
            line.color.r         = 1.0
            line.color.g         = 0.35
            line.color.b         = 0.0
            line.color.a         = 1.0
            line.lifetime        = _LIFETIME
            line.points          = [_make_point(p) for p in ee_points]
            marker_pub.publish(line)

            # 小点标注每一步
            dots = Marker()
            dots.header = line.header
            dots.ns     = "ee_predicted_dots"
            dots.id     = 2
            dots.type   = Marker.SPHERE_LIST
            dots.action = Marker.ADD
            dots.scale.x = dots.scale.y = dots.scale.z = 0.006  # 6mm 点
            dots.color   = line.color
            dots.lifetime = _LIFETIME
            dots.points  = line.points
            marker_pub.publish(dots)

    node.create_timer(1.0 / 30.0, timer_cb)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
