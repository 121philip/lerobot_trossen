#!/usr/bin/env python3
"""
TF诊断脚本：启动后发送关节状态，检查TF是否正常发布。
运行方式（在已source ROS2的终端）：
  /usr/bin/python3 important_code/rviz_config/diag_tf.py
"""
import sys, time
sys.path.insert(0, "/opt/ros/humble/local/lib/python3.10/dist-packages")
sys.path.insert(0, "/opt/ros/humble/lib/python3.10/site-packages")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

JOINT_NAMES = ["joint_0","joint_1","joint_2","joint_3","joint_4","joint_5","left_carriage_joint"]

def main():
    rclpy.init()
    node = Node("tf_diag")
    buf = Buffer()
    tf_listener = TransformListener(buf, node)

    pub = node.create_publisher(JointState, "/actual/joint_states", 10)

    def send_js():
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = [0.0] * 7
        pub.publish(msg)

    print("[DIAG] Sending joint states to /actual/joint_states for 3 seconds...")
    for i in range(90):
        send_js()
        rclpy.spin_once(node, timeout_sec=0.033)
        time.sleep(0.033)
        if i % 30 == 0:
            # Check if TF frame exists
            frames = buf.all_frames_as_string()
            if "actual/base_link" in frames:
                print(f"[DIAG] t={i}: FOUND actual/base_link in TF!")
            elif "base_link" in frames:
                print(f"[DIAG] t={i}: Found base_link (no prefix!) in TF")
            else:
                print(f"[DIAG] t={i}: No base_link found in TF at all")
                print(f"       Frames: {frames[:200] if frames else '(empty)'}")

    print("\n[DIAG] All available TF frames:")
    print(buf.all_frames_as_string())

    rclpy.shutdown()

if __name__ == "__main__":
    main()
