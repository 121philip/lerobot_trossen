"""
机器人相关工具：
  - RobotWrapper    线程安全的机器人封装（互斥锁保护并发访问）
  - 观测格式转换     机器人原始观测 → 策略模型输入格式
  - 动作格式转换     策略模型输出张量 → 机器人关节指令字典
"""

import logging
from threading import Lock

import numpy as np

from important_code.utils import JOINT_NAMES


class RobotWrapper:
    """
    用互斥锁（Lock）封装机器人对象，使推理线程和执行线程可以安全地并发访问。

    为什么需要锁？
    推理线程调用 get_observation() 读取传感器数据，
    执行线程调用 send_action() 发送关节指令，
    两者操作同一个机器人对象，必须序列化访问以防数据竞争。
    """

    def __init__(self, robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self):
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action_dict):
        with self.lock:
            self.robot.send_action(action_dict)


def robot_obs_to_policy_obs(raw_obs):
    """
    把机器人返回的原始观测字典转换为策略模型期望的格式。

    机器人返回:
        {"joint_0.pos": float, ..., "wrist": ndarray(H,W,3), "right": ndarray(H,W,3)}

    策略期望:
        {"observation.state": ndarray(7,),
         "observation.images.wrist": ndarray(H,W,3),
         "observation.images.right": ndarray(H,W,3)}
    """
    obs = {}

    # 拼接各关节位置 → observation.state，shape: [7,]
    joint_positions = []
    for name in JOINT_NAMES:
        key = f"{name}.pos"
        if key in raw_obs:
            joint_positions.append(raw_obs[key])
        else:
            # 关节数据缺失时用 0 填充（通常不应发生）
            logging.warning(f"Missing joint {key} in observation, using 0.0")
            joint_positions.append(0.0)
    obs["observation.state"] = np.array(joint_positions, dtype=np.float32)

    # 摄像头图像直接映射（key 与策略期望一致）
    if "wrist" in raw_obs:
        obs["observation.images.wrist"] = raw_obs["wrist"]
    if "right" in raw_obs:
        obs["observation.images.right"] = raw_obs["right"]

    return obs


def create_mock_observation():
    """干跑（dry-run）模式：生成全零假观测，无需真实硬件即可测试推理流程。"""
    obs = {f"{name}.pos": 0.0 for name in JOINT_NAMES}
    obs["wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
    obs["right"] = np.zeros((480, 640, 3), dtype=np.uint8)
    return obs


def policy_action_to_robot_action(action_tensor):
    """
    把策略输出的动作张量 shape=(action_dim,) 转换为机器人接受的关节指令字典。
    例: tensor([0.12, -0.05, ...]) → {"joint_0.pos": 0.12, "joint_1.pos": -0.05, ...}
    """
    action_np = action_tensor.cpu().numpy()
    return {
        f"{name}.pos": float(action_np[i])
        for i, name in enumerate(JOINT_NAMES)
        if i < len(action_np)
    }
