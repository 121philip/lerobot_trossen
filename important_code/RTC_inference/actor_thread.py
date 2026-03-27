"""
执行线程：以恒定 FPS 从动作队列取动作并发送给机器人。

时序保证：
  - 每帧固定 1/fps 秒，用 time.sleep() 补齐剩余时间，保持节拍稳定
  - 动作为 None（队列暂时为空）时跳过本帧，不发送任何指令
  - -0.001s 补偿量用于抵消系统调度抖动，避免累积漂移
"""

import logging
import time
import traceback
from threading import Event

from lerobot.policies.rtc.action_queue import ActionQueue

from important_code.RTC_inference.robot_wrapper import policy_action_to_robot_action

logger = logging.getLogger(__name__)


def actor_thread_fn(
    robot_wrapper,          # RobotWrapper 实例，None 表示干跑模式
    action_queue: ActionQueue,
    shutdown_event: Event,
    args,
):
    """
    执行线程：以恒定 FPS（默认 30Hz）从动作队列取动作并发送给机器人。

    参数:
        robot_wrapper:  线程安全的机器人封装，干跑时为 None
        action_queue:   共享动作队列（推理线程写入，本线程读取）
        shutdown_event: 全局关闭信号
        args:           命令行参数（fps 等）
    """
    try:
        logger.info("[ACTOR] Starting actor thread")

        action_count = 0
        action_interval = 1.0 / args.fps   # 目标帧间隔，例如 1/30 ≈ 0.0333s

        while not shutdown_event.is_set():
            start_time = time.perf_counter()

            # 从队列取下一步动作（队列为空时返回 None，不阻塞）
            action = action_queue.get()

            if action is not None:
                action_dict = policy_action_to_robot_action(action)

                if robot_wrapper is not None:
                    robot_wrapper.send_action(action_dict)   # 发送给真实机器人
                else:
                    # 干跑模式：每 10 步打印一次，避免日志刷屏
                    if action_count % 10 == 0:
                        logger.info(f"[DRY-RUN] Step {action_count}: {action_dict}")

                action_count += 1

            # 精确限速：补足到目标帧间隔（-1ms 补偿系统调度延迟）
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0.0, action_interval - elapsed - 0.001)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info(f"[ACTOR] Thread shutting down. Total actions: {action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Fatal error: {e}\n{traceback.format_exc()}")
        shutdown_event.set()   # 出错时广播关闭信号
