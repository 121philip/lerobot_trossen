"""
推理线程：异步持续生成动作块。

每当动作队列动作不足时，调用策略模型生成下一个动作块，
并通过 action_queue.merge() 将新块与队列中的剩余动作融合（RTC 核心操作）。

RTC 每次迭代步骤：
  A. 快照推理前队列状态（供 merge 对齐时序）
  B. 用历史最大延迟估算 inference_delay（帧数偏移，补偿推理耗时）
  C. 读取当前观测（真实机器人 or mock）
  D. 预处理观测（归一化、tokenize 任务描述）
  E. 调用 predict_action_chunk() 生成动作块（在 no_grad 外，RTC 需要 autograd）
  F. 后处理（反归一化）+ 记录本次延迟
  G. 将新块融合进动作队列
"""

import logging
import math
import time
import traceback
from copy import copy
from threading import Event

import torch

from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.latency_tracker import LatencyTracker

from important_code.utils import DEVICE, JOINT_NAMES
from important_code.Inference.robot_wrapper import create_mock_observation, robot_obs_to_policy_obs

logger = logging.getLogger(__name__)


def inference_thread_fn(
    policy,
    robot_wrapper,          # RobotWrapper 实例，None 表示干跑模式
    preprocessor,
    postprocessor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    args,
):
    """
    后台推理线程：持续监控动作队列，当队列动作不足时调用模型生成新动作块。

    参数:
        policy:          已加载的 SmolVLAPolicy
        robot_wrapper:   线程安全的机器人封装，干跑时为 None
        preprocessor:    观测预处理器（归一化、tokenize）
        postprocessor:   动作后处理器（反归一化）
        action_queue:    共享动作队列（推理线程写入，执行线程读取）
        shutdown_event:  全局关闭信号
        args:            命令行参数（fps、task、queue_threshold 等）
    """
    try:
        logger.info("[INFERENCE] Starting inference thread")

        latency_tracker = LatencyTracker()   # 维护历史推理延迟的滑动窗口
        time_per_step = 1.0 / args.fps       # 每帧时间（秒）

        # RTC 模式：队列低于阈值时提前触发推理，保持队列充足
        # 非 RTC 模式：队列完全耗尽才触发（传统块执行方式）
        queue_threshold = args.queue_threshold if args.rtc else 0

        while not shutdown_event.is_set():
            if action_queue.qsize() <= queue_threshold:
                current_time = time.perf_counter()

                # ── A. 快照推理前的队列状态（供 merge 对齐时序用）──
                action_index_before = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()  # 队列中尚未执行的剩余动作

                # ── B. 计算延迟补偿帧数 ──
                # 推理耗时期间机械臂仍在执行旧动作，inference_delay 告诉模型
                # "当这个新块被用到时，机械臂已经额外走了多少步"。
                inference_latency = latency_tracker.max() or 0.0
                inference_delay = math.ceil(inference_latency / time_per_step)
                logger.debug("inference_delay=%s", inference_delay)

                # ── C. 获取当前观测 ──
                if robot_wrapper is not None:
                    raw_obs = robot_wrapper.get_observation()
                else:
                    raw_obs = create_mock_observation()   # 干跑：全零假数据

                # ── D. 预处理观测（在 no_grad 下进行）──
                # 注意：必须用 no_grad() 而不是 inference_mode()！
                # RTC 的 denoise_step 内部调用 enable_grad() + autograd.grad() 进行梯度引导，
                # inference_mode 会完全禁用 autograd 且无法被 enable_grad() 覆盖。
                observation = robot_obs_to_policy_obs(raw_obs)
                with torch.no_grad():
                    obs_tensors = prepare_observation_for_inference(
                        copy(observation), DEVICE, task=args.task
                    )
                    # 预处理器：摄像头 key 重命名、任务文本 tokenize、状态归一化
                    obs_processed = preprocessor(obs_tensors)

                # ── E. 预测动作块（必须在 no_grad 上下文外）──
                # RTC 引导（consistency guidance）需要 autograd 计算梯度修正。
                actions = policy.predict_action_chunk(
                    obs_processed,
                    inference_delay=inference_delay,    # 延迟补偿：跳过已走的帧
                    prev_chunk_left_over=prev_actions,  # 上一块剩余动作（RTC 融合参考）
                )
                # actions.shape: [1, chunk_size, action_dim]
                # 例如: torch.Size([1, 50, 7]) → [批次=1, 未来50步, 7个关节]
                logger.debug("Raw chunk shape: %s", actions.shape)
                logger.debug("Future base joint trajectory: %s", actions[0, :, 0].cpu().numpy())

                # ── F. 保存原始动作 + 后处理 + 记录延迟 ──
                # merge 需要归一化空间的原始动作来计算引导梯度
                original_actions = actions.squeeze(0).clone()
                postprocessed_actions = postprocessor(actions).squeeze(0)

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_step)
                latency_tracker.add(new_latency)

                # 参数合理性检查：queue_threshold 应大于 execution_horizon + delay
                if args.rtc and args.queue_threshold < args.execution_horizon + new_delay:
                    logger.warning(
                        f"[INFERENCE] queue_threshold ({args.queue_threshold}) too small, "
                        f"should be > execution_horizon + delay "
                        f"({args.execution_horizon + new_delay})"
                    )

                # ── G. 融合新块进动作队列（RTC 核心操作）──
                # merge 在 execution_horizon 步的过渡区内以 guidance_weight 强度
                # 对新旧块做梯度引导融合，消除块切换处的抖动。
                action_queue.merge(
                    original_actions, postprocessed_actions,
                    new_delay, action_index_before,
                )

                logger.info(
                    f"[INFERENCE] Chunk in {new_latency:.3f}s "
                    f"(delay={new_delay}, queue={action_queue.qsize()})"
                )
            else:
                # 队列充足，短暂休眠避免 busy-waiting 浪费 CPU
                time.sleep(0.05)

        logger.info("[INFERENCE] Thread shutting down")
    except Exception as e:
        logger.error(f"[INFERENCE] Fatal error: {e}\n{traceback.format_exc()}")
        shutdown_event.set()   # 出错时广播关闭信号，避免其他线程空转
