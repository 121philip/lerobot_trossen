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
  F. 后处理（反归一化）+ 记录本次延迟 + （可选）推送给 RVizPublisher
  G. 将新块融合进动作队列

RViz 集成（rviz_publisher 不为 None 时）：
  - 每次推理后（步骤 F），将反归一化后的完整动作块推送给 RVizPublisher
  - RVizPublisher 通过 rviz_node.py 发布到 /predicted_ee_marker（未来动作末端轨迹，橙红色线）
  - 注意：推送的是后处理后的真实关节角（弧度/米），与实际执行单位一致，可直接比较
"""

import csv
import datetime
import logging
import math
import time
import traceback
from copy import copy
from pathlib import Path
from threading import Event

import numpy as np
import rerun as rr
import torch

from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.latency_tracker import LatencyTracker

from important_code.utils import DEVICE, JOINT_NAMES, get_control_fps
from important_code.inference.robot_wrapper import create_mock_observation, robot_obs_to_policy_obs
from important_code.shared_control.confidence import ConfidenceEstimator
from important_code.shared_control.sentinel import SentinelRuntime

logger = logging.getLogger(__name__)


def _save_sentinel_plot(records: list, log_dir: str) -> None:
    """运行结束后将 Sentinel 指标保存为 CSV 和时序折线图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(log_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = out / f"sentinel_metrics_{stamp}.csv"
    fieldnames = ["t", "c_action", "c_progress", "c_vlm", "r_raw", "r_smooth", "w_vla", "w_human"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    logger.info("[SENTINEL] Saved CSV → %s", csv_path)

    t          = [r["t"] for r in records]
    c_action   = [r["c_action"] for r in records]
    c_progress = [r["c_progress"] for r in records]
    c_vlm      = [r.get("c_vlm") if r.get("c_vlm") is not None else float("nan") for r in records]
    r_raw      = [r["r_raw"] for r in records]
    r_smooth   = [r["r_smooth"] for r in records]
    w_vla      = [r["w_vla"] for r in records]
    w_human    = [r["w_human"] for r in records]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Sentinel Metrics over Follower Runtime", fontsize=13)

    axes[0].plot(t, c_action,   label="c_action",   color="steelblue")
    axes[0].plot(t, c_progress, label="c_progress", color="darkorange", linestyle="--")
    axes[0].plot(t, c_vlm,      label="c_vlm",      color="green",      linestyle=":", alpha=0.6)
    axes[0].set_ylabel("Confidence")
    axes[0].set_ylim(-0.05, 1.1)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, r_raw,    label="r_raw",    color="gray",   linestyle=":")
    axes[1].plot(t, r_smooth, label="r_smooth", color="purple")
    axes[1].set_ylabel("Reliability (r)")
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, w_vla,   label="w_vla",   color="royalblue")
    axes[2].plot(t, w_human, label="w_human", color="tomato")
    axes[2].set_ylabel("Weight")
    axes[2].set_xlabel("Follower Runtime (s)")
    axes[2].set_ylim(-0.05, 1.1)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out / f"sentinel_metrics_{stamp}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("[SENTINEL] Saved plot  → %s", plot_path)


def inference_thread_fn(
    policy,
    robot_wrapper,          # RobotWrapper 实例，None 表示干跑模式
    preprocessor,
    postprocessor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    args,
    rviz_publisher=None,    # RVizPublisher 实例，None 表示不可视化
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
        rviz_publisher:  RVizPublisher 实例（由 run_inference.py 传入）
                         传入后每块推理完成后将预测动作发布到 /predicted/joint_states
                         不传（默认 None）则不启用可视化，不影响推理性能
    """
    sentinel_runtime = None
    sentinel_records: list = []
    _inference_start = time.perf_counter()
    try:
        logger.info("[INFERENCE] Starting inference thread")

        latency_tracker = LatencyTracker()   # 维护历史推理延迟的滑动窗口
        control_fps = get_control_fps(args)
        time_per_step = 1.0 / control_fps    # 控制/策略时间步（秒）
        confidence_method = getattr(
            args, "sentinel_confidence_mode",
            getattr(args, "confidence_method", "combined"),
        )
        confidence_estimator = ConfidenceEstimator(
            d=5,
            gamma=100.0,
            fps=control_fps,
            confidence_method=confidence_method,
        )
        if getattr(args, "sentinel", False):
            sentinel_runtime = SentinelRuntime.from_args(args)
            sentinel_runtime.start()

        # RTC 模式：队列低于阈值时提前触发推理，保持队列充足
        # 非 RTC 模式：队列完全耗尽才触发（传统块执行方式）
        queue_threshold = args.queue_threshold if args.rtc else 0
        logger.info("[HEALTH] Task string: %r", args.task)
        logger.info(
            "[HEALTH] RTC enabled: %s | effective queue_threshold=%s | fps=%s",
            bool(args.rtc), queue_threshold, control_fps,
        )
        logger.info(
            "[HEALTH] C_VLA method=%s",
            confidence_method,
        )
        if args.rtc:
            logger.warning("[HEALTH] RTC is enabled; first-pass grape diagnosis expects --rtc disabled.")
        else:
            logger.info("[HEALTH] RTC confirmed disabled; using synchronous/non-RTC queue refill.")

        logged_observation_health = False
        empty_since = None

        while not shutdown_event.is_set():
            if action_queue.qsize() <= queue_threshold:
                queue_size_before = action_queue.qsize()
                current_time = time.perf_counter()
                if queue_size_before == 0 and empty_since is None:
                    empty_since = current_time

                # ── A. 快照推理前的队列状态（供 merge 对齐时序用）──
                action_index_before = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()  # 队列中尚未执行的剩余动作

                # ── B. 计算延迟补偿帧数 ──
                # 推理耗时期间机械臂仍在执行旧动作，inference_delay 告诉模型
                # "当这个新块被用到时，机械臂已经额外走了多少步"。
                # Formula: delay_steps = ceil(latency_s / (1/fps))
                # Uses the *maximum* observed latency (not average) to conservatively
                # avoid misaligned merges on slow inference iterations.
                inference_latency = latency_tracker.max() or 0.0
                inference_delay = math.ceil(inference_latency / time_per_step)
                logger.debug("inference_delay=%s", inference_delay)

                # ── C. 获取当前观测 ──
                if robot_wrapper is not None:
                    raw_obs = robot_wrapper.get_observation()
                else:
                    raw_obs = create_mock_observation()   # 干跑：全零假数据

                # CroSPI 模式：用 bridge 转发的 /joint_states 覆盖 SDK/mock 关节值，
                # 确保 VLA 观测的关节状态来自 eTaSL（单一事实来源）。
                # 直接连接真实机器人时必须保留 Trossen SDK 观测，避免在线 bridge 污染状态。
                if getattr(args, "crospi", False) and rviz_publisher is not None:
                    bridge_joints = rviz_publisher.get_latest_joints()
                    if bridge_joints is not None:
                        print("Feedback: ", bridge_joints)
                        for i, name in enumerate(JOINT_NAMES):
                            raw_obs[f"{name}.pos"] = float(bridge_joints[i])

                actual_joints = rviz_publisher.get_latest_joints() if rviz_publisher is not None else None
                if actual_joints is None:
                    actual_joints = np.array(
                        [raw_obs.get(f"{name}.pos", 0.0) for name in JOINT_NAMES], dtype=float
                    )

                # ── D. 预处理观测（在 no_grad 下进行）──
                # 注意：必须用 no_grad() 而不是 inference_mode()！
                # RTC 的 denoise_step 内部调用 enable_grad() + autograd.grad() 进行梯度引导，
                # inference_mode 会完全禁用 autograd 且无法被 enable_grad() 覆盖。
                # print(raw_obs)
                observation = robot_obs_to_policy_obs(raw_obs)
                if sentinel_runtime is not None:
                    # Sentinel slow monitor 只需要相机图像；这里把最近 wrist/right 帧缓存起来。
                    # 这一步不调用云端 VLM，因此不会拖慢当前 action chunk 推理。
                    sentinel_runtime.push_observation(observation)
                with torch.no_grad():
                    obs_tensors = prepare_observation_for_inference(
                        copy(observation), DEVICE, task=args.task
                    )
                    # 预处理器：摄像头 key 重命名、任务文本 tokenize、状态归一化
                    obs_processed = preprocessor(obs_tensors)
                    if not logged_observation_health:
                        logger.info("[HEALTH] Raw observation keys: %s", sorted(observation.keys()))
                        logger.info("[HEALTH] Prepared observation keys: %s", sorted(obs_tensors.keys()))
                        logger.info("[HEALTH] Preprocessor output keys: %s", sorted(obs_processed.keys()))
                        logged_observation_health = True

                # ── E. 预测动作块（必须在 no_grad 上下文外）──
                # RTC 引导（consistency guidance）需要 autograd 计算梯度修正。
                actions = policy.predict_action_chunk(
                    obs_processed,
                    inference_delay=inference_delay,    # 延迟补偿：跳过已走的帧
                    prev_chunk_left_over=prev_actions,  # 上一块剩余动作（RTC 融合参考）
                )
                # actions.shape: [1, chunk_size, action_dim]
                # 例如: torch.Size([1, chunk_size, 7]) → [批次=1, 未来动作步, 7个关节]
                logger.debug("Raw chunk shape: %s", actions.shape)
                logger.debug("Future base joint trajectory: %s", actions[0, :, 0].cpu().numpy())
                # print("Future base joint trajectory: %s", actions[0, :, 0].cpu().numpy())

                # ── F. 保存完整预测块 + 选择实际入队动作 ──
                # full_original_chunk 保留归一化空间动作，RTC 融合需要它。
                # full_robot_chunk 是反归一化后的机器人关节目标，用于执行和可视化。
                full_original_chunk = actions.squeeze(0).clone()
                full_robot_chunk = postprocessor(actions).squeeze(0)
                # print("POST Future base joint trajectory: %s", full_robot_chunk[:, 0].cpu().numpy())

                if args.rtc:
                    queued_original_chunk = full_original_chunk
                    queued_robot_chunk = full_robot_chunk
                else:
                    # n_action_steps = int(getattr(policy.config, "n_action_steps", len(full_robot_chunk)))  # 可以调整入队的动作步数，默认为整个块
                    n_action_steps = 10  # 注意，n_action_steps 可调！！！！
                    queued_original_chunk = full_original_chunk[:n_action_steps]
                    queued_robot_chunk = full_robot_chunk[:n_action_steps]

                print(f"[DEBUG] queued_original_chunk shape={queued_original_chunk.shape}\n{queued_original_chunk.cpu().numpy()}")
                print(f"[DEBUG] queued_robot_chunk    shape={queued_robot_chunk.shape}\n{queued_robot_chunk.cpu().numpy()}")

                confidence_metrics = confidence_estimator.update(
                    queued_robot_chunk,
                    actual_joints=actual_joints,
                    delay_steps=inference_delay,
                    actions_normalized=queued_original_chunk,
                )
                vel_max = confidence_metrics.vel_max
                accel_max = confidence_metrics.accel_max
                jerk_max = confidence_metrics.jerk_max
                boundary_jump_max = confidence_metrics.boundary_jump_max

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_step)
                latency_tracker.add(new_latency)
                sentinel_result = None
                if sentinel_runtime is not None:
                    # 每个 action chunk 结束后做一次 fast arbitration：
                    # - C_action 来自 confidence_metrics.c_action / jerk / boundary jump；
                    # - C_progress 来自后台 VLM 最近一次结果；
                    # - 输出直接是 eTaSL 权重 w_vla / w_human。
                    sentinel_result = sentinel_runtime.update(
                        confidence_metrics,
                        actual_joints=actual_joints,
                        extra={
                            "chunk_latency_s": new_latency,
                            "delay_steps": new_delay,
                            "queue_size_before": queue_size_before,
                            "confidence_method": confidence_method,
                        },
                    )
                    logger.info(
                        "[SENTINEL] c_action=%.4f c_progress=%.4f c_vlm=%s "
                        "r_raw=%.4f r=%.4f w_vla=%.4f w_human=%.4f "
                        "alarm=%s progress_stale=%s vlm_latency=%s reason=%s",
                        sentinel_result.c_action,
                        sentinel_result.c_progress,
                        f"{sentinel_result.c_vlm:.4f}" if sentinel_result.c_vlm is not None else "None",
                        sentinel_result.r_raw,
                        sentinel_result.r_smooth,
                        sentinel_result.w_vla,
                        sentinel_result.w_human,
                        sentinel_result.sentinel_alarm,
                        sentinel_result.progress_stale,
                        f"{sentinel_result.vlm_latency_s:.3f}s" if sentinel_result.vlm_latency_s is not None else "None",
                        sentinel_result.reason,
                    )
                    c_vlm_text = f"{sentinel_result.c_vlm:.4f}" if sentinel_result.c_vlm is not None else "None"
                    print(
                        "[SENTINEL_VALUES] "
                        f"C_action={sentinel_result.c_action:.4f} "
                        f"C_progress={sentinel_result.c_progress:.4f} "
                        f"C_VLM={c_vlm_text} "
                        f"R_raw={sentinel_result.r_raw:.4f} "
                        f"R={sentinel_result.r_smooth:.4f}",
                        flush=True,
                    )
                    sentinel_records.append({
                        "t":          time.perf_counter() - _inference_start,
                        "c_action":   sentinel_result.c_action,
                        "c_progress": sentinel_result.c_progress,
                        "c_vlm":      sentinel_result.c_vlm,
                        "r_raw":      sentinel_result.r_raw,
                        "r_smooth":   sentinel_result.r_smooth,
                        "w_vla":      sentinel_result.w_vla,
                        "w_human":    sentinel_result.w_human,
                    })

                # 参数合理性检查：queue_threshold 应大于 execution_horizon + delay
                if args.rtc and args.queue_threshold < args.execution_horizon + new_delay:
                    logger.warning(
                        f"[INFERENCE] queue_threshold ({args.queue_threshold}) too small, "
                        f"should be > execution_horizon + delay "
                        f"({args.execution_horizon + new_delay})"
                    )

                if rviz_publisher is not None:
                    rviz_publisher.put_predicted(full_robot_chunk.cpu().numpy())
                    if (
                        sentinel_result is not None
                        and not sentinel_runtime.log_only
                        and hasattr(rviz_publisher, "put_weights")
                    ):
                        # 默认 log-only 时不会走到这里。
                        # 只有显式 --no-sentinel-log-only 才把权重送到 CroSPI/eTaSL。
                        rviz_publisher.put_weights(
                            sentinel_result.w_vla,
                            sentinel_result.w_human,
                        )

                if getattr(args, "display_data", False):
                    rr.log("inference/c_action",   rr.Scalars(confidence_metrics.c_action))
                    rr.log("inference/jerk_max",    rr.Scalars(jerk_max))
                    rr.log("inference/latency_ms",  rr.Scalars(new_latency * 1000.0))
                    rr.log("inference/queue_size",  rr.Scalars(float(action_queue.qsize())))
                    if sentinel_result is not None:
                        rr.log("sentinel/w_vla",    rr.Scalars(sentinel_result.w_vla))
                        rr.log("sentinel/w_human",  rr.Scalars(sentinel_result.w_human))
                        rr.log("sentinel/c_progress", rr.Scalars(sentinel_result.c_progress))
                        if sentinel_result.c_vlm is not None:
                            rr.log("sentinel/c_vlm", rr.Scalars(sentinel_result.c_vlm))

                # ── G. 写入动作队列 ──
                # RTC 模式：merge 会按延迟替换队列，并配合 RTC 融合新旧 chunk。
                # 非 RTC 模式：merge 只是追加 queued chunk；这里 queued chunk 已截断到 n_action_steps。
                action_queue.merge(
                    queued_original_chunk, queued_robot_chunk,
                    new_delay, action_index_before,
                )
                idle_gap_ms = 0.0
                if empty_since is not None:
                    idle_gap_ms = (time.perf_counter() - empty_since) * 1000.0
                    empty_since = None

                logger.info(
                    "[INFERENCE] Chunk in %.3fs "
                    "(delay=%s, queue_before=%s, queue_after=%s, idle_gap_ms=%.1f, "
                    "confidence_method=%s, "
                    "c_action=%.4f, c_raw=%.4f, c_speed_norm=%.4f, c_regression=%.4f, "
                    "cbc_raw_mse=%.6f, cbc_reg_residual=%.6f, "
                    "a_vi=%.6f, a_ai=%.6f, jerk=%.6f, "
                    "vel_max=%.4f, accel_max=%.4f, jerk_max=%.4f, boundary_jump_max=%.4f)",
                    new_latency,
                    new_delay,
                    queue_size_before,
                    action_queue.qsize(),
                    idle_gap_ms,
                    confidence_method,
                    confidence_metrics.c_action,
                    confidence_metrics.c_raw,
                    confidence_metrics.c_speed_norm,
                    confidence_metrics.c_regression,
                    confidence_metrics.cbc_raw_mse,
                    confidence_metrics.cbc_reg_residual,
                    confidence_metrics.a_vi,
                    confidence_metrics.a_ai,
                    confidence_metrics.jerk,
                    vel_max,
                    accel_max,
                    jerk_max,
                    boundary_jump_max,
                )
            else:
                # 队列充足，短暂休眠避免 busy-waiting 浪费 CPU
                time.sleep(0.05)

        logger.info("[INFERENCE] Thread shutting down")
    except Exception as e:
        logger.error(f"[INFERENCE] Fatal error: {e}\n{traceback.format_exc()}")
        shutdown_event.set()   # 出错时广播关闭信号，避免其他线程空转
    finally:
        if sentinel_runtime is not None:
            sentinel_runtime.stop()
        if sentinel_records:
            _save_sentinel_plot(
                sentinel_records,
                getattr(args, "sentinel_log_dir", "outputs/sentinel"),
            )
