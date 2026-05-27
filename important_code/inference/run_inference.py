"""
┌─────────────────────────────────────────────────────────────────┐
│  主线程 Main Thread                                               │
│  计时 + 监控 + 处理 Ctrl+C                                        │
└──────────────┬──────────────────────────────────────────────────┘
               │ shutdown_event (Event)
       ┌───────┴────────┐
       ▼                ▼
┌──────────────┐  ┌──────────────┐
│ 推理线程      │  │ 执行线程      │
│ Inference    │  │ Actor        │
│ Thread       │  │ Thread       │
│              │  │              │
│ 每当队列不足  │  │ 10 Hz 固定频  │
│ → 调用模型   │  │ 率从队列取动  │
│   生成新的   │  │ 作 → 发送给  │
│   动作块     │  │ 机器人       │
└──────┬───────┘  └──────┬───────┘
       │  ActionQueue     │
       └────────┬─────────┘
                ▼
         动作队列（--rtc 时融合动作块）
                │
                │ RViz publisher（始终开启）
                ▼
       ┌──────────────────┐
       │  RViz Publisher  │
       │  Thread          │
       │                  │
       │ /actual/         │
       │   joint_states   │ ← 实际执行位置（蓝色实体机器人）
       │ /predicted/      │
       │   joint_states   │ ← 预测目标位置（红色透明机器人）
       │ /predicted_      │
       │   trajectory     │ ← 完整预测轨迹
       └──────────────────┘

在 WidowX 机器人上运行 SmolVLA 策略推理。

支持两种模式（通过 --rtc 开关切换）：
  标准模式（默认）：执行完一整块动作后再生成下一块
  RTC 模式（--rtc）：异步生成并融合动作块，消除块切换时的抖动/停顿

三（或四）线程架构：
  推理线程 → ActionQueue → 执行线程 → 机器人
  主线程负责计时监控和协调关闭
  RViz 发布线程（始终启动）→ 可视化实际轨迹（蓝色机器人）与预测末端轨迹（橙红色线）

Usage:
  # 干跑（无硬件，测试推理流程）：
  python important_code/inference/run_inference.py --dry-run

  # 标准模式，连接机器人：
  python important_code/inference/run_inference.py --robot-ip 192.168.2.3

  # 开启 RTC（可选，动作更平滑）：
  python important_code/inference/run_inference.py --rtc

  # 指定训练输出目录：
  python important_code/inference/run_inference.py --train-dir outputs/train/my_run

  # 指定 Hugging Face 模型（默认已使用 fulloa10/smolVLA_grape_10hz_9000）：
  python important_code/inference/run_inference.py --train-dir fulloa10/smolVLA_grape_10hz_9000

  # RTC 调试模式：
  python important_code/inference/run_inference.py --dry-run --rtc --debug

  # RViz 可视化默认开启，启动前先在 crospi workspace 运行可视化节点：
  python important_code/inference/run_inference.py --dry-run   # 干跑验证

RViz 可视化使用步骤：
  1. 终端1（先启动，在 kaixi_crospi_ws）：
       ros2 launch crospi_application_template trossen_follower_visualization.launch.py
     该命令会启动 vla_ros_bridge_node.py + robot_state_publisher 并打开 RViz2。

  2. 终端2（后启动）：
       python important_code/inference/run_inference.py [其他参数]

  3. RViz 中可以看到：
       蓝色实体机器人        = 真实混合轨迹（VLA + SpaceMouse，来自 /joint_states 反馈）
       绿色球 + 橙红色线     = 末端执行器当前位置 + 未来 N 步预测轨迹（/predicted_ee_marker）

  4. 诊断方法：
       橙线抖动、蓝机器人也抖 → 模型预测本身不稳定，考虑调整 guidance_weight
       橙线平滑、蓝机器人抖动 → 执行层问题（max_relative_target 截断或硬件延迟）
       橙线与蓝机器人末端长期偏差 → 归一化参数或关节映射问题
"""

import argparse
from typing import cast
import logging
import sys
import time
from pathlib import Path
from threading import Event, Thread

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from peft import PeftConfig, PeftModel
except ImportError as e:
    print(f"Error importing lerobot: {e}")
    print("Requires lerobot >= 0.4.3 with RTC support.")
    sys.exit(1)

import rerun as rr
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from important_code.utils import DEVICE, resolve_checkpoint_path
from important_code.inference.robot_wrapper import RobotWrapper, create_mock_observation
from important_code.inference.inference_thread import inference_thread_fn
from important_code.inference.actor_thread import actor_thread_fn
from important_code.inference.rtc_runtime import (
    configure_policy_rtc,
    make_action_queue,
    maybe_plot_rtc_debug,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 默认配置 ===
DEFAULT_ROBOT_IP   = "192.168.2.3"
DEFAULT_CAM1_ID    = 4    # 手腕摄像头
DEFAULT_CAM2_ID    = 10   # 右侧摄像头
DEFAULT_TRAIN_DIR  = "fulloa10/smolvla_pipe_bomb_transfer_V1"
TASK_DESCRIPTION   = "pick up the pipe bomb from the box and place it at the marked location"


def load_smolvla_policy(policy_path: str):
    path = Path(policy_path)

    if (path / "adapter_config.json").exists():
        logger.info("Loading LoRA adapter policy from: %s", policy_path)
        policy_config = PreTrainedConfig.from_pretrained(policy_path)
        peft_config = PeftConfig.from_pretrained(policy_path)
        base_policy = SmolVLAPolicy.from_pretrained(
            peft_config.base_model_name_or_path,
            config=policy_config,
        )
        return PeftModel.from_pretrained(base_policy, policy_path, config=peft_config)

    logger.info("Loading full policy from: %s", policy_path)
    return SmolVLAPolicy.from_pretrained(policy_path)


def build_arg_parser():
    """Return the configured ArgumentParser (without parsing sys.argv)."""
    parser = argparse.ArgumentParser(
        description="Run SmolVLA Inference on WidowX Robot"
    )

    # 硬件
    parser.add_argument("--robot-ip",  type=str, default=DEFAULT_ROBOT_IP)
    parser.add_argument("--cam1",      type=int, default=DEFAULT_CAM1_ID,
                        help="手腕摄像头索引 (默认: 4)")
    parser.add_argument("--cam2",      type=int, default=DEFAULT_CAM2_ID,
                        help="右侧摄像头索引 (默认: 10)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="干跑模式，无需真实机器人硬件")
    parser.add_argument("--crospi",    action="store_true",
                        help="CroSPI 集成模式：观测从 trossen SDK 读取，send_action 为空操作（由 eTaSL 执行）")

    # 策略
    parser.add_argument("--train-dir", type=str, default=DEFAULT_TRAIN_DIR,
                        help="本地训练输出目录路径或 Hugging Face 模型 repo id")
    parser.add_argument("--task",      type=str, default=TASK_DESCRIPTION,
                        help="任务描述文本")

    # 控制/采集频率：相机可 30fps 后台采集，模型和机器人控制默认只跑 10Hz。
    parser.add_argument("--camera-fps", type=int, default=30,
                        help="相机后台采集帧率 fps（默认: 30）")
    parser.add_argument("--control-fps", "--fps", dest="control_fps", type=int, default=10,
                        help="模型触发和机器人控制频率 Hz（默认: 10；--fps 为兼容别名）")
    parser.add_argument("--duration", type=float, default=180.0,
                        help="最大运行时长（秒，默认: 180）")  # --duration 控制整个推理程序的最长运行时间，单位是秒（默认 180 秒 = 3 分钟）。

    # VLA confidence
    parser.add_argument("--confidence-method", type=str, default="regression_cbc",
                        choices=["raw_cbc", "speed_norm_cbc", "regression_cbc"],
                        help="C_VLA method used as the primary confidence signal")
    parser.add_argument("--sentinel-confidence-mode", type=str, default="regression_cbc",
                        choices=["raw_cbc", "speed_norm_cbc", "regression_cbc", "tracking", "combined"],
                        help="c_action mode: cbc modes measure boundary smoothness; "
                             "tracking measures joint tracking error; combined = sqrt(regression * tracking).")

    # Sentinel-style runtime arbitration. Disabled by default.
    parser.add_argument("--sentinel", action="store_true",
                        help="Enable Sentinel fast action monitor + cloud VLM progress monitor")
    parser.add_argument("--sentinel-log-only", action=argparse.BooleanOptionalAction, default=True,
                        help="Log Sentinel weights without publishing them to CroSPI/eTaSL")
    parser.add_argument("--sentinel-vlm-provider", type=str, default="openai",
                        choices=["openai", "gemini", "deepseek"],
                        help="Cloud VLM provider for Sentinel progress monitoring")
    parser.add_argument("--sentinel-vlm-model", type=str, default=None,
                        help="Cloud VLM model name; defaults to gpt-4o or gemini-3-flash-preview")
    parser.add_argument("--sentinel-interval-s", type=float, default=2.0,
                        help="Seconds between cloud VLM progress checks")
    parser.add_argument("--sentinel-timeout-s", type=float, default=5.0,
                        help="Cloud VLM request timeout in seconds")
    parser.add_argument("--sentinel-log-dir", type=str, default="outputs/sentinel",
                        help="Directory for Sentinel JSONL/CSV logs")
    parser.add_argument("--sentinel-action-tau", type=float, default=0.4,
                        help="Fast action consistency threshold for action_alarm")
    parser.add_argument("--sentinel-jerk-max", type=float, default=None,
                        help="Optional jerk_max threshold for fast Sentinel action_alarm")
    parser.add_argument("--sentinel-boundary-jump-max", type=float, default=None,
                        help="Optional boundary_jump_max threshold for fast Sentinel action_alarm")
    parser.add_argument("--sentinel-progress-threshold", type=float, default=0.7,
                        help="VLM failure_likelihood threshold for raw progress alarms")
    parser.add_argument("--sentinel-progress-alarm-count", type=int, default=2,
                        help="Consecutive raw progress alarms required to trigger progress_alarm")
    parser.add_argument("--sentinel-ema-beta", type=float, default=0.8,
                        help="EMA beta for smoothing Sentinel reliability before weight output")
    parser.add_argument("--sentinel-window-s", type=float, default=4.0,
                        help="Camera time window used for VLM progress checks")
    parser.add_argument("--sentinel-max-frames", type=int, default=6,
                        help="Maximum frames sampled into each Sentinel VLM image grid")
    parser.add_argument("--sentinel-progress-max-age-s", type=float, default=8.0,
                        help="Maximum age before a VLM progress result is considered stale")
    parser.add_argument("--sentinel-decay-lambda", type=float, default=0.1,
                        help="Exponential decay rate for c_progress when robot is stuck (half-life = ln2/lambda ≈ 14s).")
    parser.add_argument("--sentinel-stuck-threshold", type=float, default=0.05,
                        help="Max joint range (rad) across the motion window before robot is considered stuck. "
                             "Increase (e.g. 0.08) to also catch oscillatory motion.")

    # RTC 开关（默认关闭，传 --rtc 则开启）
    parser.add_argument("--rtc", action="store_true", default=False,
                        help="开启 RTC（Real-Time Chunking）动作融合，动作更平滑")

    # RTC 参数（仅 --rtc 时生效）
    parser.add_argument("--execution-horizon", type=int,   default=10,
                        help="RTC: 新旧块融合的过渡步数（默认: 10）")
    parser.add_argument("--guidance-weight",   type=float, default=10.0,
                        help="RTC: 一致性约束强度，越大新块越靠近旧块轨迹（默认: 10.0）")
    parser.add_argument("--attention-schedule", type=str,  default="EXP",
                        choices=["LINEAR", "EXP", "ONES", "ZEROS"],
                        help="RTC: 前缀注意力衰减方式（默认: EXP）")
    parser.add_argument("--queue-threshold", type=int, default=10,
                        help="RTC: 队列低于此值时触发新推理（默认: 10）")

    # 可视化
    parser.add_argument("--display-data", action="store_true",
                        help="开启 Rerun 实时可视化（相机+关节+置信度），与 teleoperation 界面相同")

    # 调试
    parser.add_argument("--debug",          action="store_true",
                        help="开启 RTC 内置调试追踪")
    parser.add_argument("--debug-maxlen",   type=int, default=100,
                        help="调试缓冲区最大条数（默认: 100）")
    parser.add_argument("--print-publish", action="store_true",
                        help="单独测试时打印 VLA 发布内容（无需 bridge_node 在线）")

    # RQ3 trial logging (B2 pure-VLA timing).
    # When --trial is given, a single-row summary CSV is appended on exit.
    # t_start is latched at the first action sent to the robot by actor_thread.
    parser.add_argument("--trial", type=int, default=None,
                        help="RQ3 trial index (e.g. 1..20). Enables trial-summary CSV.")
    parser.add_argument("--condition", type=str, default="B2",
                        choices=["B1", "B2", "B3"],
                        help="RQ3 experimental condition; used for the summary filename")
    parser.add_argument("--out-dir", type=str, default="./logs",
                        help="Directory for the per-condition summary CSV")

    return parser


def parse_args():
    """Parse sys.argv and return the argument namespace."""
    args = build_arg_parser().parse_args()
    args.fps = args.control_fps  # Backward compatibility for older helper code/tests.
    return args


def _rerun_log_thread_fn(robot_wrapper, shutdown_event, camera_fps):
    """以 camera_fps Hz 读取相机和关节观测并写入 Rerun，实现实时监控。"""
    interval = 1.0 / camera_fps
    while not shutdown_event.is_set():
        t0 = time.perf_counter()
        raw_obs = robot_wrapper.get_observation() if robot_wrapper is not None else create_mock_observation()
        log_rerun_data(observation=raw_obs)
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, interval - elapsed))


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"RTC: {'开启' if args.rtc else '关闭（标准模式）'}")

    if args.display_data:
        init_rerun(session_name="inference")

    # ── 1. 加载策略模型 ──────────────────────────────────────────
    policy_path = resolve_checkpoint_path(args.train_dir)
    logger.info(f"Loading policy from: {policy_path}")
    policy = load_smolvla_policy(policy_path)

    rtc_config = configure_policy_rtc(policy, args)
    policy.to(DEVICE).eval()
    logger.info(
        "[HEALTH] Policy config: chunk_size=%s | n_action_steps=%s | input_features=%s | output_features=%s",
        getattr(policy.config, "chunk_size", None),
        getattr(policy.config, "n_action_steps", None),
        getattr(policy.config, "input_features", None),
        getattr(policy.config, "output_features", None),
    )

    # ── 2. 加载预处理器 & 后处理器 ──────────────────────────────
    device_overrides = {"device_processor": {"device": str(DEVICE)}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides=device_overrides,
        postprocessor_overrides=device_overrides,
    )
    logger.info("Processors loaded.")
    logger.info("[HEALTH] Task string: %r", args.task)
    logger.info(
        "[HEALTH] RTC enabled: %s | queue_threshold=%s | camera_fps=%s | control_fps=%s",
        bool(args.rtc),
        args.queue_threshold,
        args.camera_fps,
        args.control_fps,
    )
    if args.rtc:
        logger.warning("[HEALTH] RTC is enabled; first-pass grape diagnosis should use the default non-RTC mode.")
    else:
        logger.info("[HEALTH] RTC confirmed disabled for the live inference path.")

    # ── 3. 连接机器人（干跑模式跳过）────────────────────────────
    robot = None
    robot_wrapper = None

    if not args.dry_run:
        try:
            from lerobot.robots.utils import make_robot_from_config

            if args.crospi:
                from lerobot_robot_trossen.config_crospi_follower import CroSPIFollowerConfig
                robot_config = CroSPIFollowerConfig(
                    ip_address=args.robot_ip,
                    cameras={
                        "wrist": OpenCVCameraConfig(args.cam1, fps=args.camera_fps, width=640, height=480, fourcc="YUYV"),
                        "right": OpenCVCameraConfig(args.cam2, fps=args.camera_fps, width=640, height=480, fourcc="YUYV"),
                    },
                )
                logger.info("CroSPI mode: observations via Trossen SDK, send_action is no-op.")
            else:
                from lerobot_robot_trossen.config_widowxai_follower import WidowXAIFollowerConfig
                robot_config = WidowXAIFollowerConfig(
                    ip_address=args.robot_ip,
                    cameras={
                        "wrist": OpenCVCameraConfig(args.cam1, fps=args.camera_fps, width=640, height=480, fourcc="YUYV"),
                        "right": OpenCVCameraConfig(args.cam2, fps=args.camera_fps, width=640, height=480, fourcc="YUYV"),
                    },
                    max_relative_target=5.0,   # 单步最大位移限制（安全保护）
                )

            robot = make_robot_from_config(robot_config)
            if args.crospi:
                from lerobot_robot_trossen.crospi_follower import CroSPIFollower
                # 跳过机械臂 SDK 连接，避免与 eTaSL 双重连接冲突；
                # 关节状态由 vla_ros_bridge_node 通过 UDP :9789 转发。
                cast(CroSPIFollower, robot).connect(connect_arm=False)
            else:
                robot.connect()
            robot_wrapper = RobotWrapper(robot)
            logger.info("Robot connected.")
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            raise
    else:
        logger.info("=== DRY RUN MODE — No robot hardware ===")

    # ── 4. 创建动作队列，启动推理线程和执行线程 ─────────────────
    shutdown_event = Event()
    action_queue = make_action_queue(rtc_config)

    # RQ3 trial timing: actor_thread latches t_start on the first send_action.
    # Only enabled when --trial is supplied on the command line.
    trial_state = {"t_start": None, "t_end": None} if args.trial is not None else None

    # 始终启动 RViz 可视化发布线程（UDP fire-and-forget，无接收方时无副作用）
    from important_code.inference.rviz_publisher import RVizPublisher
    rviz_publisher = RVizPublisher(verbose=args.print_publish)
    rviz_publisher.start()
    logger.info("RViz publisher started.")

    logger.info(
        "Task: %s | camera_fps=%s | control_fps=%s | Duration: %ss",
        args.task,
        args.camera_fps,
        args.control_fps,
        args.duration,
    )

    inf_thread = Thread(
        target=inference_thread_fn,
        args=(policy, robot_wrapper, preprocessor, postprocessor,
              action_queue, shutdown_event, args),
        kwargs={"rviz_publisher": rviz_publisher},
        daemon=True, name="InferenceThread",
    )
    act_thread = Thread(
        target=actor_thread_fn,
        args=(robot_wrapper, action_queue, shutdown_event, args),
        kwargs={"rviz_publisher": rviz_publisher, "trial_state": trial_state},
        daemon=True, name="ActorThread",
    )
    if args.display_data:
        rerun_thread = Thread(
            target=_rerun_log_thread_fn,
            args=(robot_wrapper, shutdown_event, args.camera_fps),
            daemon=True, name="RerunLogThread",
        )
        rerun_thread.start()
        logger.info("Rerun log thread started at %s Hz.", args.camera_fps)

    inf_thread.start()
    act_thread.start()
    logger.info("Threads started.")

    # ── 5. 主线程：计时监控 ──────────────────────────────────────
    start_time = time.time()
    last_logged_window = -1
    try:
        while not shutdown_event.is_set():
            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                logger.info(f"Duration limit ({args.duration}s) reached.")
                break
            window = int(elapsed) // 10
            if window > 0 and window != last_logged_window:
                logger.info(f"[MAIN] {elapsed:.0f}s elapsed, queue={action_queue.qsize()}")
                last_logged_window = window
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Ctrl+C → 推理停止。")
        if trial_state is not None:
            trial_state["t_end"] = time.time()

    # ── 6. 清理 ──────────────────────────────────────────────────
    shutdown_event.set()
    logger.info("正在等待推理/执行线程停止（最多 5s）...")
    inf_thread.join(timeout=5.0)
    if inf_thread.is_alive():
        logger.warning("推理线程未在超时内停止，继续清理。")
    act_thread.join(timeout=5.0)

    if robot:
        robot.disconnect()
        logger.info("Robot disconnected.")

    # ── 7. 调试数据输出（仅 --debug 模式）───────────────────────
    maybe_plot_rtc_debug(policy, args)

    if args.display_data:
        rr.rerun_shutdown()

    # ── 8. RQ3 trial summary (only when --trial was supplied) ───────────────
    if trial_state is not None:
        import csv
        from pathlib import Path

        t_end = trial_state.get("t_end") or time.time()
        t_start = trial_state.get("t_start")
        if t_start is None:
            t_trial = float("nan")
            logger.warning(
                "[TRIAL] No action was sent to the robot; T_trial recorded as NaN."
            )
        else:
            t_trial = t_end - t_start

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / f"{args.condition}_summary.csv"
        write_header = not summary_path.exists()
        with open(summary_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "trial_id", "condition", "T_trial_s",
                    "CE_velocity", "n_gripper_press", "gripper_rate_hz",
                    "n_mode_switch", "raw_csv",
                ])
            # B2 has no SpaceMouse logging; CE_velocity / gripper / mode fields are blank.
            w.writerow([
                int(args.trial), args.condition, f"{t_trial:.6f}",
                "", "", "", "", "",
            ])
        logger.info(
            "[TRIAL] cond=%s trial=%d T_trial=%.3f s -> %s",
            args.condition, args.trial, t_trial, summary_path,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
