"""
┌─────────────────────────────────────────────────────┐
│  主线程 Main Thread                                   │
│  计时 + 监控 + 处理 Ctrl+C                            │
└──────────────┬──────────────────────────────────────┘
               │ shutdown_event (Event)
       ┌───────┴────────┐
       ▼                ▼
┌──────────────┐  ┌──────────────┐
│ 推理线程      │  │ 执行线程      │
│ Inference    │  │ Actor        │
│ Thread       │  │ Thread       │
│              │  │              │
│ 每当队列不足  │  │ 30 Hz 固定频  │
│ → 调用模型   │  │ 率从队列取动  │
│   生成新的   │  │ 作 → 发送给  │
│   动作块     │  │ 机器人       │
└──────┬───────┘  └──────┬───────┘
       │  ActionQueue     │
       └────────┬─────────┘
                ▼
         (RTC 融合动作块)

在 WidowX 机器人上运行 SmolVLA 策略推理。

支持两种模式（通过 --rtc 开关切换）：
  标准模式（默认）：执行完一整块动作后再生成下一块
  RTC 模式（--rtc）：异步生成并融合动作块，消除块切换时的抖动/停顿

三线程架构：
  推理线程 → ActionQueue → 执行线程 → 机器人
  主线程负责计时监控和协调关闭

Usage:
  # 干跑（无硬件，测试推理流程）：
  python important_code/RTC_inference/run_inference_rtc.py --dry-run

  # 标准模式，连接机器人：
  python important_code/RTC_inference/run_inference_rtc.py --robot-ip 192.168.2.3

  # 开启 RTC（推荐，动作更平滑）：
  python important_code/RTC_inference/run_inference_rtc.py --rtc

  # 指定训练输出目录：
  python important_code/RTC_inference/run_inference_rtc.py --train-dir outputs/train/my_run --rtc

  # 调试模式：
  python important_code/RTC_inference/run_inference_rtc.py --dry-run --debug
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from threading import Event, Thread

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.action_queue import ActionQueue
except ImportError as e:
    print(f"Error importing lerobot: {e}")
    print("Requires lerobot >= 0.4.3 with RTC support.")
    sys.exit(1)

try:
    from lerobot.policies.rtc.debug_visualizer import RTCDebugVisualizer
    _HAS_DEBUG_VISUALIZER = True
except ImportError:
    _HAS_DEBUG_VISUALIZER = False

from important_code.utils import DEVICE, resolve_checkpoint_path
from important_code.Inference.robot_wrapper import RobotWrapper
from important_code.Inference.inference_thread import inference_thread_fn
from important_code.Inference.actor_thread import actor_thread_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 默认配置 ===
DEFAULT_ROBOT_IP   = "192.168.2.3"
DEFAULT_CAM1_ID    = 2    # 手腕摄像头
DEFAULT_CAM2_ID    = 10   # 右侧摄像头
DEFAULT_TRAIN_DIR  = "outputs/train/smolvla_widowx_grape_grasping"
TASK_DESCRIPTION   = "Grab the grape"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SmolVLA Inference on WidowX Robot"
    )

    # 硬件
    parser.add_argument("--robot-ip",  type=str, default=DEFAULT_ROBOT_IP)
    parser.add_argument("--cam1",      type=int, default=DEFAULT_CAM1_ID,
                        help="手腕摄像头索引 (默认: 2)")
    parser.add_argument("--cam2",      type=int, default=DEFAULT_CAM2_ID,
                        help="右侧摄像头索引 (默认: 10)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="干跑模式，无需真实机器人硬件")

    # 策略
    parser.add_argument("--train-dir", type=str, default=DEFAULT_TRAIN_DIR,
                        help="本地训练输出目录路径")
    parser.add_argument("--task",      type=str, default=TASK_DESCRIPTION,
                        help="任务描述文本")

    # 控制
    parser.add_argument("--fps",      type=int,   default=30,
                        help="机器人控制频率 Hz（默认: 30）")
    parser.add_argument("--duration", type=float, default=120.0,
                        help="最大运行时长（秒，默认: 120）")  # --duration 控制整个推理程序的最长运行时间，单位是秒（默认 120 秒 = 2 分钟）。

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
    parser.add_argument("--queue-threshold", type=int, default=30,
                        help="RTC: 队列低于此值时触发新推理（默认: 30）")

    # 调试
    parser.add_argument("--debug",          action="store_true",
                        help="开启 RTC 内置调试追踪")
    parser.add_argument("--debug-maxlen",   type=int, default=100,
                        help="调试缓冲区最大条数（默认: 100）")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"RTC: {'开启' if args.rtc else '关闭（标准模式）'}")

    # ── 1. 加载策略模型 ──────────────────────────────────────────
    policy_path = resolve_checkpoint_path(args.train_dir)
    logger.info(f"Loading policy from: {policy_path}")
    policy = SmolVLAPolicy.from_pretrained(policy_path)

    rtc_config = RTCConfig(
        enabled=args.rtc,
        execution_horizon=args.execution_horizon,
        max_guidance_weight=args.guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule[args.attention_schedule],
    )
    if args.debug:
        rtc_config.debug = True
        rtc_config.debug_maxlen = args.debug_maxlen

    policy.config.rtc_config = rtc_config
    policy.init_rtc_processor()
    policy.to(DEVICE).eval()

    # ── 2. 加载预处理器 & 后处理器 ──────────────────────────────
    device_overrides = {"device_processor": {"device": str(DEVICE)}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides=device_overrides,
        postprocessor_overrides=device_overrides,
    )
    logger.info("Processors loaded.")

    # ── 3. 连接机器人（干跑模式跳过）────────────────────────────
    robot = None
    robot_wrapper = None

    if not args.dry_run:
        try:
            from lerobot_robot_trossen.config_widowxai_follower import WidowXAIFollowerConfig
            from lerobot.robots.utils import make_robot_from_config

            robot_config = WidowXAIFollowerConfig(
                ip_address=args.robot_ip,
                cameras={
                    "wrist": OpenCVCameraConfig(args.cam1, fps=args.fps, width=640, height=480),
                    "right": OpenCVCameraConfig(args.cam2, fps=args.fps, width=640, height=480),
                },
                max_relative_target=5.0,   # 单步最大位移限制（安全保护）
            )
            robot = make_robot_from_config(robot_config)
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
    action_queue = ActionQueue(rtc_config)

    logger.info(f"Task: {args.task} | FPS: {args.fps} | Duration: {args.duration}s")

    inf_thread = Thread(
        target=inference_thread_fn,
        args=(policy, robot_wrapper, preprocessor, postprocessor,
              action_queue, shutdown_event, args),
        daemon=True, name="InferenceThread",
    )
    act_thread = Thread(
        target=actor_thread_fn,
        args=(robot_wrapper, action_queue, shutdown_event, args),
        daemon=True, name="ActorThread",
    )
    inf_thread.start()
    act_thread.start()
    logger.info("Threads started.")

    # ── 5. 主线程：计时监控，到时或 Ctrl+C 后触发关闭 ───────────
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
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Ctrl+C received, shutting down...")

    # ── 6. 清理 ──────────────────────────────────────────────────
    shutdown_event.set()
    inf_thread.join(timeout=5.0)
    act_thread.join(timeout=5.0)
    if robot:
        robot.disconnect()
        logger.info("Robot disconnected.")

    # ── 7. 调试数据输出（仅 --debug 模式）───────────────────────
    rtc_processor = getattr(policy, "rtc_processor", None)
    if args.debug and rtc_processor is not None:
        debug_data = rtc_processor.get_debug_data()
        for key, val in debug_data.items():
            logger.info(f"[DEBUG] {key}: {val}")
        if _HAS_DEBUG_VISUALIZER:
            viz = RTCDebugVisualizer()
            viz.plot(debug_data)  # type: ignore[union-attr]
        else:
            logger.info("[DEBUG] RTCDebugVisualizer not available.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
