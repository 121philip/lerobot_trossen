"""RTC-specific runtime setup for live inference."""

import logging

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)

try:
    from lerobot.policies.rtc.debug_visualizer import RTCDebugVisualizer

    _HAS_DEBUG_VISUALIZER = True
except ImportError:
    _HAS_DEBUG_VISUALIZER = False


def build_rtc_config(args) -> RTCConfig:
    rtc_config = RTCConfig(
        enabled=bool(args.rtc),
        execution_horizon=args.execution_horizon,
        max_guidance_weight=args.guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule[args.attention_schedule],
    )
    if args.debug:
        rtc_config.debug = True
        rtc_config.debug_maxlen = args.debug_maxlen
    return rtc_config


def configure_policy_rtc(policy, args) -> RTCConfig:
    rtc_config = build_rtc_config(args)
    policy.config.rtc_config = rtc_config if rtc_config.enabled else None
    policy.init_rtc_processor()
    return rtc_config


def make_action_queue(rtc_config: RTCConfig) -> ActionQueue:
    return ActionQueue(rtc_config)


def maybe_plot_rtc_debug(policy, args) -> None:
    rtc_processor = getattr(policy, "rtc_processor", None)
    if not args.debug or rtc_processor is None:
        return

    debug_data = rtc_processor.get_debug_data()
    for key, val in debug_data.items():
        logger.info("[DEBUG] %s: %s", key, val)
    if _HAS_DEBUG_VISUALIZER:
        viz = RTCDebugVisualizer()
        viz.plot(debug_data)
    else:
        logger.info("[DEBUG] RTCDebugVisualizer not available.")
