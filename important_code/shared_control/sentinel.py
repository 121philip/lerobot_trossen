"""Compact Sentinel-style monitor for VLA shared control.

Fast path:  read existing CBC/jerk metrics once per action chunk.
Slow path:  ask a cloud VLM whether recent camera frames show task progress.
Output:     direct eTaSL weights, w_vla and w_human.

The cloud VLM runs in a background thread.  If it fails or gets stale, the
arbiter falls back to the fast action score, so SmolVLA inference never waits.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Slow monitor 发给云端 VLM 的固定问题模板。
# 注意：VLM 只判断“任务有没有视觉进展”，不估计葡萄 3D 坐标，也不做相机标定。
PROMPT_TEMPLATE = """Task: {task}.

You are monitoring a robot manipulation rollout from recent camera frames.
The task is picking the grape, so the robot should reach toward the grape, grasp it, and lift it up.
Decide whether the robot is visibly making progress toward completing the task.

Return JSON only:
{{
  "progress_made": true/false,
  "stuck": true/false,
  "failure_likelihood": 0.0-1.0,
  "reason": "short explanation"
}}
"""


@dataclass(frozen=True)
class SentinelFrame:
    timestamp: float
    wrist: np.ndarray | None
    right: np.ndarray | None


@dataclass
class ProgressMonitorResult:
    # c_progress 越高，表示 VLM 越认为任务正在正常推进。
    # 如果 VLM 超时、断网或 JSON 解析失败，则 c_progress=None 且 stale=True。
    timestamp: float
    c_progress: float | None
    alarm: bool
    stuck: bool
    progress_made: bool | None
    failure_likelihood: float | None
    reason: str
    latency_s: float
    stale: bool = False
    error: str | None = None
    consecutive_alarm_count: int = 0


@dataclass(frozen=True)
class FastActionResult:
    # c_action 来自本地 action chunk 指标，例如 Regression-CBC、jerk、boundary jump。
    # 它只说明“动作是否平滑/一致”，不说明任务是否真的完成。
    c_action: float
    action_alarm: bool
    reason: str


@dataclass(frozen=True)
class SentinelArbitrationResult:
    # r_raw 是本轮未平滑的 VLA 可靠度；r_smooth 是 EMA 后的可靠度。
    # w_vla / w_human 会直接进入 eTaSL constraint weight。
    timestamp: float
    c_action: float
    action_alarm: bool
    c_progress: float | None
    progress_alarm: bool
    progress_stale: bool
    sentinel_alarm: bool
    r_raw: float
    r_smooth: float
    w_vla: float
    w_human: float
    reason: str
    vlm_latency_s: float | None


def _clip01(value: Any) -> float:
    # 所有置信度/可靠度都压到 [0, 1]，避免异常数值进入权重计算。
    return float(np.clip(float(value), 0.0, 1.0))


def _maybe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _json_object(text: str) -> dict[str, Any]:
    # 云端模型偶尔会把 JSON 包进 ```json ... ```，这里先剥掉外壳再解析。
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"No JSON object in VLM response: {text[:200]!r}")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("VLM response JSON is not an object")
    return parsed


def _stale(error: str, latency_s: float = 0.0) -> ProgressMonitorResult:
    # stale 表示 slow monitor 本轮不可用；主控制循环会退化为只相信 C_action。
    return ProgressMonitorResult(
        timestamp=time.time(),
        c_progress=None,
        alarm=False,
        stuck=False,
        progress_made=None,
        failure_likelihood=None,
        reason="",
        latency_s=latency_s,
        stale=True,
        error=error,
    )


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    # LeRobot/相机可能给 HWC、CHW、灰度或 float 图像；统一转成 uint8 RGB。
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dims, got {arr.shape}")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        scale = 255.0 if np.nanmax(arr) <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return arr


def _resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    return np.asarray(
        Image.fromarray(_as_uint8_rgb(image)).resize(size, Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def _jpeg_b64(image: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(_as_uint8_rgb(image)).save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class SentinelFrameBuffer:
    """Thread-safe cache of recent wrist/right frames."""

    def __init__(self, max_age_s: float = 8.0, maxlen: int = 240) -> None:
        self.max_age_s = float(max_age_s)
        self._frames: deque[SentinelFrame] = deque(maxlen=int(maxlen))
        self._lock = threading.Lock()

    def push(
        self,
        wrist: np.ndarray | None,
        right: np.ndarray | None,
        timestamp: float | None = None,
    ) -> None:
        # 推理线程每次拿到 observation 后调用这里。
        # 这里复制图像，避免后续 LeRobot 或相机缓冲区复用内存时污染历史帧。
        if wrist is None and right is None:
            return
        now = time.time() if timestamp is None else float(timestamp)
        frame = SentinelFrame(
            timestamp=now,
            wrist=np.array(wrist, copy=True) if wrist is not None else None,
            right=np.array(right, copy=True) if right is not None else None,
        )
        with self._lock:
            self._frames.append(frame)
            # 只保留最近 max_age_s 秒，防止 slow monitor 看到太旧的画面。
            cutoff = now - self.max_age_s
            while self._frames and self._frames[0].timestamp < cutoff:
                self._frames.popleft()

    def push_observation(self, observation: dict[str, Any]) -> None:
        self.push(
            wrist=observation.get("observation.images.wrist", observation.get("wrist")),
            right=observation.get("observation.images.right", observation.get("right")),
        )

    def sample_window(self, window_s: float = 4.0, max_frames: int = 6) -> list[SentinelFrame]:
        # 从最近 window_s 秒里均匀抽 max_frames 帧。
        # 这样 VLM 看到的是“最近一小段过程”，而不是单张静态图。
        cutoff = time.time() - float(window_s)
        with self._lock:
            frames = [frame for frame in self._frames if frame.timestamp >= cutoff]
        if len(frames) <= max_frames:
            return frames
        picks = np.linspace(0, len(frames) - 1, int(max_frames), dtype=int)
        return [frames[int(i)] for i in picks]

    def make_grid_jpeg_base64(
        self,
        window_s: float = 4.0,
        max_frames: int = 6,
        cell_size: tuple[int, int] = (320, 240),
    ) -> str | None:
        # 把每个时刻的 right/wrist 拼成一行：
        # [right_t, wrist_t]
        # 多个时刻再纵向拼成 grid，最后 JPEG + base64 发给云端 VLM。
        frames = self.sample_window(window_s, max_frames)
        if not frames:
            return None
        blank = np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)
        rows = []
        for frame in frames:
            right = _resize(frame.right, cell_size) if frame.right is not None else blank
            wrist = _resize(frame.wrist, cell_size) if frame.wrist is not None else blank
            rows.append(np.concatenate([right, wrist], axis=1))
        return _jpeg_b64(np.concatenate(rows, axis=0))


class LocalMotionDetector:
    """Estimates whether the robot is moving, used as stale c_progress fallback."""

    def __init__(self, window_s: float = 3.0, stuck_threshold: float = 0.02) -> None:
        self.window_s = float(window_s)
        self.stuck_threshold = float(stuck_threshold)
        self._history: deque[tuple[float, np.ndarray]] = deque()
        self._lock = threading.Lock()

    def push(self, joints: np.ndarray, t: float) -> None:
        joints = np.asarray(joints, dtype=np.float64).copy()
        with self._lock:
            self._history.append((float(t), joints))
            cutoff = t - self.window_s
            while self._history and self._history[0][0] < cutoff:
                self._history.popleft()

    def c_progress_local(self) -> float:
        with self._lock:
            if len(self._history) < 2:
                return 0.5
            t_oldest, j_oldest = self._history[0]
            t_newest, j_newest = self._history[-1]
            if t_newest - t_oldest < 1.0:
                return 0.5
            displacement = float(np.max(np.abs(j_newest - j_oldest)))
        return 0.2 if displacement < self.stuck_threshold else 0.5


class CloudVLMClient:
    """Minimal OpenAI/Gemini client using urllib."""

    def __init__(self, provider: str, model: str, api_key: str | None = None) -> None:
        self.provider = provider.lower()
        self.model = model
        env_name = "OPENAI_API_KEY" if self.provider == "openai" else "GEMINI_API_KEY"
        self.api_key = api_key or os.environ.get(env_name)

    def classify_progress(self, prompt: str, image_b64: str, timeout_s: float) -> ProgressMonitorResult:
        # slow monitor 的核心：把最近相机 grid 交给 VLM，让它只回答任务进展。
        # 返回的 failure_likelihood 越高，说明越可能卡住或失败。
        start = time.perf_counter()
        try:
            if not self.api_key:
                raise RuntimeError(f"Missing API key for {self.provider}")
            text = (
                self._call_gemini(prompt, image_b64, timeout_s)
                if self.provider == "gemini"
                else self._call_openai(prompt, image_b64, timeout_s)
            )
            parsed = _json_object(text)
            failure = _clip01(parsed.get("failure_likelihood", 1.0))
            # 论文里的 progress score：
            #   C_progress = 1 - failure_likelihood
            # 例如 failure=0.8，则 C_progress=0.2，说明任务进展很差。
            return ProgressMonitorResult(
                timestamp=time.time(),
                c_progress=1.0 - failure,
                alarm=False,
                stuck=bool(parsed.get("stuck", failure > 0.7)),
                progress_made=None if parsed.get("progress_made") is None else bool(parsed["progress_made"]),
                failure_likelihood=failure,
                reason=str(parsed.get("reason", "")).strip(),
                latency_s=time.perf_counter() - start,
            )
        except Exception as exc:
            # 所有云端异常都吞掉并标记 stale，绝不阻塞 SmolVLA 推理线程。
            return _stale(str(exc), time.perf_counter() - start)

    def _call_openai(self, prompt: str, image_b64: str, timeout_s: float) -> str:
        # OpenAI Responses API：同一个 user message 里放文字 prompt 和 JPEG data URL。
        data = self._post_json(
            "https://api.openai.com/v1/responses",
            {
                "model": self.model,
                "temperature": 0,
                "max_output_tokens": 300,
                "input": [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                    ],
                }],
            },
            {"Authorization": f"Bearer {self.api_key}"},
            timeout_s,
        )
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        texts = [
            content["text"]
            for output in data.get("output", [])
            for content in output.get("content", [])
            if isinstance(content.get("text"), str)
        ]
        if not texts:
            raise ValueError("OpenAI response did not contain text")
        return "\n".join(texts)

    def _call_gemini(self, prompt: str, image_b64: str, timeout_s: float) -> str:
        # Gemini REST API：图片作为 inline_data，要求 responseMimeType=application/json。
        data = self._post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            {
                "contents": [{
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                        {"text": prompt},
                    ]
                }],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
            },
            {"x-goog-api-key": str(self.api_key)},
            timeout_s,
        )
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        texts = [part["text"] for part in parts if isinstance(part.get("text"), str)]
        if not texts:
            raise ValueError("Gemini response did not contain text")
        return "\n".join(texts)

    @staticmethod
    def _post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc


class SentinelRuntime:
    """Single object used by inference_thread.py."""

    def __init__(
        self,
        task: str = "pick the grape",
        provider: str = "openai",
        model: str = "gpt-4o",
        log_dir: str | Path = "outputs/sentinel",
        log_only: bool = True,
        interval_s: float = 2.0,
        timeout_s: float = 5.0,
        window_s: float = 4.0,
        max_frames: int = 6,
        progress_max_age_s: float = 8.0,
        failure_threshold: float = 0.7,
        required_alarm_count: int = 2,
        tau_action: float = 0.4,
        jerk_max: float | None = None,
        boundary_jump_max: float | None = None,
        ema_beta: float = 0.8,
        eps: float = 1e-3,
        client: CloudVLMClient | None = None,
    ) -> None:
        # 这些参数基本对应 CLI：
        # interval/timeout/window 控制 slow VLM；
        # tau_action/jerk/boundary 控制 fast action alarm；
        # ema_beta/eps 控制最终权重平滑和避免 0 权重。
        self.frame_buffer = SentinelFrameBuffer(max_age_s=progress_max_age_s)
        self.client = client or CloudVLMClient(provider, model)
        self.task = task
        self.provider = provider
        self.model = model
        self.log_only = bool(log_only)
        self.interval_s = float(interval_s)
        self.timeout_s = float(timeout_s)
        self.window_s = float(window_s)
        self.max_frames = int(max_frames)
        self.progress_max_age_s = float(progress_max_age_s)
        self.failure_threshold = float(failure_threshold)
        self.required_alarm_count = int(required_alarm_count)
        self.tau_action = float(tau_action)
        self.jerk_max = jerk_max
        self.boundary_jump_max = boundary_jump_max
        self.ema_beta = float(ema_beta)
        self.eps = float(eps)

        self._latest_progress: ProgressMonitorResult | None = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._consecutive_failures = 0
        self._r_smooth: float | None = None
        self._motion_detector = LocalMotionDetector()

        # 每次实验创建一个时间戳目录，便于论文统计 VLM latency、报警和权重曲线。
        self.log_dir = Path(log_dir) / time.strftime("%Y%m%d-%H%M%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.log_dir / "sentinel_events.jsonl").open("a", encoding="utf-8")
        self._csv_file = (self.log_dir / "sentinel_events.csv").open("a", newline="", encoding="utf-8")
        self._csv_writer: csv.DictWriter | None = None

    @classmethod
    def from_args(cls, args: Any) -> "SentinelRuntime":
        provider = getattr(args, "sentinel_vlm_provider", "openai")
        model = getattr(args, "sentinel_vlm_model", None)
        if not model:
            model = "gpt-4o" if provider == "openai" else "gemini-3-flash-preview"
        return cls(
            task=getattr(args, "task", "pick the grape"),
            provider=provider,
            model=model,
            log_dir=getattr(args, "sentinel_log_dir", "outputs/sentinel"),
            log_only=bool(getattr(args, "sentinel_log_only", True)),
            interval_s=float(getattr(args, "sentinel_interval_s", 2.0)),
            timeout_s=float(getattr(args, "sentinel_timeout_s", 5.0)),
            window_s=float(getattr(args, "sentinel_window_s", 4.0)),
            max_frames=int(getattr(args, "sentinel_max_frames", 6)),
            progress_max_age_s=float(getattr(args, "sentinel_progress_max_age_s", 8.0)),
            failure_threshold=float(getattr(args, "sentinel_progress_threshold", 0.7)),
            required_alarm_count=int(getattr(args, "sentinel_progress_alarm_count", 2)),
            tau_action=float(getattr(args, "sentinel_action_tau", 0.4)),
            jerk_max=_maybe_float(getattr(args, "sentinel_jerk_max", None)),
            boundary_jump_max=_maybe_float(getattr(args, "sentinel_boundary_jump_max", None)),
            ema_beta=float(getattr(args, "sentinel_ema_beta", 0.8)),
            eps=float(getattr(args, "sentinel_weight_eps", 1e-3)),
        )

    def start(self) -> None:
        # slow monitor 放后台线程运行；主推理线程只读 latest_progress，不等待 VLM。
        if self._thread is None:
            self._thread = threading.Thread(target=self._slow_loop, daemon=True, name="SentinelVLM")
            self._thread.start()
        logger.info(
            "[SENTINEL] enabled provider=%s model=%s log_only=%s log_dir=%s",
            self.provider,
            self.model,
            self.log_only,
            self.log_dir,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._jsonl.close()
        self._csv_file.close()

    def push_observation(self, observation: dict[str, Any]) -> None:
        # inference_thread 在 policy preprocess 前后都能调用；这里只取 wrist/right 图像。
        self.frame_buffer.push_observation(observation)

    def update(
        self,
        confidence_metrics: Any,
        actual_joints: np.ndarray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> SentinelArbitrationResult:
        # 每个 action chunk 调一次：
        # 1. 从本地 metrics 算 C_action；
        # 2. 读取最近一次 VLM 的 C_progress；
        # 3. 融合成 w_vla/w_human 并写日志。
        if actual_joints is not None:
            self._motion_detector.push(actual_joints, time.time())
        result = self._arbitrate(self._fast_action(confidence_metrics), self._fresh_progress())
        self._log(result, extra or {})
        return result

    def _fast_action(self, metrics: Any) -> FastActionResult:
        # C_action 来自 confidence_estimator.update()，模式由 --sentinel-confidence-mode 控制。
        # 额外的 jerk/boundary_jump 阈值只负责触发 action_alarm，不直接改 C_action。
        c_action = _clip01(metrics.c_action)
        reasons = []
        if c_action < self.tau_action:
            reasons.append(f"c_action<{self.tau_action:.3f}")
        if self.jerk_max is not None and metrics.jerk_max > self.jerk_max:
            reasons.append(f"jerk_max>{self.jerk_max:.3f}")
        if self.boundary_jump_max is not None and metrics.boundary_jump_max > self.boundary_jump_max:
            reasons.append(f"boundary_jump>{self.boundary_jump_max:.3f}")
        return FastActionResult(c_action, bool(reasons), ";".join(reasons) or "ok")

    def _fresh_progress(self) -> ProgressMonitorResult | None:
        # slow monitor 比 action chunk 慢，所以这里读取“最近一次”结果。
        # 如果结果太旧，就标记 stale，让仲裁退化为只用 C_action。
        with self._latest_lock:
            progress = self._latest_progress
        if progress is None or time.time() - progress.timestamp <= self.progress_max_age_s:
            return progress
        stale = _stale("progress result is stale", progress.latency_s)
        stale.consecutive_alarm_count = progress.consecutive_alarm_count
        return stale

    def _arbitrate(
        self,
        fast: FastActionResult,
        progress: ProgressMonitorResult | None,
    ) -> SentinelArbitrationResult:
        # 核心仲裁公式：
        #   如果 VLM 结果新鲜：R_raw = min(C_action, C_progress)
        #   如果 VLM 结果 stale：R_raw = C_action
        # 取 min 的原因：动作平滑和任务进展必须同时成立，VLA 才算可靠。
        progress_ok = progress is not None and not progress.stale and progress.c_progress is not None
        c_progress = progress.c_progress if progress_ok else None
        progress_alarm = bool(progress.alarm) if progress_ok else False
        if c_progress is not None:
            r_raw = _clip01(min(fast.c_action, c_progress))
        else:
            c_fallback = self._motion_detector.c_progress_local()
            r_raw = _clip01(min(fast.c_action, c_fallback))
        # EMA 平滑，避免一次 VLM 判断抖动导致 eTaSL 权重瞬间跳变。
        self._r_smooth = r_raw if self._r_smooth is None else self.ema_beta * self._r_smooth + (1 - self.ema_beta) * r_raw
        r = _clip01(self._r_smooth)

        reason = f"action={fast.reason}"
        if progress is not None:
            reason += f" | progress_error={progress.error}" if progress.error else f" | progress={progress.reason}"

        return SentinelArbitrationResult(
            timestamp=time.time(),
            c_action=fast.c_action,
            action_alarm=fast.action_alarm,
            c_progress=c_progress,
            progress_alarm=progress_alarm,
            progress_stale=not progress_ok,
            sentinel_alarm=fast.action_alarm or progress_alarm,
            r_raw=r_raw,
            r_smooth=r,
            # 直接输出 eTaSL 权重，不再绕 alpha。
            # eps 防止某个 soft constraint 权重严格为 0。
            w_vla=r + self.eps,
            w_human=1.0 - r + self.eps,
            reason=reason,
            vlm_latency_s=progress.latency_s if progress is not None else None,
        )

    def _slow_loop(self) -> None:
        # 后台循环：每 interval_s 秒做一次 VLM progress check。
        # 若 VLM 调用很慢，下一轮会自动延后，不会堆积多个 VLM 请求。
        prompt = PROMPT_TEMPLATE.format(task=self.task)
        while not self._stop.is_set():
            started = time.monotonic()
            progress = self._check_progress(prompt)
            with self._latest_lock:
                self._latest_progress = progress
            self._stop.wait(max(0.1, self.interval_s - (time.monotonic() - started)))

    def _check_progress(self, prompt: str) -> ProgressMonitorResult:
        # 取最近几秒图像 -> grid -> VLM -> ProgressMonitorResult。
        try:
            image_b64 = self.frame_buffer.make_grid_jpeg_base64(self.window_s, self.max_frames)
        except Exception as exc:
            return _stale(f"camera grid encoding failed: {exc}")
        if image_b64 is None:
            return _stale("no camera frames available")

        progress = self.client.classify_progress(prompt, image_b64, self.timeout_s)
        if progress.error:
            self._consecutive_failures = 0
            return progress

        # VLM 一次说 stuck 还不立刻报警；必须连续 required_alarm_count 次失败。
        # 这样降低云端模型偶然误判带来的误报。
        raw_alarm = progress.stuck or (
            progress.failure_likelihood is not None
            and progress.failure_likelihood > self.failure_threshold
        )
        self._consecutive_failures = self._consecutive_failures + 1 if raw_alarm else 0
        progress.consecutive_alarm_count = self._consecutive_failures
        progress.alarm = self._consecutive_failures >= self.required_alarm_count
        return progress

    def _log(self, result: SentinelArbitrationResult, extra: dict[str, Any]) -> None:
        # JSONL 保留完整字段，CSV 只保留标量字段，方便后续画图/统计。
        row = asdict(result)
        row.update(extra)
        self._jsonl.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")
        self._jsonl.flush()

        flat = {k: v for k, v in row.items() if isinstance(v, (str, int, float, bool)) or v is None}
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(flat.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(flat)
        self._csv_file.flush()
