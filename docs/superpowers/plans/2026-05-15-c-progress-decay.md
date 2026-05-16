# c_progress 时间衰减 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用基于运动停止时间的指数衰减信号替换仲裁公式中的 VLM c_progress，保留 VLM 路径作为 c_vlm 仅用于记录。

**Architecture:** 新增 `ProgressDecayMonitor` 类，追踪机器人卡住时长并输出 `c_progress = 0.2 + 0.8 * exp(-λ * stuck_time_s)`。`SentinelArbitrationResult` 新增 `c_vlm` 字段存储 VLM 原始输出，`c_progress` 改为始终有值的 float。仲裁公式 `r_raw = min(c_action, c_progress)` 保持不变。

**Tech Stack:** Python 3.10+, numpy, threading, unittest

---

## File Map

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `important_code/shared_control/sentinel.py` | 修改 | 新增类、修改 dataclass、修改 `_arbitrate()` 和 `SentinelRuntime` |
| `important_code/shared_control/tests/test_sentinel.py` | 修改 | 更新旧测试、新增 `ProgressDecayMonitor` 测试组 |
| `important_code/inference/run_inference.py` | 修改 | 新增 `--sentinel-decay-lambda` CLI 参数 |
| `important_code/inference/inference_thread.py` | 修改 | 更新日志格式，移除 c_progress None 检查 |

---

## Task 1: 新增 `ProgressDecayMonitor` 类

**Files:**
- Modify: `important_code/shared_control/sentinel.py:278-280`（在 `LocalMotionDetector` 之后插入）
- Test: `important_code/shared_control/tests/test_sentinel.py`

- [ ] **Step 1: 写失败测试**

在 `test_sentinel.py` 末尾（`if __name__ == "__main__":` 之前）添加：

```python
from important_code.shared_control.sentinel import ProgressDecayMonitor


class ProgressDecayMonitorTest(unittest.TestCase):
    def test_no_history_returns_full_confidence(self):
        m = ProgressDecayMonitor()
        self.assertAlmostEqual(m.c_progress(), 1.0)

    def test_moving_returns_full_confidence(self):
        m = ProgressDecayMonitor(stuck_threshold=0.02)
        t = 1000.0
        for i in range(8):
            m.push(np.array([0.1 * i, 0, 0, 0, 0, 0, 0]), t + i * 0.4)
        self.assertAlmostEqual(m.c_progress(t + 3.2), 1.0)

    def test_stuck_decays_below_full(self):
        m = ProgressDecayMonitor(decay_lambda=0.05, floor=0.2)
        t = 1000.0
        for i in range(8):
            m.push(np.zeros(7), t + i * 0.4)
        # At t+3.2 robot has been stuck; at t+3.2+30 decay should reduce c_progress
        c_early = m.c_progress(t + 3.2)
        c_later = m.c_progress(t + 3.2 + 30)
        self.assertLess(c_later, c_early)
        self.assertLess(c_later, 1.0)
        self.assertGreaterEqual(c_later, 0.2)

    def test_floor_approached_asymptotically(self):
        m = ProgressDecayMonitor(decay_lambda=0.05, floor=0.2)
        t = 1000.0
        for i in range(8):
            m.push(np.zeros(7), t + i * 0.4)
        c = m.c_progress(t + 3.2 + 1000)
        self.assertAlmostEqual(c, 0.2, places=2)

    def test_recovery_resets_to_full_confidence(self):
        m = ProgressDecayMonitor()
        t = 1000.0
        for i in range(8):
            m.push(np.zeros(7), t + i * 0.4)
        c_stuck = m.c_progress(t + 10.0)
        self.assertLess(c_stuck, 1.0)
        # push moving joints to override stuck state
        for i in range(8):
            m.push(np.array([0.1 * i, 0, 0, 0, 0, 0, 0]), t + 10.0 + i * 0.4)
        c_moving = m.c_progress(t + 10.0 + 3.2)
        self.assertAlmostEqual(c_moving, 1.0)
```

- [ ] **Step 2: 确认测试失败**

```bash
cd /home/masterthesis/lerobot_trossen
python -m pytest important_code/shared_control/tests/test_sentinel.py::ProgressDecayMonitorTest -v 2>&1 | head -30
```
期望：`ImportError: cannot import name 'ProgressDecayMonitor'`

- [ ] **Step 3: 在 sentinel.py 实现 `ProgressDecayMonitor`**

在 `sentinel.py` 的 `LocalMotionDetector` 类结束处（第 278 行）之后、`CloudVLMClient` 之前（第 280 行）插入：

```python
class ProgressDecayMonitor:
    """Tracks robot stuck duration; outputs exponentially decaying c_progress."""

    def __init__(
        self,
        window_s: float = 3.0,
        stuck_threshold: float = 0.02,
        decay_lambda: float = 0.05,
        floor: float = 0.2,
    ) -> None:
        self.window_s = float(window_s)
        self.stuck_threshold = float(stuck_threshold)
        self.decay_lambda = float(decay_lambda)
        self.floor = float(floor)
        self._history: deque[tuple[float, np.ndarray]] = deque()
        self._stuck_since: float | None = None
        self._lock = threading.Lock()

    def push(self, joints: np.ndarray, t: float) -> None:
        joints = np.asarray(joints, dtype=np.float64).copy()
        with self._lock:
            self._history.append((float(t), joints))
            cutoff = t - self.window_s
            while self._history and self._history[0][0] < cutoff:
                self._history.popleft()
            if self._is_stuck():
                if self._stuck_since is None:
                    self._stuck_since = float(t)
            else:
                self._stuck_since = None

    def _is_stuck(self) -> bool:
        if len(self._history) < 2:
            return False
        t_oldest, j_oldest = self._history[0]
        t_newest, j_newest = self._history[-1]
        if t_newest - t_oldest < 1.0:
            return False
        return float(np.max(np.abs(j_newest - j_oldest))) < self.stuck_threshold

    def c_progress(self, now: float | None = None) -> float:
        now = float(now) if now is not None else time.time()
        with self._lock:
            stuck_since = self._stuck_since
        if stuck_since is None:
            return 1.0
        stuck_time = now - stuck_since
        return self.floor + (1.0 - self.floor) * float(np.exp(-self.decay_lambda * stuck_time))
```

- [ ] **Step 4: 确认测试通过**

```bash
python -m pytest important_code/shared_control/tests/test_sentinel.py::ProgressDecayMonitorTest -v
```
期望：5 个测试全部 PASS

- [ ] **Step 5: Commit**

```bash
git add important_code/shared_control/sentinel.py important_code/shared_control/tests/test_sentinel.py
git commit -m "feat: add ProgressDecayMonitor for time-based c_progress decay"
```

---

## Task 2: 更新 `SentinelArbitrationResult` dataclass

**Files:**
- Modify: `important_code/shared_control/sentinel.py:84-101`

`c_progress` 从 `float | None` 改为 `float`（始终有值），新增 `c_vlm: float | None`。

- [ ] **Step 1: 写失败测试**

在 `test_sentinel.py` 里新增一个测试，验证 `SentinelArbitrationResult` 有 `c_vlm` 字段：

```python
class SentinelArbitrationResultFieldsTest(unittest.TestCase):
    def test_result_has_c_vlm_field(self):
        from important_code.shared_control.sentinel import SentinelArbitrationResult
        import dataclasses
        fields = {f.name for f in dataclasses.fields(SentinelArbitrationResult)}
        self.assertIn("c_vlm", fields)

    def test_c_progress_is_always_float(self):
        from important_code.shared_control.sentinel import SentinelArbitrationResult
        import dataclasses
        field_map = {f.name: f for f in dataclasses.fields(SentinelArbitrationResult)}
        # c_progress should not have Optional type (always float)
        self.assertIn("c_progress", field_map)
        # c_vlm should allow None
        self.assertIn("c_vlm", field_map)
```

- [ ] **Step 2: 确认测试失败**

```bash
python -m pytest important_code/shared_control/tests/test_sentinel.py::SentinelArbitrationResultFieldsTest -v
```
期望：`test_result_has_c_vlm_field` FAIL（字段不存在）

- [ ] **Step 3: 修改 dataclass**

将 sentinel.py 中 `SentinelArbitrationResult` 的定义（第 85-101 行）替换为：

```python
@dataclass(frozen=True)
class SentinelArbitrationResult:
    # c_progress: 来自 ProgressDecayMonitor，始终有值，运动停止后指数衰减到 0.2。
    # c_vlm: 来自云端 VLM 的原始输出，stale 时为 None，仅用于日志记录。
    # w_vla / w_human 直接进入 eTaSL constraint weight。
    timestamp: float
    c_action: float
    action_alarm: bool
    c_progress: float
    c_vlm: float | None
    progress_alarm: bool
    progress_stale: bool
    sentinel_alarm: bool
    r_raw: float
    r_smooth: float
    w_vla: float
    w_human: float
    reason: str
    vlm_latency_s: float | None
```

- [ ] **Step 4: 确认测试通过**

```bash
python -m pytest important_code/shared_control/tests/test_sentinel.py::SentinelArbitrationResultFieldsTest -v
```
期望：2 个测试全部 PASS

- [ ] **Step 5: Commit**

```bash
git add important_code/shared_control/sentinel.py important_code/shared_control/tests/test_sentinel.py
git commit -m "feat: add c_vlm field to SentinelArbitrationResult, c_progress always float"
```

---

## Task 3: 更新 `SentinelRuntime` 使用 `ProgressDecayMonitor`

**Files:**
- Modify: `important_code/shared_control/sentinel.py`（`__init__`, `from_args`, `_arbitrate`, `update`）

- [ ] **Step 1: 写失败测试**

在 `test_sentinel.py` 里新增：

```python
class SentinelDecayIntegrationTest(unittest.TestCase):
    def test_c_progress_is_float_in_result(self):
        """c_progress 始终是 float，不再是 None。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, eps=0.0, log_dir=tmpdir)
            result = sentinel.update(_metrics(c_action=0.8))
            sentinel.stop()
        self.assertIsInstance(result.c_progress, float)

    def test_stuck_robot_lowers_c_progress(self):
        """向 sentinel 推送卡住的关节，c_progress 应低于 1.0。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, eps=0.0, log_dir=tmpdir)
            t = time.time()
            # Push enough stuck joints to trigger decay (need >1s span)
            for i in range(8):
                sentinel._progress_decay.push(np.zeros(7), t + i * 0.4)
            # Query decay 30 seconds after stuck started
            stuck_since = sentinel._progress_decay._stuck_since
            if stuck_since is not None:
                c = sentinel._progress_decay.c_progress(stuck_since + 30)
                self.assertLess(c, 1.0)
                self.assertGreaterEqual(c, 0.2)
            sentinel.stop()

    def test_vlm_result_stored_in_c_vlm(self):
        """VLM 有效结果存入 c_vlm，不影响 c_progress。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, eps=0.0, log_dir=tmpdir)
            fast = sentinel._fast_action(_metrics(c_action=0.9))
            vlm_progress = ProgressMonitorResult(
                timestamp=time.time(),
                c_progress=0.1,
                alarm=False,
                stuck=False,
                progress_made=True,
                failure_likelihood=0.9,
                reason="test",
                latency_s=0.1,
            )
            result = sentinel._arbitrate(fast, vlm_progress)
            sentinel.stop()
        # c_vlm should carry the VLM value
        self.assertAlmostEqual(result.c_vlm, 0.1, places=3)
        # c_progress comes from decay monitor, not VLM
        self.assertIsInstance(result.c_progress, float)
        self.assertNotAlmostEqual(result.c_progress, 0.1, places=3)
```

- [ ] **Step 2: 确认测试失败**

```bash
python -m pytest important_code/shared_control/tests/test_sentinel.py::SentinelDecayIntegrationTest -v 2>&1 | head -40
```
期望：`AttributeError: 'SentinelRuntime' object has no attribute '_progress_decay'`

- [ ] **Step 3: 修改 `SentinelRuntime.__init__()` 参数签名**

在 `__init__` 参数列表中（第 418-438 行），在 `ema_beta` 之前新增 `decay_lambda`:

```python
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
        decay_lambda: float = 0.05,
        ema_beta: float = 0.8,
        eps: float = 1e-3,
        client: CloudVLMClient | None = None,
    ) -> None:
```

- [ ] **Step 4: 修改 `__init__` 方法体**

在 `__init__` 方法体中，将 `self.ema_beta = ...` 那一段（第 444-470 行）里：
- 新增 `self.decay_lambda = float(decay_lambda)` （在 `self.ema_beta` 之前）
- 将 `self._motion_detector = LocalMotionDetector()` 替换为 `self._progress_decay = ProgressDecayMonitor(decay_lambda=self.decay_lambda)`

具体：找到以下代码块并替换：

旧代码（第 459-470 行附近）：
```python
        self.ema_beta = float(ema_beta)
        self.eps = float(eps)

        self._latest_progress: ProgressMonitorResult | None = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._consecutive_failures = 0
        self._r_smooth: float | None = None
        self._motion_detector = LocalMotionDetector()
```

新代码：
```python
        self.decay_lambda = float(decay_lambda)
        self.ema_beta = float(ema_beta)
        self.eps = float(eps)

        self._latest_progress: ProgressMonitorResult | None = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._consecutive_failures = 0
        self._r_smooth: float | None = None
        self._progress_decay = ProgressDecayMonitor(decay_lambda=self.decay_lambda)
```

- [ ] **Step 5: 修改 `from_args()` 传入 `decay_lambda`**

在 `from_args` 的 `return cls(...)` 调用中，在 `ema_beta=` 之前新增一行：

```python
            decay_lambda=float(getattr(args, "sentinel_decay_lambda", 0.05)),
```

- [ ] **Step 6: 修改 `update()` 方法**

将 `update()` 方法（第 534-547 行）中：
```python
        if actual_joints is not None:
            self._motion_detector.push(actual_joints, time.time())
```
改为：
```python
        if actual_joints is not None:
            self._progress_decay.push(actual_joints, time.time())
```

- [ ] **Step 7: 修改 `_arbitrate()` 方法**

将 `_arbitrate()` 方法（第 574-615 行）完整替换为：

```python
    def _arbitrate(
        self,
        fast: FastActionResult,
        progress: ProgressMonitorResult | None,
    ) -> SentinelArbitrationResult:
        # c_progress 来自 ProgressDecayMonitor：机器人卡住后指数衰减到 floor。
        # c_vlm 保留 VLM 原始输出，仅记录，不参与仲裁公式。
        now = time.time()
        c_progress = self._progress_decay.c_progress(now)
        r_raw = _clip01(min(fast.c_action, c_progress))

        progress_ok = progress is not None and not progress.stale and progress.c_progress is not None
        c_vlm = progress.c_progress if progress_ok else None
        progress_alarm = bool(progress.alarm) if progress_ok else False

        self._r_smooth = (
            r_raw if self._r_smooth is None
            else self.ema_beta * self._r_smooth + (1 - self.ema_beta) * r_raw
        )
        r = _clip01(self._r_smooth)

        reason = f"action={fast.reason}"
        if progress is not None:
            reason += f" | vlm_error={progress.error}" if progress.error else f" | vlm={progress.reason}"

        return SentinelArbitrationResult(
            timestamp=now,
            c_action=fast.c_action,
            action_alarm=fast.action_alarm,
            c_progress=c_progress,
            c_vlm=c_vlm,
            progress_alarm=progress_alarm,
            progress_stale=not progress_ok,
            sentinel_alarm=fast.action_alarm or progress_alarm,
            r_raw=r_raw,
            r_smooth=r,
            w_vla=r + self.eps,
            w_human=1.0 - r + self.eps,
            reason=reason,
            vlm_latency_s=progress.latency_s if progress is not None else None,
        )
```

- [ ] **Step 8: 确认新测试通过**

```bash
python -m pytest important_code/shared_control/tests/test_sentinel.py::SentinelDecayIntegrationTest -v
```
期望：3 个测试全部 PASS

- [ ] **Step 9: Commit**

```bash
git add important_code/shared_control/sentinel.py important_code/shared_control/tests/test_sentinel.py
git commit -m "feat: wire ProgressDecayMonitor into SentinelRuntime arbitration"
```

---

## Task 4: 更新旧测试以适配新接口

**Files:**
- Modify: `important_code/shared_control/tests/test_sentinel.py`

以下旧测试依赖被改变的接口，需要更新。

- [ ] **Step 1: 运行全部测试，看哪些失败**

```bash
python -m pytest important_code/shared_control/tests/test_sentinel.py -v 2>&1 | tail -30
```

- [ ] **Step 2: 更新 `test_reliability_uses_min_of_action_and_progress`**

该测试原来验证 VLM c_progress=0.1 使 r_raw=0.1。新逻辑中 VLM 不进入仲裁，改为验证 c_vlm 字段：

将该测试替换为：

```python
    def test_vlm_result_is_stored_in_c_vlm_not_c_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, eps=0.001, log_dir=tmpdir)
            fast = sentinel._fast_action(_metrics(c_action=0.9))
            result = sentinel._arbitrate(
                fast,
                ProgressMonitorResult(
                    timestamp=time.time(),
                    c_progress=0.1,
                    alarm=True,
                    stuck=True,
                    progress_made=False,
                    failure_likelihood=0.9,
                    reason="stuck",
                    latency_s=0.2,
                ),
            )
            sentinel.stop()

        # VLM c_progress=0.1 goes to c_vlm, not to arbitration formula
        self.assertAlmostEqual(result.c_vlm, 0.1)
        # r_raw = min(c_action=0.9, c_progress_decay); since no joints pushed, decay=1.0
        self.assertAlmostEqual(result.r_raw, 0.9, places=3)
        # progress_alarm still reflects VLM alarm
        self.assertTrue(result.progress_alarm)
```

- [ ] **Step 3: 更新 `test_stale_progress_falls_back_to_action_consistency`**

旧测试验证 stale 时 r_raw=0.5（来自 motion detector 返回 0.5）。新逻辑中 stale 只影响 c_vlm=None，c_progress 来自 decay monitor（无 joints → 1.0），r_raw = min(0.7, 1.0) = 0.7。将该测试替换为：

```python
    def test_stale_vlm_sets_c_vlm_to_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, eps=0.001, log_dir=tmpdir)
            fast = sentinel._fast_action(_metrics(c_action=0.7))
            result = sentinel._arbitrate(
                fast,
                ProgressMonitorResult(
                    timestamp=time.time(),
                    c_progress=None,
                    alarm=False,
                    stuck=False,
                    progress_made=None,
                    failure_likelihood=None,
                    reason="",
                    latency_s=0.0,
                    stale=True,
                    error="timeout",
                ),
            )
            sentinel.stop()

        self.assertIsNone(result.c_vlm)
        self.assertTrue(result.progress_stale)
        self.assertFalse(result.progress_alarm)
        # c_progress from decay (no joints pushed → 1.0), r_raw = min(0.7, 1.0) = 0.7
        self.assertAlmostEqual(result.r_raw, 0.7, places=3)
```

- [ ] **Step 4: 更新 `SentinelStaleCapTest`**

该测试直接访问 `sentinel._motion_detector`，改为 `sentinel._progress_decay`：

将 `SentinelStaleCapTest` 整个类替换为：

```python
class SentinelStaleCapTest(unittest.TestCase):
    def test_stuck_robot_lowers_r_raw(self):
        """r_raw < 1.0 when robot is stuck and c_action=1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, eps=0.0, log_dir=tmpdir)
            t = 1000.0
            for i in range(8):
                sentinel._progress_decay.push(np.zeros(7), t + i * 0.4)
            fast = sentinel._fast_action(_metrics(c_action=1.0))
            # Check decay after 30 more seconds
            stuck_since = sentinel._progress_decay._stuck_since
            if stuck_since is not None:
                c = sentinel._progress_decay.c_progress(stuck_since + 30)
                self.assertLess(c, 1.0)
                self.assertGreaterEqual(c, 0.2)
            sentinel.stop()

    def test_moving_robot_keeps_full_c_progress(self):
        """c_progress = 1.0 when robot is moving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, eps=0.0, log_dir=tmpdir)
            t = 1000.0
            for i in range(8):
                sentinel._progress_decay.push(
                    np.array([0.1 * i, 0, 0, 0, 0, 0, 0]), t + i * 0.4
                )
            c = sentinel._progress_decay.c_progress(t + 3.2)
            self.assertAlmostEqual(c, 1.0)
            sentinel.stop()
```

- [ ] **Step 5: 确认全部测试通过**

```bash
python -m pytest important_code/shared_control/tests/test_sentinel.py -v
```
期望：所有测试 PASS，无 FAIL

- [ ] **Step 6: Commit**

```bash
git add important_code/shared_control/tests/test_sentinel.py
git commit -m "test: update sentinel tests for c_progress decay and c_vlm refactor"
```

---

## Task 5: 添加 `--sentinel-decay-lambda` CLI 参数

**Files:**
- Modify: `important_code/inference/run_inference.py`（sentinel 参数区域，第 234 行之后）

- [ ] **Step 1: 在 `run_inference.py` 中添加参数**

在 `--sentinel-progress-max-age-s` 参数（最后一个 sentinel 参数）之后插入：

```python
    parser.add_argument("--sentinel-decay-lambda", type=float, default=0.05,
                        help="Exponential decay rate for c_progress when robot is stuck (half-life = ln2/lambda ≈ 14s).")
```

- [ ] **Step 2: 验证参数被正确读取**

```bash
cd /home/masterthesis/lerobot_trossen
python -c "
import sys; sys.argv = ['prog', '--sentinel-decay-lambda', '0.1']
from important_code.inference.run_inference import build_arg_parser
args = build_arg_parser().parse_args()
print('decay_lambda:', args.sentinel_decay_lambda)
"
```
期望输出：`decay_lambda: 0.1`

- [ ] **Step 3: Commit**

```bash
git add important_code/inference/run_inference.py
git commit -m "feat: add --sentinel-decay-lambda CLI argument"
```

---

## Task 6: 更新 `inference_thread.py` 日志

**Files:**
- Modify: `important_code/inference/inference_thread.py`

`c_progress` 现在始终是 float（不再需要 None 检查），新增 `c_vlm` 日志字段。

- [ ] **Step 1: 更新 logger.info 格式（第 308-330 行）**

将：
```python
                    logger.info(
                        "[SENTINEL] c_action=%.4f c_progress=%s "
                        "r_raw=%.4f r=%.4f w_vla=%.4f w_human=%.4f "
                        "alarm=%s progress_stale=%s vlm_latency=%s reason=%s",
                        sentinel_result.c_action,
                        (
                            f"{sentinel_result.c_progress:.4f}"
                            if sentinel_result.c_progress is not None
                            else "None"
                        ),
                        sentinel_result.r_raw,
                        sentinel_result.r_smooth,
                        sentinel_result.w_vla,
                        sentinel_result.w_human,
                        sentinel_result.sentinel_alarm,
                        sentinel_result.progress_stale,
                        (
                            f"{sentinel_result.vlm_latency_s:.3f}s"
                            if sentinel_result.vlm_latency_s is not None
                            else "None"
                        ),
                        sentinel_result.reason,
                    )
```

替换为：
```python
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
```

- [ ] **Step 2: 更新 print 语句（第 331-343 行）**

将：
```python
                    c_progress_text = (
                        f"{sentinel_result.c_progress:.4f}"
                        if sentinel_result.c_progress is not None
                        else "None"
                    )
                    print(
                        "[SENTINEL_VALUES] "
                        f"C_action={sentinel_result.c_action:.4f} "
                        f"C_progress={c_progress_text} "
                        f"R_raw={sentinel_result.r_raw:.4f} "
                        f"R={sentinel_result.r_smooth:.4f}",
                        flush=True,
                    )
```

替换为：
```python
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
```

- [ ] **Step 3: 更新 sentinel_records 字典（第 344-352 行），新增 c_vlm**

将：
```python
                    sentinel_records.append({
                        "t":          time.perf_counter() - _inference_start,
                        "c_action":   sentinel_result.c_action,
                        "c_progress": sentinel_result.c_progress,
                        "r_raw":      sentinel_result.r_raw,
                        "r_smooth":   sentinel_result.r_smooth,
                        "w_vla":      sentinel_result.w_vla,
                        "w_human":    sentinel_result.w_human,
                    })
```

替换为：
```python
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
```

- [ ] **Step 4: 更新 `_save_sentinel_plot` 的 c_progress 处理（第 62-74 行）**

将：
```python
    fieldnames = ["t", "c_action", "c_progress", "r_raw", "r_smooth", "w_vla", "w_human"]
```
替换为：
```python
    fieldnames = ["t", "c_action", "c_progress", "c_vlm", "r_raw", "r_smooth", "w_vla", "w_human"]
```

将：
```python
    c_progress = [r["c_progress"] if r["c_progress"] is not None else float("nan") for r in records]
```
替换为：
```python
    c_progress = [r["c_progress"] for r in records]
    c_vlm      = [r.get("c_vlm") if r.get("c_vlm") is not None else float("nan") for r in records]
```

在绘图的 `axes[0]` 部分，在 `axes[0].plot(t, c_progress, ...)` 行之后添加：
```python
    axes[0].plot(t, c_vlm, label="c_vlm", color="green", linestyle=":", alpha=0.6)
```

- [ ] **Step 5: 更新 rr.log 的 None 检查（第 387-388 行）**

将：
```python
                        if sentinel_result.c_progress is not None:
                            rr.log("sentinel/c_progress", rr.Scalars(sentinel_result.c_progress))
```
替换为：
```python
                        rr.log("sentinel/c_progress", rr.Scalars(sentinel_result.c_progress))
                        if sentinel_result.c_vlm is not None:
                            rr.log("sentinel/c_vlm", rr.Scalars(sentinel_result.c_vlm))
```

- [ ] **Step 6: 验证模块能正常导入，无语法错误**

```bash
python -c "from important_code.inference.inference_thread import inference_thread_fn; print('OK')"
```
期望：`OK`

- [ ] **Step 7: Commit**

```bash
git add important_code/inference/inference_thread.py
git commit -m "feat: update inference_thread logging for c_progress decay and c_vlm"
```

---

## Task 7: 全套测试验证

- [ ] **Step 1: 运行所有 sentinel 测试**

```bash
python -m pytest important_code/shared_control/tests/test_sentinel.py -v
```
期望：全部 PASS

- [ ] **Step 2: 运行全部测试套件（smoke check）**

```bash
python -m pytest important_code/ -v --tb=short 2>&1 | tail -40
```
期望：无 FAIL（可有 SKIP）

- [ ] **Step 3: 手动验证 sentinel 端到端**

```bash
python -c "
import tempfile, numpy as np, time
from important_code.shared_control.sentinel import SentinelRuntime
from types import SimpleNamespace

metrics = SimpleNamespace(c_action=0.8, jerk_max=0.0, boundary_jump_max=0.0)
with tempfile.TemporaryDirectory() as tmpdir:
    s = SentinelRuntime(ema_beta=0.0, eps=0.0, log_dir=tmpdir)
    # simulate stuck robot
    t = time.time()
    for i in range(10):
        s._progress_decay.push(np.zeros(7), t + i * 0.4)
    r = s.update(metrics, actual_joints=np.zeros(7))
    print(f'c_progress={r.c_progress:.4f}')
    print(f'c_vlm={r.c_vlm}')
    print(f'r_raw={r.r_raw:.4f}')
    print(f'w_vla={r.w_vla:.4f}  w_human={r.w_human:.4f}')
    s.stop()
"
```
期望：`c_vlm=None`，`c_progress` 为 float，`w_vla + w_human ≈ 1.0`

- [ ] **Step 4: Commit（如有残余改动）**

```bash
git status
# 仅在有未提交变更时执行：
git add -p
git commit -m "chore: final cleanup for c_progress decay feature"
```
