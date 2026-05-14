# Sentinel Score Deflation — Design Spec

**Date:** 2026-05-14  
**Status:** Approved, pending implementation  
**Scope:** Targeted refactor (B-level) — no architectural changes, 4 files

---

## Problem Summary

Data from `outputs/sentinel/` shows two systematic score-inflation failures:

1. **c_action saturates to ≥0.95 within 2 chunks and never drops** — Regression-CBC measures chunk-boundary smoothness, which SmolVLA guarantees by design. A robot moving smoothly in the wrong direction scores c_action ≈ 1.0.

2. **c_progress is unreliable in three ways:**
   - Gemini latency 3–10 s exceeds the 2 s monitor interval → stale cascade
   - VLM outputs only 3 discrete levels (0.1 / 0.5 / 0.9) — not a continuous signal
   - VLM is optimistically biased; it reads "arm moving" as "progress" regardless of direction

3. **Stale fallback = c_action ≈ 1.0** — when Gemini is unavailable, `r_raw` collapses to c_action, and EMA (β=0.8) locks r_smooth near 1.0, driving w_human → 0.

---

## Goals

- c_action must reflect **task-space tracking**, not just trajectory smoothness
- c_progress must be a **continuous float** and have a **meaningful stale fallback**
- Support **DeepSeek** as a third VLM provider
- All three c_action modes selectable at runtime for ablation experiments

---

## Files Changed (4 total)

| File | What changes |
|------|-------------|
| `important_code/shared_control/confidence.py` | Rename class, add tracking-error mode, 3-mode selector |
| `important_code/shared_control/sentinel.py` | LocalMotionDetector, prompt fix, DeepSeek, EMA β param |
| `important_code/inference/inference_thread.py` | Pass `actual_joints` + `delay_steps` to estimator and sentinel |
| `important_code/inference/run_inference.py` | 3 new CLI args |

---

## Design

### 1. Data Flow

```
rviz_publisher.get_latest_joints()  →  [7,] actual joint positions (rad)
        │
        ├──→  ConfidenceEstimator.update(chunk, actual_joints, delay_steps)
        │           └──→  c_tracking, ConfidenceMetrics.c_action
        │
        └──→  SentinelRuntime.update(..., actual_joints=...)
                    └──→  LocalMotionDetector.push(joints, t)
                                └──→  c_progress_local  (stale fallback)
```

---

### 2. c_action — Three Modes (`confidence.py`)

#### 2a. Rename

- `CBCConfidenceEstimator` → `ConfidenceEstimator`
- `CBCConfidenceMetrics` → `ConfidenceMetrics`
- `ConfidenceMetrics.c_cbc` → `ConfidenceMetrics.c_action`

All downstream references updated accordingly.

#### 2b. Tracking-Error Signal

At each chunk boundary, `ConfidenceEstimator.update()` receives:
- `actual_joints: np.ndarray | None` — real robot joint positions from `rviz_publisher.get_latest_joints()`
- `delay_steps: int` — `inference_delay` computed before the inference started

```python
predicted_pos  = prev_chunk[min(delay_steps, N-1)]   # [7,]
tracking_MSE   = mean((actual_joints - predicted_pos) ** 2)
c_tracking     = exp(-gamma_t * tracking_MSE)          # gamma_t default 1.0
```

`delay_steps` uses `inference_delay` (pre-inference estimate of how many steps of
the previous chunk were consumed), so `prev_chunk[delay_steps]` is the VLA's
predicted robot position at the moment the new chunk arrives.

When `actual_joints is None` (no robot / first chunk): `c_tracking = 0.5`.

#### 2c. Three Modes

```python
CONFIDENCE_METHODS = (
    "raw_cbc", "speed_norm_cbc", "regression_cbc",   # existing
    "tracking",   # c_tracking only
    "combined",   # sqrt(c_regression * c_tracking) — geometric mean
)
```

`combined` (default) ensures both signals must be high; if either is low, c_action drops.

#### 2d. Updated Signature

```python
def update(
    self,
    actions: Any,
    actual_joints: np.ndarray | None = None,
    delay_steps: int = 0,
) -> ConfidenceMetrics:
```

`ConfidenceMetrics` gains two new fields: `c_tracking: float`, `tracking_mse: float`.

---

### 3. c_progress — Three Fixes (`sentinel.py`)

#### 3a. Continuous Prompt

Replace the `failure_likelihood` line in `PROMPT_TEMPLATE`:

```
"failure_likelihood": <float 0.00–1.00, use fine-grained values like 0.23 or 0.71, NOT just 0.1/0.5/0.9>,
```

Parser keeps existing JSON extraction logic; `round(..., 4)` preserves precision.

#### 3b. DeepSeek Provider

DeepSeek's API is OpenAI-compatible; reuse `_call_openai()` with different base URL:

```python
if provider == "deepseek":
    base_url = "https://api.deepseek.com/v1"
    api_key  = os.environ.get("DEEPSEEK_API_KEY", "")
    model    = "deepseek-chat"   # DeepSeek-V3, vision-capable
```

Missing API key → stale + error log, no crash (same behaviour as Gemini).

#### 3c. LocalMotionDetector (stale fallback)

New lightweight class inside `sentinel.py`:

```python
class LocalMotionDetector:
    def __init__(self, window_s: float = 3.0, stuck_threshold: float = 0.02):
        ...
    def push(self, joints: np.ndarray, t: float) -> None:
        # append (t, joints) to internal deque, evict entries older than window_s
    def c_progress_local(self) -> float:
        # max joint displacement over window < stuck_threshold  →  0.2 (stuck)
        # otherwise                                             →  0.5 (moving, unknown direction)
        # fewer than 1 s of data                               →  0.5 (neutral)
```

**Usage in `SentinelRuntime._arbiter()`:**

```python
# Old stale fallback:
r_raw = fast.c_action   # always ≈ 1.0 — bad

# New stale fallback:
c_fallback = self._motion_detector.c_progress_local()
r_raw = min(fast.c_action, c_fallback)   # max 0.5 when moving, 0.2 when stuck
```

`SentinelRuntime.update()` gains `actual_joints: np.ndarray | None = None` parameter,
calls `self._motion_detector.push(actual_joints, time.time())` before arbitration.

#### 3d. EMA β Adjustment

Default β lowered from 0.8 → 0.6. Configurable via `--sentinel-ema-beta`.

| β | Steps to drop 0.9 → 0.5 (@3 s/step) |
|---|--------------------------------------|
| 0.8 (old) | ~3.1 steps (~9 s) |
| 0.6 (new) | ~1.8 steps (~5 s) |

---

### 4. inference_thread.py Changes

Three targeted changes:

```python
# After Step C (get observation):
actual_joints = rviz_publisher.get_latest_joints() if rviz_publisher else None

# Step F — confidence update:
confidence_metrics = confidence_estimator.update(
    queued_robot_chunk,
    actual_joints=actual_joints,
    delay_steps=inference_delay,   # pre-inference estimate
)

# Sentinel update:
sentinel_result = sentinel_runtime.update(
    confidence_metrics,
    actual_joints=actual_joints,
    extra={...},
)
```

All `confidence_metrics.c_cbc` references → `confidence_metrics.c_action`.

---

### 5. run_inference.py — New CLI Args

```bash
--sentinel-confidence-mode  {cbc,tracking,combined}   default=combined
--sentinel-vlm-provider     {openai,gemini,deepseek}   # deepseek added
--sentinel-ema-beta          float                      default=0.6
```

---

## Error Handling

| Failure | Behaviour |
|---------|-----------|
| `actual_joints` is None | `c_tracking = 0.5` (neutral); LocalMotionDetector skipped |
| DeepSeek API key missing | stale + log, no crash |
| LocalMotionDetector < 1 s data | returns 0.5 (neutral) |
| VLM stale | `r_raw = min(c_action, c_progress_local)` instead of `c_action` |

---

## Ablation Experiment Support

Three c_action modes map directly to three CLI values:

```bash
--sentinel-confidence-mode cbc       # baseline (existing behaviour)
--sentinel-confidence-mode tracking  # tracking error only
--sentinel-confidence-mode combined  # proposed default
```

Results can be compared across `outputs/sentinel/` sessions.

---

## Out of Scope

- Replacing EMA with adaptive/learned arbitration
- End-effector position feedback from CroSPI (not yet available)
- Training a failure-likelihood probe (SAFE-style)
- Changing eTaSL Lua or the CroSPI bridge
