# c_progress 时间衰减设计文档

**日期:** 2026-05-15  
**文件:** `important_code/shared_control/sentinel.py`

## 问题背景

VLM 驱动的 `c_progress` 存在延迟、断续、跳变问题，导致机器人在任务中卡住时权重无法及时降低，机器人停在某位置不动或晃动。

## 目标

用基于运动停止时间的指数衰减信号替换仲裁公式中的 VLM c_progress，保留 VLM 路径用于日志记录和未来研究。

## 设计

### 新增类：`ProgressDecayMonitor`

位置：`sentinel.py`，紧跟现有 `LocalMotionDetector` 之后。

```
c_progress(t) = floor + (1 - floor) * exp(-lambda * stuck_time_s)
            where floor = 0.2, lambda = 0.05
```

- **`push(joints, t)`**：更新运动历史，检测卡住状态并记录 `stuck_since`
- **`c_progress(now)`**：返回当前衰减值
- 运动恢复时立即重置（`stuck_since = None` → 返回 1.0）
- 参数：`window_s=3.0`, `stuck_threshold=0.02`, `decay_lambda=0.05`, `floor=0.2`

半衰期：`ln(2) / 0.05 ≈ 14s`  
在 30s 静止后：`0.2 + 0.8 * exp(-1.5) ≈ 0.38`（渐近趋向 0.2）

### 数据流变化

```
before:  r_raw = min(c_action, c_progress_vlm)     [VLM 驱动]
after:   r_raw = min(c_action, c_progress_decay)   [时间衰减驱动]
         c_vlm = VLM result                         [仅日志记录]
```

### `SentinelArbitrationResult` 字段变化

| 字段 | 变化 |
|------|------|
| `c_progress` | 语义变为衰减值（始终有值，从无到有） |
| `c_vlm` | **新增**，存储 VLM 原始输出，仅用于日志 |

### `SentinelRuntime` 变化

- `_motion_detector: LocalMotionDetector` → `_progress_decay: ProgressDecayMonitor`
- 新 CLI 参数：`--sentinel-decay-lambda`（默认 0.05）
- `LocalMotionDetector` 类保留（不删除）
- `_arbitrate()` 移除 VLM/fallback 分支，始终用衰减值

### 仲裁逻辑简化

```python
# 变更前
if c_progress is not None:
    r_raw = min(c_action, c_progress)
else:
    r_raw = min(c_action, self._motion_detector.c_progress_local())

# 变更后
c_progress = self._progress_decay.c_progress()
r_raw = min(c_action, c_progress)
c_vlm = progress.c_progress if (progress and not progress.stale) else None
```

## 受影响的文件

1. `important_code/shared_control/sentinel.py` — 主要修改
2. `important_code/inference/inference_thread.py` — 添加 `--sentinel-decay-lambda` 参数
3. `important_code/shared_control/tests/test_sentinel.py` — 更新测试
