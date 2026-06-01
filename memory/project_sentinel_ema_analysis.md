---
name: sentinel-ema-analysis
description: Sentinel EMA beta 选型分析、c_progress 恢复行为、r_raw 移除及画图简化（2026-06）
metadata: 
  node_type: memory
  type: project
  originSessionId: 8f5e8de8-15ff-4c32-99d0-21ddcaca30f0
---

## Sentinel 推理频率

非 RTC 模式下 `sentinel.update()` 的调用频率 = `control_fps ÷ n_action_steps`。

当前配置（`run_inference.py`）：
- `control_fps = 10`（`--control-fps`，默认 10）
- `n_action_steps = 10`（`inference_thread.py:266`，硬编码）
- 结果：**1 Hz**，即每秒调用一次

**Why:** actor_thread 以 10 Hz 消耗动作，10 步耗尽队列需 1 秒，队列空了才触发下一次推理。

## c_progress 恢复行为（结论已验证）

- `ProgressDecayMonitor.c_progress()` 当 `_stuck_since is None` 时**立即返回 1.0**（无渐变）
- `push()` 一旦检测到不再卡住，立刻 `_stuck_since = None`
- **因此 c_progress 在机器人恢复运动的下一个 push() 调用后就立即跳变到 1.0**
- 真正平滑恢复的是 `r_smooth`（EMA），它是驱动 `w_vla/w_human` 的实际信号

## EMA beta 时间常数（1 Hz 更新频率下）

公式：`τ = 1 / ln(1/β)` 步（1 Hz 下 1 步 = 1 秒）

| beta | τ（秒） | 恢复到 ~95% |
|------|---------|------------|
| 0.5  | ~1.4 s  | ~4 s       |
| 0.6  | ~2.0 s  | ~6 s       |
| 0.8  | ~4.5 s  | ~13 s      |

**决策：采用 `beta = 0.5`**（4 秒恢复足够，响应更快）。

## 代码不一致 bug（已修复）

- `sentinel.py` 类默认参数：`ema_beta=0.8`
- `run_inference.py` CLI 默认：原为 `0.6`，已改为 `0.5`
- 实际运行走 `SentinelRuntime.from_args(args)`，所以生效的是 CLI 默认值
- **How to apply:** 判断实际 beta 时以 `run_inference.py` 的 CLI 默认为准，不看类默认值

## r_raw 移除（设计简化）

**决策：移除 `r_raw` 中间变量，`c_progress` 直接送入 EMA。**

原来：`r_raw = clip(c_progress)` → EMA → `r_smooth`
现在：`c_progress` → EMA → `r_smooth`

**Why:** `r_raw` 只是 `clip(c_progress)`，clip 在 c_progress 已在 [0,1] 时无意义，是多余的抽象层。

改动文件：
- `sentinel.py`：删除 `SentinelArbitrationResult.r_raw`，`_arbitrate` 直接用 `c_progress` 喂 EMA
- `inference_thread.py`：sentinel_records / logger / print 全部移除 r_raw

## 画图简化

原：3 子图（confidence：c_action+c_progress+c_vlm；reliability：r_raw+r_smooth；weights）
新：2 子图（c_progress + r 合并一图；weights 一图）

**Why:** c_action 和 c_vlm 不参与仲裁公式，r_raw 已删除，plot 只需展示决策相关信号。
