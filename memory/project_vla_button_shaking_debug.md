---
name: vla-button-shaking-debug
description: SpaceMouse 左键/右键触发的 follower 抖动问题：根因分析、已实施的全部修复、设计决策
metadata: 
  node_type: memory
  type: project
  originSessionId: a137fec7-920b-4b90-bac9-e7090422f3c5
---

## 系统频率全景（已验证）

| 组件 | 频率 | 来源 |
|---|---|---|
| CroSPI 真实机器人驱动 | **200 Hz** | setup.json `periodicity: 0.005` |
| eTaSL QP 求解器 | ~100 Hz | setup.json `periodicity: 0.01` |
| VLA 动作执行 (actor_thread) | **10 Hz** | `run_inference.py --control-fps` 默认 10 |
| VLA 单次推理延迟 | **≈ 0.4s = 4步** | `Indexes diff: 0, real delay: 4` 警告确认 |
| 相机采集 | 30 fps | 独立后台，与推理无关 |

**Why:** 频率差异（200Hz vs 10Hz）意味着每个 VLA 目标点在 eTaSL 内执行 20 个控制周期。若 VLA 目标突变，eTaSL 在接下来的 20 个周期全力追赶 → 抖动感明显。

## eTaSL 权重语义（重要澄清）

eTaSL WLN-QP 权重是**相对**的，非绝对：
- `w_vla=1, w_human=1` → VLA 和 SpaceMouse 各占 50% 输出
- `w_vla=2, w_human=1` → VLA 占 2/3，SpaceMouse 占 1/3
- `w_vla=0` → 纯 SpaceMouse（HUMAN_ONLY 模式）

**How to apply:** 设计权重时以相对比例思考，不要用"满权重"概念。

## 抖动根因的统一分析

**关键观察（已验证）：**
- 正常 SpaceMouse+VLA 共享控制：**不抖**
- 抖动仅在按下左键（gripper 切换）时发生
- 即使不按左键，HUMAN_ONLY → SHARED 切换也**不抖**
- **但只要按过左键激活 gripper 覆盖，无论是在 SHARED 里持续运行还是从 HUMAN_ONLY 切回，都会抖**

**统一根因：VLA 模型的 observation-action mismatch**

SmolVLA 训练数据中，gripper 始终由模型自己控制（arm joints 和 gripper 之间有学到的联合分布）。当 gripper 被外部干预（joint_6 通过 ramp 渐变到新目标），VLA 通过 UDP 9789 观测到 joint_6 变化，其下一个 chunk 预测中 arm joints 0-5 也随之改变（模型试图与 gripper 状态保持一致）→ arm 目标振荡 → 抖动。

**根本原因：** VLA 模型未见过外部 gripper 干预场景。软件层面难以完全消除，模型重训练才能根治。

## 设计决策：左键仅在 HUMAN_ONLY 模式下有效

**决策（已实施）：**
- HUMAN_ONLY 模式：左键有效，SpaceMouse 控制 gripper 开合
- SHARED 模式：左键无效，gripper 完全由 VLA 输出控制
- HUMAN_ONLY → SHARED（右键）：自动清除 `gripper_override_active`，VLA 接管 gripper

**Why:** 在 SHARED 模式下按左键会导致 VLA observation-action mismatch，引发手臂抖动。从架构上禁止操作比软件缓解更干净。

## ramp 和 alignment 的含义

| 机制 | 控制的是 | 作用 |
|---|---|---|
| **ramp** (`_RESUME_RAMP_S=1.5s`) | eTaSL 中 VLA 的权重大小（w_vla: 0→1） | 防止 VLA 影响力突变 |
| **alignment** (`_ALIGN_DURATION_S=3.0s`) | 发给 eTaSL 的 VLA 目标关节角 | 发布 actual joints 作为目标，使 VLA 跟踪误差=0 |

**时序设计（alignment > ramp）：**
```
t=0       t=1.5s              t=3.0s
|         |                   |
对齐开始  ramp结束             对齐结束
          w_vla=1（已满）       VLA 真实预测开始
          但误差仍=0            VLA 已观测手臂在当前位置 1.5s
          → 无力，不抖          预测从当前位置出发，误差小 → 平滑
```

关键：ramp 结束时（t=1.5s）w_vla 已满，但 alignment 仍在（VLA 误差=0，无驱动力）。这额外 1.5s 给 VLA 时间适应当前位置。SmolVLA 约 2s 历史窗口（10Hz × 20帧），1.5s 足够填满当前位置的观测。

## 已实施的全部修复（vla_ros_bridge_node.py）

### 常量
| 常量 | 旧值 | 新值 | 说明 |
|---|---|---|---|
| `_RESUME_RAMP_S` | 0.4 | 1.5 | VLA 权重 ramp 时长 |
| `_GRIPPER_RAMP_S` | 无 | 0.6 | 夹爪目标渐变时长（新增） |
| `_ALIGN_DURATION_S` | 无 | 3.0 | VLA 目标对齐窗口（新增） |

### `_OperatorAuthority` 类变更
- 新增 `_gripper_ramp_s`, `_gripper_ramp_start`, `_gripper_ramp_started_at`：支持夹爪目标线性插值
- 新增 `_align_duration_s`, `_align_until`：对齐窗口状态
- `toggle_gripper(current_pos, now)`：新增参数，从实际位置开始 ramp
- `apply_gripper_override(joints, now=None)`：线性插值替代 step change
- `note_vla_command`：切回 SHARED 后第一条 VLA 到达时设置 `_align_until = now + 3.0s`
- `toggle_human_only`：切回 SHARED 时清除 `gripper_override_active = False`
- 新增 `is_aligning` property

### `VLABridgeNode` 类变更
- `_publish_vla_cmd`：对齐窗口内 joints[0:6] = actual joints；使用统一 `mono = time.monotonic()`
- `_timer_cb`：
  - 新增 `vla_cmd_published` 标志；无 VLA 时 gripper override 用 `_publish_gripper_hold_cmd` 发布
  - 左键处理加门控：`and self._operator.mode == _MODE_HUMAN_ONLY`
- 新增 `_publish_gripper_hold_cmd`：VLA 未运行时发布手臂保持+夹爪目标
- 新增 `_get_current_gripper_pos`：读取实际 joint_6 位置

## 被排除的根因

`Action queue empty / Indexes diff: 0, real delay: 4` 警告来自 lerobot 推理侧：
- 非 RTC 模式（`--rtc` 默认 False）下，队列耗尽才推理，推理期间队列空 → indexes_diff=0
- 但这会导致**持续性抖动**，而实际观测是**按钮特异性抖动** → 排除为主因
- 注意：`run_inference_rtc.py` **不存在**；实际入口是 `run_inference.py`，需显式加 `--rtc` 标志

## 未来改进

- **根治左键抖动**：重训练 VLA 模型，加入外部 gripper 干预场景的训练数据
- **缓解左键抖动（未实施）**：gripper ramp 期间停止向 VLA 转发实际 joint_6，让 VLA 看不到 gripper 变化，等 ramp 完成后恢复
