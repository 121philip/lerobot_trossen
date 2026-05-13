---
name: vla-2026-05-13
description: SpaceMouse 左键/右键触发的 follower 抖动问题：根因分析、实施修复和待改进项
metadata: 
  node_type: memory
  type: project
  originSessionId: 144b2eb3-bfc3-4dd9-a8f3-41543c390509
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

## 已修复：无 VLA 时左键无法控制夹爪

**根因：** `apply_gripper_override` 只在 `_publish_vla_cmd`（VLA UDP 到达时）内调用。无 VLA 运行时 `/joint_states_VLA` 从不发布，夹爪目标永远不会送达 eTaSL。

**修复（vla_ros_bridge_node.py）：**
- 新增 `_publish_gripper_hold_cmd`：当 `gripper_override_active=True` 且本 tick 无 VLA UDP 到达时，用实际关节角（手臂原地）+ 夹爪目标发布 `/joint_states_VLA`
- 新增 `_get_current_gripper_pos` 辅助方法

## 已修复：夹爪目标跳变（左键 step change）

**根因：** `toggle_gripper` 直接把 joint_6 目标从 VLA 命令值跳变到 0.001/0.035，eTaSL K=2 以高增益追赶 → 瞬间速度命令大。

**修复：** 新增 gripper ramp（`_GRIPPER_RAMP_S=0.6s`）。`toggle_gripper(current_pos, now)` 记录当前实际夹爪位置，`apply_gripper_override(joints, now)` 做线性插值。

## 已修复：HUMAN_ONLY→SHARED 大误差跳变

**根因：** HUMAN_ONLY 期间 SpaceMouse 带动手臂移到位置 A，VLA 追踪约束权重为 0 时 VLA 预测可能漂移到位置 B。切回 SHARED 时，VLA 权重从 0 上升，eTaSL 开始追赶 B，误差大 → 手臂加速 → 抖动。

**修复：**
- `_RESUME_RAMP_S` 从 0.4s 延长到 1.5s
- 新增对齐窗口（`_ALIGN_DURATION_S=1.5s`）：切回 SHARED 后第一条 VLA UDP 到达时启动，期间 `_publish_vla_cmd` 将 arm joints 0-5 替换为实际关节角，VLA 跟踪约束误差=0
- 新增 `_OperatorAuthority.is_aligning` property

## 残留问题：按钮特异性抖动（VLA 推理运行中）

**关键观察：** 正常 SpaceMouse+VLA 共享控制**不抖**；抖动仅在左键/右键按下时发生。这排除了 RTC 队列清空作为主因（队列问题会导致持续性抖动）。

### 左键（夹爪切换）抖动的真实根因

**根因：eTaSL 中 VLA 关节跟踪约束与 SpaceMouse 笛卡尔约束的耦合，叠加 VLA observation-action correlation。**

#### eTaSL 约束结构

WidowX 的 `tcp_frame` 定义在 joint_5 之后（joint_6 是夹爪，不影响 TCP 位置）。但 eTaSL QP 求解器将所有约束**联合求解**：

```lua
-- VLA 跟踪约束（joint space，7个关节各一条）：
Constraint{ expr = joint_i - target_joint_i, weight = w_vla }

-- SpaceMouse 笛卡尔约束（task space，TCP 速度）：
Constraint{ expr = coord_x(origin(tcp_frame)) - desired_vel * time, weight = w_human }
```

虽然 joint_6 不影响 TCP 位置，但 QP 是对**所有关节的联合优化**，joint_6 的约束误差会与其他关节约束共同影响整体求解结果。

#### VLA observation-action correlation（主要原因）

当夹爪 joint_6 开始运动（即使是 ramp 渐变），实际机器人的 `/joint_states` 反馈中 joint_6 变化 → VLA 通过 UDP 9789 观测到 joint_6 变化。

SmolVLA 是序列模型，训练时 gripper 是模型自己控制的，arm joints 和 gripper 之间有**学到的联合分布**。当 ramp 改变 joint_6：
- VLA 的下一个 chunk 预测中 joint_0–joint_5 的轨迹也会随之改变
- 模型试图与 gripper 状态保持一致，产生了预期外的手臂目标
- eTaSL 追踪这些变化的目标 → 抖动

这是模型内部的 **observation-action correlation**：ramp 减慢了变化速度，但只要 joint_6 在变化，VLA 就会持续产生不同的手臂预测。

**根本原因：** VLA 模型训练时 gripper 是模型自己控制的，未见过外部 gripper 干预场景。难以从软件层面完全消除，模型重训练才能根治。

**潜在缓解方案（未实施）：** 在 gripper ramp 期间（0.6s），停止通过 UDP 9789 向 VLA 转发实际 joint_6 位置（用 ramp 前的值替代），让 VLA 看不到 gripper 在变化，等 ramp 完成后再恢复真实反馈。

### 右键（HUMAN_ONLY→SHARED）抖动的真实根因

#### 三个阶段（当前实现，alignment=ramp=1.5s）

```
阶段 1: HUMAN_ONLY
  w_vla=0, w_human=1
  手臂 100% 跟 SpaceMouse 走，VLA 约束关闭

阶段 2: 对齐窗口（0 ≤ t < 1.5s，右键按下后第一条 VLA UDP 到达时启动）
  w_vla: 0 → 1（ramp），w_human=1
  但 /joint_states_VLA 目标被强制 = 实际关节角
    → eTaSL 的 VLA 跟踪误差 = actual - actual = 0
    → VLA 约束贡献的力 = w_vla × K × 0 = 0
    → 手臂实际上仍然 100% 跟 SpaceMouse 走（和 HUMAN_ONLY 一样顺畅）

阶段 3: t = 1.5s — 危险时刻
  两件事同时发生：
    ① ramp 结束，w_vla 已达最大值（w_vla=1，与 w_human=1 各占50%）
    ② is_aligning=False，/joint_states_VLA 目标从"实际位置"切换为"VLA真实预测"

  如果 VLA 真实预测 ≠ 实际位置（VLA 认为下一步要抬手臂/移到某处）：
    误差 = VLA预测 - actual ≠ 0
    eTaSL 立即以 50% 权重追这个误差
    → 手臂突然被拉向 VLA 的目标 → 抖动
```

#### 为什么正常运行不抖？

正常 SHARED 模式下，VLA 预测是连续更新的，误差始终很小（VLA 一直在观测实际位置并输出平滑轨迹）。不存在"误差从 0 突变为大值"的瞬间。

#### 为什么按右键才抖？

HUMAN_ONLY 期间，SpaceMouse 带着手臂去了新位置 A。VLA 虽然也在持续推理（通过 UDP 9789 看到手臂在 A），但其 action chunk 是任务驱动的——它"知道"下一步该做什么（比如抬手臂去 B），不一定从 A 出发。alignment 结束时，这个"去 B"的预测以满权重激活 → 误差大 → 抖动。

### 待改进：alignment 应比 ramp 长

正确的时序：`_ALIGN_DURATION_S = 3.0s`（ramp 的 2 倍）：

```
t=0       t=1.5s              t=3.0s
|         |                   |
对齐开始  ramp结束             对齐结束
          w_vla已最大          VLA 已看到手臂在当前位置 1.5s
          VLA target=actual    VLA 预测已逐渐收敛到当前位置出发
          误差仍=0 → 不抖      小误差 → 平滑过渡
```

关键：ramp 结束后（t=1.5s），w_vla 虽然已满，但对齐仍在（误差=0，无力）。VLA 在这额外 1.5s 里持续观测手臂在当前位置 A，其预测逐渐从"去 B"变为"从 A 出发"。t=3s 对齐结束时，VLA 预测 ≈ actual → 误差小 → 不抖。

SmolVLA 使用约 2s 历史窗口（10Hz × 20帧），额外 1.5s 足够让历史窗口填满当前位置的观测。

**How to apply:** 修改 `_ALIGN_DURATION_S = 1.5` → `3.0` 并测试。

## 被排除的根因

`Action queue empty / Indexes diff: 0, real delay: 4` 警告来自 lerobot 推理侧：
- 非 RTC 模式（`--rtc` 默认 False）下，队列耗尽才推理，推理期间队列空 → indexes_diff=0
- 但这会导致**持续性抖动**，而用户观测是**按钮特异性抖动** → 排除为主因
- 注意：`run_inference_rtc.py` **不存在**；实际入口是 `run_inference.py`，需显式加 `--rtc` 标志

## 关键文件变更摘要（本次 session）

`vla_ros_bridge_node.py` 新增/修改：
- `_RESUME_RAMP_S`: 0.4 → 1.5
- `_GRIPPER_RAMP_S = 0.6`（新增常量）
- `_ALIGN_DURATION_S = 1.5`（新增常量，待改为 3.0）
- `_OperatorAuthority`: 新增 `_gripper_ramp_*`、`_align_until`、`is_aligning`
- `toggle_gripper(current_pos, now)`: 新增参数，启动 ramp
- `apply_gripper_override(joints, now=None)`: 线性插值替代 step change
- `note_vla_command`: 新增 `_align_until = now + _align_duration_s`
- `_publish_vla_cmd`: 对齐窗口内 joints[0:6] = actual joints
- `_publish_gripper_hold_cmd`: 新方法，VLA 未运行时发布手臂保持+夹爪目标
- `_get_current_gripper_pos`: 新辅助方法
