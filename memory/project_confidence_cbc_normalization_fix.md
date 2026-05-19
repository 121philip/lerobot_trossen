---
name: project-confidence-cbc-normalization-fix
description: regression-CBC 置信度计算从机器人空间改为归一化空间的问题分析与修复记录
metadata: 
  node_type: memory
  type: project
  originSessionId: 01379b47-c08d-4410-ada6-9478c16b5aaf
---

## queued_original_chunk 与 queued_robot_chunk 的区别

两者均 shape=[n_action_steps, 7]，来自同一次模型推理，但处于不同空间：

| 变量 | 来源 | 值域 | 用途 |
|------|------|------|------|
| `queued_original_chunk` | `actions.squeeze(0).clone()`，模型直接输出 | 归一化范围（≈[-1, 1]） | RTC chunk 融合（必须在归一化空间插值） |
| `queued_robot_chunk` | `postprocessor(actions).squeeze(0)`，经反归一化 | 真实关节角（rad），可直接发给机器人 | 执行、confidence 估计、action_queue 写入 |

**为何两者都入队**：RTC 融合必须在归一化空间操作；执行必须用真实关节值。两者在 inference_thread.py 中同步进入 `action_queue.merge()`，并分别存入各自的 `prev_chunk_norm` / `prev_chunk` 缓冲。

---

## 问题

`ConfidenceEstimator.update()` 原先接收 `queued_robot_chunk`（反归一化，单位 rad），各关节尺度差异极大：

| 关节 | 机器人空间值 | 归一化空间值 |
|------|------------|------------|
| 0–2  | ~1.67–1.85 rad | ~-0.30–0.68 |
| 4    | ~-0.08 rad     | ~-0.27      |
| 6（夹爪）| ~0.0001 rad | ~-1.11   |

CBC 计算 `mean((boundary - fitted)²)` 对所有关节均匀平均，夹爪关节（joint 6）几乎对残差毫无贡献，即使它发生了大幅相对变化。

**Why:** 模型（SmolVLA）在归一化空间输出动作，归一化空间中各关节的变化量才能代表等价的模型不确定性；机器人空间中相同绝对变化量对不同关节意义完全不同。

## 解决方案

为 `ConfidenceEstimator.update()` 新增可选参数 `actions_normalized`：

- **CBC 边界指标**（raw_mse、reg_residual、speed_norm、boundary_jump_max）→ 使用归一化空间
- **Tracking 误差**（actual_joints vs predicted）→ 保留机器人空间（actual_joints 单位是 rad，无法归一化）
- **Instability 指标**（vel/accel/jerk）→ 保留机器人空间（物理意义：rad/s 等）

调用侧传入 `actions_normalized=queued_original_chunk`（模型原始输出，未经 postprocessor 反归一化）。

## 关键文件（修改后）

- `important_code/shared_control/confidence.py`
  - `__init__`：新增 `self.prev_chunk_norm: np.ndarray | None = None`
  - `reset()`：同步清除 `prev_chunk_norm`
  - `update(actions, ..., actions_normalized=None)`：CBC 路由至 `chunk_norm`/`prev_chunk_norm`，tracking 仍用 `chunk`
- `important_code/inference/inference_thread.py`
  - `confidence_estimator.update(...)` 调用处新增 `actions_normalized=queued_original_chunk`
- `important_code/shared_control/tests/test_confidence.py`
  - 新增 `test_normalized_space_detects_small_joint_jump`：验证夹爪小尺度跳变在归一化模式下可被检测
  - 新增 `test_normalized_fallback_equals_robot_space_when_not_provided`：向后兼容性验证

## 关于仿射回归框架是否能真正检测跳变

结论：可以。affine-regression 在归一化空间同样有效。d=5、30fps 下窗口约 1/6 秒，真实机械臂运动在此窗口内近似线性，因此 boundary kink 和跳变会清晰显现为残差。

**Why:** 仿射拟合跨越 boundary 窗口（2d 步），若轨迹有不连续，单条直线无法同时拟合 boundary 两侧，残差必然偏大。

## Tracking 误差计算方式

```python
idx = min(max(0, int(delay_steps)), self.prev_chunk.shape[0] - 1)
predicted = self.prev_chunk[idx]   # 上一块第 delay_steps 步的预测关节位置
tracking_mse = mean((actual_joints - predicted)²)
c_tracking = exp(-γ × tracking_mse)
```

逻辑：推理时延为 `delay_steps` 帧，所以机器人此刻"应该"处于上一块第 `delay_steps` 步的位置。将其与 `actual_joints`（机器人实际关节角，rad）比较，MSE 大说明 VLA 对当前状态估计偏差大（obs-action mismatch 或外力干扰）。

**同样存在尺度偏差**：tracking_mse 也在机器人空间（rad）均匀平均，夹爪贡献极小。修复 tracking 比 CBC 复杂——`actual_joints` 无对应归一化版本，需从 postprocessor 提取 per-joint mean/std 手动归一化，暂不处理。

## 注意

- `queued_original_chunk`（归一化）与 `queued_robot_chunk`（机器人空间）的分离早已存在（RTC 融合也使用归一化空间）；本次改动只是将这个分离扩展到 confidence 计算
- Tracking 误差目前无法用归一化空间计算（actual_joints 无对应归一化版本）；若将来需要，需从 postprocessor 提取 per-joint mean/std 并手动归一化 actual_joints

**How to apply:** 若将来修改 confidence 计算，记住：CBC 在归一化空间，tracking 在机器人空间，两者各有独立缓冲 `prev_chunk_norm` 和 `prev_chunk`。
