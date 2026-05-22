---
name: bridge node FK 正向运动学分析结论
description: vla_ros_bridge_node.py 中自写 FK 的参数正确性验证结果，以及 predicted_ee_marker 偏差的根本原因分析
type: project
originSessionId: c4f953f4-c5a0-453f-9c3e-48a83989233e
---
## 结论：FK 参数和公式均正确

已逐一比对 `wxai_follower.urdf`（两个副本 MD5 完全相同）：

| 关节 | xyz 偏移 | 旋转轴 | 验证 |
|------|---------|--------|------|
| joint_0 | [0, 0, 0.05725] | Z | ✓ |
| joint_1 | [0.02, 0, 0.04625] | Y | ✓ |
| joint_2 | [-0.264, 0, 0] | -Y | ✓ |
| joint_3 | [0.245, 0, 0.06] | -Y | ✓ |
| joint_4 | [0.06775, 0, 0.0455] | -Z | ✓ |
| joint_5 | [0.02895, 0, -0.0455] | X | ✓ |

EE offset [0.156062, 0, 0] 对应 `ee_gripper` 固定关节（直接 parent = link_6，不经过 carriage 关节）✓

FK 公式 `Tji = [[R(axis,q) | xyz], [0|1]]` 等价于 URDF 标准变换 `T_translate @ T_rotate`（rpy=0 时成立）✓

关节顺序：lerobot `JOINT_NAMES` 与 bridge `_JOINT_NAMES_7` 完全一致 ✓

Chunk 数值空间：`postprocessed_actions`（反归一化真实弧度）直接用于 FK ✓

坐标系：预测 marker 使用 `actual/base_link`，与 robot_state_publisher 的 TF 根帧一致 ✓

## 预测轨迹偏差的根本原因

**不是 FK Bug，是共享控制的预期行为。**

时序根因：
1. 时刻 T₀：VLA 读取 bridge_joints（真实状态）→ 开始推理（耗时 1-3s）
2. 推理期间：SpaceMouse 持续移动机器人（w_human 越大，移动越多）
3. 时刻 T₀+L：推理完成，发布 chunk（基于 T₀ 状态的预测轨迹）
4. 橙色轨迹 = "VLA 从 T₀ 状态出发的意图轨迹"；蓝色机器人 = "T₀+L 实际位置"

w_human 越大 → 机器人被 SpaceMouse 推得越远 → 轨迹脱节越明显。这是正确的可视化，不是错误。

## 潜在改进方向（非 Bug 修复）

1. **Chunk 截断展示**：只展示前 5-10 步而非全部 50 步，近期预测更准确
2. **观测延迟补偿**：`inference_delay` 已计算（inference_thread.py:125-127），目前只用于 RTC action_queue 对齐，未用于选取更新的观测——可以用历史关节状态估算推理完成时的机器人状态作为 VLA 输入

**How to apply:** 若再次遇到"预测轨迹偏离真实机器人"的 issue，先确认 w_human 值，若 w_human 大则属正常行为；若 w_human=0 时仍偏离才需排查归一化参数或关节映射。
