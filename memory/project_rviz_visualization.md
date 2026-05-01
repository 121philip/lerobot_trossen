---
name: RViz 推理轨迹可视化系统
description: RViz2 可视化 SmolVLA 推理的实际执行轨迹与预测末端轨迹，用于诊断抖动/偏差问题
type: project
originSessionId: 191a4f2e-05f3-49c7-985e-c8a061f7bfe6
---
## 架构

**问题背景**：SmolVLA 策略运行在 Python 3.12 venv（无 rclpy），ROS2 Humble 的 rclpy 编译目标是系统 Python 3.10，两者不兼容。

**解决方案**：UDP socket 桥接（端口 9788，localhost）：
- Python 3.12 侧：`important_code/inference/rviz_publisher.py` — 推理/执行线程通过 `put_actual()` / `put_predicted()` 发数据
- Python 3.10 侧：`important_code/inference/rviz_node.py` — 由 `launch_viz.sh` 用 `/usr/bin/python3` 启动，接收 UDP 并发布 ROS2 话题

**UDP 协议**：1 字节 msg_type + pickle（protocol=4）序列化 numpy 数组
- `0x00` + shape [7]    → 实际执行关节角（来自 actor_thread，每帧）
- `0x01` + shape [N,7]  → 预测动作块（来自 inference_thread，每次推理）

## 文件清单

| 文件 | 作用 |
|------|------|
| `important_code/inference/rviz_node.py` | ROS2 节点（系统 Python 3.10），发布话题和 Marker |
| `important_code/inference/rviz_publisher.py` | UDP 发送端（Python 3.12 venv），非阻塞队列 |
| `important_code/visualization/launch_viz.sh` | 启动脚本：rviz_node + robot_state_publisher + RViz2 |
| `important_code/visualization/trajectory_viz.rviz` | RViz 配置文件 |
| `important_code/visualization/test_rviz_motion.py` | 独立测试脚本（正弦波，验证可视化管线） |

## ROS2 话题

- `/actual/joint_states_VLA` — 蓝色实体机器人（robot_state_publisher 订阅）
- `/predicted_ee_marker` — 绿色球（当前 EE 位置） + 橙红色线段+点（50步预测轨迹）

**Why:** `joint_states_VLA` 后缀区分 VLA 推理输出与其他可能存在的 joint_states 话题。

## URDF TF 处理

`launch_viz.sh` 用 `sed` 直接将 `actual/` 前缀烘焙进 URDF link 名称（`<link name="actual/base_link">`），比 `frame_prefix` 参数或 RViz TF Prefix 更可靠。

`rviz_node.py` 发布静态变换 `actual/base_link → predicted/base_link`（identity），连接两棵 TF 树（保留兼容性，即使不渲染预测机器人）。

## 正向运动学

`rviz_node.py` 用纯 numpy Rodrigues 公式计算 FK（无需 `kdl_parser_py`），参数硬编码自 URDF：
- 6 个旋转关节（joint_0~5）+ 固定末端偏移 `[0.156062, 0.0, 0.0]`
- 第 7 个关节（left_carriage_joint，夹爪）不影响末端位置

## 关键 Bug 记录

**CRITICAL — 目录大小写**：代码重构后目录从 `Inference/`（大写 I）改为 `inference/`（小写）、`rviz_config/` 改为 `visualization/`。`launch_viz.sh` 里仍有旧路径时，`rviz_node.py` 会静默失败（后台进程，`set -e` 不影响），导致：
- 无 `/actual/joint_states_VLA` 数据 → 机器人不动
- 无 `/predicted_ee_marker` → 无预测轨迹
- 症状：蓝色机器人可见（robot_state_publisher 正常）但静止不动，无橙色线

**现象与正确路径**：
```bash
RVIZ_NODE="$SCRIPT_DIR/../inference/rviz_node.py"   # 正确（小写 i）
```

## 使用方法

```bash
# 终端1（先启动）
bash important_code/visualization/launch_viz.sh

# 终端2（后启动）
python important_code/inference/run_inference_rtc.py --rviz [其他参数]
# 或干跑验证
python important_code/inference/run_inference_rtc.py --dry-run --rviz
```

## 诊断方法

| 现象 | 根因 |
|------|------|
| 橙线抖动 + 蓝机器人也抖 | 模型预测本身不稳定，调整 `guidance_weight` |
| 橙线平滑 + 蓝机器人抖动 | 执行层问题（`max_relative_target` 截断或硬件延迟）|
| 橙线与蓝机器人末端长期偏差 | 归一化参数或关节映射问题 |

**Why:** 建立可视化的初衷是诊断 WidowX 抓葡萄任务中的抖动和位置偏差。
**How to apply:** 运行推理时加 `--rviz`，先确认橙线在蓝机器人末端前方，再判断稳定性。

---

## 2026-05-01: CroSPI shared-control orange trajectory detachment diagnosis

Problem observed during real follower tests:

- RViz orange predicted EE trajectory can be far from the blue actual follower EE, especially when `w_human` is large in `vla_spacemouse_blend.etasl.lua`.
- Even with SpaceMouse untouched, follower can initially move in a strange posture/high position and not follow the orange trajectory.
- After stopping SmolVLA inference, the follower can continue moving slowly toward the last orange trajectory; speed decreases as it approaches.
- Stopping `trossen_vla_shared_control_runner.py` does not necessarily stop the follower.
- Stopping `vla_ros_bridge_node.py` removes the orange marker and can make motion appear to stop; restarting the bridge does not restore the orange marker unless a new predicted chunk arrives, but CroSPI may still hold the previous target.

Important data-flow distinction:

- Orange trajectory is **not** the measured future path of the real robot. It is FK of `postprocessed_actions` predicted by SmolVLA, sent through UDP as `_MSG_PREDICTED`, cached by `vla_ros_bridge_node.py`, and drawn as `/predicted_ee_marker`.
- Blue robot is actual `/joint_states` after eTaSL solves its QP and the Trossen driver executes integrated velocity setpoints.
- The real trajectory only matches orange if observation timing is current, SmolVLA is in-distribution, and eTaSL executes the VLA joint targets without strong conflicting constraints.

Findings:

1. FK/URDF parameters were not the primary cause. The FK constants in `vla_ros_bridge_node.py` match the follower URDF joint origins/axes for `joint_0..joint_5`, and joint order matches `lerobot_trossen` (`joint_0..joint_5`, `left_carriage_joint`).

2. Alpha is currently not actually wired into eTaSL:
   - `vla_spacemouse_blend.etasl.lua` has `ctx:createInputChannelScalar("alpha_input", 0.5)` commented out and uses `alpha = 0.5`.
   - Effective weights are constants: `w_vla = 1`, `w_human = 10`.
   - `trossen_vla_shared_control.setup.json` has no `TopicInputHandler` for `/shared_control/alpha`, even though the bridge publishes it.
   - Therefore `run_inference_rtc.py --alpha-const 0.5` and `/override/alpha` do not change eTaSL behavior unless the alpha input handler is restored.

3. SpaceMouse constraints remain active even when the SpaceMouse is not moved:
   - With zero joystick input, the linear Cartesian velocity constraints impose near-zero TCP velocity with `weight = w_human`.
   - These constraints have the same priority as VLA joint tracking but much higher weight (`10` vs `1`).
   - The solver therefore compromises: it tries to reduce VLA joint error while keeping Cartesian motion/orientation constrained. This can create strange postures and EE paths that do not match the orange VLA FK trajectory.

4. `/joint_states_VLA` is latched semantically by CroSPI input state:
   - `JointStateInputHandler` updates `target_joint_1..6` only when a new message arrives.
   - When messages stop, the input channels keep the previous values; they do not automatically reset to current robot state or zero.
   - VLA stopping therefore leaves eTaSL chasing the last target. Because the joint constraint has `K=0.5`, convergence slows as error decreases, explaining the observed asymptotic slow motion.

5. Startup default can pull the robot away before the first VLA action:
   - setup uses `default_joint_states: [0,0,0,0,0,0]` for `/joint_states_VLA`.
   - MovingHome sends the robot to `[0, 1.0472, 0.5236, 0.6283, 0, 0]`.
   - If `spacemouse_shared_control_vla` activates before the first VLA message, eTaSL initially sees all-zero VLA targets and may start pulling away from the home pose.

6. Stopping the runner is not a stop command for CroSPI:
   - `spacemouse_shared_control_vla` has `execution_time = 0`, so it does not finish automatically.
   - `eTaSL_StateMachine` defaults to `deactivate_last=False`.
   - Killing `trossen_vla_shared_control_runner.py` can leave `/crospi_node` active and still running the last eTaSL task.

7. OOD/training-distribution shift is plausible but not the only root cause:
   - SmolVLA observes images plus joint state and predicts an action chunk. If shared-control/eTaSL first drives the follower to a pose absent from training demonstrations, the policy can become unreliable.
   - In OOD states, predicted chunks may jump toward familiar demo-manifold states, start far from current EE, or produce inconsistent wrist/head posture.
   - However OOD alone does not explain continued motion after inference stops or the immediate dependence on `w_human`; those point to eTaSL target locking and constraint conflict.
   - Best interpretation: eTaSL/shared-control deviation pushes the robot off the training manifold, then SmolVLA OOD prediction amplifies the orange/blue detachment in a closed-loop covariate-shift cycle.

Useful diagnostic commands:

```bash
ros2 topic hz /joint_states_VLA
ros2 topic echo /joint_states_VLA --once
ros2 topic hz /shared_control/alpha
ros2 topic echo /shared_control/alpha_monitor --once
ros2 lifecycle get /crospi_node
```

Minimal experiments to isolate root cause:

- Pure VLA tracking test: temporarily set `w_human = 0` or remove/comment SpaceMouse Cartesian constraints. If blue follows orange much better, the main issue is eTaSL constraint conflict.
- Alpha wiring test: restore `alpha_input` in Lua and add `/shared_control/alpha` `TopicInputHandler` in setup. Verify changing `/override/alpha` changes behavior.
- Startup target test: start VLA before activating `spacemouse_shared_control_vla`, or change `default_joint_states` to the MovingHome pose. If initial strange jump disappears, the all-zero default was contributing.
- OOD test: compare predictions from a training-like home/grasp pose versus the abnormal high/twisted pose. If orange is reasonable in-distribution but jumps in the abnormal pose, SmolVLA OOD is confirmed.
- Latch test: stop VLA and confirm `/joint_states_VLA` stops publishing. If follower still moves, CroSPI is chasing the last latched target.

Recommended fixes:

- Wire alpha end-to-end: Lua input channel + setup `TopicInputHandler` for `/shared_control/alpha`; make `w_vla` and `w_human` functions of alpha.
- Do not let zero SpaceMouse input dominate VLA. Gate or down-weight Cartesian constraints when human input magnitude is near zero.
- On VLA shutdown, publish a hold-current `/joint_states_VLA` target or explicitly deactivate/cleanup `/crospi_node`.
- Avoid all-zero default targets after MovingHome; use current/home pose or wait for the first VLA target before activation.
- Treat orange marker as "VLA intent" only, not guaranteed executed trajectory. Compare three layers separately: predicted chunk FK, `/joint_states_VLA`, and actual `/joint_states`.

