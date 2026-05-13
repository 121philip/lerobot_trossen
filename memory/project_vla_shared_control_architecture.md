---
name: VLA 共享控制系统架构
description: kaixi_crospi_ws 的 VLA+SpaceMouse 共享控制系统的整体架构、数据流和关键文件
type: project
originSessionId: a0411ee3-9f34-42f4-a7f6-b87b0845d408
---
## 系统架构

VLA（SmolVLA）+ SpaceMouse 共享控制，运行在 WidowX AI 机械臂上，通过 eTaSL/CroSPI 执行。

### CrOSPI 四层架构与职责

```
Layer 1 [可视化]  trossen_follower_visualization.launch.py
                  → robot_state_publisher + RViz2，订阅 /joint_states

Layer 2 [硬件层]  crospi_node  ← trossen_vla_shared_control.setup.json
                  → eTaSL QP 求解器 + 真实/仿真驱动，发布 /joint_states
                  → IO: 接收 /spacenav/twist、/joint_states_vla、/shared_control/alpha

Layer 3 [编排层]  trossen_vla_shared_control_runner.py (BeTFSM 状态机)
                  → 加载 tasks/trossen_vla_shared_control.json
                  → 状态机路径: MovingHome → Wait(5s) → spacemouse_control → spacemouse_shared_control_vla

Layer 4 [输入层]  spacenav classic-launch.py → 发布 /spacenav/twist
```

### 推荐启动顺序（命令间依赖关系）

```bash
# Step 1: 可视化（独立，无依赖）
ros2 launch crospi_application_template trossen_follower_visualization.launch.py

# Step 2: SpaceMouse 驱动（独立，无依赖）
ros2 launch spacenav classic-launch.py

# Step 3: CrOSPI 核心节点（依赖 setup.json，提供 /joint_states）
ros2 run crospi_core crospi_node --ros-args \
  -p config_file:="$[crospi_application_template]/applications/trossen_applications/trossen_vla_shared_control.setup.json" \
  -p simulation:=false

# Step 4: BeTFSM 状态机（必须在 crospi_node 就绪后启动）
python3 trossen_vla_shared_control_runner.py
```

**Why:** Step 3 必须在 Step 4 之前，因为 runner.py 通过 ROS2 服务向 crospi_node 提交 eTaSL 任务，若 crospi_node 未就绪则服务调用失败。Step 1、2 顺序无关紧要。

**How to apply:** 调试启动问题时按此顺序逐步验证各层是否正常。

### 关键文件

| 文件 | 职责 |
|------|------|
| `kaixi_crospi_ws/src/.../example_nodes/vla_ros_bridge_node.py` | UDP↔ROS2 桥：接收 lerobot UDP，发布 /joint_states_VLA、/shared_control/alpha、/predicted_ee_marker；订阅 /joint_states 发布 /actual/joint_states_rviz（蓝色机器人可视化）；内联 `_SpaceMouseArbiter`（按钮长/短按检测）|
| `kaixi_crospi_ws/src/.../trossen_applications/trossen_follower_visualization.launch.py` | 启动 bridge + robot_state_publisher + RViz2 |
| `kaixi_crospi_ws/src/.../trossen_applications/trossen_vla_shared_control.setup.json` | CroSPI 配置：arm IP 192.168.2.3、InputHandlers、eTaSL 参数 |
| `kaixi_crospi_ws/src/.../dummy_lib/task_specifications/vla_spacemouse_blend.etasl.lua` | eTaSL 任务规范：VLA 关节跟踪 + SpaceMouse 笛卡尔速度混合 |
| `kaixi_crospi_ws/src/.../dummy_lib/task_json_schemas/vla_spacemouse_blend.etasl.json` | 上述 Lua 任务规范的 JSON Schema 参数校验文件；定义并验证启动该任务所需的配置参数 |
| `kaixi_crospi_ws/src/.../test_trossen/trossen_vla_shared_control_runner.py` | betfsm 入口：MovingHome → TimedWait(5s) → spacemouse_control → spacemouse_shared_control_vla |
| `kaixi_crospi_ws/src/.../test_trossen/tasks/trossen_vla_shared_control.json` | betfsm 任务列表：指向 vla_spacemouse_blend.etasl.lua |
| `lerobot_trossen/important_code/inference/run_inference.py` | VLA 推理主入口（Python 3.12 venv）；`run_inference_rtc.py` 不存在，勿混淆；`--crospi` 切换机器人类；`--rtc` 默认 False |
| `lerobot_trossen/important_code/inference/actor_thread.py` | 执行线程：**10 Hz**（默认 --control-fps=10）从队列取动作 → send_action() → put_actual() |
| `lerobot_trossen/important_code/inference/rviz_publisher.py` | UDP 发送端：三类消息（ACTUAL/PREDICTED/ALPHA）始终启动 |
| `lerobot_trossen/packages/.../crospi_follower.py` | 新机器人类：观测从 trossen SDK 读取（只读），send_action() 为空操作 |
| `lerobot_trossen/packages/.../config_crospi_follower.py` | CroSPIFollower 配置（无运动控制参数） |

### UDP 通信协议（端口 9788，反向 9789）

- 类型 0 (MSG_ACTUAL)：shape [7,]，actor_thread 当前步 VLA 目标关节角 → /joint_states_VLA → eTaSL
- 类型 1 (MSG_PREDICTED)：shape [N,7]，完整预测 chunk → RViz 橙色轨迹线
- 类型 2 (MSG_ALPHA)：shape [1,]，共享控制 alpha → /shared_control/alpha → eTaSL
- 反向 9789（已实施）：vla_ros_bridge_node `_joint_states_cb` → pickle([7,] float64) → lerobot `rviz_publisher._recv_loop` → `get_latest_joints()` → inference_thread 覆盖 raw_obs joint 字段

### eTaSL 混合权重（当前状态：动态权重已激活）

WLN-QP 最小化 `sum_i(weight_i × expr_i²)`，权重放在 Constraint{weight=} 字段：
- VLA 关节跟踪：`weight = w_vla`（arm joints 0-5），`weight = w_gripper`（joint_6）
- SpaceMouse 笛卡尔：`weight = w_human`

**权重是相对的（非绝对）：** `w_vla=w_human=1` → VLA:SpaceMouse = 50:50；`w_vla=2,w_human=1` → VLA 占 2/3。
默认值：`w_vla=1, w_human=1, w_gripper=1`（初始）。

CroSPI 驱动频率：**200 Hz**（setup.json `periodicity: 0.005`）。eTaSL QP：~100 Hz（`periodicity: 0.01`）。

**当前 lua 文件实际状态**（`vla_spacemouse_blend.etasl.lua` 第 58-67 行）：
```lua
-- alpha = ctx:createInputChannelScalar("alpha_input", 0.5)  ← 注释掉
alpha = 0.5  ← 硬编码
local eps     = constant(1e-6)
-- local w_vla   = constant(1.0) - alpha + eps  ← 注释掉
local w_vla   = 1  ← 硬编码
-- local w_human = alpha + eps  ← 注释掉
local w_human = 1  ← 硬编码
```

**激活动态 α 需要两步**（TODO，未做）：
1. lua 文件：取消注释 `alpha = ctx:createInputChannelScalar(...)` 和两个 `w_vla/w_human` 行，注释掉硬编码值
2. setup.json：在 inputhandlers 末尾加入 `{"is-TopicInputHandler": true, "topic-name": "/shared_control/alpha"}`

注：bridge 已在 30Hz 发布 `/shared_control/alpha`（CrospiInput 格式），接收端管道配置好后直接可用。

**已删除**：`/shared_control/mode` 话题及 `_control_mode` 状态（FSM 是过度工程，α 值本身即控制比；α=0 即纯 VLA，α=1 即纯 SpaceMouse）

### vla_spacemouse_blend 任务规范参数（JSON Schema 定义）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `activate_linear` | `true` | 是否启用 SpaceMouse 线速度控制（必填） |
| `activate_angular` | `false` | 是否启用 SpaceMouse 角速度控制（必填） |
| `linear_scale` | `0.3` | 线速度缩放比例 |
| `angular_scale` | `0.3` | 角速度缩放比例 |
| `execution_time` | `0` | 任务执行时间（秒），0 表示不自动停止 |
| `task_frame` | `"tcp_frame"` | 笛卡尔空间控制所用的坐标系名称 |

每个参数支持三种赋值方式：直接值、`$blackboard/...` 路径（运行时从黑板读取）、`$application_defined_parameter`（由上层应用决定）。

### 两处 IP 192.168.2.3 连接

- **CroSPI 侧（主控制）**：`trossen_vla_shared_control.setup.json` → `trossen_widowx_driver_crospi`，eTaSL 发 q_cmd
- **lerobot 侧（只读观测）**：`run_inference_rtc.py --robot-ip` → `CroSPIFollowerConfig.ip_address` → `driver.configure(serv_ip=...)`

`--crospi` 模式下通过 `connect(connect_arm=False)` 跳过 SDK 连接，关节状态改由 UDP :9789 反向通道从 bridge 获取，避免双重连接冲突。

**Why:** Python 3.12 (lerobot venv) 无法直接 import rclpy（rclpy 编译针对系统 Python 3.10），因此通过 UDP 桥接两侧。

**How to apply:** 涉及可视化、推理流程或共享控制调参时参考此架构。
