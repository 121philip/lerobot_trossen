---
name: VLA shared-control system architecture
description: Current VLA + SpaceMouse + CroSPI/eTaSL runtime architecture, startup order, data flow, and active interfaces.
type: project
originSessionId: a0411ee3-9f34-42f4-a7f6-b87b0845d408
last_updated: 2026-05-28
---

## System Architecture

SmolVLA and SpaceMouse run in parallel on the WidowX AI follower arm. SmolVLA produces action chunks and optional Sentinel reliability weights in `lerobot_trossen`; CroSPI/eTaSL executes the task in `kaixi_crospi_ws`.

Active control is direct-weight based:

- UDP `MSG_ACTUAL=0`: current VLA action / target source.
- UDP `MSG_PREDICTED=1`: predicted chunk for RViz marker only.
- UDP `MSG_WEIGHTS=2`: Sentinel arm weights `[w_vla, w_human]`.
- ROS2 `/shared_control/weights`: `crospi_interfaces/Input(names=["w_vla","w_human","w_gripper"], data=[...])`.
- `MSG_ALPHA`, `/shared_control/alpha`, and `/shared_control/alpha_monitor` are historical/deprecated.

## Runtime Layers

```text
SpaceMouse driver
  -> /spacenav/twist
  -> /spacenav/joy

CroSPI core
  -> trossen_vla_shared_control.setup.json
  -> eTaSL task + Trossen CroSPI driver
  -> publishes /joint_states

VLA bridge / visualization
  -> receives UDP from lerobot_trossen
  -> publishes /joint_states_VLA, /pose_VLA, /shared_control/weights, /shared_control/mode
  -> publishes /actual/joint_states_rviz and /predicted_ee_marker for RViz

BeTFSM runner
  -> MovingHome -> Wait(5s) -> spacemouse_control -> spacemouse_shared_control_vla

LeRobot inference
  -> run_inference.py --sentinel --no-sentinel-log-only --crospi --display-data
```

## Canonical Startup Order

Run the CroSPI side first:

```bash
ros2 launch spacenav classic-launch.py
```

```bash
ros2 run crospi_core crospi_node --ros-args \
  -p config_file:="$[crospi_application_template]/applications/trossen_applications/trossen_vla_shared_control.setup.json" \
  -p simulation:=false
```

```bash
ros2 launch crospi_application_template trossen_follower_visualization.launch.py
```

```bash
python3 kaixi_crospi_ws/src/crospi_application_template/skill_specifications/libraries/test_trossen/skill_specifications/trossen_vla_shared_control_runner.py
```

Optional baseline logger:

```bash
ros2 run crospi_application_template spacemouse_logger_node.py \
  --condition B1 \
  --trial 1 \
  --out-dir ./logs \
  --vmax-yaml ./vmax.yaml
```

Then run the VLA side from `lerobot_trossen` for B2/B3:

```bash
python important_code/inference/run_inference.py \
  --sentinel \
  --no-sentinel-log-only \
  --crospi \
  --display-data
```

## Key Files

| File | Responsibility |
| --- | --- |
| `kaixi_crospi_ws/src/crospi_application_template/example_nodes/vla_ros_bridge_node.py` | UDP -> ROS2 bridge; publishes VLA targets, pose target, weights, mode, and RViz marker topics. |
| `kaixi_crospi_ws/src/crospi_application_template/example_nodes/spacemouse_logger_node.py` | B1/B3 SpaceMouse logger for control-effort metrics. |
| `kaixi_crospi_ws/src/crospi_application_template/example_nodes/calibrate_spacemouse_vmax.py` | SpaceMouse vmax calibration for logger normalization. |
| `kaixi_crospi_ws/src/crospi_application_template/example_nodes/check_csv.py` | Trial CSV sanity checker and metric recomputation helper. |
| `kaixi_crospi_ws/src/crospi_application_template/applications/trossen_applications/trossen_follower_visualization.launch.py` | Starts bridge, robot_state_publisher, and RViz2. |
| `kaixi_crospi_ws/src/crospi_application_template/applications/trossen_applications/trossen_vla_shared_control.setup.json` | CroSPI setup; consumes `/spacenav/twist`, `/joint_states_VLA`, `/pose_VLA`, and `/shared_control/weights`. |
| `kaixi_crospi_ws/src/crospi_application_template/task_specifications/libraries/dummy_lib/task_specifications/vla_spacemouse_blend.etasl.lua` | eTaSL task; blends VLA pose/joint tracking with SpaceMouse Cartesian constraints using direct weights. |
| `kaixi_crospi_ws/src/crospi_application_template/skill_specifications/libraries/test_trossen/skill_specifications/trossen_vla_shared_control_runner.py` | BeTFSM runner for homing and task activation. |
| `lerobot_trossen/important_code/inference/run_inference.py` | SmolVLA inference entry point; B2/B3 live command uses Sentinel direct weights. |
| `lerobot_trossen/important_code/inference/rviz_publisher.py` | UDP sender for actual/predicted/weights and reverse joint-state receiver. |

## eTaSL Weights

The active Lua task reads:

```lua
local w_vla = ctx:createInputChannelScalar("w_vla", 1.0)
local w_human = ctx:createInputChannelScalar("w_human", 1.0)
local w_gripper = ctx:createInputChannelScalar("w_gripper", 1.0)
```

Weights are used in `Constraint{weight=...}`. The bridge derives `w_gripper` locally from SpaceMouse gripper override state; the VLA side only sends `[w_vla, w_human]`.

## Python Runtime Boundary

ROS2 Humble `rclpy` targets system Python 3.10, while LeRobot runs in Python 3.12. The local UDP bridge keeps those runtimes isolated. If the project later moves to a ROS2 distribution whose `rclpy` ABI matches the LeRobot environment, the UDP bridge could be reconsidered.
