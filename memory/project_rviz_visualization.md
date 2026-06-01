---
name: RViz inference trajectory visualization
description: Current RViz2 visualization and UDP bridge facts for SmolVLA + CroSPI shared control.
type: project
originSessionId: 191a4f2e-05f3-49c7-985e-c8a061f7bfe6
last_updated: 2026-05-28
---

## Current Architecture

SmolVLA runs in the LeRobot Python 3.12 environment; ROS2 Humble/CroSPI runs with system Python 3.10. The two sides communicate through local UDP.

Active files:

- VLA side: `lerobot_trossen/important_code/inference/rviz_publisher.py`
- CroSPI side: `kaixi_crospi_ws/src/crospi_application_template/example_nodes/vla_ros_bridge_node.py`
- Launch: `kaixi_crospi_ws/src/crospi_application_template/applications/trossen_applications/trossen_follower_visualization.launch.py`

Active UDP protocol on `127.0.0.1:9788`:

- `MSG_ACTUAL=0`: current VLA action / target source, shape `[7]`.
- `MSG_PREDICTED=1`: predicted action chunk for RViz orange marker, shape `[N,7]`.
- `MSG_WEIGHTS=2`: Sentinel direct arm weights `[w_vla,w_human]`.

Reverse UDP on `127.0.0.1:9789` sends actual follower joint state from bridge back to LeRobot observation code.

## Active ROS2 Topics

- `/joint_states_VLA`: VLA joint target, including gripper.
- `/pose_VLA`: VLA Cartesian end-effector pose target.
- `/actual/joint_states_rviz`: blue actual robot state for RViz.
- `/predicted_ee_marker`: green current EE marker plus orange predicted EE trajectory.
- `/shared_control/weights`: `crospi_interfaces/Input(names=["w_vla","w_human","w_gripper"], data=[...])`.
- `/shared_control/mode`: `SHARED` or `HUMAN_ONLY`.

`MSG_ALPHA`, `/shared_control/alpha`, `/shared_control/alpha_monitor`, `run_inference_rtc.py`, `important_code/inference/rviz_node.py`, and `important_code/visualization/launch_viz.sh` are historical/deprecated for the current runtime.

## Startup Reference

Use the canonical system startup order:

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

From `lerobot_trossen` for B2/B3:

```bash
python important_code/inference/run_inference.py \
  --sentinel \
  --no-sentinel-log-only \
  --crospi \
  --display-data
```

## Diagnostic Commands

```bash
ros2 topic hz /joint_states_VLA
ros2 topic echo /joint_states_VLA --once
ros2 topic echo /pose_VLA --once
ros2 topic hz /shared_control/weights
ros2 topic echo /shared_control/weights --once
ros2 topic echo /shared_control/mode --once
ros2 lifecycle get /crospi_node
```

## Interpretation

The orange trajectory is VLA intent, not a measured future path of the real robot. It is FK of the predicted SmolVLA action chunk. The blue robot is the measured follower after CroSPI/eTaSL solves its constraints and the Trossen driver executes.

If orange is smooth but blue diverges, compare:

- `/joint_states_VLA` and `/pose_VLA` publication freshness.
- `/shared_control/weights` values.
- SpaceMouse input on `/spacenav/twist`.
- Actual `/joint_states` feedback.
- Whether the robot has moved outside the training distribution.

Stopping the runner is not necessarily a stop command for CroSPI because the active eTaSL task can keep running with `execution_time=0`. Check `/crospi_node` lifecycle when shutting down.
