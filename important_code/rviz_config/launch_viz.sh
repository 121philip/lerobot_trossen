#!/bin/bash
# 启动 RViz 可视化：rviz_node + 两个 robot_state_publisher + RViz2
#
# 使用方式：
#   终端1 (先启动): bash important_code/rviz_config/launch_viz.sh
#   终端2 (后启动): python important_code/Inference/run_inference_rtc.py --rviz [其他参数]
#
# RViz 中：
#   蓝色实体机器人          = 实际执行位置（/actual/joint_states）
#   绿色球 + 橙红色线 + 点  = 预测末端轨迹（/predicted_ee_marker）

set -e

source /opt/ros/humble/setup.bash
source /home/masterthesis/workspaces_ros2/kaixi_ws/install/setup.bash

URDF="/home/masterthesis/workspaces_ros2/kaixi_ws/install/trossen_arm_description/share/trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RVIZ_NODE="$SCRIPT_DIR/../Inference/rviz_node.py"

if [ ! -f "$URDF" ]; then
    echo "[ERROR] URDF not found: $URDF"
    exit 1
fi

# 修复 URDF 颜色，并将 link 名称直接内嵌前缀（最可靠的双机器人可视化方式）：
# 1. 替换顶层 texture → color rgba
# 2. 将 visual 中的 <material name="trossen_black"/> 内联颜色
# 3. 将所有 <link name="...">、<parent link="...">、<child link="..."> 加上命名空间前缀
#    这样 robot_state_publisher 直接发布带前缀的 TF 帧，RViz 无需 TF Prefix 就能正确渲染
URDF_ACTUAL=$(sed \
    's|<texture filename="package://trossen_arm_description/meshes/trossen_black.png"/>|<color rgba="0.2 0.3 0.8 1.0"/>|g;
     s|<material name="trossen_black"/>|<material name="trossen_black"><color rgba="0.2 0.3 0.8 1.0"/></material>|g;
     s|<link name="\([^"]*\)"|<link name="actual/\1"|g;
     s|<parent link="\([^"]*\)"|<parent link="actual/\1"|g;
     s|<child link="\([^"]*\)"|<child link="actual/\1"|g' \
    "$URDF")
# 预测轨迹现在用 /predicted_ee_marker（线段 + 小点），不再需要预测机器人模型

# 清理可能残留的旧 rviz_node 进程（避免端口 9788 冲突）
pkill -f "rviz_node.py" 2>/dev/null && sleep 0.5 || true

echo "[VIZ] Starting rviz_node (UDP receiver -> ROS2 publisher)..."
/usr/bin/python3 "$RVIZ_NODE" &
PID_NODE=$!

echo "[VIZ] Starting robot_state_publisher for actual robot (dark grey)..."
ros2 run robot_state_publisher robot_state_publisher \
    --ros-args \
    -r __ns:=/actual \
    -r joint_states:=/actual/joint_states \
    -p robot_description:="$URDF_ACTUAL" &
PID_ACTUAL=$!

PID_PREDICTED=""  # 预测轨迹改用 EE marker，无需 robot_state_publisher

RVIZ_CONFIG="$SCRIPT_DIR/trajectory_viz.rviz"

echo "[VIZ] Starting RViz2..."
echo "[VIZ] Now run: python important_code/Inference/run_inference_rtc.py --rviz"
rviz2 -d "$RVIZ_CONFIG"

# 关闭 RViz 后清理所有后台进程
kill $PID_NODE $PID_ACTUAL $PID_PREDICTED 2>/dev/null
wait 2>/dev/null
echo "[VIZ] Shutdown complete."
