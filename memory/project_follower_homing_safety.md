---
name: Follower 安全回正问题及解决方案
description: 推理退出时 follower 机械臂撞击桌面的根因分析、disconnect() 修复、键盘交互控制实现
type: project
originSessionId: 45fb720f-394c-4378-861d-e0e5751a5c54
---
## 问题

推理过程中按 Ctrl+C 退出，follower 机械臂在回正时末端执行器撞击桌面，损伤硬件。

**Why:** `disconnect()` 直接从推理时的任意低位跳到 staged 位置，运动轨迹可能穿越桌面。  
**How to apply:** 凡是修改 disconnect/homing 逻辑时，必须保证先做安全预步骤再回正。

---

## 根因分析

文件：`packages/lerobot_robot_trossen/src/lerobot_robot_trossen/widowxai_follower.py`

原 `disconnect()` 两步：
1. 当前位置 → staged `[0, π/3, π/6, π/5, 0, 0, 0]`
2. staged → sleep `[0,0,0,0,0,0,0]`

经关节角度测试确认：joint_3（前臂滚转）在推理时偏离 staged 角度（π/5 = 36°）时，直接移动到 staged 会导致末端执行器扫过桌面。

---

## 解决方案：disconnect() 三步回正

在 `widowxai_follower.py` 的 `disconnect()` 中，在原有两步之前插入预步骤：

```
Step 0（新增）: 仅将 joint_3 调到 staged_positions[3]（π/5 = 36°），其余关节不动
                goal_time=2.0s, blocking=True
Step 1（原有）: 移动到 staged 位置 [0, π/3, π/6, π/5, 0, 0, 0]
Step 2（原有）: 移动到 sleep 位置 [0,0,0,0,0,0,0]
```

**Why joint_3:** 通过 `joint_angle_test.py` 目视确认，joint_3 是导致末端撞击的关键关节。  
**How to apply:** 如果 staged_positions 或机械臂配置改变，需重新用关节测试脚本验证回正路径。

---

## 推理过程中的键盘安全控制

文件：`important_code/inference/run_inference_rtc.py`

新增交互：
- **Q 键**：停止推理线程，机械臂就地保持（不回正）
- **R 键**：执行安全回正（三步 disconnect）
- **Ctrl+C**：等同 Q（冻结，等待 R）
- **时间到**（--duration）：正常回正

实现要点：
- `_keyboard_listener(shutdown_event, home_event, kb_stop)` 后台线程，使用 `tty.setcbreak()`
- 用 `atexit` 注册终端设置还原，防止 daemon 线程退出时终端损坏
- join 推理/执行线程后先 `kb_stop.set()` 还原终端，再重启第二个键盘线程等待 R
- 冻结等待块条件改为 `if not home_event.is_set()`（不依赖 robot 是否为 None，干跑也等待）

---

## 诊断工具

| 脚本 | 用途 |
|------|------|
| `important_code/diagnostics/joint_angle_test.py` | 交互式测试 follower 各关节角度（目视确认） |
| `important_code/diagnostics/joint_angle_test_leader.py` | 同上，针对 leader（支持 position/gc 模式切换） |
| `important_code/diagnostics/motor_health_check.py` | 单次或持续监控电机温度/力矩/错误信息 |
| `important_code/diagnostics/teleoperation_with_health.py` | 遥操作同时后台监控电机健康 |

### trossen_arm JointLimit 正确属性名

```python
lim.position_max   # 不是 position_upper
lim.position_min   # 不是 position_lower
lim.effort_max     # 不是 effort
lim.velocity_max   # 不是 velocity
```

### 健康监控阈值（经验值）

| 指标 | 警告 | 危险 |
|------|------|------|
| 驱动/转子温度 | > 55°C | > 70°C |
| 力矩占额定比 | > 40% | > 70% |
