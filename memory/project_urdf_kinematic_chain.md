---
name: WidowX AI Follower 运动学链结构
description: wxai_follower.urdf 的完整关节链、EE 路径和 gripper 分支结构
type: project
originSessionId: c4f953f4-c5a0-453f-9c3e-48a83989233e
---
## 主运动学链（base → EE）

```
base_link
  → joint_0 (revolute, Z轴, xyz=[0,0,0.05725])
    → link_1
      → joint_1 (revolute, Y轴, xyz=[0.02,0,0.04625])
        → link_2
          → joint_2 (revolute, -Y轴, xyz=[-0.264,0,0])
            → link_3
              → joint_3 (revolute, -Y轴, xyz=[0.245,0,0.06])
                → link_4
                  → joint_4 (revolute, -Z轴, xyz=[0.06775,0,0.0455])
                    → link_5
                      → joint_5 (revolute, X轴, xyz=[0.02895,0,-0.0455])
                        → link_6
                          → ee_gripper (fixed, xyz=[0.156062,0,0])
                            → ee_gripper_link  ← EE 末端
```

## Gripper 分支（不在 EE 路径上）

```
link_6
  → right_carriage_joint (prismatic, xyz=[0.0865,-0.023,0]) → carriage_right
  → left_carriage_joint  (prismatic, xyz=[0.0865, 0.023,0]) → carriage_left
```

关键：`left_carriage_joint` 是夹爪运动关节（第7个自由度），不影响 EE 位置 FK 计算。

## 7-DOF 关节顺序

索引 0-5：joint_0 ~ joint_5（旋转）
索引 6：left_carriage_joint（夹爪开合，prismatic）

URDF 路径（两个副本 MD5 相同）：
- `src/trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf`
- `workspaces_ros2/kaixi_ws/install/trossen_arm_description/.../wxai_follower.urdf`

**How to apply:** 修改 FK、添加关节约束或检查 URDF 时直接参考此结构。
