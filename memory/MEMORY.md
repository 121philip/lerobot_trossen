# Memory Index

- **维护规则**：以后新增或更新 Claude 本地记忆（`~/.claude/projects/-home-masterthesis-lerobot-trossen/memory/`）时，必须同步更新本仓库的 `memory/` 目录，确保可提交到 GitHub 的版本保持最新。

- [RealSense D405 Camera Configuration](project_realsense_camera_config.md) — video node selection (YUYV vs UYVY), Path vs int index, lerobot occupancy false-negative, unsupported controls
- [fourcc: YUY2 vs YUYV on Linux](project_fourcc_linux_windows.md) — YUY2 silently fails on Linux V4L2; use YUYV instead
- [Follower 安全回正与诊断工具](project_follower_homing_safety.md) — disconnect() 三步回正(joint_3预步骤)、Q/R键盘控制、电机健康监控脚本、JointLimit正确属性名
- [RViz 推理轨迹可视化系统](project_rviz_visualization.md) — UDP桥接架构、预测EE marker、CroSPI shared-control 橙蓝轨迹脱节诊断、alpha/target锁存/eTaSL约束问题
- [SmolVLA LoRA 训练配置与冻结策略](project_lora_training.md) — PEFT/LoRA 支持、target_modules、train_expert_only/freeze_vision_encoder/train_state_proj 三个冻结参数（默认均True）
