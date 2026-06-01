# Memory Index

- **维护规则**：以后新增或更新 Claude 本地记忆（`~/.claude/projects/-home-masterthesis-lerobot-trossen/memory/`）时，必须同步更新本仓库的 `memory/` 目录，确保可提交到 GitHub 的版本保持最新。

- [RealSense D405 Camera Configuration](project_realsense_camera_config.md) — video node selection (YUYV vs UYVY), Path vs int index, lerobot occupancy false-negative, unsupported controls
- [fourcc: YUY2 vs YUYV on Linux](project_fourcc_linux_windows.md) — YUY2 silently fails on Linux V4L2; use YUYV instead
- [Follower 安全回正与诊断工具](project_follower_homing_safety.md) — disconnect() 三步回正(joint_3预步骤)、Q/R键盘控制、电机健康监控脚本、JointLimit正确属性名
- [RViz 推理轨迹可视化系统](project_rviz_visualization.md) — UDP桥接架构、预测EE marker、CroSPI shared-control 橙蓝轨迹脱节诊断、alpha/target锁存/eTaSL约束问题
- [SmolVLA LoRA 训练配置与冻结策略](project_lora_training.md) — PEFT/LoRA 支持、target_modules、train_expert_only/freeze_vision_encoder/train_state_proj 三个冻结参数（默认均True）
- [SmolVLA V4 数据集处理、微调与离线验证](project_smolvla_v4_training_validation.md) — V4 position2/3/4 分层 split/merge、Trossen checkpoint LoRA 微调、train/val 离线 checkpoint 评估、CSV/JSON/Matplotlib 图输出、checkpoints/last 含义、LoRA adapter 加载修复、HF 文件小原因、自动化脚本与 skip-data-prep 注意事项
- [SmolVLA 10Hz 推理运行时分析](project_smolvla_10hz_inference_runtime.md) — `chunk_size=25`/`n_action_steps=10` 语义、async 与 RTC 区别、30fps 相机 + 10Hz 控制部署、非 RTC 截断和运行命令
- [VLA 共享控制系统架构](project_vla_shared_control_architecture.md) — 关键文件、UDP 协议、eTaSL 权重配置、双进程桥接原因、CroSPI 200Hz/VLA 10Hz 频率
- [bridge node FK 正向运动学分析结论](project_fk_analysis_verified.md) — FK 参数已验证正确；predicted_ee_marker 偏差是共享控制预期行为，非 Bug；改进方向
- [WidowX AI Follower 运动学链结构](project_urdf_kinematic_chain.md) — 完整关节链、EE 路径、gripper 分支、7-DOF 顺序
- [VLA 按钮触发抖动调试记录](project_vla_button_shaking_debug.md) — 统一根因(VLA obs-action mismatch)、左键仅HUMAN_ONLY有效、alignment=3s>ramp=1.5s设计、全部修复清单
- [regression-CBC 归一化空间修复](project_confidence_cbc_normalization_fix.md) — 夹爪关节尺度偏差问题、CBC 改用 actions_normalized（归一化空间）、tracking 保留机器人空间、关键文件与设计约束
- [Sentinel EMA beta 分析与简化](project_sentinel_ema_analysis.md) — 推理频率 1 Hz 计算、c_progress 瞬时恢复+r_smooth 渐进恢复、beta=0.5 选型、r_raw 移除、CLI vs 类默认值不一致 bug、画图简化为 2 子图
