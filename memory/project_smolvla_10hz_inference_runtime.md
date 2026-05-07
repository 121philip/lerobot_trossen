---
name: SmolVLA 10Hz 推理运行时分析
description: 记录 fulloa10/smolVLA_grape_10hz_9000 的 chunk_size/n_action_steps 语义、异步/RTC 区别、30fps 相机与 10Hz 控制部署方案
type: project
originSessionId: 2026-05-07-smolvla-10hz-inference
---

## 背景

目标模型：`fulloa10/smolVLA_grape_10hz_9000`

训练数据集：`kaixiyao/widowxai_grape_grasping_V4_pos234_train_10hz`

原始 30Hz 数据集：`kaixiyao/widowxai_grape_grasping_V4_pos234_train`

关键配置：

- `chunk_size = 25`
- `n_action_steps = 10`
- `n_obs_steps = 1`
- `rtc_config = null`
- 10Hz 数据集 `meta/info.json` 中 `fps = 10`
- 原始数据集 `fps = 30`，10Hz 数据集总帧数约为原始帧数的 1/3

导师要求：确保 follower 推理控制能跑在 10Hz。相机可以 30fps 采集，但模型只应以 10Hz 用最新帧触发。

## 问题 1：`chunk_size=25, n_action_steps=10` 的含义

结论：标准 LeRobot 语义下，模型每次预测 25 个未来动作，但普通非 RTC 执行只使用前 10 个动作，然后重新推理。

在 10Hz 控制频率下：

- `chunk_size=25` 表示模型一次看当前 observation 后预测未来 2.5 秒动作。
- `n_action_steps=10` 表示每次实际执行 1.0 秒动作，然后重新规划。
- 因此“预测 25 步，执行前 10 步”是合理理解。

重要批判点：

- `predict_action_chunk()` 只返回完整 chunk，本身不会自动截断到 `n_action_steps`。
- LeRobot 的标准 `select_action()` 路径内部才会把 chunk 中前 `n_action_steps` 放进 policy action queue。
- 本仓库自定义推理线程使用 `predict_action_chunk()` + `ActionQueue.merge()`，因此必须显式处理截断，否则非 RTC 模式会执行完整 25 步，和训练/标准推理语义不一致。

解决方案：

- 在 `important_code/inference/inference_thread.py` 中区分完整预测块与实际入队块：
  - `full_original_chunk`: 模型输出的完整归一化动作块，shape `[chunk_size, action_dim]`
  - `full_robot_chunk`: 后处理/反归一化后的完整机器人动作块
  - `queued_original_chunk` / `queued_robot_chunk`: 真正放进 `ActionQueue` 的动作
- 非 RTC 模式下只入队前 `policy.config.n_action_steps` 个动作。
- RTC 模式下保留完整 chunk，因为 RTC 需要完整未来轨迹做延迟补偿和块融合。

核心语义：

```text
模型输出:
[ action_0, ..., action_24 ]      # 25 steps

非 RTC 入队:
[ action_0, ..., action_9 ]       # 10 steps

RTC 入队:
[ action_0, ..., action_24 ]      # full chunk
```

## 问题 2：这是否等同于 LeRobot Async Inference

结论：不等同，但概念相关。

`chunk_size/n_action_steps` 是 policy 的 action chunk 语义；async inference 是系统部署架构。

区别：

- `chunk_size/n_action_steps` 决定模型一次预测多少未来动作、执行多少步后重规划。
- async inference 让 robot/client 和 policy/server 解耦：机器人持续消耗 action queue，policy 后台提前生成下一块动作，减少等待推理导致的 idle frame。
- RTC 不是 async 本身。RTC 的重点是 chunk 切换和推理延迟下的新旧动作块融合，减少边界不连续。

本仓库情况：

- 当前 live inference 已经是本地多线程结构：
  - inference thread 生成 chunk
  - actor thread 按固定控制频率从 queue 取动作并发送给机器人
- 这接近“本地异步队列推理”，但不等于 LeRobot 官方 async server/client 部署。
- 非 RTC 模式默认 `queue_threshold=0`，队列空了才触发下一次推理，仍可能出现等待推理的空队列。
- RTC 模式下队列低于阈值会提前触发推理，并使用 RTC 融合新旧 chunk。

批判性建议：

- 不要因为模型有 `chunk_size=25, n_action_steps=10` 就认为已经实现了 async inference。
- 初次真实机器人测试应优先使用非 RTC baseline：
  - 控制 10Hz
  - 相机 30fps
  - 非 RTC 截断到 `n_action_steps=10`
  - 记录 inference latency、queue empty 次数、实际 actor 频率
- 如果非 RTC 经常出现 queue empty 或卡顿，再考虑 RTC 或官方 async server/client，并系统调参：
  - `queue_threshold`
  - `execution_horizon`
  - `guidance_weight`
  - 官方 async 中的 `actions_per_chunk` / `chunk_size_threshold`

## 问题 3：相机 30fps 与模型 10Hz 的关系

导师说法：“You can get frames at 30, just discard them and trigger the model only at 10 with the latest frame.”

解释：

- 相机后台可以 30fps 连续采集。
- 推理/控制 loop 只在每 0.1 秒触发一次。
- 每次触发时读取相机后台线程缓存的最新帧。
- 两次 10Hz 触发之间多采到的中间帧直接丢弃。

为什么合理：

- 该模型 `n_obs_steps=1`，只用当前 observation，不需要历史帧序列。
- 因此只要 10Hz 时刻拿到的是“最新帧”，中间帧不进入模型不会破坏输入格式。

关键风险：

- 不能简单把旧参数 `--fps` 改成 30。
- 旧代码中 `args.fps` 同时控制相机 fps、actor 执行动作频率、推理延迟换算。
- 如果 `--fps=30`，机器人控制也会变成 30Hz，导致 10Hz 训练出来的动作时间尺度压缩 3 倍：
  - 10 个动作从 1.0 秒变成 0.33 秒
  - 25 步 horizon 从 2.5 秒变成 0.83 秒
- 这会造成明显的 distribution/time-scale mismatch。

解决方案：

- 拆分运行时参数：
  - `--camera-fps`: 相机后台采集频率，默认 30
  - `--control-fps`: 模型触发、actor 执行动作、延迟换算频率，默认 10
  - `--fps`: 保留为 `--control-fps` 的兼容别名
- `inference_thread.py` 和 `actor_thread.py` 都使用 `get_control_fps(args)`。
- 相机配置只使用 `args.camera_fps`。

推荐运行命令：

```bash
python important_code/inference/run_inference.py \
  --train-dir fulloa10/smolVLA_grape_10hz_9000 \
  --camera-fps 30 \
  --control-fps 10
```

如果需要 RTC：

```bash
python important_code/inference/run_inference.py \
  --train-dir fulloa10/smolVLA_grape_10hz_9000 \
  --camera-fps 30 \
  --control-fps 10 \
  --rtc
```

## 数据集 30Hz 下采样到 10Hz 的影响

观察：

- 10Hz 数据集 meta 中 `fps=10`。
- 原始 30Hz 数据集 meta 中 `fps=30`。
- 10Hz 数据集帧数约等于原始数据集 1/3，说明大概率是真的按 stride 下采样，而不只是改 metadata。

判断：

- 如果图像、状态、动作和 timestamp 是同步下采样的，那么训练 10Hz 模型并以 10Hz 部署是合理的。
- 如果只是改了 metadata 而没有同步处理动作/视频，则会产生严重时间尺度错误。
- 本次核查更支持“已重建为 10Hz 数据集”的解释。

残余风险：

- 30Hz 到 10Hz 下采样会丢失快速接触/抓取瞬间细节。
- 如果抓葡萄动作中关键阶段变化很快，10Hz 模型可能更难学到精细闭合时机。
- 但这和导师要求 10Hz 部署一致，测试时应先验证 10Hz baseline，而不是用 30Hz 控制掩盖问题。

## 代码调整记录

主要文件：

- `important_code/inference/run_inference.py`
- `important_code/inference/inference_thread.py`
- `important_code/inference/actor_thread.py`
- `important_code/inference/rtc_runtime.py`
- `important_code/utils.py`

调整：

1. 将 `run_inference_rtc.py` 重命名为 `run_inference.py`，因为 RTC 现在只是可选功能，不应体现在主入口文件名中。
2. 将 RTC 配置、ActionQueue 创建和 debug 可视化整理到 `rtc_runtime.py`。
3. 默认不向 policy 注入 RTC config；只有 `--rtc` 时才设置 `policy.config.rtc_config`。
4. 将 `get_control_fps(args)` 放入 `important_code/utils.py`，供 inference 和 actor 共享。
5. 删除单独的 `runtime_timing.py`。
6. 非 RTC 模式下将模型完整 chunk 截断到 `n_action_steps` 后再入队。

## 当前推荐测试顺序

1. Dry run 检查模型加载、预处理、推理线程和 action queue：

```bash
python important_code/inference/run_inference.py \
  --train-dir fulloa10/smolVLA_grape_10hz_9000 \
  --dry-run \
  --camera-fps 30 \
  --control-fps 10
```

2. 真实 follower 非 RTC baseline：

```bash
python important_code/inference/run_inference.py \
  --train-dir fulloa10/smolVLA_grape_10hz_9000 \
  --robot-ip 192.168.2.3 \
  --camera-fps 30 \
  --control-fps 10
```

3. 观察日志中的关键指标：

- `control_fps=10`
- `camera_fps=30`
- `chunk_size=25`
- `n_action_steps=10`
- `Action queue empty`
- `Chunk in ...s`
- `delay=...`
- `idle_gap_ms=...`

4. 如果非 RTC baseline 经常空队列或块边界明显停顿，再启用 RTC 做第二阶段对比。

## 易错点总结

- 不要把 `--camera-fps 30` 理解为控制也 30Hz。
- 不要用 `--control-fps 30` 测试 10Hz 训练模型，除非明确是在做消融实验。
- `chunk_size=25` 不是“执行 25 步”的充分条件；非 RTC 应按 `n_action_steps=10` 执行。
- async inference、RTC、action chunking 是三个不同层面的概念：
  - action chunking: 模型输出和重规划步数
  - async inference: 部署架构，避免推理阻塞执行
  - RTC: 新旧 chunk 的实时融合和延迟补偿
- 橙色预测轨迹只是 VLA intent，不保证真实 robot 会完全执行，尤其在 CroSPI/eTaSL shared-control 约束存在时。
