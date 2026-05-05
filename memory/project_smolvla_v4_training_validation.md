---
name: SmolVLA V4 数据集处理、微调与离线验证
description: V4 position2/3/4 数据集分层 train/validation split、merge、Trossen checkpoint LoRA 微调、自动训练和离线 checkpoint 评估流程
type: project
---
# SmolVLA V4 训练与验证流程

## 背景问题

用户准备使用三个新录制的 LeRobot 数据集微调 SmolVLA：

- `kaixiyao/widowxai_grape_grasping_V4_position2`
- `kaixiyao/widowxai_grape_grasping_V4_position3`
- `kaixiyao/widowxai_grape_grasping_V4_position4`

Hub 元数据确认三者兼容：同为 `widowxai_follower_robot`，30 FPS，7 维 `observation.state` / `action`，两路 480x640 视频：`observation.images.wrist` 与 `observation.images.right`。

本地 LeRobot 训练配置不支持直接传多个 `dataset.repo_id`；`TrainPipelineConfig.validate()` 会对 list repo 抛 `NotImplementedError("LeRobotMultiDataset is not currently implemented.")`。因此必须先 split/merge 成单个 train repo。

## 数据集 split/merge 决策

最初方案是每个源数据集按 episode 顺序末尾 20% 做 validation。后来用户澄清：

- 50 episodes 数据集 = 5 个目标位置，每个位置 10 次动作
- 40 episodes 数据集 = 4 个目标位置，每个位置 10 次动作

所以最终采用“每个目标位置内部分层抽样”：

- 每个目标位置 10 条 episode
- 每个位置前 8 条进 train
- 每个位置后 2 条进 validation

具体 split：

- 50 episodes：train `[0-7,10-17,20-27,30-37,40-47]`，val `[8,9,18,19,28,29,38,39,48,49]`
- 40 episodes：train `[0-7,10-17,20-27,30-37]`，val `[8,9,18,19,28,29,38,39]`

合并输出 repo：

- train: `kaixiyao/widowxai_grape_grasping_V4_pos234_train`
- val: `kaixiyao/widowxai_grape_grasping_V4_pos234_val`

实现文件：

- `important_code/training/prepare_smolvla_v4_datasets.py`
- `important_code/training/tests/test_prepare_smolvla_v4_datasets.py`

运行数据准备：

```bash
uv run python important_code/training/prepare_smolvla_v4_datasets.py --dry-run --push-to-hub
uv run python important_code/training/prepare_smolvla_v4_datasets.py --push-to-hub
```

`prepare_smolvla_v4_datasets.sh` 已删除；用户更倾向直接跑 Python。

## 微调配置

用户决定从 `TrossenRoboticsCommunity/smolvla_solo_red_block` 继续微调，而不是 `lerobot/smolvla_base`，因为该 checkpoint 与当前硬件更接近：同为 WidowXAI、7 维 state/action、两路相机。

训练脚本：

- `important_code/training/fine_tune_smolvla.sh`
- 通过 `uv run python important_code/training/run_train_clean.py` 启动
- 用户终端环境中无需 `UV_CACHE_DIR=/tmp/uv-cache`

关键训练参数：

```bash
--policy.path=TrossenRoboticsCommunity/smolvla_solo_red_block
--dataset.repo_id=kaixiyao/widowxai_grape_grasping_V4_pos234_train
--policy.repo_id=kaixiyao/smolvla_widowx_grape_grasping_V4_pos234_lora
--output_dir=outputs/train/smolvla_widowx_grape_grasping_V4_pos234_lora
--job_name=smolvla_widowx_grape_grasping_V4_pos234_lora
--peft.method_type=LORA
--peft.r=16
--batch_size=8
--steps=120000
--save_freq=5000
--policy.scheduler_decay_steps=120000
```

相机键名必须对齐 Trossen checkpoint：

```json
{
  "observation.images.right": "observation.images.cam_main",
  "observation.images.wrist": "observation.images.cam_wrist"
}
```

注意：`fine_tune_smolvla.sh` 只使用 train set，这是正确的。validation set 不应参与训练，否则不再是 validation。

## validation set 的用途与 NXP 文章结论

用户引用 Hugging Face/NXP 文章：

- 最终 checkpoint 不应只看 training loss
- 应根据 training set 和 validation set 的表现选择 checkpoint
- 文章还指出，训练略微超过开始 overfit 的点，有时会提高整体准确率

因此推荐训练到 `120000` steps，并保存每 `5000` steps checkpoint。重点比较：

```text
40000, 60000, 80000, 100000, 120000
```

训练步数估算基于分层 split 后约 `112` 个 train episodes、约 `6.7 万` train frames；batch size 8 时，`120000` steps 大约提供 `96 万` frame-samples。`scheduler_decay_steps` 同步设为 `120000`，避免默认 `30000` 后学习率过早降到底。

用户明确评估时不会开真实机器人。因此实现的是离线 validation proxy，而不是真实 success rate。

## 离线 checkpoint 评估

新增脚本：

- `important_code/training/evaluate_smolvla_v4_checkpoints.py`
- `important_code/training/tests/test_evaluate_smolvla_v4_checkpoints.py`

作用：

- 加载指定训练目录下的 checkpoint
- 同时评估 train dataset 和 validation dataset
- 对 recorded frames 做 policy action chunk prediction
- 与 dataset 中的 ground-truth `action` 计算 MAE/RMSE/max error/latency
- 不连接真实机器人

默认输入：

```text
train_dir: outputs/train/smolvla_widowx_grape_grasping_V4_pos234_lora
train_dataset: kaixiyao/widowxai_grape_grasping_V4_pos234_train
val_dataset: kaixiyao/widowxai_grape_grasping_V4_pos234_val
steps: 40000,60000,80000,100000,120000
```

输出：

```text
outputs/validation/smolvla_widowx_grape_grasping_V4_pos234_lora/offline_checkpoint_eval.csv
outputs/validation/smolvla_widowx_grape_grasping_V4_pos234_lora/offline_checkpoint_recommendation.json
outputs/validation/smolvla_widowx_grape_grasping_V4_pos234_lora/checkpoint_metrics_overview.png
outputs/validation/smolvla_widowx_grape_grasping_V4_pos234_lora/checkpoint_joint_mae.png
outputs/validation/smolvla_widowx_grape_grasping_V4_pos234_lora/checkpoint_joint_rmse.png
```

单独评估已有 checkpoint：

```bash
uv run python important_code/training/evaluate_smolvla_v4_checkpoints.py
```

可调参数示例：

```bash
uv run python important_code/training/evaluate_smolvla_v4_checkpoints.py \
  --steps all \
  --frame-stride 50 \
  --train-episodes all \
  --val-episodes all
```

重要限制：离线 action error 只能作为 proxy，用来辅助排除明显差的 checkpoint。真实机器人 success rate 仍是最终指标；如果不开机器人，就用离线 validation MAE/RMSE 作为当前可行的替代标准。

### validation 评估细节

`evaluate_smolvla_v4_checkpoints.py` 的离线评估逻辑：

1. 加载一个 checkpoint，例如 `040000/pretrained_model`。
2. 加载 train/val LeRobot dataset。
3. 对每个 episode 按 `--frame-stride` 选取起始帧。
4. 对该起始帧调用模型预测一个 action chunk。
5. 与 dataset 中后续 ground-truth `action` 对齐比较。
6. 汇总 MAE/RMSE/max error/latency。

`--frame-stride` 不是 FPS，而是“每隔多少个 dataset frame 取一个起始帧”。当前数据 30 FPS、模型 `chunk_size=50`，所以默认 `--frame-stride 50` 表示每约 `50/30=1.67s` 做一次非重叠 chunk 评估：

```text
起始帧 0   -> 比较 0-49
起始帧 50  -> 比较 50-99
起始帧 100 -> 比较 100-149
```

如果设为 `30`，表示每 1 秒一个起点，窗口会重叠；如果设为 `1`，几乎每帧都作为起点，评估最密集但很慢。

日志中的：

```text
train episode 13: 599 frames
```

表示 train dataset 里 `episode_index == 13` 的样本行数是 599；在 30 FPS 下约 `599/30=19.97s`。每行包含同一时刻的图像、state、action、timestamp、episode/frame index 等。

### CSV / JSON / 图像输出说明

`offline_checkpoint_eval.csv` 每一行是某个 checkpoint 在某个 split 上的统计结果。主要字段：

- `split`: `train` 或 `val`
- `checkpoint_step`: checkpoint step，例如 `40000`
- `checkpoint_path`: 实际加载的 `pretrained_model` 路径
- `dataset_repo`: 使用的数据集 repo
- `episodes`: 评估过的 episode 列表
- `num_episodes`: 评估 episode 数
- `num_chunks`: 预测 action chunk 次数
- `num_action_frames`: 参与误差统计的 action frame 数
- `mae_mean`: 全 joint 平均绝对误差
- `rmse_mean`: 全 joint 均方根误差
- `max_abs_mean`: 各 joint 最大绝对误差的均值
- `latency_ms_mean` / `latency_ms_max`: 推理延迟
- `mae_<joint>` / `rmse_<joint>` / `max_abs_<joint>`: 每个 joint 的误差

`offline_checkpoint_recommendation.json` 按最低 validation `mae_mean` 给出离线推荐 checkpoint，例如：

```json
{
  "selection_metric": "lowest offline validation MAE",
  "best_checkpoint_step": 40000,
  "best_checkpoint_path": "outputs/train/.../checkpoints/040000/pretrained_model",
  "val_mae_mean": 0.028856069381747926,
  "train_mae_mean": null,
  "train_val_mae_gap": null
}
```

如果只评估了 val，没有评估 train，则 `train_mae_mean` 和 `train_val_mae_gap` 会是 `null`。

评估脚本使用 `matplotlib` 生成 PNG 折线图：

- `checkpoint_metrics_overview.png`: `mae_mean`、`rmse_mean`、`max_abs_mean`、`latency_ms_mean` 随 checkpoint step 变化
- `checkpoint_joint_mae.png`: 每个 joint 的 MAE 随 checkpoint step 变化
- `checkpoint_joint_rmse.png`: 每个 joint 的 RMSE 随 checkpoint step 变化

如果 CSV 只有一个 checkpoint/split row，图里只有一个点；评估多个 checkpoint 后才会形成折线。

### checkpoints/last 的含义

`outputs/train/.../checkpoints/last` 是 LeRobot 训练脚本维护的 symlink，不是 validation 最优 checkpoint。每次保存 checkpoint 后，LeRobot 调用 `update_last_checkpoint(checkpoint_dir)`，把 `last` 指向最新保存的 step 目录。

例如当前检查结果：

```text
005000
010000
015000
020000
025000
030000
035000
040000
last -> 040000
```

因此：

```text
checkpoints/last/pretrained_model
```

等价于：

```text
checkpoints/040000/pretrained_model
```

但如果后续继续训练到 `045000`，`last` 会改指向 `045000`。选择 checkpoint 应看 validation/offline 评估结果或真实 rollout success，而不是直接把 `last` 当最佳模型。

## LoRA checkpoint 加载报错与修复

用户运行：

```bash
/home/masterthesis/lerobot_trossen/.venv/bin/python \
  /home/masterthesis/lerobot_trossen/important_code/training/evaluate_smolvla_v4_checkpoints.py
```

报错：

```text
FileNotFoundError: outputs/train/smolvla_widowx_grape_grasping_V4_pos234_lora/checkpoints/040000/pretrained_model/model.safetensors
```

同时脚本输出：

```text
Skipping missing checkpoint: .../060000/pretrained_model
Skipping missing checkpoint: .../080000/pretrained_model
Skipping missing checkpoint: .../100000/pretrained_model
Skipping missing checkpoint: .../120000/pretrained_model
Checkpoints: [40000]
```

分析结论：

- `Skipping missing checkpoint` 本身不是错误；当前本地训练只到 `040000`，所以 `060000` 到 `120000` 不存在。
- `040000/pretrained_model` 目录下有 `adapter_model.safetensors` 和 `adapter_config.json`，但没有 `model.safetensors`。
- 这是 PEFT/LoRA adapter checkpoint，不是完整 SmolVLA 权重。
- 原评估脚本直接调用 `SmolVLAPolicy.from_pretrained(checkpoint_path)`，该路径按完整模型加载，因此会去找 `model.safetensors` 并失败。
- 正确加载方式是用 LeRobot 的 `make_policy(...)` factory：当 `config.use_peft=true` 时，factory 会先读取 `adapter_config.json` 里的 `base_model_name_or_path`，加载 base policy，再用 `PeftModel.from_pretrained(...)` 挂载 adapter。

实际修复：

- 在 `important_code/training/evaluate_smolvla_v4_checkpoints.py` 中新增 `load_policy_config(checkpoint_path)`。
- 从 checkpoint 的 `config.json` 读取 policy config 后，将 `config.pretrained_path` 强制设为当前 checkpoint 路径。
- `load_policy_bundle(...)` 改为调用 `make_policy(policy_config, ds_meta=..., rename_map=...)`，不再直接调用 `SmolVLAPolicy.from_pretrained(...)`。
- 传入训练时一致的相机 rename map：

```python
{
    "observation.images.right": "observation.images.cam_main",
    "observation.images.wrist": "observation.images.cam_wrist",
}
```

否则会在 policy/dataset feature consistency check 处报：

```text
Missing features: ['observation.images.cam_main', 'observation.images.cam_wrist']
Extra features: ['observation.images.right', 'observation.images.wrist']
```

验证命令：

```bash
.venv/bin/python -m unittest important_code.training.tests.test_evaluate_smolvla_v4_checkpoints
.venv/bin/python important_code/training/evaluate_smolvla_v4_checkpoints.py \
  --steps 40000 \
  --skip-train \
  --val-episodes 0 \
  --max-frames-per-episode 1
```

验证结果：单测通过；smoke test 成功加载 `040000` LoRA checkpoint，完成一次 validation action prediction，并写出：

```text
outputs/validation/smolvla_widowx_grape_grasping_V4_pos234_lora/offline_checkpoint_eval.csv
outputs/validation/smolvla_widowx_grape_grasping_V4_pos234_lora/offline_checkpoint_recommendation.json
```

如果只想评估本地已有 checkpoint，推荐：

```bash
.venv/bin/python important_code/training/evaluate_smolvla_v4_checkpoints.py --steps all
```

## Hugging Face 模型文件很小的原因

用户查看 Hugging Face repo：

```text
https://huggingface.co/kaixiyao/smolvla_widowx_grape_grasping_V4_pos234_lora/tree/main
```

发现文件很小。检查结果：

- repo: `kaixiyao/smolvla_widowx_grape_grasping_V4_pos234_lora`
- Hub storage 约 `2.99 MB`
- 文件包括 `adapter_model.safetensors`、`adapter_config.json`、`config.json`、processor config 和 README
- 没有完整 base model 的 `model.safetensors`

这是正常的，因为上传的是 LoRA adapter。`adapter_config.json` 中记录：

```json
"base_model_name_or_path": "TrossenRoboticsCommunity/smolvla_solo_red_block"
```

因此推理/评估时必须按“base model + adapter”组合加载。LoRA 仓库小不代表训练没保存成功；它只保存微调增量权重。

## 自动化流程

新增：

- `important_code/training/prepare_then_fine_tune_smolvla_v4.sh`

默认流程：

```text
prepare datasets -> fine-tune -> offline train/val checkpoint evaluation
```

数据已准备好时，避免重复 split/merge，必须用：

```bash
bash important_code/training/prepare_then_fine_tune_smolvla_v4.sh --skip-data-prep
```

只训练、不做离线评估：

```bash
bash important_code/training/prepare_then_fine_tune_smolvla_v4.sh --skip-data-prep --skip-eval
```

预览数据准备命令，不训练：

```bash
bash important_code/training/prepare_then_fine_tune_smolvla_v4.sh --dry-run
```

自动脚本支持：

```bash
--skip-data-prep
--skip-info
--skip-eval
--eval-steps STEPS
--eval-frame-stride N
--eval-max-frames-per-episode N
```

如果数据集已经存在，不要裸跑 `prepare_then_fine_tune_smolvla_v4.sh`，因为会重新执行 split/merge，可能与 Hub repo 或本地 cache 冲突。

## 已知验证命令

```bash
uv run python -m unittest important_code.training.tests.test_prepare_smolvla_v4_datasets -v
uv run python -m unittest important_code.training.tests.test_evaluate_smolvla_v4_checkpoints -v
uv run python -m py_compile important_code/training/prepare_smolvla_v4_datasets.py
uv run python -m py_compile important_code/training/evaluate_smolvla_v4_checkpoints.py
bash -n important_code/training/fine_tune_smolvla.sh
bash -n important_code/training/prepare_then_fine_tune_smolvla_v4.sh
```
