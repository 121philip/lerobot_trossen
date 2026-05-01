---
name: SmolVLA LoRA 训练配置与冻结策略
description: LeRobot 内置 PEFT/LoRA 支持、SmolVLA 默认 target_modules、fine_tune_smolvla.sh 修改方法、train_expert_only 冻结策略
type: project
originSessionId: ba8c9eb5-573a-41fd-bb7b-6bee63f0b26c
---
# SmolVLA LoRA 训练

## 背景

在 SmolVLA 全量微调基础上，改用 LoRA（Low-Rank Adaptation）以降低显存、减少对 VLM 骨干的灾难性遗忘。

## LeRobot LoRA 支持机制

- `lerobot/configs/default.py` 中定义 `PeftConfig` dataclass
- `TrainPipelineConfig.peft: PeftConfig | None = None`（默认 None = 禁用）
- 训练脚本中：`if cfg.peft is not None: policy.wrap_with_peft(peft_cli_overrides=...)`
- SmolVLA 在 `modeling_smolvla.py:482` 预定义了默认 target_modules：
  - VLM expert 的 q/v 注意力投影（`lm_expert.*.(q|v)_proj`）
  - Action 投影层（`state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out`）

## PeftConfig 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `method_type` | `"LORA"` | PEFT 方法类型 |
| `r` | `16` | LoRA rank，越大越接近全量微调 |
| `target_modules` | SmolVLA 有内置默认值 | 无需手动指定 |
| `full_training_modules` | SmolVLA 有内置默认值 | 新增层全量训练 |

## 对 fine_tune_smolvla.sh 的修改

在 `--dataset.repo_id` 之后、训练超参之前，添加：

```bash
--peft.method_type=LORA \
--peft.r=16 \
```

同时将 `output_dir` 和 `job_name` 加 `_lora` 后缀，避免覆盖全量微调结果。

**Why:** LoRA 只训练约 1-5% 的参数，大幅降低显存，且保留 VLM 预训练能力，适合机器人任务适配场景。

**How to apply:** 每次新建 LoRA 实验时，参考此模式修改 output_dir/job_name/peft 参数；可通过调整 `--peft.r` 控制参数量（r=8 更轻量，r=32 更接近全量）。

---

## SmolVLA 冻结策略（train_expert_only 等）

SmolVLA `SmolVLAConfig`（`configuration_smolvla.py:70-74`）有三个冻结相关字段，**全部默认为 True**：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `freeze_vision_encoder` | `True` | 冻结视觉编码器（无论其他设置如何） |
| `train_expert_only` | `True` | 只训练 action expert 层，冻结 VLM 语言主干 |
| `train_state_proj` | `True` | 是否训练 state 投影层 |

这三个参数在 `modeling_smolvla.py:558-584` 被传入 `SmolVLMWithExpertModel` 并通过 `set_requires_grad()` 生效。

### 典型使用场景

- **默认（train_expert_only=True）：** 只更新 action expert，VLM 骨干完全冻结，显存最省、速度最快。
- **全量微调（train_expert_only=False）：** VLM 主干也参与训练，需要更多显存，但对任务语义理解更强。
- **视觉编码器解冻（freeze_vision_encoder=False）：** 极少用，除非数据集图像分布与预训练差异极大。

### 配置方式（命令行覆盖）

```bash
# 默认行为（无需显式设置）
--policy.train_expert_only=true

# 全量微调 VLM 主干
--policy.train_expert_only=false

# 同时解冻视觉编码器（慎用）
--policy.freeze_vision_encoder=false
```

**Why:** `train_expert_only=True` 是官方推荐的微调起点，因为 action expert 是与机器人任务直接相关的新增模块，而 VLM 骨干已具备强大的视觉-语言理解能力，无需大幅更新。

**How to apply:** 初始实验用默认值（train_expert_only=True），若任务泛化性不足再考虑设为 False 并配合 LoRA 使用。
