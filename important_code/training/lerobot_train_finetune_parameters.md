# LeRobot 训练与微调参数说明

本文面向 `lerobot-train` / `TrainPipelineConfig`，重点覆盖当前项目里的 SmolVLA 微调脚本：

- `important_code/training/fine_tune_smolvla.sh`
- 本地环境：`lerobot==0.5.0`
- 官方最新稳定包：`lerobot==0.5.1`，与本地训练字段基本一致；`0.5.1` 的 `WandBConfig` 新增 `wandb.add_tags`
- 资料源：本地源码、PyPI `lerobot==0.5.1` 源码、Hugging Face LeRobot 文档、Hugging Face PEFT 文档、LeRobot GitHub 源码

官方文档里也明确建议用 `lerobot-train --help` 查看完整微调选项；但当前环境中 help 输出会被 `draccus/argparse` 的 `%` 字符格式化问题打断，所以本文以源码 dataclass 字段为准。

## CLI 规则

LeRobot 使用 `draccus` 解析配置，命令行参数就是 dataclass 字段路径。

示例：

```bash
--batch_size=64
--dataset.repo_id=kaixiyao/widowxai_grape_grasping_V4_pos234_train_10hz
--policy.path=TrossenRoboticsCommunity/smolvla_solo_red_block
--policy.optimizer_lr=2e-4
--wandb.enable=true
--rename_map='{"old.key": "new.key"}'
```

常见规则：

- 顶层字段直接写：`--steps=40000`
- 嵌套字段用点号：`--policy.chunk_size=25`
- dict/list/tuple 建议用 JSON/YAML 风格字符串，并用单引号保护 shell：`--policy.resize_imgs_with_padding='[512, 512]'`
- `--policy.path=...` 表示从 Hub 或本地 checkpoint 加载已有 policy config 和权重；命令行里的 `--policy.*` 会覆盖加载到的配置字段
- `--config_path=... --resume=true` 用于从训练配置恢复；恢复时默认以 checkpoint 保存的配置为准

## 顶层训练参数

| 参数 | 默认值 | 含义 | 调大/调小的影响 |
|---|---:|---|---|
| `dataset` | 必填 | 训练数据集配置，见下一节。 | 数据集字段、fps、camera key、action/state shape 必须和 policy 输入输出匹配。 |
| `env` | `None` | 仿真/RL 环境配置。纯离线模仿学习通常不需要；LIBERO、PushT、Aloha、MetaWorld 等训练/评测会用到。 | 配了 env 后可周期性评测成功率；会增加运行成本。 |
| `policy` | `None` | 策略模型配置。训练必须通过 `--policy.path` 加载预训练模型，或通过 `--policy.type=...` 从头建模。 | 决定模型结构、输入输出、优化器 preset。 |
| `output_dir` | 自动生成 | 训练输出目录。 | 已存在且 `resume=false` 会报错，避免覆盖旧实验。 |
| `job_name` | 自动生成 | 实验名，用于输出目录、日志、Hub 标识。 | 建议固定且可读，便于比较实验。 |
| `resume` | `False` | 是否从已有 checkpoint 恢复。 | `true` 会读取 checkpoint config；命令行新参数不一定生效，除非源码支持覆盖。 |
| `seed` | `1000` | 随机种子，用于初始化、shuffle、env。 | 固定后更可复现；不同 seed 可评估稳定性。 |
| `cudnn_deterministic` | `False` | 是否启用确定性 cuDNN。 | `true` 更可复现，但可能慢约 10-20%。 |
| `num_workers` | `4` | PyTorch DataLoader worker 数。 | 调大可加速解码/预处理；过大可能抢 CPU/RAM 或导致 IO 抖动。 |
| `batch_size` | `8` | 每个训练 step 的样本数。多 GPU 下 LeRobot 不自动缩放学习率或 steps。 | 调大梯度更稳、吞吐更高，但显存增加；调小时噪声更大，可能需要降学习率或累积更多 steps。 |
| `steps` | `100000` | 总训练迭代数。 | 调大更充分但易过拟合/耗时；调小适合快速验证。SmolVLA 官方示例常用 20k 起步。 |
| `eval_freq` | `20000` | 每多少 step 做一次 env evaluation。 | 无 `env` 时影响很小；有 env 时会显著增加耗时。 |
| `log_freq` | `200` | 每多少 step 打日志。 | 调小便于观察但日志更多；调大日志更稀疏。 |
| `tolerance_s` | `1e-4` | 训练循环时间同步/容忍阈值。 | 通常不改；实时/环境相关流程才可能需要关注。 |
| `save_checkpoint` | `True` | 是否保存 checkpoint。 | 关闭会省磁盘，但无法恢复和挑选中间模型。 |
| `save_freq` | `20000` | checkpoint 保存间隔；最后一步也会保存。 | 调小便于回滚和选最优，但占磁盘；调大更省空间。 |
| `use_policy_training_preset` | `True` | 是否使用 policy 内置优化器和 scheduler preset。 | `true` 时 `policy.optimizer_*` / `policy.scheduler_*` 生效；`false` 时必须显式传 `optimizer` 和 `scheduler`。 |
| `optimizer` | `None` | 手动优化器配置。 | 仅当 `use_policy_training_preset=false` 时需要。 |
| `scheduler` | `None` | 手动学习率调度器配置。 | 仅当 `use_policy_training_preset=false` 时需要。 |
| `eval` | `EvalConfig` | 评测配置。 | 只在有环境评测时重要。 |
| `wandb` | `WandBConfig` | Weights & Biases 日志配置。 | 不影响模型数学结果，但影响实验追踪。 |
| `peft` | `None` | 参数高效微调配置，如 LoRA。 | 可显著减少可训练参数和显存；适应能力取决于 target module 和 rank。 |
| `use_rabc` | `False` | Reward-Aligned Behavior Cloning，按奖励/进度加权样本。 | 需要 `sarm_progress.parquet`；打开后高质量样本权重更大。 |
| `rabc_progress_path` | `None` | RA-BC 进度 parquet 路径。 | 不填时会从数据集目录或 Hub 自动推断。 |
| `rabc_kappa` | `0.01` | RA-BC 高质量样本阈值。 | 调大更严格，调小更宽松。 |
| `rabc_epsilon` | `1e-6` | RA-BC 数值稳定常数。 | 通常不改。 |
| `rabc_head_mode` | `"sparse"` | 双头模型中使用 sparse 或 dense head 的选择。 | 只对支持双头/奖励对齐的模型有意义。 |
| `rename_map` | `{}` | 训练时把数据集字段名重命名成 policy 需要的字段名。 | 对迁移已有 checkpoint 很关键；映射错会导致 feature mismatch 或相机错位。 |

## 数据集参数

| 参数 | 默认值 | 含义 | 调整影响 |
|---|---:|---|---|
| `dataset.repo_id` | 必填 | Hub 数据集 repo id；源码也预留了 list，但当前 `TrainPipelineConfig.validate()` 对 list 抛 `NotImplementedError`。 | 决定训练数据来源。 |
| `dataset.root` | `None` | 本地数据集目录。`None` 时从 `$HF_LEROBOT_HOME` 缓存或 Hub 下载。 | 用本地数据可避免重复下载，也便于调试未上传数据。 |
| `dataset.episodes` | `None` | 只使用指定 episode index。 | 可做小样本调试、验证集切分；重复或负数在 0.5.1 会报错。 |
| `dataset.image_transforms` | 默认关闭 | 图像增强配置。 | 增强可提高泛化；过强会破坏视觉任务细节。 |
| `dataset.revision` | `None` | Hub dataset revision/branch/commit。 | 固定 commit 可保证复现实验。 |
| `dataset.use_imagenet_stats` | `True` | 图像归一化是否使用 ImageNet stats。 | 预训练视觉 encoder 通常保持 `true`；从头训练或特殊图像分布可评估关闭。 |
| `dataset.video_backend` | 安全默认 codec | 视频解码 backend。 | 影响兼容性和速度；通常不改。 |
| `dataset.streaming` | `False` | 是否流式读取 Hub 数据。 | 大数据集可减少本地磁盘占用；训练吞吐可能受网络影响。 |

### 图像增强参数

| 参数 | 默认值 | 含义 | 调整影响 |
|---|---:|---|---|
| `dataset.image_transforms.enable` | `False` | 是否启用训练时图像增强。 | 开启可抗光照/颜色变化，但不应让画面偏离真实部署分布太多。 |
| `dataset.image_transforms.max_num_transforms` | `3` | 每帧最多随机应用多少个 transform。 | 调大增强更强；过大可能损害动作学习。 |
| `dataset.image_transforms.random_order` | `False` | 是否随机打乱 transform 顺序。 | 增加随机性；可复现性和可解释性稍差。 |
| `dataset.image_transforms.tfs` | 默认 brightness/contrast/saturation/hue/sharpness | transform 字典。每项是 `ImageTransformConfig`。 | 控制具体增强类型和强度。 |
| `ImageTransformConfig.weight` | `1.0` | 该 transform 被采样的权重。 | 越大越常被用到。 |
| `ImageTransformConfig.type` | `"Identity"` | torchvision transform 名称。 | 决定增强类型。 |
| `ImageTransformConfig.kwargs` | `{}` | 传给 transform 的参数。 | 决定增强强度范围。 |

## Eval 与 WandB

| 参数 | 默认值 | 含义 | 调整影响 |
|---|---:|---|---|
| `eval.n_episodes` | `50` | 每次评测 episode 数。 | 调大评测更稳定但更慢。 |
| `eval.batch_size` | `50` | vector env 并行评测环境数。 | 不应大于 `n_episodes`；调大更快但占更多资源。 |
| `eval.use_async_envs` | `False` | 是否使用异步多进程 env。 | 可加速复杂 env；调试更困难。 |
| `wandb.enable` | `False` | 是否启用 W&B。 | 打开后记录 loss、lr、checkpoint artifact 等。 |
| `wandb.disable_artifact` | `False` | 即使保存 checkpoint，也不上传 artifact。 | 省上传时间/空间，但 Hub/W&B 上不可追溯 checkpoint。 |
| `wandb.project` | `"lerobot"` | W&B project 名。 | 用于组织实验。 |
| `wandb.entity` | `None` | W&B team/user。 | 多账号时需要。 |
| `wandb.notes` | `None` | run 备注。 | 记录数据集、相机、超参变更。 |
| `wandb.run_id` | `None` | 固定 run id。 | 恢复或合并日志时使用。 |
| `wandb.mode` | `None` | `online`、`offline` 或 `disabled`。 | 离线机器可用 `offline`。 |
| `wandb.add_tags` | `True` | 0.5.1 新增；把配置保存为 tags。 | 便于筛选实验；本地 0.5.0 没有此字段。 |

## PEFT / LoRA 参数

官方 PEFT 文档示例：

```bash
--peft.method_type=LORA
--peft.r=64
--peft.target_modules='...'
--peft.full_training_modules='["state_proj"]'
```

| 参数 | 默认值 | 含义 | 调整影响 |
|---|---:|---|---|
| `peft.target_modules` | `None` | 要加 adapter 的模块名后缀、列表、`all-linear` 或 regex。SmolVLA 默认会选择 LM expert 的 `q_proj`/`v_proj` 以及 state/action projection。 | 选得越多，可训练参数越多、适应力越强、显存越高；选错会训练不到关键层。 |
| `peft.full_training_modules` | `None` | 完全训练并随 adapter 保存的模块，PEFT 中对应 `modules_to_save`。 | 适合任务相关的新 projection/head；增多会提高适配力但失去部分 PEFT 省参优势。 |
| `peft.method_type` | `"LORA"` | PEFT 方法类型。当前高层接口主要支持 LoRA。 | 不同方法有不同参数和可训练结构。 |
| `peft.init_type` | `None` | adapter 初始化方式。 | 通常使用 PEFT 默认；特殊初始化可能影响初期稳定性。 |
| `peft.r` | `16` | LoRA rank。 | 越大越接近全量微调，效果上限和显存/参数量上升；越小更省但容量有限。 |

PEFT 调参经验：官方文档建议 LoRA 学习率通常可比全量微调大约高 10 倍，例如全量 `1e-4`，LoRA 可从 `1e-3` 试起。

## Policy 通用参数

所有内置 policy 都继承 `PreTrainedConfig`。

| 参数 | 默认值 | 含义 | 调整影响 |
|---|---:|---|---|
| `policy.n_obs_steps` | `1` | 使用多少个历史 observation step。 | 调大可利用时序信息，但输入更大、延迟/显存更高。 |
| `policy.input_features` | `{}` 或 `null` | 输入字段定义，key 是数据字段名，value 是 `{type, shape}`。`null` 可从数据集推断。 | 微调预训练模型时建议显式匹配 checkpoint；shape/key 错会直接失败或学错输入。 |
| `policy.output_features` | `{}` 或 `null` | 输出字段定义，通常是 `action`。 | action 维度必须和机器人控制维度一致。 |
| `policy.device` | 自动选择 | `cuda`、`cuda:0`、`cpu`、`mps` 等。 | GPU 快；CPU 仅适合调试。不可用时会自动回退。 |
| `policy.use_amp` | `False` | 自动混合精度。 | 可降显存、提速；少数模型/设备可能数值不稳。使用 `accelerate` 时由 accelerate 控制 mixed precision。 |
| `policy.use_peft` | `False` | 标记该 policy 是否使用 PEFT。 | 通常由训练流程设置；不要手动伪造。 |
| `policy.push_to_hub` | `True` | 训练结束是否上传 policy。 | `true` 时必须提供 `policy.repo_id`。 |
| `policy.repo_id` | `None` | 上传到 Hub 的模型 repo id。 | 建议包含任务、机器人、数据版本、fps。 |
| `policy.private` | `None` | Hub repo 是否私有。 | 需要保护数据/模型时设 `true`。 |
| `policy.tags` | `None` | Hub 模型 tags。 | 便于检索。 |
| `policy.license` | `None` | Hub 模型 license。 | 发布共享时建议明确。 |
| `policy.pretrained_path` | `None` | 实际加载的 checkpoint path/repo。 | 由 `--policy.path` 或恢复逻辑设置。 |

`PolicyFeature`：

| 字段 | 含义 |
|---|---|
| `type` | `STATE`、`VISUAL`、`ACTION`、`ENV` 等特征类型。 |
| `shape` | tensor shape。图像通常是 `[3, H, W]`，state/action 是关节或控制维度。 |

## 优化器与学习率调度

默认情况下你不直接传 `optimizer.*` / `scheduler.*`，而是调 `policy.optimizer_*` 和 `policy.scheduler_*`。只有 `--use_policy_training_preset=false` 时才手写下面这些配置。

### OptimizerConfig

| 类型 | 参数 | 含义与影响 |
|---|---|---|
| `adam` | `lr`, `betas`, `eps`, `weight_decay`, `grad_clip_norm` | 通用 Adam。`lr` 控制步长；`betas` 控制动量平滑；`eps` 防止除零；`weight_decay` 正则；`grad_clip_norm` 限制梯度爆炸。 |
| `adamw` | `lr`, `betas`, `eps`, `weight_decay`, `grad_clip_norm` | AdamW 把 weight decay 解耦，更常用于 transformer/VLA。 |
| `sgd` | `lr`, `momentum`, `dampening`, `nesterov`, `weight_decay`, `grad_clip_norm` | 传统 SGD。通常不作为 VLA 微调首选。 |
| `xvla-adamw` | `lr`, `betas`, `eps`, `weight_decay`, `grad_clip_norm`, `soft_prompt_lr_scale`, `soft_prompt_warmup_lr_scale` | X-VLA 专用，VLM 参数用 0.1 倍 lr，soft prompt 可单独缩放。 |
| `multi_adam` | `lr`, `weight_decay`, `grad_clip_norm`, `optimizer_groups` | 多优化器配置，主要用于 SAC 等多网络 RL。 |

### SchedulerConfig

| 类型 | 参数 | 含义与影响 |
|---|---|---|
| `diffuser` | `name`, `num_warmup_steps` | 使用 diffusers 的 scheduler，如 cosine。 |
| `vqbet` | `num_warmup_steps`, `num_vqvae_training_steps`, `num_cycles` | VQ-BeT 先训练 VQ-VAE，再 warmup/cosine。 |
| `cosine_decay_with_warmup` | `num_warmup_steps`, `num_decay_steps`, `peak_lr`, `decay_lr` | 先线性 warmup 到 peak lr，再 cosine 衰减到 decay lr。若训练 steps 少于 decay steps，会自动按比例缩放。 |

## SmolVLA 参数详解

SmolVLA 是你当前脚本使用的核心。官方建议用 `lerobot/smolvla_base` 或兼容 checkpoint 微调；20k steps 在 A100 上约数小时，真实任务需按验证表现调 steps。

### 输入输出与动作时域

| 参数 | 默认值 | 含义 | 调整影响 |
|---|---:|---|---|
| `policy.n_obs_steps` | `1` | 输入多少帧 observation。SmolVLA 默认只用当前帧。 | 调大需要模型/数据支持更多历史帧；显存增加。 |
| `policy.chunk_size` | `50` | 模型一次预测的 action chunk 长度。 | 越大一次生成更长动作，但更难学；必须 `n_action_steps <= chunk_size`。 |
| `policy.n_action_steps` | `50` | 每次模型调用后实际执行/训练使用的动作步数。 | 小于 `chunk_size` 可提高重规划频率；太小可能浪费预测，太大响应变慢。 |
| `policy.normalization_mapping` | VISUAL identity, STATE/ACTION mean_std | 各特征类型的归一化方式。 | 不建议随意改；与预训练 checkpoint 不一致会严重影响微调。 |
| `policy.max_state_dim` | `32` | state 向量最大维度，较短会 padding。 | 必须覆盖机器人 state 维度；调大增加 projection 输入冗余。 |
| `policy.max_action_dim` | `32` | action 向量最大维度，较短会 padding。 | 必须覆盖 action 维度；你的 WidowX 7 维 action 没问题。 |
| `policy.resize_imgs_with_padding` | `(512, 512)` | resize 图像并 padding 到固定大小。 | 调大保留更多细节但显存/计算上升；调小更快但可能丢细节。 |
| `policy.empty_cameras` | `0` | 添加空相机占位。 | 用于 checkpoint 期望更多相机但数据缺相机的场景；比随意删特征更安全。 |
| `policy.adapt_to_pi_aloha` | `False` | Aloha 动作空间适配到 PI 内部运行时空间。 | 只对特定 Aloha/PI 兼容流程使用。 |
| `policy.use_delta_joint_actions_aloha` | `False` | Aloha joint action 转相对 delta。 | 当前源码中打开会 `NotImplementedError`，不要用于本项目。 |

### 语言、解码与注意力

| 参数 | 默认值 | 含义 | 调整影响 |
|---|---:|---|---|
| `policy.tokenizer_max_length` | `48` | 任务语言指令最大 token 长度。 | 长任务描述可调大，但显存略增；过短会截断关键信息。 |
| `policy.num_steps` | `10` | action generation/denoising steps。 | 调大可能更准但推理更慢；调小更快但动作质量可能下降。 |
| `policy.use_cache` | `True` | attention/cache 使用开关。 | 通常保持打开以提速；调试显存/兼容性问题时可关。 |
| `policy.add_image_special_tokens` | `False` | 是否在图像特征周围加特殊 token。 | 必须与 VLM/tokenizer 预训练方式兼容。 |
| `policy.attention_mode` | `"cross_attn"` | VLM 与 action expert 的注意力交互方式。 | 改动属于结构变化，微调时不建议随意改。 |
| `policy.prefix_length` | `-1` | prefix token 长度控制，`-1` 表示默认。 | 高级结构参数，通常不改。 |
| `policy.pad_language_to` | `"longest"` | 语言 padding 到 batch 最长或 max length。 | `max_length` 更固定但浪费；`longest` 更省。 |

### 微调冻结范围

| 参数 | 默认值 | 含义 | 调整影响 |
|---|---:|---|---|
| `policy.freeze_vision_encoder` | `True` | 冻结视觉 encoder。 | 省显存、防止视觉表征遗忘；相机/画面分布差异大时可尝试关闭，但风险和成本更高。 |
| `policy.train_expert_only` | `True` | 只训练 action expert。 | 适合小数据微调；全模型训练适应力更强但更易过拟合/遗忘。 |
| `policy.train_state_proj` | `True` | 训练 state projection。 | 机器人 state 维度/语义变化时应保持打开。 |

### SmolVLA preset 优化器与调度

| 参数 | 默认值 | 含义 | 调整影响 |
|---|---:|---|---|
| `policy.optimizer_lr` | `1e-4` | AdamW peak/base lr。 | 全量微调常从 `1e-4` 或更低试；LoRA 可从 `1e-3` 附近试。你脚本用 `2e-4`，属于更激进微调。 |
| `policy.optimizer_betas` | `(0.9, 0.95)` | AdamW 动量参数。 | beta2 较低对非平稳微调更灵敏；通常不改。 |
| `policy.optimizer_eps` | `1e-8` | AdamW 数值稳定项。 | 通常不改。 |
| `policy.optimizer_weight_decay` | `1e-10` | AdamW 权重衰减。 | SmolVLA 默认几乎不做 weight decay；数据少时可小幅试验，但过大会欠拟合。 |
| `policy.optimizer_grad_clip_norm` | `10` | 梯度裁剪阈值。 | loss spike/梯度爆炸可调低；太低会限制学习。 |
| `policy.scheduler_warmup_steps` | `1000` | warmup step 数。 | 数据少或 lr 大时增加 warmup 更稳；太长会拖慢前期学习。 |
| `policy.scheduler_decay_steps` | `30000` | cosine decay 总步数。 | 应接近或略小于 `steps`；若 `steps < decay_steps`，调度器会自动缩放。 |
| `policy.scheduler_decay_lr` | `2.5e-6` | 最终学习率。 | 调高后期继续更新更强，可能更适应也更易过拟合；调低更保守。 |

### 架构与高级参数

| 参数 | 默认值 | 含义 | 调整影响 |
|---|---:|---|---|
| `policy.vlm_model_name` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | VLM backbone 名称。 | 改 backbone 基本等于换模型结构；微调 checkpoint 时不要改。 |
| `policy.load_vlm_weights` | `False` | 从 VLM backbone 加载权重；预训练 SmolVLA 权重加载时通常由 checkpoint 处理。 | 从头训练 expert 时可关；训练新模型时才考虑。 |
| `policy.num_expert_layers` | `-1` | action expert 层数；`<=0` 表示与 VLM 层数一致。 | 减少层数省显存但容量下降。 |
| `policy.num_vlm_layers` | `16` | 使用 VLM 前多少层。 | 减少可省算力但可能损失语义/视觉能力。 |
| `policy.self_attn_every_n_layers` | `2` | action expert 中 self-attention 插入频率。 | 结构参数，微调不建议改。 |
| `policy.expert_width_multiplier` | `0.75` | action expert hidden size 相对 VLM 的倍率。 | 越大容量越高、显存越高。 |
| `policy.min_period` | `0.004` | timestep sin-cos 位置编码最小周期。 | 影响 diffusion/flow 时间嵌入尺度；通常不改。 |
| `policy.max_period` | `4.0` | timestep sin-cos 位置编码最大周期。 | 通常不改。 |
| `policy.rtc_config` | `None` | Real-Time Chunking 配置。 | 用于实时分块推理策略；训练普通 BC 时通常不设。 |
| `policy.compile_model` | `False` | 是否 `torch.compile`。 | 可能加速，但首次编译慢且兼容性不一定稳定。 |
| `policy.compile_mode` | `"max-autotune"` | torch compile mode。 | 更激进优化可能更快但编译更慢。 |

## 你的脚本参数解读

当前 `fine_tune_smolvla.sh` 的关键选择：

| 参数 | 当前值 | 解读 |
|---|---:|---|
| `policy.path` | `TrossenRoboticsCommunity/smolvla_solo_red_block` | 从已有 SmolVLA checkpoint 微调，而不是从 `lerobot/smolvla_base` 起步。 |
| `policy.input_features` | state 7 + main/wrist 两路图像 | 显式覆盖 checkpoint 输入，适配 WidowX 数据。 |
| `policy.output_features` | action 7 | 对应 7 维关节/夹爪 action。 |
| `rename_map` | right->cam_main, wrist->cam_wrist | 把数据集原相机名映射到 policy 期望名。这个映射必须和 `input_features` 完全一致。 |
| `batch_size` | `64` | 对 SmolVLA 较大，吞吐好但显存压力大。 |
| `steps` | `40000` | 比官方示例 20k 更充分，适合多位置/多数据版本；需要用验证集防过拟合。 |
| `save_freq` | `5000` | 中间 checkpoint 较密，便于选最优 step。 |
| `policy.scheduler_decay_steps` | `40000` | 学习率衰减覆盖完整训练。 |
| `policy.optimizer_lr` | `2e-4` | 比默认 `1e-4` 激进；如果出现 loss 震荡或遗忘，可降到 `1e-4` / `5e-5`。 |
| `policy.n_action_steps` | `10` | 训练/推理每次执行 10 个动作后重规划。10Hz 数据下约 1 秒一段。 |
| `policy.chunk_size` | `25` | 每次预测 25 步，但只使用 10 步；提供未来动作监督，同时保持较高重规划频率。 |
| `policy.resize_imgs_with_padding` | `[512, 512]` | 与 SmolVLA 默认一致。 |
| `policy.use_amp` | `true` | 降显存、提速；如果出现 NaN/不稳定，先尝试关闭。 |
| `wandb.project` | `smolvla_widowx_grape_grasping` | 实验分组明确。 |

建议优先调参顺序：

1. 先固定数据和 `rename_map`，确认 feature shape、相机画面、action 量纲完全正确。
2. 用当前 `save_freq=5000` 跑完整训练，离线评估 5k/10k/.../40k checkpoint。
3. 如果后期验证集误差上升，减少 `steps` 或降低 `optimizer_lr`。
4. 如果训练集也学不动，先检查 action/state 归一化、相机映射和任务语言，再考虑提高 lr 或解冻更多模块。
5. 如果显存紧张，优先降 `batch_size`，其次考虑 LoRA/PEFT；不要先改模型结构参数。

## 内置 policy 参数完整索引

下面是 `lerobot==0.5.1` 内置 policy 配置字段。所有 policy 还共享上文的 `policy.n_obs_steps/input_features/output_features/device/use_amp/use_peft/push_to_hub/repo_id/private/tags/license/pretrained_path`。

### ACT (`policy.type=act`)

完整字段：

`n_obs_steps`, `chunk_size`, `n_action_steps`, `normalization_mapping`, `vision_backbone`, `pretrained_backbone_weights`, `replace_final_stride_with_dilation`, `pre_norm`, `dim_model`, `n_heads`, `dim_feedforward`, `feedforward_activation`, `n_encoder_layers`, `n_decoder_layers`, `use_vae`, `latent_dim`, `n_vae_encoder_layers`, `temporal_ensemble_coeff`, `dropout`, `kl_weight`, `optimizer_lr`, `optimizer_weight_decay`, `optimizer_lr_backbone`.

调参含义：

- `chunk_size` / `n_action_steps` 控制 action chunk 与执行步数；更长动作更平滑但重规划慢。
- `vision_backbone` / `pretrained_backbone_weights` 控制视觉特征提取器。
- `dim_model`, `n_heads`, `dim_feedforward`, `n_encoder_layers`, `n_decoder_layers` 控制 transformer 容量。
- `use_vae`, `latent_dim`, `n_vae_encoder_layers`, `kl_weight` 控制 ACT 的 CVAE 行为；`kl_weight` 过大可能动作保守，过小潜变量约束弱。
- `temporal_ensemble_coeff` 控制 temporal ensemble 平滑强度。
- `optimizer_lr`, `optimizer_weight_decay`, `optimizer_lr_backbone` 控制主干和 backbone 学习率。

### Diffusion (`policy.type=diffusion`)

完整字段：

`n_obs_steps`, `horizon`, `n_action_steps`, `normalization_mapping`, `drop_n_last_frames`, `vision_backbone`, `resize_shape`, `crop_ratio`, `crop_shape`, `crop_is_random`, `pretrained_backbone_weights`, `use_group_norm`, `spatial_softmax_num_keypoints`, `use_separate_rgb_encoder_per_camera`, `down_dims`, `kernel_size`, `n_groups`, `diffusion_step_embed_dim`, `use_film_scale_modulation`, `noise_scheduler_type`, `num_train_timesteps`, `beta_schedule`, `beta_start`, `beta_end`, `prediction_type`, `clip_sample`, `clip_sample_range`, `num_inference_steps`, `compile_model`, `compile_mode`, `do_mask_loss_for_padding`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `scheduler_name`, `scheduler_warmup_steps`.

调参含义：

- `horizon`, `n_obs_steps`, `n_action_steps` 决定条件窗口和预测窗口。
- crop/backbone/group norm/keypoints 控制视觉 encoder 和图像增强。
- `down_dims`, `kernel_size`, `n_groups`, `diffusion_step_embed_dim`, `use_film_scale_modulation` 控制 U-Net 容量。
- `noise_scheduler_type`, `num_train_timesteps`, `beta_*`, `prediction_type`, `num_inference_steps` 控制 diffusion 噪声过程；推理 steps 越多通常越慢但更准。
- `do_mask_loss_for_padding` 用于 padding action 的 loss mask。

### SmolVLA (`policy.type=smolvla`)

完整字段已在上文详解：

`n_obs_steps`, `chunk_size`, `n_action_steps`, `normalization_mapping`, `max_state_dim`, `max_action_dim`, `resize_imgs_with_padding`, `empty_cameras`, `adapt_to_pi_aloha`, `use_delta_joint_actions_aloha`, `tokenizer_max_length`, `num_steps`, `use_cache`, `freeze_vision_encoder`, `train_expert_only`, `train_state_proj`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `optimizer_grad_clip_norm`, `scheduler_warmup_steps`, `scheduler_decay_steps`, `scheduler_decay_lr`, `vlm_model_name`, `load_vlm_weights`, `add_image_special_tokens`, `attention_mode`, `prefix_length`, `pad_language_to`, `num_expert_layers`, `num_vlm_layers`, `self_attn_every_n_layers`, `expert_width_multiplier`, `min_period`, `max_period`, `rtc_config`, `compile_model`, `compile_mode`.

### Pi0 / Pi05 / Pi0Fast

这些是 Physical Intelligence 系列 VLA。字段含义与 SmolVLA 类似，但模型更大，显存和推理成本更高。

Pi0 完整字段：

`paligemma_variant`, `action_expert_variant`, `dtype`, `n_obs_steps`, `chunk_size`, `n_action_steps`, `max_state_dim`, `max_action_dim`, `num_inference_steps`, `time_sampling_beta_alpha`, `time_sampling_beta_beta`, `time_sampling_scale`, `time_sampling_offset`, `min_period`, `max_period`, `use_relative_actions`, `relative_exclude_joints`, `action_feature_names`, `rtc_config`, `image_resolution`, `empty_cameras`, `normalization_mapping`, `gradient_checkpointing`, `compile_model`, `compile_mode`, `device`, `freeze_vision_encoder`, `train_expert_only`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `optimizer_grad_clip_norm`, `scheduler_warmup_steps`, `scheduler_decay_steps`, `scheduler_decay_lr`, `tokenizer_max_length`.

Pi05 完整字段：

`paligemma_variant`, `action_expert_variant`, `dtype`, `n_obs_steps`, `chunk_size`, `n_action_steps`, `max_state_dim`, `max_action_dim`, `num_inference_steps`, `time_sampling_beta_alpha`, `time_sampling_beta_beta`, `time_sampling_scale`, `time_sampling_offset`, `min_period`, `max_period`, `use_relative_actions`, `relative_exclude_joints`, `action_feature_names`, `rtc_config`, `image_resolution`, `empty_cameras`, `tokenizer_max_length`, `normalization_mapping`, `gradient_checkpointing`, `compile_model`, `compile_mode`, `device`, `freeze_vision_encoder`, `train_expert_only`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `optimizer_grad_clip_norm`, `scheduler_warmup_steps`, `scheduler_decay_steps`, `scheduler_decay_lr`, `tokenizer_max_length`.

Pi0Fast 完整字段：

`paligemma_variant`, `action_expert_variant`, `dtype`, `chunk_size`, `n_action_steps`, `max_state_dim`, `max_action_dim`, `max_action_tokens`, `use_relative_actions`, `relative_exclude_joints`, `action_feature_names`, `rtc_config`, `image_resolution`, `empty_cameras`, `tokenizer_max_length`, `text_tokenizer_name`, `action_tokenizer_name`, `temperature`, `max_decoding_steps`, `fast_skip_tokens`, `validate_action_token_prefix`, `use_kv_cache`, `normalization_mapping`, `gradient_checkpointing`, `compile_model`, `compile_mode`, `device`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `optimizer_grad_clip_norm`, `scheduler_warmup_steps`, `scheduler_decay_steps`, `scheduler_decay_lr`.

调参含义：

- `paligemma_variant`, `action_expert_variant`, `dtype`, `image_resolution` 控制 backbone/专家模型变体、精度和视觉分辨率。
- `num_inference_steps`, `time_sampling_*`, `min_period`, `max_period` 控制 flow/diffusion 时间采样和推理步数。
- `use_relative_actions`, `relative_exclude_joints`, `action_feature_names` 控制动作表示，必须和机器人控制约定一致。
- `train_expert_only`, `freeze_vision_encoder`, `gradient_checkpointing` 决定微调范围、显存和训练速度。

### X-VLA (`policy.type=xvla`)

完整字段：

`n_obs_steps`, `chunk_size`, `n_action_steps`, `dtype`, `normalization_mapping`, `florence_config`, `tokenizer_name`, `tokenizer_max_length`, `tokenizer_padding_side`, `pad_language_to`, `hidden_size`, `depth`, `num_heads`, `mlp_ratio`, `num_domains`, `len_soft_prompts`, `dim_time`, `max_len_seq`, `use_hetero_proj`, `action_mode`, `num_denoising_steps`, `use_proprio`, `max_state_dim`, `max_action_dim`, `domain_feature_key`, `resize_imgs_with_padding`, `num_image_views`, `empty_cameras`, `freeze_vision_encoder`, `freeze_language_encoder`, `train_policy_transformer`, `train_soft_prompts`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `optimizer_grad_clip_norm`, `optimizer_soft_prompt_lr_scale`, `optimizer_soft_prompt_warmup_lr_scale`, `scheduler_warmup_steps`, `scheduler_decay_steps`, `scheduler_decay_lr`.

调参含义：

- `florence_config` 和 tokenizer 字段控制视觉语言 backbone。
- `hidden_size/depth/num_heads/mlp_ratio` 控制 policy transformer 容量。
- `num_domains/len_soft_prompts/domain_feature_key/train_soft_prompts` 控制多机器人/多任务 soft prompt。
- `action_mode`, `num_denoising_steps`, `use_proprio` 控制动作表示和生成。

### Wall-X (`policy.type=wall_x`)

完整字段：

`n_obs_steps`, `chunk_size`, `n_action_steps`, `max_action_dim`, `max_state_dim`, `normalization_mapping`, `pretrained_name_or_path`, `action_tokenizer_path`, `prediction_mode`, `attn_implementation`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `optimizer_grad_clip_norm`, `scheduler_warmup_steps`, `scheduler_decay_steps`, `scheduler_decay_lr`.

调参含义：

- `prediction_mode` 在 diffusion/fast 动作预测之间切换。
- `action_tokenizer_path` 只对 tokenized/fast 动作模式关键。
- `attn_implementation` 可选 eager/flash_attention_2/sdpa，影响速度和依赖。

### Groot (`policy.type=groot`)

完整字段：

`n_obs_steps`, `chunk_size`, `n_action_steps`, `max_state_dim`, `max_action_dim`, `normalization_mapping`, `image_size`, `base_model_path`, `tokenizer_assets_repo`, `embodiment_tag`, `tune_llm`, `tune_visual`, `tune_projector`, `tune_diffusion_model`, `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_full_model`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `warmup_ratio`, `use_bf16`, `video_backend`, `balance_dataset_weights`, `balance_trajectory_weights`, `dataset_paths`, `output_dir`, `save_steps`, `max_steps`, `batch_size`, `dataloader_num_workers`, `report_to`, `resume`.

调参含义：

- `base_model_path`, `tokenizer_assets_repo`, `embodiment_tag` 选择 GROOT 基座、tokenizer 资产和机器人 embodiment。
- `tune_llm`, `tune_visual`, `tune_projector`, `tune_diffusion_model` 控制微调哪些子模块。
- `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_full_model` 控制 LoRA 容量和应用范围。
- `balance_dataset_weights`, `balance_trajectory_weights`, `dataset_paths` 用于多数据源平衡。
- `save_steps`, `max_steps`, `batch_size`, `dataloader_num_workers`, `report_to`, `resume` 是该 policy 自带训练参数。

### MultiTaskDiT (`policy.type=multi_task_dit`)

完整字段：

`n_obs_steps`, `horizon`, `n_action_steps`, `objective`, `noise_scheduler_type`, `num_train_timesteps`, `beta_schedule`, `beta_start`, `beta_end`, `prediction_type`, `clip_sample`, `clip_sample_range`, `num_inference_steps`, `sigma_min`, `num_integration_steps`, `integration_method`, `timestep_sampling_strategy`, `timestep_sampling_s`, `timestep_sampling_alpha`, `timestep_sampling_beta`, `hidden_dim`, `num_layers`, `num_heads`, `dropout`, `use_positional_encoding`, `timestep_embed_dim`, `use_rope`, `rope_base`, `vision_encoder_name`, `use_separate_rgb_encoder_per_camera`, `vision_encoder_lr_multiplier`, `image_resize_shape`, `image_crop_shape`, `image_crop_is_random`, `text_encoder_name`, `tokenizer_max_length`, `tokenizer_padding`, `tokenizer_padding_side`, `tokenizer_truncation`, `normalization_mapping`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `scheduler_name`, `scheduler_warmup_steps`, `do_mask_loss_for_padding`, `drop_n_last_frames`.

调参含义：

- `horizon`, `n_obs_steps`, `n_action_steps` 控制动作预测时域。
- `objective`, `noise_scheduler_type`, `num_train_timesteps`, `beta_*`, `sigma_min`, `num_integration_steps`, `integration_method`, `timestep_sampling_*` 控制 diffusion/flow 训练目标和时间采样。
- `hidden_dim`, `num_layers`, `num_heads`, `dropout`, `use_rope` 控制 DiT 容量。
- `vision_encoder_*`, `image_*`, `text_encoder_name`, `tokenizer_*` 控制视觉/语言编码。

### VQ-BeT (`policy.type=vqbet`)

完整字段：

`n_obs_steps`, `n_action_pred_token`, `action_chunk_size`, `normalization_mapping`, `vision_backbone`, `crop_shape`, `crop_is_random`, `pretrained_backbone_weights`, `use_group_norm`, `spatial_softmax_num_keypoints`, `n_vqvae_training_steps`, `vqvae_n_embed`, `vqvae_embedding_dim`, `vqvae_enc_hidden_dim`, `gpt_block_size`, `gpt_input_dim`, `gpt_output_dim`, `gpt_n_layer`, `gpt_n_head`, `gpt_hidden_dim`, `dropout`, `offset_loss_weight`, `primary_code_loss_weight`, `secondary_code_loss_weight`, `bet_softmax_temperature`, `sequentially_select`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `optimizer_vqvae_lr`, `optimizer_vqvae_weight_decay`, `scheduler_warmup_steps`.

调参含义：

- `n_vqvae_training_steps` 和 `vqvae_*` 控制动作 tokenization。
- `gpt_*` 控制行为 transformer 容量。
- `*_loss_weight` 控制 code/offset 训练目标权重。

### TDMPC (`policy.type=tdmpc`)

完整字段：

`n_obs_steps`, `n_action_repeats`, `horizon`, `n_action_steps`, `normalization_mapping`, `image_encoder_hidden_dim`, `state_encoder_hidden_dim`, `latent_dim`, `q_ensemble_size`, `mlp_dim`, `discount`, `use_mpc`, `cem_iterations`, `max_std`, `min_std`, `n_gaussian_samples`, `n_pi_samples`, `uncertainty_regularizer_coeff`, `n_elites`, `elite_weighting_temperature`, `gaussian_mean_momentum`, `max_random_shift_ratio`, `reward_coeff`, `expectile_weight`, `value_coeff`, `consistency_coeff`, `advantage_scaling`, `pi_coeff`, `temporal_decay_coeff`, `target_model_momentum`, `optimizer_lr`.

调参含义：

- MPC/CEM 字段控制规划质量和速度。
- reward/value/consistency/pi coeff 控制 TD-MPC 各损失项权重。
- `target_model_momentum` 控制 target network 更新平滑度。

### SAC (`policy.type=sac`)

完整字段：

`normalization_mapping`, `dataset_stats`, `device`, `storage_device`, `vision_encoder_name`, `freeze_vision_encoder`, `image_encoder_hidden_dim`, `shared_encoder`, `num_discrete_actions`, `image_embedding_pooling_dim`, `online_steps`, `online_buffer_capacity`, `offline_buffer_capacity`, `async_prefetch`, `online_step_before_learning`, `policy_update_freq`, `discount`, `temperature_init`, `num_critics`, `num_subsample_critics`, `critic_lr`, `actor_lr`, `temperature_lr`, `critic_target_update_weight`, `utd_ratio`, `state_encoder_hidden_dim`, `latent_dim`, `target_entropy`, `use_backup_entropy`, `grad_clip_norm`, `critic_network_kwargs`, `actor_network_kwargs`, `policy_kwargs`, `discrete_critic_network_kwargs`, `actor_learner_config`, `concurrency`, `use_torch_compile`.

调参含义：

- SAC 是 RL policy，很多字段服务于 actor/critic/entropy/buffer，而不是离线 BC。
- `discount`, `target_entropy`, `temperature_init`, `temperature_lr` 控制长期奖励与熵正则。
- `online_buffer_capacity`, `offline_buffer_capacity`, `online_steps`, `online_step_before_learning`, `utd_ratio` 控制样本收集和 replay 训练强度。
- `critic_lr`, `actor_lr`, `critic_target_update_weight`, `num_critics`, `num_subsample_critics` 控制 actor/critic 学习动态。

### SARM (`policy.type=sarm`)

完整字段：

`annotation_mode`, `n_obs_steps`, `frame_gap`, `max_rewind_steps`, `image_dim`, `text_dim`, `hidden_dim`, `num_heads`, `num_layers`, `max_state_dim`, `drop_n_last_frames`, `batch_size`, `clip_batch_size`, `dropout`, `stage_loss_weight`, `rewind_probability`, `language_perturbation_probability`, `num_sparse_stages`, `sparse_subtask_names`, `sparse_temporal_proportions`, `num_dense_stages`, `dense_subtask_names`, `dense_temporal_proportions`, `pretrained_model_path`, `device`, `image_key`, `state_key`, `input_features`, `output_features`, `normalization_mapping`.

调参含义：

- `annotation_mode`, `num_sparse_stages`, `sparse_subtask_names`, `dense_subtask_names`, `*_temporal_proportions` 控制阶段/子任务标注。
- `frame_gap`, `max_rewind_steps`, `rewind_probability` 控制 temporal rewind 训练。
- `image_dim`, `text_dim`, `hidden_dim`, `num_heads`, `num_layers` 控制模型容量。
- `stage_loss_weight`, `language_perturbation_probability` 控制阶段监督和语言扰动强度。

### RewardClassifierConfig

完整字段：

`name`, `num_classes`, `hidden_dim`, `latent_dim`, `image_embedding_pooling_dim`, `dropout_rate`, `model_name`, `device`, `model_type`, `num_cameras`, `learning_rate`, `weight_decay`, `grad_clip_norm`, `normalization_mapping`.

含义：用于 SAC 奖励分类器；`num_classes` 控制分类类别，`hidden_dim/latent_dim/image_embedding_pooling_dim/dropout_rate` 控制模型容量和正则，`model_name/model_type/num_cameras` 控制视觉模型与相机数，`learning_rate/weight_decay/grad_clip_norm` 控制训练稳定性。

## 环境参数索引

离线微调真实机器人数据通常不需要 `env.*`。如果你训练/评测仿真或 RL，才需要这些。

通用 `EnvConfig`：

`task`, `fps`, `features`, `features_map`, `max_parallel_tasks`, `disable_env_checker`.

环境类型与主要字段：

| `env.type` | 完整字段 |
|---|---|
| `aloha` | `task`, `fps`, `episode_length`, `obs_type`, `observation_height`, `observation_width`, `render_mode`, `features`, `features_map` |
| `pusht` | `task`, `fps`, `episode_length`, `obs_type`, `render_mode`, `visualization_width`, `visualization_height`, `observation_height`, `observation_width`, `features`, `features_map` |
| `libero` | `task`, `task_ids`, `fps`, `episode_length`, `obs_type`, `render_mode`, `camera_name`, `init_states`, `camera_name_mapping`, `observation_height`, `observation_width`, `features`, `features_map`, `control_mode` |
| `metaworld` | `task`, `fps`, `episode_length`, `obs_type`, `render_mode`, `multitask_eval`, `features`, `features_map` |
| `gym_manipulator` | `robot`, `teleop`, `processor`, `name` |
| `isaaclab_arena` | `hub_path`, `episode_length`, `num_envs`, `embodiment`, `object`, `mimic`, `teleop_device`, `seed`, `device`, `disable_fabric`, `enable_cameras`, `headless`, `enable_pinocchio`, `environment`, `task`, `state_dim`, `action_dim`, `camera_height`, `camera_width`, `video`, `video_length`, `video_interval`, `state_keys`, `camera_keys`, `features`, `features_map`, `kwargs` |

环境字段影响：

- `task` 选择任务。
- `fps` 必须和数据/控制频率一致，否则 temporal delta 和动作执行时长会错。
- `obs_type` 决定使用 pixels、state 或两者。
- `features` / `features_map` 决定 env 输出如何映射到 policy 输入。
- `episode_length` 控制 episode 最大长度。
- camera height/width/name/key 控制视觉输入，必须和 policy image feature 兼容。
- `control_mode` 控制动作是 absolute 还是 relative，这对真实机器人迁移非常关键。

## Multi-GPU / Accelerate 参数

这些不是 `TrainPipelineConfig` 字段，而是 `accelerate launch` 的外层参数。

示例：

```bash
accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --mixed_precision=fp16 \
  $(which lerobot-train) \
  --dataset.repo_id=${HF_USER}/my_dataset \
  --policy.type=act \
  --batch_size=8 \
  --steps=50000
```

关键点：

- `--multi_gpu`：启用多 GPU。
- `--num_processes`：使用几个 GPU/process。
- `--mixed_precision=fp16|bf16`：混合精度由 accelerate 控制；此时 `policy.use_amp` 不再控制 mixed precision。
- LeRobot 不会自动按 GPU 数缩放学习率或 steps。有效 batch size 约为 `batch_size * num_processes`，是否线性放大学习率、是否按比例减少 steps，需要你手动决定。

## 参考链接

- LeRobot 文档首页：https://huggingface.co/docs/lerobot/index
- SmolVLA 文档：https://huggingface.co/docs/lerobot/smolvla
- PEFT 微调文档：https://huggingface.co/docs/lerobot/peft_training
- Multi-GPU 文档：https://huggingface.co/docs/lerobot/multi_gpu_training
- LeRobot GitHub：https://github.com/huggingface/lerobot
- PyPI `lerobot==0.5.1`：https://pypi.org/project/lerobot/
