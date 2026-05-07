# LeRobot Training and Fine-Tuning Parameter Guide

This document covers `lerobot-train` / `TrainPipelineConfig`, with emphasis on the SmolVLA fine-tuning script used in this project:

- `important_code/training/fine_tune_smolvla.sh`
- Local environment: `lerobot==0.5.0`
- Latest official stable package checked: `lerobot==0.5.1`. Its training fields are mostly consistent with the local environment; `0.5.1` adds `wandb.add_tags` to `WandBConfig`.
- Sources: local source code, PyPI `lerobot==0.5.1` source, Hugging Face LeRobot docs, Hugging Face PEFT docs, and the LeRobot GitHub source.

The official docs also recommend using `lerobot-train --help` to inspect all fine-tuning options. In this environment, however, the help output is interrupted by a `draccus/argparse` `%` formatting issue, so this document uses the source dataclass fields as the source of truth.

## CLI Rules

LeRobot uses `draccus` for configuration parsing. CLI parameters map directly to dataclass field paths.

Example:

```bash
--batch_size=64
--dataset.repo_id=kaixiyao/widowxai_grape_grasping_V4_pos234_train_10hz
--policy.path=TrossenRoboticsCommunity/smolvla_solo_red_block
--policy.optimizer_lr=2e-4
--wandb.enable=true
--rename_map='{"old.key": "new.key"}'
```

Common rules:

- Top-level fields are passed directly: `--steps=40000`
- Nested fields use dot notation: `--policy.chunk_size=25`
- For dict/list/tuple values, use JSON/YAML-style strings and protect them from the shell with single quotes: `--policy.resize_imgs_with_padding='[512, 512]'`
- `--policy.path=...` loads an existing policy config and weights from the Hub or a local checkpoint. CLI `--policy.*` values override fields from the loaded config.
- `--config_path=... --resume=true` resumes from a saved training config. When resuming, the checkpoint config is used by default.

## Top-Level Training Parameters

| Parameter | Default | Meaning | Effect of increasing/decreasing |
|---|---:|---|---|
| `dataset` | Required | Training dataset configuration. See the next section. | Dataset fields, fps, camera keys, and action/state shapes must match the policy inputs and outputs. |
| `env` | `None` | Simulation/RL environment configuration. Usually not needed for pure offline imitation learning; used by LIBERO, PushT, Aloha, MetaWorld, and similar training/evaluation flows. | Enables periodic success-rate evaluation when configured, but increases runtime cost. |
| `policy` | `None` | Policy model configuration. Training must load a pretrained model with `--policy.path`, or create one from scratch with `--policy.type=...`. | Determines model architecture, inputs/outputs, and optimizer presets. |
| `output_dir` | Auto-generated | Directory for training outputs. | If it already exists and `resume=false`, training errors out to avoid overwriting prior experiments. |
| `job_name` | Auto-generated | Experiment name used in output paths, logs, and Hub metadata. | Use a stable, readable name for experiment comparison. |
| `resume` | `False` | Whether to resume from an existing checkpoint. | `true` loads the checkpoint config; new CLI values may not take effect unless the source explicitly supports overriding. |
| `seed` | `1000` | Random seed for initialization, shuffling, and envs. | Fixed seeds improve reproducibility; multiple seeds help assess stability. |
| `cudnn_deterministic` | `False` | Enables deterministic cuDNN algorithms. | More reproducible, but can slow training by about 10-20%. |
| `num_workers` | `4` | Number of PyTorch DataLoader workers. | More workers can speed up decoding/preprocessing; too many can contend for CPU/RAM or cause IO jitter. |
| `batch_size` | `8` | Number of samples per training step. LeRobot does not automatically scale learning rate or steps under multi-GPU training. | Larger batches give steadier gradients and better throughput but use more memory; smaller batches add gradient noise and may need lower LR or more steps. |
| `steps` | `100000` | Total number of training iterations. | More steps train more thoroughly but can overfit and cost more time; fewer steps are useful for quick validation. Official SmolVLA examples commonly start around 20k. |
| `eval_freq` | `20000` | Run env evaluation every N steps. | Little effect without `env`; with an env, it can add significant runtime. |
| `log_freq` | `200` | Log every N steps. | Lower values give finer monitoring but more logs; higher values make logs sparser. |
| `tolerance_s` | `1e-4` | Training-loop timing/synchronization tolerance. | Usually left unchanged; mainly relevant to realtime or env-based workflows. |
| `save_checkpoint` | `True` | Whether to save checkpoints. | Disabling saves disk but prevents resuming and selecting intermediate models. |
| `save_freq` | `20000` | Checkpoint save interval; the final step is also saved. | Smaller intervals ease rollback and model selection but use more disk. |
| `use_policy_training_preset` | `True` | Whether to use the policy's built-in optimizer and scheduler presets. | If `true`, tune `policy.optimizer_*` / `policy.scheduler_*`; if `false`, you must explicitly pass `optimizer` and `scheduler`. |
| `optimizer` | `None` | Manual optimizer configuration. | Needed only when `use_policy_training_preset=false`. |
| `scheduler` | `None` | Manual learning-rate scheduler configuration. | Needed only when `use_policy_training_preset=false`. |
| `eval` | `EvalConfig` | Evaluation configuration. | Important only when env evaluation is used. |
| `wandb` | `WandBConfig` | Weights & Biases logging configuration. | Does not change the model math, but affects experiment tracking. |
| `peft` | `None` | Parameter-efficient fine-tuning configuration, such as LoRA. | Can greatly reduce trainable parameters and memory; adaptation capacity depends on target modules and rank. |
| `use_rabc` | `False` | Reward-Aligned Behavior Cloning, which weights samples by reward/progress. | Requires `sarm_progress.parquet`; high-quality samples receive more weight when enabled. |
| `rabc_progress_path` | `None` | Path to the RA-BC progress parquet file. | If omitted, LeRobot infers it from the dataset directory or Hub path. |
| `rabc_kappa` | `0.01` | High-quality sample threshold for RA-BC. | Larger values are stricter; smaller values are more permissive. |
| `rabc_epsilon` | `1e-6` | Numerical stability constant for RA-BC. | Usually left unchanged. |
| `rabc_head_mode` | `"sparse"` | Choice of sparse or dense head for dual-head models. | Only meaningful for models that support dual-head or reward-aligned training. |
| `rename_map` | `{}` | Renames dataset fields into the field names expected by the policy. | Critical when adapting an existing checkpoint; incorrect mappings can cause feature mismatches or swapped cameras. |

## Dataset Parameters

| Parameter | Default | Meaning | Effect |
|---|---:|---|---|
| `dataset.repo_id` | Required | Hub dataset repo id. The source has a placeholder for lists, but current `TrainPipelineConfig.validate()` raises `NotImplementedError` for lists. | Selects the training data source. |
| `dataset.root` | `None` | Local dataset directory. If `None`, LeRobot uses `$HF_LEROBOT_HOME` cache or downloads from the Hub. | Local data avoids repeated downloads and helps debug unpublished datasets. |
| `dataset.episodes` | `None` | Use only the specified episode indices. | Useful for small-sample debugging or train/validation splits; duplicates or negative indices error in 0.5.1. |
| `dataset.image_transforms` | Disabled by default | Image augmentation configuration. | Improves generalization, but overly strong augmentation can damage task-relevant visual details. |
| `dataset.revision` | `None` | Hub dataset revision/branch/commit. | Pin a commit for reproducible experiments. |
| `dataset.use_imagenet_stats` | `True` | Whether image normalization uses ImageNet statistics. | Usually keep `true` for pretrained vision encoders; evaluate disabling only for from-scratch training or unusual image distributions. |
| `dataset.video_backend` | Safe default codec | Video decoding backend. | Affects compatibility and speed; usually left unchanged. |
| `dataset.streaming` | `False` | Whether to stream data from the Hub. | Can reduce local disk use for large datasets; throughput may depend on network quality. |

### Image Augmentation Parameters

| Parameter | Default | Meaning | Effect |
|---|---:|---|---|
| `dataset.image_transforms.enable` | `False` | Enables image augmentation during training. | Helps with lighting/color variation, but should not move images too far from the deployment distribution. |
| `dataset.image_transforms.max_num_transforms` | `3` | Maximum number of transforms randomly applied to each frame. | Larger values increase augmentation strength; too large can hurt action learning. |
| `dataset.image_transforms.random_order` | `False` | Randomizes transform order. | Adds randomness; slightly worse for reproducibility and interpretability. |
| `dataset.image_transforms.tfs` | Default brightness/contrast/saturation/hue/sharpness | Dictionary of transforms. Each item is an `ImageTransformConfig`. | Controls augmentation type and strength. |
| `ImageTransformConfig.weight` | `1.0` | Sampling weight for this transform. | Higher means the transform is used more often. |
| `ImageTransformConfig.type` | `"Identity"` | Torchvision transform name. | Selects the augmentation type. |
| `ImageTransformConfig.kwargs` | `{}` | Arguments passed to the transform. | Controls augmentation strength/range. |

## Eval and WandB

| Parameter | Default | Meaning | Effect |
|---|---:|---|---|
| `eval.n_episodes` | `50` | Number of evaluation episodes per evaluation run. | More episodes give more stable evaluation but take longer. |
| `eval.batch_size` | `50` | Number of vectorized envs used for parallel evaluation. | Should not exceed `n_episodes`; higher values can be faster but use more resources. |
| `eval.use_async_envs` | `False` | Uses asynchronous multiprocessing envs. | Can speed up complex envs; harder to debug. |
| `wandb.enable` | `False` | Enables W&B logging. | Records loss, LR, checkpoint artifacts, and related metadata. |
| `wandb.disable_artifact` | `False` | Does not upload artifacts even when checkpoints are saved. | Saves upload time/storage, but checkpoints become less traceable on Hub/W&B. |
| `wandb.project` | `"lerobot"` | W&B project name. | Organizes experiments. |
| `wandb.entity` | `None` | W&B team/user. | Needed when multiple accounts/entities are available. |
| `wandb.notes` | `None` | Run notes. | Useful for recording dataset, camera, and hyperparameter changes. |
| `wandb.run_id` | `None` | Fixed run id. | Used to resume or merge logs. |
| `wandb.mode` | `None` | `online`, `offline`, or `disabled`. | Use `offline` on machines without network access. |
| `wandb.add_tags` | `True` | Added in 0.5.1; saves configuration as run tags. | Helps filter experiments; not available in local 0.5.0. |

## PEFT / LoRA Parameters

Official PEFT docs example:

```bash
--peft.method_type=LORA
--peft.r=64
--peft.target_modules='...'
--peft.full_training_modules='["state_proj"]'
```

| Parameter | Default | Meaning | Effect |
|---|---:|---|---|
| `peft.target_modules` | `None` | Module suffix, list, `all-linear`, or regex specifying where adapters are inserted. SmolVLA defaults typically target the LM expert `q_proj`/`v_proj` and state/action projections. | More targets increase trainable parameters, adaptation capacity, and memory; wrong targets may miss critical layers. |
| `peft.full_training_modules` | `None` | Modules that are fully trained and saved with adapter weights; PEFT calls this `modules_to_save`. | Useful for task-specific new projections/heads; more modules improve adaptation but reduce PEFT's parameter-saving benefit. |
| `peft.method_type` | `"LORA"` | PEFT method type. The high-level interface mainly targets LoRA. | Different methods expose different parameters and trainable structures. |
| `peft.init_type` | `None` | Adapter initialization method. | Usually leave at PEFT default; special initialization can affect early stability. |
| `peft.r` | `16` | LoRA rank. | Higher ranks approach full fine-tuning with higher capacity and memory/parameter count; lower ranks are cheaper but less expressive. |

PEFT tuning note: the official docs suggest LoRA learning rates can often be roughly 10x higher than full fine-tuning. For example, if full fine-tuning uses `1e-4`, LoRA can start around `1e-3`.

## Common Policy Parameters

All built-in policies inherit from `PreTrainedConfig`.

| Parameter | Default | Meaning | Effect |
|---|---:|---|---|
| `policy.n_obs_steps` | `1` | Number of historical observation steps used by the policy. | More history can help temporal reasoning, but increases input size, latency, and memory. |
| `policy.input_features` | `{}` or `null` | Input feature definitions. Keys are dataset field names, values are `{type, shape}`. `null` can infer from the dataset. | For pretrained fine-tuning, explicitly match the checkpoint; wrong keys/shapes either fail or train on the wrong input. |
| `policy.output_features` | `{}` or `null` | Output feature definitions, usually `action`. | Action dimension must match the robot control dimension. |
| `policy.device` | Auto-selected | `cuda`, `cuda:0`, `cpu`, `mps`, etc. | GPU is fast; CPU is mainly for debugging. Unavailable devices fall back automatically. |
| `policy.use_amp` | `False` | Automatic mixed precision. | Can reduce memory and speed up training; some models/devices may become numerically unstable. Under `accelerate`, mixed precision is controlled by accelerate instead. |
| `policy.use_peft` | `False` | Marks whether the policy uses PEFT. | Usually set by the training flow; do not fake it manually. |
| `policy.push_to_hub` | `True` | Uploads the trained policy at the end. | Requires `policy.repo_id=true` when enabled. |
| `policy.repo_id` | `None` | Hub model repo id for upload. | Include task, robot, dataset version, and fps in the name. |
| `policy.private` | `None` | Whether the Hub repo is private. | Set `true` for protected data/models. |
| `policy.tags` | `None` | Hub model tags. | Improves discoverability. |
| `policy.license` | `None` | Hub model license. | Recommended when publishing or sharing. |
| `policy.pretrained_path` | `None` | Actual checkpoint path/repo loaded. | Set by `--policy.path` or resume logic. |

`PolicyFeature`:

| Field | Meaning |
|---|---|
| `type` | Feature type such as `STATE`, `VISUAL`, `ACTION`, or `ENV`. |
| `shape` | Tensor shape. Images are usually `[3, H, W]`; state/action features are joint or control dimensions. |

## Optimizer and Learning-Rate Scheduler

By default, you do not pass `optimizer.*` / `scheduler.*` directly. Instead, tune `policy.optimizer_*` and `policy.scheduler_*`. Only write the following configs manually when `--use_policy_training_preset=false`.

### OptimizerConfig

| Type | Parameters | Meaning and effect |
|---|---|---|
| `adam` | `lr`, `betas`, `eps`, `weight_decay`, `grad_clip_norm` | General Adam. `lr` controls step size; `betas` control momentum smoothing; `eps` prevents division by zero; `weight_decay` regularizes; `grad_clip_norm` limits gradient explosions. |
| `adamw` | `lr`, `betas`, `eps`, `weight_decay`, `grad_clip_norm` | AdamW decouples weight decay and is common for transformer/VLA training. |
| `sgd` | `lr`, `momentum`, `dampening`, `nesterov`, `weight_decay`, `grad_clip_norm` | Classic SGD. Usually not the first choice for VLA fine-tuning. |
| `xvla-adamw` | `lr`, `betas`, `eps`, `weight_decay`, `grad_clip_norm`, `soft_prompt_lr_scale`, `soft_prompt_warmup_lr_scale` | X-VLA-specific AdamW. VLM parameters use 0.1x LR, and soft prompts can be scaled separately. |
| `multi_adam` | `lr`, `weight_decay`, `grad_clip_norm`, `optimizer_groups` | Multi-optimizer setup, mainly for SAC and other multi-network RL policies. |

### SchedulerConfig

| Type | Parameters | Meaning and effect |
|---|---|---|
| `diffuser` | `name`, `num_warmup_steps` | Uses a diffusers scheduler such as cosine. |
| `vqbet` | `num_warmup_steps`, `num_vqvae_training_steps`, `num_cycles` | VQ-BeT trains VQ-VAE first, then uses warmup/cosine. |
| `cosine_decay_with_warmup` | `num_warmup_steps`, `num_decay_steps`, `peak_lr`, `decay_lr` | Linear warmup to peak LR, then cosine decay to decay LR. If training steps are fewer than decay steps, the schedule is automatically scaled. |

## SmolVLA Parameter Details

SmolVLA is the core policy used by your current script. The official docs recommend fine-tuning from `lerobot/smolvla_base` or a compatible checkpoint; 20k steps on an A100 takes roughly several hours, and real tasks should tune steps based on validation performance.

### Inputs, Outputs, and Action Horizon

| Parameter | Default | Meaning | Effect |
|---|---:|---|---|
| `policy.n_obs_steps` | `1` | Number of observation frames used as input. SmolVLA defaults to the current frame only. | Increasing requires model/data support for more history and raises memory use. |
| `policy.chunk_size` | `50` | Length of the action chunk predicted in one model call. | Larger chunks predict longer sequences but are harder to learn; must satisfy `n_action_steps <= chunk_size`. |
| `policy.n_action_steps` | `50` | Number of action steps actually used/executed after each model call. | Smaller than `chunk_size` increases replanning frequency; too small wastes predictions, too large reduces responsiveness. |
| `policy.normalization_mapping` | VISUAL identity, STATE/ACTION mean_std | Normalization mode per feature type. | Do not change casually; mismatch with the pretrained checkpoint can severely harm fine-tuning. |
| `policy.max_state_dim` | `32` | Maximum state vector dimension; shorter states are padded. | Must cover the robot state dimension; increasing adds projection input slack. |
| `policy.max_action_dim` | `32` | Maximum action vector dimension; shorter actions are padded. | Must cover the action dimension; your 7-D WidowX action fits. |
| `policy.resize_imgs_with_padding` | `(512, 512)` | Resizes images with padding to a fixed size. | Larger sizes preserve detail but increase memory/compute; smaller sizes are faster but can lose details. |
| `policy.empty_cameras` | `0` | Adds empty camera placeholders. | Useful when a checkpoint expects more cameras than the dataset provides; safer than arbitrarily deleting features. |
| `policy.adapt_to_pi_aloha` | `False` | Adapts Aloha action space to the PI internal runtime space. | Only for specific Aloha/PI-compatible flows. |
| `policy.use_delta_joint_actions_aloha` | `False` | Converts Aloha joint actions to relative deltas. | Enabling this currently raises `NotImplementedError`; do not use it for this project. |

### Language, Decoding, and Attention

| Parameter | Default | Meaning | Effect |
|---|---:|---|---|
| `policy.tokenizer_max_length` | `48` | Maximum token length for task language instructions. | Increase for long task descriptions; too short truncates key information. |
| `policy.num_steps` | `10` | Action generation/denoising steps. | More steps may improve quality but slow inference; fewer steps are faster but can reduce action quality. |
| `policy.use_cache` | `True` | Attention/cache switch. | Usually keep enabled for speed; disable only for debugging memory or compatibility issues. |
| `policy.add_image_special_tokens` | `False` | Whether to add special tokens around image features. | Must match the VLM/tokenizer pretraining scheme. |
| `policy.attention_mode` | `"cross_attn"` | Attention interaction mode between the VLM and action expert. | This is a structural change; avoid changing it during fine-tuning. |
| `policy.prefix_length` | `-1` | Prefix token length control; `-1` means default. | Advanced structural parameter; usually unchanged. |
| `policy.pad_language_to` | `"longest"` | Pads language to batch longest or max length. | `max_length` is more fixed but wasteful; `longest` is more efficient. |

### Fine-Tuning Scope and Freezing

| Parameter | Default | Meaning | Effect |
|---|---:|---|---|
| `policy.freeze_vision_encoder` | `True` | Freezes the vision encoder. | Saves memory and reduces visual representation forgetting; if camera/image distribution differs heavily, disabling can help but costs more and is riskier. |
| `policy.train_expert_only` | `True` | Trains only the action expert. | Good for small-data fine-tuning; full-model training adapts more but is more prone to overfitting/forgetting. |
| `policy.train_state_proj` | `True` | Trains the state projection. | Keep enabled when robot state dimension or semantics differ. |

### SmolVLA Preset Optimizer and Scheduler

| Parameter | Default | Meaning | Effect |
|---|---:|---|---|
| `policy.optimizer_lr` | `1e-4` | AdamW peak/base learning rate. | Full fine-tuning often starts at `1e-4` or lower; LoRA can start around `1e-3`. Your script uses `2e-4`, which is more aggressive. |
| `policy.optimizer_betas` | `(0.9, 0.95)` | AdamW momentum parameters. | Lower beta2 reacts more quickly to non-stationary fine-tuning; usually unchanged. |
| `policy.optimizer_eps` | `1e-8` | AdamW numerical stability term. | Usually unchanged. |
| `policy.optimizer_weight_decay` | `1e-10` | AdamW weight decay. | SmolVLA default is almost no weight decay; small increases can be tested on small data, but too much can underfit. |
| `policy.optimizer_grad_clip_norm` | `10` | Gradient clipping threshold. | Lower it if loss spikes or gradients explode; too low limits learning. |
| `policy.scheduler_warmup_steps` | `1000` | Number of warmup steps. | More warmup is more stable for small data or high LR; too much slows early learning. |
| `policy.scheduler_decay_steps` | `30000` | Total cosine decay steps. | Should be close to or slightly below `steps`; if `steps < decay_steps`, the scheduler auto-scales. |
| `policy.scheduler_decay_lr` | `2.5e-6` | Final learning rate. | Higher values keep updating more strongly late in training, which can help adaptation but increase overfitting; lower values are more conservative. |

### Architecture and Advanced Parameters

| Parameter | Default | Meaning | Effect |
|---|---:|---|---|
| `policy.vlm_model_name` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | VLM backbone name. | Changing it is effectively changing the architecture; avoid during checkpoint fine-tuning. |
| `policy.load_vlm_weights` | `False` | Loads weights from the VLM backbone; pretrained SmolVLA weights are usually handled by the checkpoint. | Can stay off for training an expert from scratch; consider only when training a new model. |
| `policy.num_expert_layers` | `-1` | Number of action expert layers; `<=0` means match the VLM layer count. | Fewer layers save memory but reduce capacity. |
| `policy.num_vlm_layers` | `16` | Number of VLM layers used. | Reducing saves compute but can lose semantic/visual capability. |
| `policy.self_attn_every_n_layers` | `2` | Frequency of self-attention layers in the action expert. | Structural parameter; avoid changing for fine-tuning. |
| `policy.expert_width_multiplier` | `0.75` | Action expert hidden size relative to the VLM. | Larger means more capacity and memory. |
| `policy.min_period` | `0.004` | Minimum period for timestep sin-cos positional encoding. | Affects diffusion/flow time embedding scale; usually unchanged. |
| `policy.max_period` | `4.0` | Maximum period for timestep sin-cos positional encoding. | Usually unchanged. |
| `policy.rtc_config` | `None` | Real-Time Chunking configuration. | Used for realtime chunked inference; usually unset for ordinary BC training. |
| `policy.compile_model` | `False` | Enables `torch.compile`. | May speed up training/inference, but first compilation is slow and compatibility can vary. |
| `policy.compile_mode` | `"max-autotune"` | Torch compile mode. | More aggressive optimization may be faster but compiles more slowly. |

## Parameters in Your Script

Key choices in the current `fine_tune_smolvla.sh`:

| Parameter | Current value | Interpretation |
|---|---:|---|
| `policy.path` | `TrossenRoboticsCommunity/smolvla_solo_red_block` | Fine-tunes from an existing SmolVLA checkpoint instead of starting from `lerobot/smolvla_base`. |
| `policy.input_features` | state 7 + main/wrist images | Explicitly overrides checkpoint inputs to fit WidowX data. |
| `policy.output_features` | action 7 | Matches a 7-D joint/gripper action. |
| `rename_map` | right->cam_main, wrist->cam_wrist | Maps dataset camera names to the names expected by the policy. This must match `input_features` exactly. |
| `batch_size` | `64` | Large for SmolVLA; good throughput but high memory pressure. |
| `steps` | `40000` | More than the common 20k example; suitable for multi-position/multi-version data, but needs validation to avoid overfitting. |
| `save_freq` | `5000` | Dense intermediate checkpoints, useful for selecting the best step. |
| `policy.scheduler_decay_steps` | `40000` | LR decay covers the full training run. |
| `policy.optimizer_lr` | `2e-4` | More aggressive than the default `1e-4`; if loss oscillates or forgetting appears, try `1e-4` or `5e-5`. |
| `policy.n_action_steps` | `10` | Train/inference replans after executing 10 actions. At 10 Hz, this is roughly 1 second per chunk. |
| `policy.chunk_size` | `25` | Predicts 25 steps but uses only 10; provides future-action supervision while keeping high replanning frequency. |
| `policy.resize_imgs_with_padding` | `[512, 512]` | Matches SmolVLA default. |
| `policy.use_amp` | `true` | Reduces memory and improves speed; if NaNs or instability appear, try disabling it first. |
| `wandb.project` | `smolvla_widowx_grape_grasping` | Clear experiment grouping. |

Recommended tuning order:

1. First lock the dataset and `rename_map`, and verify feature shapes, camera images, and action scales are all correct.
2. Run full training with the current `save_freq=5000`, then evaluate 5k/10k/.../40k checkpoints offline.
3. If validation error rises later in training, reduce `steps` or lower `optimizer_lr`.
4. If even training performance does not improve, first inspect action/state normalization, camera mapping, and task language before raising LR or unfreezing more modules.
5. If memory is tight, reduce `batch_size` first, then consider LoRA/PEFT; do not start by changing model architecture parameters.

## Complete Built-In Policy Parameter Index

The following are built-in policy configuration fields from `lerobot==0.5.1`. All policies also share the common fields described above: `policy.n_obs_steps/input_features/output_features/device/use_amp/use_peft/push_to_hub/repo_id/private/tags/license/pretrained_path`.

### ACT (`policy.type=act`)

Complete fields:

`n_obs_steps`, `chunk_size`, `n_action_steps`, `normalization_mapping`, `vision_backbone`, `pretrained_backbone_weights`, `replace_final_stride_with_dilation`, `pre_norm`, `dim_model`, `n_heads`, `dim_feedforward`, `feedforward_activation`, `n_encoder_layers`, `n_decoder_layers`, `use_vae`, `latent_dim`, `n_vae_encoder_layers`, `temporal_ensemble_coeff`, `dropout`, `kl_weight`, `optimizer_lr`, `optimizer_weight_decay`, `optimizer_lr_backbone`.

Tuning meaning:

- `chunk_size` / `n_action_steps` control action chunk length and execution steps; longer chunks are smoother but replan more slowly.
- `vision_backbone` / `pretrained_backbone_weights` control the visual feature extractor.
- `dim_model`, `n_heads`, `dim_feedforward`, `n_encoder_layers`, `n_decoder_layers` control transformer capacity.
- `use_vae`, `latent_dim`, `n_vae_encoder_layers`, `kl_weight` control ACT's CVAE behavior; too much `kl_weight` can make actions conservative, too little weakens latent regularization.
- `temporal_ensemble_coeff` controls temporal ensemble smoothing.
- `optimizer_lr`, `optimizer_weight_decay`, `optimizer_lr_backbone` control main and backbone learning rates.

### Diffusion (`policy.type=diffusion`)

Complete fields:

`n_obs_steps`, `horizon`, `n_action_steps`, `normalization_mapping`, `drop_n_last_frames`, `vision_backbone`, `resize_shape`, `crop_ratio`, `crop_shape`, `crop_is_random`, `pretrained_backbone_weights`, `use_group_norm`, `spatial_softmax_num_keypoints`, `use_separate_rgb_encoder_per_camera`, `down_dims`, `kernel_size`, `n_groups`, `diffusion_step_embed_dim`, `use_film_scale_modulation`, `noise_scheduler_type`, `num_train_timesteps`, `beta_schedule`, `beta_start`, `beta_end`, `prediction_type`, `clip_sample`, `clip_sample_range`, `num_inference_steps`, `compile_model`, `compile_mode`, `do_mask_loss_for_padding`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `scheduler_name`, `scheduler_warmup_steps`.

Tuning meaning:

- `horizon`, `n_obs_steps`, `n_action_steps` define the conditioning window and prediction window.
- Crop/backbone/group norm/keypoints control the visual encoder and image augmentation.
- `down_dims`, `kernel_size`, `n_groups`, `diffusion_step_embed_dim`, `use_film_scale_modulation` control U-Net capacity.
- `noise_scheduler_type`, `num_train_timesteps`, `beta_*`, `prediction_type`, `num_inference_steps` control the diffusion noise process; more inference steps are usually slower but more accurate.
- `do_mask_loss_for_padding` masks loss for padded actions.

### SmolVLA (`policy.type=smolvla`)

Complete fields, already detailed above:

`n_obs_steps`, `chunk_size`, `n_action_steps`, `normalization_mapping`, `max_state_dim`, `max_action_dim`, `resize_imgs_with_padding`, `empty_cameras`, `adapt_to_pi_aloha`, `use_delta_joint_actions_aloha`, `tokenizer_max_length`, `num_steps`, `use_cache`, `freeze_vision_encoder`, `train_expert_only`, `train_state_proj`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `optimizer_grad_clip_norm`, `scheduler_warmup_steps`, `scheduler_decay_steps`, `scheduler_decay_lr`, `vlm_model_name`, `load_vlm_weights`, `add_image_special_tokens`, `attention_mode`, `prefix_length`, `pad_language_to`, `num_expert_layers`, `num_vlm_layers`, `self_attn_every_n_layers`, `expert_width_multiplier`, `min_period`, `max_period`, `rtc_config`, `compile_model`, `compile_mode`.

### Pi0 / Pi05 / Pi0Fast

These are Physical Intelligence VLA-family policies. Their fields are conceptually similar to SmolVLA, but the models are larger and have higher memory/inference cost.

Pi0 complete fields:

`paligemma_variant`, `action_expert_variant`, `dtype`, `n_obs_steps`, `chunk_size`, `n_action_steps`, `max_state_dim`, `max_action_dim`, `num_inference_steps`, `time_sampling_beta_alpha`, `time_sampling_beta_beta`, `time_sampling_scale`, `time_sampling_offset`, `min_period`, `max_period`, `use_relative_actions`, `relative_exclude_joints`, `action_feature_names`, `rtc_config`, `image_resolution`, `empty_cameras`, `normalization_mapping`, `gradient_checkpointing`, `compile_model`, `compile_mode`, `device`, `freeze_vision_encoder`, `train_expert_only`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `optimizer_grad_clip_norm`, `scheduler_warmup_steps`, `scheduler_decay_steps`, `scheduler_decay_lr`, `tokenizer_max_length`.

Pi05 complete fields:

`paligemma_variant`, `action_expert_variant`, `dtype`, `n_obs_steps`, `chunk_size`, `n_action_steps`, `max_state_dim`, `max_action_dim`, `num_inference_steps`, `time_sampling_beta_alpha`, `time_sampling_beta_beta`, `time_sampling_scale`, `time_sampling_offset`, `min_period`, `max_period`, `use_relative_actions`, `relative_exclude_joints`, `action_feature_names`, `rtc_config`, `image_resolution`, `empty_cameras`, `tokenizer_max_length`, `normalization_mapping`, `gradient_checkpointing`, `compile_model`, `compile_mode`, `device`, `freeze_vision_encoder`, `train_expert_only`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `optimizer_grad_clip_norm`, `scheduler_warmup_steps`, `scheduler_decay_steps`, `scheduler_decay_lr`, `tokenizer_max_length`.

Pi0Fast complete fields:

`paligemma_variant`, `action_expert_variant`, `dtype`, `chunk_size`, `n_action_steps`, `max_state_dim`, `max_action_dim`, `max_action_tokens`, `use_relative_actions`, `relative_exclude_joints`, `action_feature_names`, `rtc_config`, `image_resolution`, `empty_cameras`, `tokenizer_max_length`, `text_tokenizer_name`, `action_tokenizer_name`, `temperature`, `max_decoding_steps`, `fast_skip_tokens`, `validate_action_token_prefix`, `use_kv_cache`, `normalization_mapping`, `gradient_checkpointing`, `compile_model`, `compile_mode`, `device`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `optimizer_grad_clip_norm`, `scheduler_warmup_steps`, `scheduler_decay_steps`, `scheduler_decay_lr`.

Tuning meaning:

- `paligemma_variant`, `action_expert_variant`, `dtype`, `image_resolution` control backbone/expert variants, precision, and visual resolution.
- `num_inference_steps`, `time_sampling_*`, `min_period`, `max_period` control flow/diffusion time sampling and inference steps.
- `use_relative_actions`, `relative_exclude_joints`, `action_feature_names` control action representation and must match the robot control convention.
- `train_expert_only`, `freeze_vision_encoder`, `gradient_checkpointing` affect fine-tuning scope, memory, and speed.

### X-VLA (`policy.type=xvla`)

Complete fields:

`n_obs_steps`, `chunk_size`, `n_action_steps`, `dtype`, `normalization_mapping`, `florence_config`, `tokenizer_name`, `tokenizer_max_length`, `tokenizer_padding_side`, `pad_language_to`, `hidden_size`, `depth`, `num_heads`, `mlp_ratio`, `num_domains`, `len_soft_prompts`, `dim_time`, `max_len_seq`, `use_hetero_proj`, `action_mode`, `num_denoising_steps`, `use_proprio`, `max_state_dim`, `max_action_dim`, `domain_feature_key`, `resize_imgs_with_padding`, `num_image_views`, `empty_cameras`, `freeze_vision_encoder`, `freeze_language_encoder`, `train_policy_transformer`, `train_soft_prompts`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `optimizer_grad_clip_norm`, `optimizer_soft_prompt_lr_scale`, `optimizer_soft_prompt_warmup_lr_scale`, `scheduler_warmup_steps`, `scheduler_decay_steps`, `scheduler_decay_lr`.

Tuning meaning:

- `florence_config` and tokenizer fields control the vision-language backbone.
- `hidden_size/depth/num_heads/mlp_ratio` control policy transformer capacity.
- `num_domains/len_soft_prompts/domain_feature_key/train_soft_prompts` control multi-robot/multi-task soft prompts.
- `action_mode`, `num_denoising_steps`, `use_proprio` control action representation and generation.

### Wall-X (`policy.type=wall_x`)

Complete fields:

`n_obs_steps`, `chunk_size`, `n_action_steps`, `max_action_dim`, `max_state_dim`, `normalization_mapping`, `pretrained_name_or_path`, `action_tokenizer_path`, `prediction_mode`, `attn_implementation`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `optimizer_grad_clip_norm`, `scheduler_warmup_steps`, `scheduler_decay_steps`, `scheduler_decay_lr`.

Tuning meaning:

- `prediction_mode` switches between diffusion and fast action prediction.
- `action_tokenizer_path` matters only for tokenized/fast action modes.
- `attn_implementation` can be eager/flash_attention_2/sdpa, affecting speed and dependencies.

### Groot (`policy.type=groot`)

Complete fields:

`n_obs_steps`, `chunk_size`, `n_action_steps`, `max_state_dim`, `max_action_dim`, `normalization_mapping`, `image_size`, `base_model_path`, `tokenizer_assets_repo`, `embodiment_tag`, `tune_llm`, `tune_visual`, `tune_projector`, `tune_diffusion_model`, `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_full_model`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `warmup_ratio`, `use_bf16`, `video_backend`, `balance_dataset_weights`, `balance_trajectory_weights`, `dataset_paths`, `output_dir`, `save_steps`, `max_steps`, `batch_size`, `dataloader_num_workers`, `report_to`, `resume`.

Tuning meaning:

- `base_model_path`, `tokenizer_assets_repo`, `embodiment_tag` select the GROOT base model, tokenizer assets, and robot embodiment.
- `tune_llm`, `tune_visual`, `tune_projector`, `tune_diffusion_model` control which submodules are fine-tuned.
- `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_full_model` control LoRA capacity and coverage.
- `balance_dataset_weights`, `balance_trajectory_weights`, `dataset_paths` are used for multi-source dataset balancing.
- `save_steps`, `max_steps`, `batch_size`, `dataloader_num_workers`, `report_to`, `resume` are training parameters embedded in this policy config.

### MultiTaskDiT (`policy.type=multi_task_dit`)

Complete fields:

`n_obs_steps`, `horizon`, `n_action_steps`, `objective`, `noise_scheduler_type`, `num_train_timesteps`, `beta_schedule`, `beta_start`, `beta_end`, `prediction_type`, `clip_sample`, `clip_sample_range`, `num_inference_steps`, `sigma_min`, `num_integration_steps`, `integration_method`, `timestep_sampling_strategy`, `timestep_sampling_s`, `timestep_sampling_alpha`, `timestep_sampling_beta`, `hidden_dim`, `num_layers`, `num_heads`, `dropout`, `use_positional_encoding`, `timestep_embed_dim`, `use_rope`, `rope_base`, `vision_encoder_name`, `use_separate_rgb_encoder_per_camera`, `vision_encoder_lr_multiplier`, `image_resize_shape`, `image_crop_shape`, `image_crop_is_random`, `text_encoder_name`, `tokenizer_max_length`, `tokenizer_padding`, `tokenizer_padding_side`, `tokenizer_truncation`, `normalization_mapping`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `scheduler_name`, `scheduler_warmup_steps`, `do_mask_loss_for_padding`, `drop_n_last_frames`.

Tuning meaning:

- `horizon`, `n_obs_steps`, `n_action_steps` control the action prediction horizon.
- `objective`, `noise_scheduler_type`, `num_train_timesteps`, `beta_*`, `sigma_min`, `num_integration_steps`, `integration_method`, `timestep_sampling_*` control the diffusion/flow objective and time sampling.
- `hidden_dim`, `num_layers`, `num_heads`, `dropout`, `use_rope` control DiT capacity.
- `vision_encoder_*`, `image_*`, `text_encoder_name`, `tokenizer_*` control vision/language encoding.

### VQ-BeT (`policy.type=vqbet`)

Complete fields:

`n_obs_steps`, `n_action_pred_token`, `action_chunk_size`, `normalization_mapping`, `vision_backbone`, `crop_shape`, `crop_is_random`, `pretrained_backbone_weights`, `use_group_norm`, `spatial_softmax_num_keypoints`, `n_vqvae_training_steps`, `vqvae_n_embed`, `vqvae_embedding_dim`, `vqvae_enc_hidden_dim`, `gpt_block_size`, `gpt_input_dim`, `gpt_output_dim`, `gpt_n_layer`, `gpt_n_head`, `gpt_hidden_dim`, `dropout`, `offset_loss_weight`, `primary_code_loss_weight`, `secondary_code_loss_weight`, `bet_softmax_temperature`, `sequentially_select`, `optimizer_lr`, `optimizer_betas`, `optimizer_eps`, `optimizer_weight_decay`, `optimizer_vqvae_lr`, `optimizer_vqvae_weight_decay`, `scheduler_warmup_steps`.

Tuning meaning:

- `n_vqvae_training_steps` and `vqvae_*` control action tokenization.
- `gpt_*` control behavior transformer capacity.
- `*_loss_weight` controls code/offset training objective weights.

### TDMPC (`policy.type=tdmpc`)

Complete fields:

`n_obs_steps`, `n_action_repeats`, `horizon`, `n_action_steps`, `normalization_mapping`, `image_encoder_hidden_dim`, `state_encoder_hidden_dim`, `latent_dim`, `q_ensemble_size`, `mlp_dim`, `discount`, `use_mpc`, `cem_iterations`, `max_std`, `min_std`, `n_gaussian_samples`, `n_pi_samples`, `uncertainty_regularizer_coeff`, `n_elites`, `elite_weighting_temperature`, `gaussian_mean_momentum`, `max_random_shift_ratio`, `reward_coeff`, `expectile_weight`, `value_coeff`, `consistency_coeff`, `advantage_scaling`, `pi_coeff`, `temporal_decay_coeff`, `target_model_momentum`, `optimizer_lr`.

Tuning meaning:

- MPC/CEM fields control planning quality and speed.
- reward/value/consistency/pi coefficients control TD-MPC loss weights.
- `target_model_momentum` controls target network update smoothing.

### SAC (`policy.type=sac`)

Complete fields:

`normalization_mapping`, `dataset_stats`, `device`, `storage_device`, `vision_encoder_name`, `freeze_vision_encoder`, `image_encoder_hidden_dim`, `shared_encoder`, `num_discrete_actions`, `image_embedding_pooling_dim`, `online_steps`, `online_buffer_capacity`, `offline_buffer_capacity`, `async_prefetch`, `online_step_before_learning`, `policy_update_freq`, `discount`, `temperature_init`, `num_critics`, `num_subsample_critics`, `critic_lr`, `actor_lr`, `temperature_lr`, `critic_target_update_weight`, `utd_ratio`, `state_encoder_hidden_dim`, `latent_dim`, `target_entropy`, `use_backup_entropy`, `grad_clip_norm`, `critic_network_kwargs`, `actor_network_kwargs`, `policy_kwargs`, `discrete_critic_network_kwargs`, `actor_learner_config`, `concurrency`, `use_torch_compile`.

Tuning meaning:

- SAC is an RL policy; many fields serve actor/critic/entropy/buffer behavior rather than offline BC.
- `discount`, `target_entropy`, `temperature_init`, `temperature_lr` control long-horizon reward and entropy regularization.
- `online_buffer_capacity`, `offline_buffer_capacity`, `online_steps`, `online_step_before_learning`, `utd_ratio` control sample collection and replay training intensity.
- `critic_lr`, `actor_lr`, `critic_target_update_weight`, `num_critics`, `num_subsample_critics` control actor/critic learning dynamics.

### SARM (`policy.type=sarm`)

Complete fields:

`annotation_mode`, `n_obs_steps`, `frame_gap`, `max_rewind_steps`, `image_dim`, `text_dim`, `hidden_dim`, `num_heads`, `num_layers`, `max_state_dim`, `drop_n_last_frames`, `batch_size`, `clip_batch_size`, `dropout`, `stage_loss_weight`, `rewind_probability`, `language_perturbation_probability`, `num_sparse_stages`, `sparse_subtask_names`, `sparse_temporal_proportions`, `num_dense_stages`, `dense_subtask_names`, `dense_temporal_proportions`, `pretrained_model_path`, `device`, `image_key`, `state_key`, `input_features`, `output_features`, `normalization_mapping`.

Tuning meaning:

- `annotation_mode`, `num_sparse_stages`, `sparse_subtask_names`, `dense_subtask_names`, `*_temporal_proportions` control stage/subtask annotations.
- `frame_gap`, `max_rewind_steps`, `rewind_probability` control temporal rewind training.
- `image_dim`, `text_dim`, `hidden_dim`, `num_heads`, `num_layers` control model capacity.
- `stage_loss_weight`, `language_perturbation_probability` control stage supervision and language perturbation strength.

### RewardClassifierConfig

Complete fields:

`name`, `num_classes`, `hidden_dim`, `latent_dim`, `image_embedding_pooling_dim`, `dropout_rate`, `model_name`, `device`, `model_type`, `num_cameras`, `learning_rate`, `weight_decay`, `grad_clip_norm`, `normalization_mapping`.

Meaning: used for the SAC reward classifier. `num_classes` controls class count; `hidden_dim/latent_dim/image_embedding_pooling_dim/dropout_rate` control model capacity and regularization; `model_name/model_type/num_cameras` control the visual model and camera count; `learning_rate/weight_decay/grad_clip_norm` control training stability.

## Environment Parameter Index

Offline fine-tuning on real robot data usually does not require `env.*`. These parameters matter when training/evaluating in simulation or RL.

Common `EnvConfig`:

`task`, `fps`, `features`, `features_map`, `max_parallel_tasks`, `disable_env_checker`.

Environment types and main fields:

| `env.type` | Complete fields |
|---|---|
| `aloha` | `task`, `fps`, `episode_length`, `obs_type`, `observation_height`, `observation_width`, `render_mode`, `features`, `features_map` |
| `pusht` | `task`, `fps`, `episode_length`, `obs_type`, `render_mode`, `visualization_width`, `visualization_height`, `observation_height`, `observation_width`, `features`, `features_map` |
| `libero` | `task`, `task_ids`, `fps`, `episode_length`, `obs_type`, `render_mode`, `camera_name`, `init_states`, `camera_name_mapping`, `observation_height`, `observation_width`, `features`, `features_map`, `control_mode` |
| `metaworld` | `task`, `fps`, `episode_length`, `obs_type`, `render_mode`, `multitask_eval`, `features`, `features_map` |
| `gym_manipulator` | `robot`, `teleop`, `processor`, `name` |
| `isaaclab_arena` | `hub_path`, `episode_length`, `num_envs`, `embodiment`, `object`, `mimic`, `teleop_device`, `seed`, `device`, `disable_fabric`, `enable_cameras`, `headless`, `enable_pinocchio`, `environment`, `task`, `state_dim`, `action_dim`, `camera_height`, `camera_width`, `video`, `video_length`, `video_interval`, `state_keys`, `camera_keys`, `features`, `features_map`, `kwargs` |

Environment field effects:

- `task` selects the task.
- `fps` must match the data/control frequency; otherwise temporal deltas and action execution duration will be wrong.
- `obs_type` selects pixels, state, or both.
- `features` / `features_map` determine how env outputs map to policy inputs.
- `episode_length` controls maximum episode length.
- Camera height/width/name/key fields control visual input and must be compatible with policy image features.
- `control_mode` controls whether actions are absolute or relative, which is critical for real-robot transfer.

## Multi-GPU / Accelerate Parameters

These are not `TrainPipelineConfig` fields. They are outer `accelerate launch` parameters.

Example:

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

Key points:

- `--multi_gpu`: enables multi-GPU training.
- `--num_processes`: number of GPU/process workers.
- `--mixed_precision=fp16|bf16`: mixed precision is controlled by accelerate; `policy.use_amp` no longer controls mixed precision in this setup.
- LeRobot does not automatically scale learning rate or steps by GPU count. Effective batch size is approximately `batch_size * num_processes`; decide manually whether to scale LR linearly or reduce steps proportionally.

## References

- LeRobot documentation home: https://huggingface.co/docs/lerobot/index
- SmolVLA documentation: https://huggingface.co/docs/lerobot/smolvla
- PEFT fine-tuning documentation: https://huggingface.co/docs/lerobot/peft_training
- Multi-GPU documentation: https://huggingface.co/docs/lerobot/multi_gpu_training
- LeRobot GitHub: https://github.com/huggingface/lerobot
- PyPI `lerobot==0.5.1`: https://pypi.org/project/lerobot/
