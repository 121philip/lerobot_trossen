from pathlib import Path

import torch

# 机器人关节名称（7 个自由度：6 旋转关节 + 夹爪）
JOINT_NAMES = [
    "joint_0", "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "left_carriage_joint",
]

# 优先使用 GPU，否则回退到 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint_path(train_dir: str) -> str:
    """Return path to the pretrained_model dir inside the last checkpoint.

    If train_dir looks like a HuggingFace repo ID (e.g. 'username/model-name'),
    return it as-is so from_pretrained() can download it directly from the Hub.
    """
    # HF repo ID has exactly one '/' (owner/repo) and is not a local path
    p = Path(train_dir)
    if not p.exists() and train_dir.count("/") == 1 and not train_dir.startswith("/"):
        return train_dir  # treat as HF Hub repo ID

    last_ptr = p / "checkpoints" / "last"
    if not last_ptr.exists():
        raise FileNotFoundError(
            f"No checkpoint 'last' pointer found at: {last_ptr}\n"
            f"Tip: pass a HuggingFace repo ID (e.g. 'username/model-name') "
            f"to load directly from the Hub.")
    if last_ptr.is_dir():
        checkpoint_dir = last_ptr
    elif last_ptr.is_file():
        checkpoint_dir = last_ptr.parent / last_ptr.read_text().strip()
    else:
        raise ValueError(
            f"'last' at {last_ptr} is neither a file nor a directory.")
    pretrained = checkpoint_dir / "pretrained_model"
    if not pretrained.exists():
        raise FileNotFoundError(f"pretrained_model not found at: {pretrained}")
    return str(pretrained)
