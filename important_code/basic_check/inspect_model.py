"""
Script to inspect a fine-tuned SmolVLA policy and print its details.
Usage: python inspect_model.py --train-dir outputs/train/smolvla_widowx_aluminum
"""

import argparse
import sys

# Inspect 'lerobot' imports
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
except ImportError as e:
    print(f"Error importing lerobot: {e}")
    sys.exit(1)

from utils import resolve_checkpoint_path

DEFAULT_TRAIN_DIR = "outputs/train/smolvla_widowx_grape_grasping"

def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")

def inspect_model(train_dir):
    print(f"Resolving policy from: {train_dir}")
    try:
        policy_path = resolve_checkpoint_path(train_dir)
        print(f"Found checkpoint at: {policy_path}")
    except Exception as e:
        print(f"Error resolving checkpoint: {e}")
        return

    print("Loading policy...")
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.eval()
    
    config = policy.config
    
    # 1. Model Configuration
    print_section("Model Configuration")
    print(f"Policy Class: {type(policy).__name__}")
    print(f"VLM Backbone: {config.vlm_model_name}")
    print(f"Chunk Size: {config.chunk_size}")
    print(f"Input Sequence Length: {config.n_obs_steps}")
    action_shape = config.output_features.get('action')
    print(f"Action Dimension: {action_shape.shape if action_shape else 'Unknown'}")
    print(f"Max Action Dim (padded): {config.max_action_dim}")
    print(f"Max State Dim (padded): {config.max_state_dim}")

    # 2. Input Features
    print_section("Input Features")
    for name, feature in config.input_features.items():
        print(f"- {name}:")
        print(f"  Shape: {feature.shape}")
        if hasattr(feature, 'dtype'):
            print(f"  Dtype: {feature.dtype}")
        # Show normalization mode for this feature type
        if hasattr(feature, 'type') and config.normalization_mapping:
            norm_mode = config.normalization_mapping.get(feature.type.value)
            if norm_mode:
                print(f"  Normalization: {norm_mode}")

    # 3. Output Features
    print_section("Output Features")
    for name, feature in config.output_features.items():
        print(f"- {name}:")
        print(f"  Shape: {feature.shape}")
        if hasattr(feature, 'dtype'):
            print(f"  Dtype: {feature.dtype}")
        if hasattr(feature, 'type') and config.normalization_mapping:
            norm_mode = config.normalization_mapping.get(feature.type.value)
            if norm_mode:
                print(f"  Normalization: {norm_mode}")

    # 4. Parameters
    print_section("Model Parameters")
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    # Use actual dtype size per parameter for accurate calculation
    model_size_bytes = sum(p.numel() * p.element_size() for p in policy.parameters())
    print(f"Model Size (actual): {model_size_bytes / (1024**2):.2f} MB")
    print(f"Model Size (if float32): {total_params * 4 / (1024**2):.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Fine-tuned Model Details")
    parser.add_argument("--train-dir", type=str, default=DEFAULT_TRAIN_DIR, help="Path to training output directory")
    args = parser.parse_args()
    
    inspect_model(args.train_dir)
