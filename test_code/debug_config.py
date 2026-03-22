import sys
import os
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs import parser

# Mock sys.argv to simulate the command
sys.argv = [
    "lerobot_train",
    "--policy.path=lerobot/smolvla_base",
    "--policy.push_to_hub=true",
    "--policy.repo_id=kaixiyao/smolvla_widowx_aluminum",
    '--policy.input_features={"observation.state": {"type": "STATE", "shape": [7]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 480, 640]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 480, 640]}}',
    '--policy.output_features={"action": {"type": "ACTION", "shape": [7]}}',
    "--dataset.repo_id=kaixiyao/grab_aluminum_profile",
    "--output_dir=outputs/train/debug_config",
    "--job_name=debug",
    "--batch_size=8",
    "--steps=1",
    "--save_freq=5000",
    "--policy.n_action_steps=50",
    "--policy.chunk_size=50",
    "--policy.device=cpu",
    "--wandb.enable=false",
    '--rename_map={"observation.images.cam_high": "observation.images.camera1", "observation.images.cam_wrist": "observation.images.camera2"}'
]

def main():
    print("Debug: Parsing configuration...")
    try:
        cfg = parser.parse(TrainPipelineConfig)
        
        print("\n--- Policy Configuration ---")
        print(f"Policy Type: {cfg.policy.type}")
        print("\n--- Input Features ---")
        for key, value in cfg.policy.input_features.items():
            print(f"Key: {key}")
            print(f"  Type: {value.type}")
            print(f"  Shape: {value.shape}")
            
        print("\n--- Analysis ---")
        has_camera3 = "observation.images.camera3" in cfg.policy.input_features
        if has_camera3:
            print("WARNING: 'observation.images.camera3' IS PRESENT in the config!")
        else:
            print("SUCCESS: 'observation.images.camera3' is NOT present.")

    except Exception as e:
        print(f"Error parsing config: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
