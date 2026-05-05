from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("kaixiyao/widowxai_grape_grasping_V4_position3_10hz")
dataset.push_to_hub(private=False)
