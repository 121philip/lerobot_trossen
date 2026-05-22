from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("kaixiyao/widowxai_pipe_bomb_transfer_v1_fixed")
dataset.push_to_hub(private=False)
