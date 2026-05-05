import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tempfile import TemporaryDirectory

from important_code.training.evaluate_smolvla_v4_checkpoints import (
    find_checkpoint_paths,
    load_policy_config,
    parse_episode_selector,
    parse_step_selector,
    write_metric_plots,
)


class EvaluateSmolVLAV4CheckpointsTest(unittest.TestCase):
    def test_parse_episode_selector_supports_all_ranges_and_lists(self):
        self.assertEqual(parse_episode_selector("all", total_episodes=5), [0, 1, 2, 3, 4])
        self.assertEqual(parse_episode_selector("0,2-4,6", total_episodes=8), [0, 2, 3, 4, 6])

    def test_parse_step_selector_supports_all_and_comma_lists(self):
        self.assertIsNone(parse_step_selector("all"))
        self.assertEqual(parse_step_selector("40000,60000,120000"), [40000, 60000, 120000])

    def test_find_checkpoint_paths_filters_missing_requested_steps(self):
        with TemporaryDirectory() as tmp:
            train_dir = Path(tmp)
            for step in (5000, 10000, 20000):
                checkpoint = train_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
                checkpoint.mkdir(parents=True)
                (checkpoint / "config.json").write_text("{}", encoding="utf-8")

            paths = find_checkpoint_paths(train_dir, requested_steps=[5000, 20000, 120000])

        self.assertEqual([item.step for item in paths], [5000, 20000])
        self.assertEqual(paths[0].path.name, "pretrained_model")

    def test_load_policy_config_uses_checkpoint_as_pretrained_path(self):
        checkpoint_path = Path("outputs/train/example/checkpoints/040000/pretrained_model")
        config = SimpleNamespace(pretrained_path="base-model", use_peft=True)

        with patch("lerobot.configs.policies.PreTrainedConfig.from_pretrained", return_value=config) as from_pretrained:
            loaded = load_policy_config(checkpoint_path)

        from_pretrained.assert_called_once_with(checkpoint_path)
        self.assertIs(loaded, config)
        self.assertEqual(loaded.pretrained_path, checkpoint_path)

    def test_write_metric_plots_creates_overview_and_joint_plots(self):
        rows = [
            {
                "split": "train",
                "checkpoint_step": 5000,
                "mae_mean": 0.3,
                "rmse_mean": 0.4,
                "max_abs_mean": 0.8,
                "latency_ms_mean": 100.0,
                "mae_joint_0": 0.2,
                "rmse_joint_0": 0.3,
                "mae_left_carriage_joint": 0.01,
                "rmse_left_carriage_joint": 0.02,
            },
            {
                "split": "val",
                "checkpoint_step": 5000,
                "mae_mean": 0.35,
                "rmse_mean": 0.45,
                "max_abs_mean": 0.9,
                "latency_ms_mean": 110.0,
                "mae_joint_0": 0.25,
                "rmse_joint_0": 0.35,
                "mae_left_carriage_joint": 0.02,
                "rmse_left_carriage_joint": 0.03,
            },
            {
                "split": "val",
                "checkpoint_step": 10000,
                "mae_mean": 0.2,
                "rmse_mean": 0.3,
                "max_abs_mean": 0.7,
                "latency_ms_mean": 120.0,
                "mae_joint_0": 0.15,
                "rmse_joint_0": 0.25,
                "mae_left_carriage_joint": 0.015,
                "rmse_left_carriage_joint": 0.025,
            },
        ]

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_metric_plots(rows, output_dir)

            self.assertTrue((output_dir / "checkpoint_metrics_overview.png").exists())
            self.assertTrue((output_dir / "checkpoint_joint_mae.png").exists())
            self.assertTrue((output_dir / "checkpoint_joint_rmse.png").exists())


if __name__ == "__main__":
    unittest.main()
