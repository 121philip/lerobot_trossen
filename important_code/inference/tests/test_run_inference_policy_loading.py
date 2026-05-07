import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from important_code.inference import run_inference


class RunInferencePolicyLoadingTest(unittest.TestCase):
    def test_default_train_dir_points_to_latest_lora_run(self):
        self.assertEqual(
            run_inference.DEFAULT_TRAIN_DIR,
            "fulloa10/smolVLA_grape_10hz_9000",
        )

    def test_default_control_rate_is_10hz(self):
        with patch("sys.argv", ["run_inference.py"]):
            args = run_inference.parse_args()

        self.assertEqual(args.camera_fps, 30)
        self.assertEqual(args.control_fps, 10)
        self.assertEqual(args.fps, 10)
        self.assertEqual(args.queue_threshold, 10)
        self.assertEqual(args.task, "pick the grape")

    def test_fps_alias_sets_control_rate_without_changing_camera_rate(self):
        with patch("sys.argv", ["run_inference.py", "--fps", "12"]):
            args = run_inference.parse_args()

        self.assertEqual(args.camera_fps, 30)
        self.assertEqual(args.control_fps, 12)
        self.assertEqual(args.fps, 12)

    def test_camera_fps_is_configured_separately_from_control_rate(self):
        with patch("sys.argv", ["run_inference.py", "--camera-fps", "30", "--control-fps", "10"]):
            args = run_inference.parse_args()

        self.assertEqual(args.camera_fps, 30)
        self.assertEqual(args.control_fps, 10)
        self.assertEqual(args.fps, 10)

    def test_load_smolvla_policy_attaches_lora_adapter_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir)
            (policy_path / "adapter_config.json").write_text("{}", encoding="utf-8")

            config = SimpleNamespace(device="cuda")
            peft_config = SimpleNamespace(base_model_name_or_path="base/model")
            base_policy = object()
            lora_policy = object()

            with (
                patch.object(run_inference.PreTrainedConfig, "from_pretrained", return_value=config) as load_config,
                patch.object(run_inference.PeftConfig, "from_pretrained", return_value=peft_config) as load_peft_config,
                patch.object(run_inference.SmolVLAPolicy, "from_pretrained", return_value=base_policy) as load_base,
                patch.object(run_inference.PeftModel, "from_pretrained", return_value=lora_policy) as load_adapter,
            ):
                policy = run_inference.load_smolvla_policy(str(policy_path))

        self.assertIs(policy, lora_policy)
        load_config.assert_called_once_with(str(policy_path))
        load_peft_config.assert_called_once_with(str(policy_path))
        load_base.assert_called_once_with("base/model", config=config)
        load_adapter.assert_called_once_with(base_policy, str(policy_path), config=peft_config)

    def test_load_smolvla_policy_uses_regular_loader_for_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            full_policy = object()

            with patch.object(run_inference.SmolVLAPolicy, "from_pretrained", return_value=full_policy) as load_policy:
                policy = run_inference.load_smolvla_policy(tmpdir)

        self.assertIs(policy, full_policy)
        load_policy.assert_called_once_with(tmpdir)


if __name__ == "__main__":
    unittest.main()
