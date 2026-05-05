import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from important_code.inference import run_inference_rtc


class RunInferenceRTCPolicyLoadingTest(unittest.TestCase):
    def test_default_train_dir_points_to_latest_lora_run(self):
        self.assertEqual(
            run_inference_rtc.DEFAULT_TRAIN_DIR,
            "outputs/train/smolvla_widowx_grape_grasping_V4_pos234_lora",
        )

    def test_load_smolvla_policy_attaches_lora_adapter_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir)
            (policy_path / "adapter_config.json").write_text("{}", encoding="utf-8")

            config = SimpleNamespace(device="cuda")
            peft_config = SimpleNamespace(base_model_name_or_path="base/model")
            base_policy = object()
            lora_policy = object()

            with (
                patch.object(run_inference_rtc.PreTrainedConfig, "from_pretrained", return_value=config) as load_config,
                patch.object(run_inference_rtc.PeftConfig, "from_pretrained", return_value=peft_config) as load_peft_config,
                patch.object(run_inference_rtc.SmolVLAPolicy, "from_pretrained", return_value=base_policy) as load_base,
                patch.object(run_inference_rtc.PeftModel, "from_pretrained", return_value=lora_policy) as load_adapter,
            ):
                policy = run_inference_rtc.load_smolvla_policy(str(policy_path))

        self.assertIs(policy, lora_policy)
        load_config.assert_called_once_with(str(policy_path))
        load_peft_config.assert_called_once_with(str(policy_path))
        load_base.assert_called_once_with("base/model", config=config)
        load_adapter.assert_called_once_with(base_policy, str(policy_path), config=peft_config)

    def test_load_smolvla_policy_uses_regular_loader_for_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            full_policy = object()

            with patch.object(run_inference_rtc.SmolVLAPolicy, "from_pretrained", return_value=full_policy) as load_policy:
                policy = run_inference_rtc.load_smolvla_policy(tmpdir)

        self.assertIs(policy, full_policy)
        load_policy.assert_called_once_with(tmpdir)


if __name__ == "__main__":
    unittest.main()
