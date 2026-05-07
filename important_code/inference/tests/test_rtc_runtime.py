import unittest
from types import SimpleNamespace

from important_code.inference.rtc_runtime import configure_policy_rtc


class _FakePolicy:
    def __init__(self):
        self.config = SimpleNamespace(rtc_config="existing")
        self.init_calls = 0

    def init_rtc_processor(self):
        self.init_calls += 1


def _args(rtc):
    return SimpleNamespace(
        rtc=rtc,
        execution_horizon=10,
        guidance_weight=10.0,
        attention_schedule="EXP",
        debug=False,
        debug_maxlen=100,
    )


class RTCRuntimeTest(unittest.TestCase):
    def test_policy_rtc_config_is_cleared_when_rtc_is_disabled(self):
        policy = _FakePolicy()

        rtc_config = configure_policy_rtc(policy, _args(rtc=False))

        self.assertFalse(rtc_config.enabled)
        self.assertIsNone(policy.config.rtc_config)
        self.assertEqual(policy.init_calls, 1)

    def test_policy_rtc_config_is_attached_when_rtc_is_enabled(self):
        policy = _FakePolicy()

        rtc_config = configure_policy_rtc(policy, _args(rtc=True))

        self.assertTrue(rtc_config.enabled)
        self.assertIs(policy.config.rtc_config, rtc_config)
        self.assertEqual(policy.init_calls, 1)


if __name__ == "__main__":
    unittest.main()
