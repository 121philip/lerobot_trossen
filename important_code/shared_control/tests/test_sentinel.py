import tempfile
import time
import unittest
from types import SimpleNamespace

import numpy as np

from important_code.shared_control.sentinel import (
    CloudVLMClient,
    LocalMotionDetector,
    ProgressDecayMonitor,
    ProgressMonitorResult,
    SentinelFrameBuffer,
    SentinelRuntime,
)


def _metrics(c_action=0.5):
    return SimpleNamespace(c_action=c_action, jerk_max=0.0, boundary_jump_max=0.0)


class SentinelFrameBufferTest(unittest.TestCase):
    def test_push_observation_accepts_policy_image_keys(self):
        buffer = SentinelFrameBuffer(max_age_s=8.0)
        image = np.zeros((4, 4, 3), dtype=np.uint8)

        buffer.push_observation(
            {
                "observation.images.wrist": image,
                "observation.images.right": image + 1,
            }
        )

        frames = buffer.sample_window(window_s=8.0, max_frames=6)
        self.assertEqual(len(frames), 1)
        self.assertIsNotNone(frames[0].wrist)
        self.assertIsNotNone(frames[0].right)


class SentinelRuntimeTest(unittest.TestCase):
    def test_action_alarm_fires_on_low_consistency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(tau_action=0.4, log_dir=tmpdir)
            fast = sentinel._fast_action(_metrics(c_action=0.2))
            sentinel.stop()

        self.assertTrue(fast.action_alarm)
        self.assertEqual(fast.c_action, 0.2)

    def test_vlm_result_is_stored_in_c_vlm_not_c_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, eps=0.001, log_dir=tmpdir)
            fast = sentinel._fast_action(_metrics(c_action=0.9))
            result = sentinel._arbitrate(
                fast,
                ProgressMonitorResult(
                    timestamp=time.time(),
                    c_progress=0.1,
                    alarm=True,
                    stuck=True,
                    progress_made=False,
                    failure_likelihood=0.9,
                    reason="stuck",
                    latency_s=0.2,
                ),
            )
            sentinel.stop()

        # VLM c_progress=0.1 goes to c_vlm, not to arbitration formula
        self.assertAlmostEqual(result.c_vlm, 0.1)
        # r_raw = min(c_action=0.9, c_progress_decay); since no joints pushed, decay=1.0
        self.assertAlmostEqual(result.r_raw, 0.9, places=3)
        # progress_alarm still reflects VLM alarm
        self.assertTrue(result.progress_alarm)

    def test_stale_vlm_sets_c_vlm_to_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, eps=0.001, log_dir=tmpdir)
            fast = sentinel._fast_action(_metrics(c_action=0.7))
            result = sentinel._arbitrate(
                fast,
                ProgressMonitorResult(
                    timestamp=time.time(),
                    c_progress=None,
                    alarm=False,
                    stuck=False,
                    progress_made=None,
                    failure_likelihood=None,
                    reason="",
                    latency_s=0.0,
                    stale=True,
                    error="timeout",
                ),
            )
            sentinel.stop()

        self.assertIsNone(result.c_vlm)
        self.assertTrue(result.progress_stale)
        self.assertFalse(result.progress_alarm)
        # c_progress from decay (no joints pushed → 1.0), r_raw = min(0.7, 1.0) = 0.7
        self.assertAlmostEqual(result.r_raw, 0.7, places=3)

    def test_progress_alarm_requires_consecutive_failures(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        failing = [
            ProgressMonitorResult(1.0, 0.1, False, True, False, 0.9, "stuck", 0.1),
            ProgressMonitorResult(2.0, 0.1, False, True, False, 0.9, "stuck", 0.1),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(
                client=_FakeClient(failing),
                required_alarm_count=2,
                log_dir=tmpdir,
            )
            sentinel.frame_buffer.push(wrist=image, right=image)
            first = sentinel._check_progress("prompt")
            second = sentinel._check_progress("prompt")
            sentinel.stop()

        self.assertFalse(first.alarm)
        self.assertTrue(second.alarm)
        self.assertEqual(second.consecutive_alarm_count, 2)

    def test_logger_writes_jsonl_and_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, log_dir=tmpdir)
            sentinel.update(_metrics(c_action=0.5), extra={"chunk_latency_s": 0.01})
            log_dir = sentinel.log_dir
            sentinel.stop()

            self.assertTrue((log_dir / "sentinel_events.jsonl").exists())
            self.assertTrue((log_dir / "sentinel_events.csv").exists())


class _FakeClient:
    def __init__(self, results):
        self.results = list(results)

    def classify_progress(self, prompt, image_b64, timeout_s):
        return self.results.pop(0)


class LocalMotionDetectorTest(unittest.TestCase):
    def test_insufficient_data_returns_neutral(self):
        det = LocalMotionDetector(window_s=3.0, stuck_threshold=0.02)
        det.push(np.zeros(7), time.time())
        self.assertAlmostEqual(det.c_progress_local(), 0.5)

    def test_stuck_returns_low_confidence(self):
        det = LocalMotionDetector(window_s=3.0, stuck_threshold=0.02)
        t = time.time()
        for i in range(8):
            det.push(np.zeros(7), t + i * 0.4)  # 3.2 s span, no movement
        self.assertAlmostEqual(det.c_progress_local(), 0.2)

    def test_moving_returns_mid_confidence(self):
        det = LocalMotionDetector(window_s=3.0, stuck_threshold=0.02)
        t = time.time()
        for i in range(8):
            det.push(np.array([0.1 * i, 0, 0, 0, 0, 0, 0]), t + i * 0.4)
        self.assertAlmostEqual(det.c_progress_local(), 0.5)


class SentinelStaleCapTest(unittest.TestCase):
    def test_stuck_robot_lowers_r_raw(self):
        """r_raw < 1.0 when robot is stuck and c_action=1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, eps=0.0, log_dir=tmpdir)
            t = 1000.0
            for i in range(8):
                sentinel._progress_decay.push(np.zeros(7), t + i * 0.4)
            stuck_since = sentinel._progress_decay._stuck_since
            self.assertIsNotNone(stuck_since)
            c = sentinel._progress_decay.c_progress(stuck_since + 30)
            self.assertLess(c, 1.0)
            self.assertGreaterEqual(c, 0.2)
            sentinel.stop()

    def test_moving_robot_keeps_full_c_progress(self):
        """c_progress = 1.0 when robot is moving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, eps=0.0, log_dir=tmpdir)
            t = 1000.0
            for i in range(8):
                sentinel._progress_decay.push(
                    np.array([0.1 * i, 0, 0, 0, 0, 0, 0]), t + i * 0.4
                )
            c = sentinel._progress_decay.c_progress(t + 3.2)
            self.assertAlmostEqual(c, 1.0)
            sentinel.stop()


class ProgressDecayMonitorTest(unittest.TestCase):
    def test_no_history_returns_full_confidence(self):
        m = ProgressDecayMonitor()
        self.assertAlmostEqual(m.c_progress(), 1.0)

    def test_moving_returns_full_confidence(self):
        m = ProgressDecayMonitor(stuck_threshold=0.02)
        t = 1000.0
        for i in range(8):
            m.push(np.array([0.1 * i, 0, 0, 0, 0, 0, 0]), t + i * 0.4)
        self.assertAlmostEqual(m.c_progress(t + 3.2), 1.0)

    def test_stuck_decays_below_full(self):
        m = ProgressDecayMonitor(decay_lambda=0.05, floor=0.2)
        t = 1000.0
        for i in range(8):
            m.push(np.zeros(7), t + i * 0.4)
        c_early = m.c_progress(t + 3.2)
        c_later = m.c_progress(t + 3.2 + 30)
        self.assertLess(c_early, 1.0)
        self.assertLess(c_later, c_early)
        self.assertLess(c_later, 1.0)
        self.assertGreaterEqual(c_later, 0.2)

    def test_past_now_clamps_to_full_confidence(self):
        """now earlier than stuck_since must not exceed 1.0."""
        m = ProgressDecayMonitor(decay_lambda=0.05, floor=0.2)
        t = 1000.0
        for i in range(8):
            m.push(np.zeros(7), t + i * 0.4)
        # Pass a now earlier than when stuck was first detected
        c = m.c_progress(t)  # earlier than stuck_since (~t+1.2)
        self.assertLessEqual(c, 1.0)
        self.assertGreaterEqual(c, 0.2)

    def test_floor_approached_asymptotically(self):
        m = ProgressDecayMonitor(decay_lambda=0.05, floor=0.2)
        t = 1000.0
        for i in range(8):
            m.push(np.zeros(7), t + i * 0.4)
        c = m.c_progress(t + 3.2 + 1000)
        self.assertAlmostEqual(c, 0.2, places=2)

    def test_recovery_resets_to_full_confidence(self):
        m = ProgressDecayMonitor()
        t = 1000.0
        for i in range(8):
            m.push(np.zeros(7), t + i * 0.4)
        c_stuck = m.c_progress(t + 10.0)
        self.assertLess(c_stuck, 1.0)
        for i in range(8):
            m.push(np.array([0.1 * i, 0, 0, 0, 0, 0, 0]), t + 10.0 + i * 0.4)
        c_moving = m.c_progress(t + 10.0 + 3.2)
        self.assertAlmostEqual(c_moving, 1.0)


class CloudVLMClientTest(unittest.TestCase):
    def test_deepseek_missing_key_returns_stale(self):
        client = CloudVLMClient(provider="deepseek", model="deepseek-chat", api_key="")
        result = client.classify_progress("test", "fake_b64", timeout_s=1.0)
        self.assertTrue(result.stale)
        self.assertIn("Missing API key", result.error)

    def test_openai_missing_key_returns_stale(self):
        client = CloudVLMClient(provider="openai", model="gpt-4o", api_key="")
        result = client.classify_progress("test", "fake_b64", timeout_s=1.0)
        self.assertTrue(result.stale)
        self.assertIn("Missing API key", result.error)


class SentinelArbitrationResultFieldsTest(unittest.TestCase):
    def test_result_has_c_vlm_field(self):
        from important_code.shared_control.sentinel import SentinelArbitrationResult
        import dataclasses
        fields = {f.name for f in dataclasses.fields(SentinelArbitrationResult)}
        self.assertIn("c_vlm", fields)

    def test_c_progress_and_c_vlm_are_present(self):
        from important_code.shared_control.sentinel import SentinelArbitrationResult
        import dataclasses
        fields = {f.name for f in dataclasses.fields(SentinelArbitrationResult)}
        self.assertIn("c_progress", fields)
        self.assertIn("c_vlm", fields)


class SentinelDecayIntegrationTest(unittest.TestCase):
    def test_c_progress_is_float_in_result(self):
        """c_progress 始终是 float，不再是 None。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, eps=0.0, log_dir=tmpdir)
            result = sentinel.update(_metrics(c_action=0.8))
            sentinel.stop()
        self.assertIsInstance(result.c_progress, float)

    def test_stuck_robot_lowers_c_progress(self):
        """向 sentinel 推送卡住的关节，c_progress 应低于 1.0。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, eps=0.0, log_dir=tmpdir)
            t = time.time()
            for i in range(8):
                sentinel._progress_decay.push(np.zeros(7), t + i * 0.4)
            stuck_since = sentinel._progress_decay._stuck_since
            self.assertIsNotNone(stuck_since, "stuck detection should have fired after 8 zero-joint pushes spanning >1s")
            c = sentinel._progress_decay.c_progress(stuck_since + 30)
            self.assertLess(c, 1.0)
            self.assertGreaterEqual(c, 0.2)
            sentinel.stop()

    def test_vlm_result_stored_in_c_vlm(self):
        """VLM 有效结果存入 c_vlm，不影响 c_progress。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = SentinelRuntime(ema_beta=0.0, eps=0.0, log_dir=tmpdir)
            fast = sentinel._fast_action(_metrics(c_action=0.9))
            vlm_progress = ProgressMonitorResult(
                timestamp=time.time(),
                c_progress=0.1,
                alarm=False,
                stuck=False,
                progress_made=True,
                failure_likelihood=0.9,
                reason="test",
                latency_s=0.1,
            )
            result = sentinel._arbitrate(fast, vlm_progress)
            sentinel.stop()
        self.assertAlmostEqual(result.c_vlm, 0.1, places=3)
        self.assertIsInstance(result.c_progress, float)
        self.assertNotAlmostEqual(result.c_progress, 0.1, places=3)


if __name__ == "__main__":
    unittest.main()
