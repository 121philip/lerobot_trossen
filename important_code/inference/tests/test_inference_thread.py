import unittest
from types import SimpleNamespace
from threading import Event
from unittest.mock import patch

import numpy as np
import torch

from important_code.inference.inference_thread import inference_thread_fn
from important_code.utils import JOINT_NAMES


class _FakeRobotWrapper:
    def get_observation(self):
        obs = {f"{name}.pos": float(i) for i, name in enumerate(JOINT_NAMES)}
        obs["wrist"] = np.zeros((4, 4, 3), dtype=np.uint8)
        obs["right"] = np.zeros((4, 4, 3), dtype=np.uint8)
        return obs


class _FakeRVizPublisher:
    def get_latest_joints(self):
        return np.arange(100.0, 107.0, dtype=np.float64)

    def put_predicted(self, chunk):
        pass

    def put_alpha(self, alpha):
        pass


class _FakePolicy:
    config = SimpleNamespace(n_action_steps=10)

    def predict_action_chunk(self, obs_processed, inference_delay, prev_chunk_left_over):
        return torch.zeros((1, 25, len(JOINT_NAMES)), dtype=torch.float32)


class _FakeActionQueue:
    def __init__(self, shutdown_event):
        self.shutdown_event = shutdown_event
        self.merged_original_shape = None
        self.merged_processed_shape = None

    def qsize(self):
        return 0

    def get_action_index(self):
        return 0

    def get_left_over(self):
        return None

    def merge(self, original_actions, postprocessed_actions, new_delay, action_index_before):
        self.merged_original_shape = tuple(original_actions.shape)
        self.merged_processed_shape = tuple(postprocessed_actions.shape)
        self.shutdown_event.set()


class InferenceThreadBridgeOverrideTest(unittest.TestCase):
    def test_non_crospi_mode_keeps_sdk_joint_observation_when_bridge_is_online(self):
        captured = {}
        shutdown_event = Event()

        def capture_preprocessor(observation):
            captured["state"] = observation["observation.state"].copy()
            return observation

        args = SimpleNamespace(
            fps=10,
            rtc=False,
            queue_threshold=30,
            task="test task",
            confidence_method="regression_cbc",
            alpha_mode="constant",
            alpha_const=0.5,
            alpha_tau_c=0.4,
            alpha_k_c=8.0,
            execution_horizon=10,
            crospi=False,
        )

        with patch(
            "important_code.inference.inference_thread.prepare_observation_for_inference",
            side_effect=lambda observation, device, task: observation,
        ):
            inference_thread_fn(
                _FakePolicy(),
                _FakeRobotWrapper(),
                capture_preprocessor,
                lambda actions: actions.squeeze(0),
                _FakeActionQueue(shutdown_event),
                shutdown_event,
                args,
                rviz_publisher=_FakeRVizPublisher(),
            )

        np.testing.assert_array_equal(
            captured["state"],
            np.arange(len(JOINT_NAMES), dtype=np.float32),
        )

    def test_non_rtc_mode_enqueues_only_configured_action_steps(self):
        shutdown_event = Event()
        action_queue = _FakeActionQueue(shutdown_event)

        args = SimpleNamespace(
            fps=10,
            rtc=False,
            queue_threshold=30,
            task="test task",
            confidence_method="regression_cbc",
            alpha_mode="constant",
            alpha_const=0.5,
            alpha_tau_c=0.4,
            alpha_k_c=8.0,
            execution_horizon=10,
            crospi=False,
        )

        with patch(
            "important_code.inference.inference_thread.prepare_observation_for_inference",
            side_effect=lambda observation, device, task: observation,
        ):
            inference_thread_fn(
                _FakePolicy(),
                _FakeRobotWrapper(),
                lambda observation: observation,
                lambda actions: actions.squeeze(0),
                action_queue,
                shutdown_event,
                args,
                rviz_publisher=_FakeRVizPublisher(),
            )

        self.assertEqual(action_queue.merged_original_shape, (10, len(JOINT_NAMES)))
        self.assertEqual(action_queue.merged_processed_shape, (10, len(JOINT_NAMES)))


if __name__ == "__main__":
    unittest.main()
