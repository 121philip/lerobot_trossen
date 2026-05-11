import numpy as np

from important_code.inference import rviz_publisher


def test_actual_joint_payload_preserves_gripper_as_seventh_joint():
    joints = rviz_publisher._normalize_actual_joints(
        [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.037]]
    )

    assert joints.shape == (7,)
    assert joints[-1] == 0.037


def test_actual_joint_payload_rejects_arm_only_six_joint_commands():
    try:
        rviz_publisher._normalize_actual_joints(np.arange(6))
    except ValueError as exc:
        assert "Expected 7 joints" in str(exc)
    else:
        raise AssertionError("six-joint commands must not be sent to CroSPI")


def test_predicted_chunk_requires_gripper_column():
    try:
        rviz_publisher._normalize_predicted_chunk(np.zeros((4, 6)))
    except ValueError as exc:
        assert "Expected predicted chunk shape (N, 7)" in str(exc)
    else:
        raise AssertionError("predicted chunks without gripper must not be sent")


def test_sentinel_weights_are_flat_non_negative_pair():
    weights = rviz_publisher._normalize_weights(-0.2, 0.9)

    assert weights.shape == (2,)
    np.testing.assert_array_equal(weights, np.array([0.0, 0.9]))
