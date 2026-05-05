import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _camera_paths(script_relative_path: str) -> dict[str, str]:
    source = (REPO_ROOT / script_relative_path).read_text()
    tree = ast.parse(source)
    paths: dict[str, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or key.value not in {"wrist", "right"}:
                continue
            if not isinstance(value, ast.Call):
                continue
            for keyword in value.keywords:
                if keyword.arg == "index_or_path" and isinstance(keyword.value, ast.Call):
                    path_arg = keyword.value.args[0]
                    if isinstance(path_arg, ast.Constant):
                        paths[str(key.value)] = str(path_arg.value)

    return paths


def _module_constants(script_relative_path: str) -> dict[str, int]:
    source = (REPO_ROOT / script_relative_path).read_text()
    tree = ast.parse(source)
    constants: dict[str, int] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, int):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = int(node.value.value)

    return constants


class CameraBindingsTest(unittest.TestCase):
    def test_record_episode_uses_flipped_wrist_and_right_camera_paths(self):
        self.assertEqual(
            _camera_paths("important_code/data_collection/record_episode.py"),
            {"wrist": "/dev/video4", "right": "/dev/video10"},
        )

    def test_teleoperation_uses_flipped_wrist_and_right_camera_paths(self):
        self.assertEqual(
            _camera_paths("important_code/diagnostics/teleoperation.py"),
            {"right": "/dev/video10", "wrist": "/dev/video4"},
        )

    def test_rtc_inference_defaults_match_flipped_camera_paths(self):
        constants = _module_constants("important_code/inference/run_inference_rtc.py")

        self.assertEqual(constants["DEFAULT_CAM1_ID"], 4)
        self.assertEqual(constants["DEFAULT_CAM2_ID"], 10)


if __name__ == "__main__":
    unittest.main()
