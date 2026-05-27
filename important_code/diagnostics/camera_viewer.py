import rerun as rr
from pathlib import Path
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.utils.visualization_utils import init_rerun

cameras = {
    "right": OpenCVCamera(OpenCVCameraConfig(index_or_path=Path("/dev/video10"), width=640, height=480, fps=30, warmup_s=3)),
    "wrist": OpenCVCamera(OpenCVCameraConfig(index_or_path=Path("/dev/video4"),  width=640, height=480, fps=30, warmup_s=3)),
}

init_rerun(session_name="camera_viewer")

for cam in cameras.values():
    cam.connect()

try:
    while True:
        for name, cam in cameras.items():
            frame = cam.read()
            rr.log(f"camera/{name}", rr.Image(frame))
except KeyboardInterrupt:
    pass
finally:
    rr.rerun_shutdown()
    for cam in cameras.values():
        cam.disconnect()
