---
name: RealSense D405 Camera Configuration
description: Key findings about RealSense D405 video streams, fourcc formats, and lerobot OpenCVCameraConfig usage on Linux
type: project
originSessionId: 4335d8dd-22f6-4ed6-8f30-c792fee7e645
---
Intel RealSense D405 on Linux exposes two /dev/video* nodes per camera:
- **UYVY stream** (e.g. /dev/video2): stereo/depth imager — flat, dull image, not suitable for robot observation
- **YUYV stream** (e.g. /dev/video4): RGB sensor — bright, natural image, recommended for data collection and inference

**Why:** The Trossen docs recommend using the YUY2 (= YUYV on Linux) RGB stream. The UYVY stereo stream is primarily for depth debugging.

**How to apply:** Always configure the right camera to use the YUYV device path (e.g. /dev/video4), not the UYVY one (e.g. /dev/video2). Current assignment:
- wrist: /dev/video10 (YUYV, regular USB webcam or RealSense RGB)
- right: /dev/video4 (YUYV, RealSense D405 RGB sensor)

## Critical: Use Path object, not integer index

```python
from pathlib import Path
OpenCVCameraConfig(index_or_path=Path("/dev/video4"), width=640, height=480, fps=30, fourcc="YUYV")
```

Using integer `4` maps unreliably; `Path("/dev/video4")` is explicit and matches Python test behavior.

## lerobot-find-cameras occupancy bug

When `lerobot-find-cameras opencv` runs, it opens /dev/video2 first, which blocks /dev/video4 (same physical RealSense). video4 appears to fail — this is a false negative. Test video4 in isolation to confirm it works.

## Unsupported camera controls

For these cameras, `cap.get()` returns -1.0 for: AUTOFOCUS, FOCUS, AUTO_EXPOSURE, EXPOSURE. Only BRIGHTNESS and GAIN are adjustable via OpenCV. Focus is fixed; blur issues must be solved physically (lighting, lens cleaning).

---

## 2026-05-01: Two-camera OpenCV runtime diagnosis

Problem observed during `important_code/diagnostics/teleoperation.py`:

- `lerobot-find-cameras opencv` listed `/dev/video10`, `/dev/video2`, `/dev/video4`, `/dev/video8`.
- It then logged failures for `/dev/video4` and `/dev/video8`.
- During teleoperation, `/dev/video4` did not fail at `VideoCapture.open`; instead LeRobot timed out while waiting for the first frame:
  `Timed out waiting for frame from camera OpenCVCamera(/dev/video4) after 1000 ms. Read thread alive: True.`

Analysis:

- Each RealSense D405 exposes multiple V4L2 nodes. In this setup:
  - serial `235123074636`: `/dev/video2` and `/dev/video4`
  - serial `235123074875`: `/dev/video8` and `/dev/video10`
- `lerobot-find-cameras opencv` is misleading because it tries to open every detected node. Opening one node on a physical RealSense can block another node on the same camera, so some reported failures are false negatives.
- `/dev/video2` and `/dev/video8` are UYVY/stereo/depth-style streams. They can sometimes be opened by OpenCV, but they are not the desired RGB observation streams.
- `/dev/video4` and `/dev/video10` are the intended YUYV RGB streams for robot observation. They were verified together by the user and by an OpenCV/LeRobot read test.
- LeRobot's default `warmup_s=1` can be too short for these cameras. Use `warmup_s=3` in teleoperation to avoid first-frame timeout.

Solution applied:

- Use `/dev/video4` for the right camera.
- Use `/dev/video10` for the wrist camera.
- Prefer explicit `Path("/dev/videoN")` in Python configs.
- Keep CLI integer camera IDs aligned as `right=4`, `wrist=10`.
- Add `warmup_s=3` for teleoperation camera configs.

Relevant files:

- `important_code/diagnostics/teleoperation.py`
- `important_code/data_collection/record_episode.py`
- `important_code/inference/run_inference_rtc.py`

Current intended mapping:

```python
"right": OpenCVCameraConfig(index_or_path=Path("/dev/video4"), width=640, height=480, fps=30, warmup_s=3)
"wrist": OpenCVCameraConfig(index_or_path=Path("/dev/video10"), width=640, height=480, fps=30, warmup_s=3)
```

Verification command used:

```python
from pathlib import Path
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

configs = {
    "right": OpenCVCameraConfig(index_or_path=Path("/dev/video4"), width=640, height=480, fps=30, warmup_s=3),
    "wrist": OpenCVCameraConfig(index_or_path=Path("/dev/video10"), width=640, height=480, fps=30, warmup_s=3),
}
```

Observed verification result:

```text
connecting right /dev/video4
connected right
connecting wrist /dev/video10
connected wrist
read right (480, 640, 3)
read wrist (480, 640, 3)
```
