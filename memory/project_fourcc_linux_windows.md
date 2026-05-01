---
name: fourcc format naming: YUY2 vs YUYV
description: YUY2 (Windows/DirectShow) and YUYV (Linux/V4L2) are the same pixel format — using YUY2 on Linux silently fails
type: project
originSessionId: 4335d8dd-22f6-4ed6-8f30-c792fee7e645
---
YUY2 and YUYV are the same packed YUV 4:2:2 pixel format with different platform names:
- **YUY2**: Windows / DirectShow / Trossen docs name
- **YUYV**: Linux / V4L2 name

**Why:** OpenCV on Linux with V4L2 backend does not recognize "YUY2" as a valid fourcc and silently falls back to the camera's default format (success=True but actual≠requested).

**How to apply:** Always use `fourcc="YUYV"` in `OpenCVCameraConfig` on Linux, even when vendor docs say "YUY2".
