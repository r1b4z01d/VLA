"""Minimal USB/UVC camera capture for hardware teleop (scene + wrist).

`cv2.VideoCapture` by device index OR a stable device path; returns RGB HWC frames resized to the
requested size. Works for USB webcams and RealSense-as-UVC. Matches the sim's render-fn contract —
`read(w, h) -> RGB array` — so the teleop panel / recorder are unchanged between sim and hardware.

Prefer the stable `/dev/v4l/by-id/...` paths below: numeric `/dev/videoN` indices are reassigned
whenever USB devices are (un)plugged, but the by-id names are tied to the camera's USB descriptor.

Capture at 1280x720 via MJPG (default) and let `read()` downscale to the dataset size. 720p keeps
3-camera capture real-time on the robot PC (1080p is ~2x the per-frame CPU) while still exceeding the
960x540 dataset size; raise capture_w/h if you record larger than 720p.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# All three cams are now the SAME wide-FOV model with no serial, so their by-id COLLIDES — address them
# by USB-PORT path (stable per physical port; keep each camera in its port). Verify with
# `ls -l /dev/v4l/by-path/*video-index0`, and grab a frame per port to confirm which is which.
_BYPATH = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:{}:1.0-video-index0"
# Re-identified 2026-09-09 after a re-cabling moved all three off their original ports (confirmed via
# lens-cover test + content inspection over the port5/port6 pair — see chat history, not a doc).
SCENE_CAM = _BYPATH.format("5")     # 3rd-person scene (port 0:5)  — "left eye", robot's right side
WRIST_CAM = _BYPATH.format("2")     # eye-in-hand      (port 0:2)
SIDE_CAM = _BYPATH.format("6")      # 2nd 3rd-person   (port 0:6)  — "right eye", robot's left side

# Per-camera mount orientation, clockwise degrees (0/90/180/270). Adjust after mounting by watching the
# live panel — the scene/side cams currently look rotated ~90°, so these likely need 90 or 270.
SCENE_ROTATE = 270   # scene (port 0:5): came out upside-down at 90 -> flipped 180 to 270
WRIST_ROTATE = 0
SIDE_ROTATE = 90     # side (port 0:6): came out upside-down at 270 -> flipped 180 to 90

# Per-camera GAIN (V4L2). None = camera default / let auto-exposure meter — correct with adequate
# workspace lighting (a forced gain on TOP of auto-exposure overexposes). Only raise these if the
# scene is genuinely dim; the real fix for a dark view is lighting, not gain (which also adds noise).
SCENE_GAIN = None
WRIST_GAIN = None
SIDE_GAIN = None


def _fit_pad(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize `rgb` to fit inside (w, h) preserving aspect, centered on a black canvas (letterbox)."""
    ih, iw = rgb.shape[:2]
    if (iw, ih) == (w, h):
        return np.ascontiguousarray(rgb)
    scale = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
    resized = cv2.resize(rgb, (nw, nh))
    if (nw, nh) == (w, h):
        return np.ascontiguousarray(resized)
    canvas = np.zeros((h, w, 3), dtype=rgb.dtype)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return np.ascontiguousarray(canvas)


class UsbCamera:
    def __init__(self, device, capture_w: int = 1280, capture_h: int = 720, fps: int = 30,
                 rotate: int = 0, gain: "int | None" = None,
                 auto_exposure: float = 3.0, exposure: "int | None" = None):
        # device: an int index, a digit string, or a /dev path (e.g. a stable /dev/v4l/by-id/ symlink)
        self.device = device
        self.capture_w, self.capture_h, self.fps = capture_w, capture_h, fps
        self.rotate = rotate % 360  # mounting orientation, clockwise degrees: 0 / 90 / 180 / 270
        assert self.rotate in (0, 90, 180, 270), "rotate must be one of 0/90/180/270"
        self.gain = gain  # None = camera default; raise to brighten dim scenes (adds noise)
        self.auto_exposure = auto_exposure  # V4L2: 3 = auto (default), 1 = manual (use `exposure`)
        self.exposure = exposure  # 100µs units; only applied when auto_exposure == 1 (flaky on some cams)
        self._cap: "cv2.VideoCapture | None" = None

    def connect(self) -> "UsbCamera":
        dev = self.device
        if isinstance(dev, str) and dev.isdigit():
            dev = int(dev)
        if isinstance(dev, str):
            cap = cv2.VideoCapture(os.path.realpath(dev), cv2.CAP_V4L2)  # resolve by-id symlink
        else:
            cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            raise RuntimeError(f"camera {self.device!r} did not open (check the device / path)")
        # MJPG unlocks the high-res / high-fps modes (raw YUYV caps ~640x480 over USB bandwidth);
        # set it BEFORE width/height. BUFFERSIZE=1 keeps reads fresh (no stale buffered frames).
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.auto_exposure)  # 3 = auto, 1 = manual
        if self.auto_exposure == 1 and self.exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        if self.gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, self.gain)  # lift brightness in dim light (adds noise)
        for _ in range(15):  # warm up so auto-exposure + gain settle before the first real read
            cap.read()
        self._cap = cap
        return self

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self, w: int, h: int) -> np.ndarray:
        """Grab a frame -> RGB HWC uint8 fit into (h, w), aspect-PRESERVED (letterboxed, never squished).

        Rotating a 16:9 frame 90/270 makes it portrait, so a plain resize to a landscape (w,h) would
        squish it. Fit-and-pad keeps geometry correct for any rotation/aspect at the cost of black bars.
        """
        if self._cap is None:
            raise RuntimeError(f"camera {self.device!r} not connected — call connect() first")
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"camera {self.device!r} read failed")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.rotate == 90:
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == 180:
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)  # camera mounted upside down
        elif self.rotate == 270:
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return _fit_pad(rgb, w, h)

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
