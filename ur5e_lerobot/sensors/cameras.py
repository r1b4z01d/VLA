"""Minimal USB/UVC camera capture for hardware teleop (scene + wrist).

`cv2.VideoCapture` by device index OR a stable device path; returns RGB HWC frames resized to the
requested size. Works for USB webcams and RealSense-as-UVC. Matches the sim's render-fn contract —
`read(w, h) -> RGB array` — so the teleop panel / recorder are unchanged between sim and hardware.

Prefer the stable `/dev/v4l/by-id/...` paths below: numeric `/dev/videoN` indices are reassigned
whenever USB devices are (un)plugged, but the by-id names are tied to the camera's USB descriptor.

Capture at a higher native resolution (e.g. 640x480) and let `read()` downscale to the dataset
size; you can't recover detail you didn't capture.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# Stable device paths (by USB descriptor) — survive /dev/videoN re-enumeration across replug/reboot.
SCENE_CAM = "/dev/v4l/by-id/usb-4K_USB_CAMERA_HD_USB_CAMERA_01.00.00-video-index0"    # 4K wide (3rd-person)
WRIST_CAM = "/dev/v4l/by-id/usb-HD_Camera_Manufacturer_USB_2.0_Camera-video-index0"   # ARC (eye-in-hand)


class UsbCamera:
    def __init__(self, device, capture_w: int = 640, capture_h: int = 480, fps: int = 30,
                 rotate180: bool = False):
        # device: an int index, a digit string, or a /dev path (e.g. a stable /dev/v4l/by-id/ symlink)
        self.device = device
        self.capture_w, self.capture_h, self.fps = capture_w, capture_h, fps
        self.rotate180 = rotate180  # True if the camera is mounted upside down
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        for _ in range(5):  # warm up so auto-exposure settles before the first real read
            cap.read()
        self._cap = cap
        return self

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self, w: int, h: int) -> np.ndarray:
        """Grab a frame -> RGB HWC uint8 resized to (h, w)."""
        if self._cap is None:
            raise RuntimeError(f"camera {self.device!r} not connected — call connect() first")
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"camera {self.device!r} read failed")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.rotate180:
            rgb = cv2.flip(rgb, -1)  # 180° — camera mounted upside down
        if (rgb.shape[1], rgb.shape[0]) != (w, h):
            rgb = cv2.resize(rgb, (w, h))
        return np.ascontiguousarray(rgb)

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
