"""Minimal USB/UVC camera capture for hardware teleop: three identical wide-FOV webcams, scene / side /
wrist (eye-in-hand). In normal operation the vla_cameras service (camera_stream.py) owns them and every
consumer subscribes over UDP (`--cameras udp`); opening them here directly is the `--cameras local` path.

`cv2.VideoCapture` by device index OR a stable device path; returns RGB HWC frames resized to the
requested size. Works for USB webcams and RealSense-as-UVC. Matches the sim's render-fn contract —
`read(w, h) -> RGB array` — so the teleop panel / recorder are unchanged between sim and hardware.

Cameras are addressed by stable `/dev/v4l/by-path/...` USB-PORT paths below: numeric `/dev/videoN`
indices are reassigned whenever USB devices are (un)plugged, and by-id names collide because all three
cameras are the same model with no serial — so each camera must stay in its port.

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
import platform as _platform

if _platform.machine() == "aarch64":
    # Jetson AGX Orin robot PC (since 2026-10-10): the cams hang off the Tegra xHCI at 3610000.usb.
    # Identified 2026-10-11 from lit frames + operator confirmation: 4.3 shows the AmazingHand's
    # white/pink fingers at the top of the frame (eye-in-hand), 4.2 has the arm links in the
    # foreground (scene), 4.1 looks across the workspace from the body with the banana in front (side).
    # Bandwidth: a hub port on this box carries at most TWO of these cameras (each reserves its
    # maximum USB 2.0 alt-setting whatever the mode), and the dock's sub-hubs carried none reliably.
    # Keep the cameras on different hubs, never behind the dock's own sub-hubs.
    _BYPATH = "/dev/v4l/by-path/platform-3610000.usb-usb-0:{}:1.0-video-index0"
    # Roles checked against tigger_1 training frames (2026-10-11): the straight-on bookshelf view
    # (cabinet at the bottom, arm entering top-right) is SCENE = port 2.4; the close Tigger-shelf view
    # with the window at the right is SIDE = port 4.2. With the roles right, the Intel-era rotations
    # (SCENE 270 / SIDE 90) give upright feeds; swapped roles made both look upside down.
    # 2026-10-30, final layout — one webcam per USB 2 bus, because all four USB-A jacks of the devkit
    # hang off ONE onboard hub (root port 4) and a second of these cameras on the same bus fails
    # stream-on with ENOSPC at 1280x720 (the uvcvideo FIX_BANDWIDTH quirk only helps at 640x480):
    #   SCENE 1.1  = the VIA dock on a USB-C port (the RealSense rides the dock's USB 3 port at 5 Gbps)
    #   WRIST 2    = alone on the other USB-C port (A-to-C adapter)
    #   SIDE  4.2  = a USB-A jack (shared bus with the lidar adapter, Stream Deck, receivers — fine)
    # Roles from frames: bookshelf straight-on = scene, AmazingHand fingers = wrist, arm-in-foreground
    # shelf = side. Photos + kernel map: rd_ws/provision/jetson/usb-ports-*.png, usb-layout.png.
    # 2026-11-02 re-cabling (kept): the two USB-C cameras swapped places and the one on USB-C "L" now
    # sits behind a 4-port hub (port 2.4) — one webcam per bus still holds, so that is fine:
    #   WRIST 1.1  = the VIA dock on a USB-C port
    #   SIDE  2.4  = hub on the other USB-C port, hub port 4
    #   SCENE 4.2  = a USB-A jack
    # (scene and side were first mapped the other way round; that put each camera under the other's
    # rotation and both came out upside down — the 270/90 rotations below are right for these ports.)
    _PORTS = {"scene": "4.2", "wrist": "1.1", "side": "2.4"}
else:
    # Intel robot PC (retired 2026-10). Re-identified 2026-09-09 after a re-cabling moved all three
    # off their original ports (confirmed via lens-cover test + content inspection over the
    # port5/port6 pair — see chat history, not a doc).
    _BYPATH = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:{}:1.0-video-index0"
    _PORTS = {"scene": "5", "wrist": "2", "side": "6"}

SCENE_CAM = _BYPATH.format(_PORTS["scene"])   # 3rd-person scene — "left eye", robot's right side
WRIST_CAM = _BYPATH.format(_PORTS["wrist"])   # eye-in-hand
SIDE_CAM = _BYPATH.format(_PORTS["side"])     # 2nd 3rd-person  — "right eye", robot's left side

# Per-camera mount orientation, clockwise degrees (0/90/180/270). Adjust after mounting by watching the
# live panel — the scene/side cams currently look rotated ~90°, so these likely need 90 or 270.
SCENE_ROTATE = 270   # scene mount: came out upside-down at 90 -> flipped 180 to 270
WRIST_ROTATE = 0     # matches training (hand fingers at the top of the frame)
SIDE_ROTATE = 90     # side mount: came out upside-down at 270 -> flipped 180 to 90

# Per-camera GAIN (V4L2). None = camera default / let auto-exposure meter — correct with adequate
# workspace lighting (a forced gain on TOP of auto-exposure overexposes). Only raise these if the
# scene is genuinely dim; the real fix for a dark view is lighting, not gain (which also adds noise).
SCENE_GAIN = None
WRIST_GAIN = None
SIDE_GAIN = None

# Per-camera EXPOSURE. AUTO_EXPOSURE is the V4L2 mode: 3 = auto (driver meters), 1 = manual. EXPOSURE
# is only applied in manual mode, in 100us units (so 60 ~= 6 ms).
#
# Auto meters over the WHOLE sensor frame, which here is dominated by white cabinets, shelving and
# floor — and the square crop throws most of that away. Measured on captured frames: 4-6% of content
# pixels sit at >=250 with p99 = 255, i.e. the highlights are clipped and that detail is gone before
# the policy ever sees it. Locking manual exposure lets you expose for the workspace instead of the
# room. UVC support for these controls is inconsistent, so use `scripts/tune_exposure.py`, which
# sweeps values, reports clipped-%, and verifies the camera actually honours the setting.
SCENE_AUTO_EXPOSURE, SCENE_EXPOSURE = 3.0, None
WRIST_AUTO_EXPOSURE, WRIST_EXPOSURE = 3.0, None
SIDE_AUTO_EXPOSURE, SIDE_EXPOSURE = 3.0, None

# Per-camera SQUARE CROP of the UPRIGHT (post-rotation) frame. None = off (letterbox as before).
# Otherwise a fraction saying where the square sits along the camera's LONG axis: 0.0 = top/left,
# 0.5 = centre, 1.0 = bottom/right. For the sideways scene/side cams the upright frame is 720x1280
# portrait, so the square is 720 tall sliding over 560 px: rows round(560*at) .. +720 (0.40 -> 224..944,
# 0.65 -> 364..1084, 1.0 -> the bottom 720 rows = the floor).
#
# TUNING: the camera service applies the crop, so nothing downstream ever sees the pixels outside it.
# Tune through the service, live, without restarting it:
#     run.sh scripts/camera_viewer.py --host 127.0.0.1 --squares            # current values
#     run.sh scripts/camera_viewer.py --host 127.0.0.1 --square scene 0.65  # move it, takes effect now
#     http://<rd>:5602/scene.raw.jpg      # the FULL upright frame with the current square drawn on it
#     http://<rd>:5602/scene.mjpg         # what the policy will actually see
# Then write the chosen value HERE so it survives a service restart — the service reads these at start.
#
# Why: the scene/side cams are mounted sideways, so after ROTATE they are 720x1280 PORTRAIT, and
# fitting that into the landscape dataset frame leaves ~68% of every stored frame black. SmolVLA then
# pads again into 512x512, so only ~18% of what the policy sees is real pixels. Cropping the upright
# frame to a square and storing it square removes BOTH paddings — ~100% of the policy's input becomes
# workspace — and spends the resolution on the reachable area instead of bookshelf and floor.
#
# CAUTION: this DISCARDS field of view. The square must still contain the object at every position you
# record; your placements span ~913mm. Verify in the panel before recording, and note that changing it
# changes the observation geometry, so it needs a fresh dataset + retrain (an existing checkpoint
# expects the geometry it was trained on).
SCENE_SQUARE = 0.40  # scene: keeps the shelf Tigger sits on plus the arm's working volume
WRIST_SQUARE = None  # wrist is already landscape and unpadded — nothing to reclaim
SIDE_SQUARE = 0.30   # side: the robot base dominates lower down; 0.30 keeps Tigger and the upper arm


def square_crop(rgb: np.ndarray, at: "float | None") -> np.ndarray:
    """Crop `rgb` to a centred square, positioned by `at` along the long axis. `at=None` -> unchanged."""
    if at is None:
        return rgb
    ih, iw = rgb.shape[:2]
    side = min(iw, ih)
    at = min(max(float(at), 0.0), 1.0)
    if ih >= iw:                                  # portrait: full width, choose the vertical band
        y0 = int(round((ih - side) * at))
        return np.ascontiguousarray(rgb[y0:y0 + side, :])
    x0 = int(round((iw - side) * at))             # landscape: full height, choose the horizontal band
    return np.ascontiguousarray(rgb[:, x0:x0 + side])


def out_size(rotate: int, square: "float | None", w: int, h: int,
             cap_w: int = 1280, cap_h: int = 720) -> "tuple[int, int]":
    """The (w, h) a camera with this rotation/crop actually returns for a requested (w, h) box.

    Callers need it BEFORE the first frame, to declare the dataset's per-camera image shape
    (`sim/record_sim.build_features`) and to size deploy-side buffers. Shapes now differ per camera:
    a square-cropped cam returns a square, a rotated one a portrait, an unrotated one the full box.
    """
    if square is None:
        return w, h        # legacy path: read() fit-and-PADS onto the full (w, h) canvas
    iw = ih = min(cap_w, cap_h)   # square crop of the upright frame, then fit WITHOUT padding
    scale = min(w / iw, h / ih)
    return max(1, round(iw * scale)), max(1, round(ih * scale))


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
                 auto_exposure: float = 3.0, exposure: "int | None" = None,
                 square: "float | None" = None):
        # device: an int index, a digit string, or a /dev path (e.g. a stable /dev/v4l/by-id/ symlink)
        self.device = device
        self.capture_w, self.capture_h, self.fps = capture_w, capture_h, fps
        self.rotate = rotate % 360  # mounting orientation, clockwise degrees: 0 / 90 / 180 / 270
        self.square = square  # None = letterbox as before; else crop upright frame to a square (see SCENE_SQUARE)
        assert self.rotate in (0, 90, 180, 270), "rotate must be one of 0/90/180/270"
        self.gain = gain  # None = camera default; raise to brighten dim scenes (adds noise)
        self.auto_exposure = auto_exposure  # V4L2: 3 = auto (default), 1 = manual (use `exposure`)
        self.exposure = exposure  # 100µs units; only applied when auto_exposure == 1 (flaky on some cams)
        self._cap: "cv2.VideoCapture | None" = None
        self.last_full: "np.ndarray | None" = None  # upright, uncropped frame from the last read()

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
        """Grab a frame -> RGB HWC uint8 fit inside (h, w), aspect-PRESERVED (never squished).

        Rotating a 16:9 frame 90/270 makes it portrait, so a plain resize to a landscape (w,h) would
        squish it. With `square=None` we fit-and-pad, which keeps geometry correct at the cost of black
        bars (the legacy path — existing checkpoints were trained on it). With a square crop set we fit
        WITHOUT padding, so the frame is all real pixels; `out_size()` predicts the returned shape.
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
        # keep the UPRIGHT, UNCROPPED frame: the camera service's /<cam>.raw.jpg shows it with the
        # current square drawn on, which is how SCENE_SQUARE/SIDE_SQUARE are chosen (nothing else
        # downstream ever sees pixels outside the square)
        self.last_full = rgb
        if self.square is None:
            return _fit_pad(rgb, w, h)
        rgb = square_crop(rgb, self.square)
        ih, iw = rgb.shape[:2]
        scale = min(w / iw, h / ih)
        return np.ascontiguousarray(
            cv2.resize(rgb, (max(1, round(iw * scale)), max(1, round(ih * scale)))))

    # --- runtime exposure (the launch-time "auto-expose then trim then lock" flow) --------------
    def read_exposure(self) -> "float | None":
        """What the driver currently reports for exposure. None if it won't say."""
        if self._cap is None:
            return None
        v = self._cap.get(cv2.CAP_PROP_EXPOSURE)
        return float(v) if v and v > 0 else None

    def set_exposure(self, value: float) -> float:
        """Switch to MANUAL and set exposure (100us units). Returns what the driver reports back.

        Read-back matters: UVC drivers silently clamp to their own supported range, so the value you
        asked for and the value in effect are often different.
        """
        if self._cap is None:
            raise RuntimeError(f"camera {self.device!r} not connected")
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)   # 1 = manual
        self._cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
        self.auto_exposure, self.exposure = 1.0, float(value)
        return self.read_exposure() or float(value)

    def set_auto_exposure(self) -> None:
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)  # 3 = auto
        self.auto_exposure, self.exposure = 3.0, None

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ExposureSession:
    """Launch-time exposure flow shared by the teleop panel and eval: auto -> trim -> lock.

    The cameras start on AUTO so the driver picks a sane working point, the operator nudges all of
    them together with two deck keys while watching the live feeds, and a third key LOCKS the current
    values to manual. Locking matters beyond the brightness itself: auto-exposure re-meters as the arm
    sweeps through frame, so image brightness becomes a function of arm position — one more spurious
    correlate in a dataset where the policy has already been shown to prefer spurious correlates over
    looking at the object. A locked exposure depends on the scene alone.

    Steps are MULTIPLICATIVE because exposure is perceptually logarithmic; +/- of a fixed number of
    microseconds is a huge change when dark and imperceptible when bright.

    All cameras move together by the same factor, each from its own auto-converged starting point, so
    their relative differences (the wrist looks at a lit close-up, the scene at a room) are preserved.
    """

    STEP = 1.25           # one press = +/-25%
    FALLBACK = 78.0       # 100us units, used if the driver won't report its auto-converged value

    def __init__(self, cams: "dict"):
        self.cams = {k: c for k, c in cams.items() if c is not None}
        self.locked = False
        self.values: dict = {}

    @classmethod
    def for_cams(cls, cams: "dict"):
        """The right session for whatever `make_engine` handed back.

        Local UsbCameras -> a normal ExposureSession that drives them directly. Cameras served by the
        camera service (`make_engine(cameras="udp")`) carry a `remote_session`: the same flow, but
        every step runs INSIDE the service, against the real devices, and the locked value outlives
        this process — the next launch starts from it instead of re-metering. Running the search from
        here through the proxies would also work, but each settle read would then wait a full
        publish period on the wire instead of a grab, and the result would die with this process.
        """
        live = {k: c for k, c in cams.items() if c is not None}
        remote = {k: getattr(c, "remote_session", None) for k, c in live.items()}
        if live and all(remote.values()):
            return next(iter(remote.values()))
        return cls(live)

    def begin(self) -> None:
        """Put every camera on auto so the driver converges before the operator trims."""
        for c in self.cams.values():
            try:
                c.set_auto_exposure()
            except Exception:  # noqa: BLE001 — a camera that refuses is handled at lock time
                pass
        self.locked = False

    # Coarse ladder searched at launch, brightest first. 100us units.
    LADDER = (625.0, 312.0, 156.0, 78.0, 39.0, 20.0, 10.0, 5.0)
    TARGET_MEDIAN = 120.0  # aim the BULK of the frame here (0-255)
    TARGET_CLIP = 1.0      # advisory only — reported, not optimised against (see auto_expose)

    def adopt_auto(self) -> dict:
        """Snapshot what auto settled on — the starting point for +/-."""
        for name, c in self.cams.items():
            try:
                self.values[name] = c.read_exposure() or self.FALLBACK
            except Exception:  # noqa: BLE001
                self.values[name] = self.FALLBACK
        return dict(self.values)

    @staticmethod
    def clip_pct(img) -> float:
        """% of CONTENT pixels at >=250 — blown highlights carry no information at all."""
        nz = img.max(axis=2) > 12
        px = img[nz] if nz.any() else img.reshape(-1, 3)
        return 100.0 * float((px.max(axis=-1) >= 250).mean())

    def auto_expose(self, w: int, h: int, settle: int = 2) -> dict:
        """Expose the BULK of each frame near TARGET_MEDIAN, per camera.

        Time budget: ~1-1.5 s, down from ~7 s. The search is LADDER (8) x (settle + 1) reads per
        camera; it used to run the cameras one after another with settle=6, i.e. 168 reads at
        ~45 ms. Two measured facts bring it down without weakening the search: the sensor reflects
        a new exposure from the 2nd frame after the change (frame 0 = old value, frame 1 =
        transitional, frame 2+ = settled — so settle=2 reads the first trustworthy frame), and the
        three cameras are independent devices, so they are metered in parallel threads and the
        wall time is one camera's search (24 reads), not three.

        Two things this must survive, both measured on this rig:

        The driver's own auto is not usable — it settles around 625 and clips 81% of the scene view,
        turning the white cabinets the workspace sits against into a featureless slab. Stepping down
        from there by hand would take ~15 presses, hence a coarse search with +/- left as fine trim.

        And the obvious criterion, "brightest setting that doesn't clip", is WRONG here: the wrist
        camera has a daylight window in one corner, which clips at every exposure that leaves the
        workspace visible. Optimising against clipping drove the wrist to the floor of the ladder and
        left the whole working area at mean 25 — technically unclipped, practically black. Matching a
        MEDIAN ignores a small blown region and exposes for what actually fills the frame.
        """
        import threading

        def measure(c, v):
            c.set_exposure(v)
            for _ in range(settle):
                c.read(w, h)
            img = c.read(w, h)
            nz = img.max(axis=2) > 12
            px = img[nz] if nz.any() else img.reshape(-1, 3)
            return float(np.median(px))

        def search(name, c):
            best, best_err, best_med = self.LADDER[-1], None, None
            for v in self.LADDER:
                try:
                    med = measure(c, v)
                    err = abs(med - self.TARGET_MEDIAN)
                    if best_err is None or err < best_err:
                        best, best_err, best_med = v, err, med
                except Exception:  # noqa: BLE001
                    break
            # Refine: the ladder is x2 coarse, so two cameras can land a stop apart in brightness
            # (2026-10-26: scene median 176 vs side 45 after a global trim). Walk +/-25% toward the
            # target while it keeps improving — at most 3 extra measurements per camera.
            if best_err is not None:
                for _ in range(3):
                    if best_med is None or abs(best_med - self.TARGET_MEDIAN) < 8:
                        break
                    cand = max(1.0, best * (self.STEP if best_med < self.TARGET_MEDIAN else 1.0 / self.STEP))
                    try:
                        med = measure(c, cand)
                    except Exception:  # noqa: BLE001
                        break
                    if abs(med - self.TARGET_MEDIAN) < best_err:
                        best, best_err, best_med = cand, abs(med - self.TARGET_MEDIAN), med
                    else:
                        break
            try:
                self.values[name] = c.set_exposure(best)
            except Exception:  # noqa: BLE001
                self.values[name] = best

        # one thread per camera: each UsbCamera (or LockedCamera in the service) is its own device
        # and its own lock, so the searches don't contend; dict writes to self.values are per-key
        threads = [threading.Thread(target=search, args=(n, c), daemon=True, name=f"expose-{n}")
                   for n, c in self.cams.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return dict(self.values)

    def nudge(self, factor: float, cam: "str | None" = None) -> dict:
        """Scale every camera's exposure (or just `cam`'s). Applies immediately so the live feed
        shows the result."""
        if not self.values:
            self.adopt_auto()
        for name, c in self.cams.items():
            if cam is not None and name != cam:
                continue
            want = max(1.0, self.values.get(name, self.FALLBACK) * factor)
            try:
                self.values[name] = c.set_exposure(want)   # keep the DRIVER's value, not ours
            except Exception:  # noqa: BLE001
                self.values[name] = want
        return dict(self.values)

    def lock(self) -> dict:
        """Freeze the current values in manual mode. Returns {camera: exposure} for the annotation."""
        if not self.values:
            self.adopt_auto()
        for name, c in self.cams.items():
            try:
                self.values[name] = c.set_exposure(self.values[name])
            except Exception:  # noqa: BLE001
                pass
        self.locked = True
        return dict(self.values)

    def summary(self) -> str:
        return "  ".join(f"{k}={v:.0f}" for k, v in sorted(self.values.items())) or "(none)"
