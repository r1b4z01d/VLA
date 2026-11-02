"""The teleop / recording panel — the app that collects demonstrations, in sim or on the hardware.

Action source (`--input`): spacemouse (default) | gamepad (Xbox) | freedrive (hand-guide the real arm)
| sliders (on-screen, no device). Engine (`--engine`): mujoco (default, a rendered UR5e + AmazingHand)
| kinematic | hardware (the real UR5e over RTDE + the hand over UDP + the cameras). Episodes are
recorded into a LeRobotDataset with a per-frame task string (what the language-conditioned VLA trains on).

UI (`--ui`): web (default) serves a browser page on http://<this host>:8090 with the Tk window hidden;
tk is the classic Tk window; both. The Tk window keeps a few controls of its own (Set Home, the
randomize buttons), so use --ui tk|both for those. The Stream Deck, the 1-5 star rating after each save
and the background upload to the GPU box all work the same whichever UI is shown.

    .venv/bin/python -m ur5e_lerobot.teleop.manual_panel --input sliders                 # sim, in the browser
    .venv/bin/python -m ur5e_lerobot.teleop.manual_panel --input sliders --ui tk         # sim, Tk window
    ~/rd_ws/VLA/run.sh -m ur5e_lerobot.teleop.manual_panel --engine hardware --cameras udp   # robot (or the vla_teleop unit)
    .venv/bin/python -m ur5e_lerobot.teleop.manual_panel --selftest                      # headless check
"""
from __future__ import annotations

import argparse
import math
import os
import threading as _threading_mod
import time as _ptime

import numpy as np
from scipy.spatial.transform import Rotation

from ..robot.workspace import load_workspace_calib
from ..schema import ACTION_NAMES, ARM_POSE_NAMES, STATE_NAMES, Action
from ..sim.record_sim import make_engine

SLIDER_ORDER = ("x", "y", "z", "roll", "pitch", "yaw")
# Teleop target is clamped to home ± delta, anchored at the arm's pose WHEN THE PANEL STARTED (not the
# live pose). Keep this box INSIDE the arm's well-conditioned workspace: driving the target near full
# extension / a singularity makes servoL oscillate -> the arm jerks. Start the panel with the EE
# centered over the workspace so a modest box reaches everything. Bump one axis at a time + test.
# Per-axis overrides from outputs/workspace_calib.json["pos_deltas"] when present (calibrate_workspace.py).
POS_DELTAS = {**{"x": 0.40, "y": 0.40, "z": 0.45}, **(load_workspace_calib().get("pos_deltas") or {})}

# SpaceMouse teleop gains (per second at full deflection); tune to taste.
SM_POS_GAIN = 0.131  # m/s  (was 0.175; -25%)
SM_ROT_GAIN = 0.45   # rad/s (was 0.6;   -25%)
SM_GRASP_CLOSE_RATE = 0.104  # grasp close per tick while the close button is held
SM_GRASP_OPEN_RATE = 0.060   # grasp open per tick while the open button is held
SM_DEADBAND = 0.12    # ignore tiny center deflections (helps isolate a single axis)
SM_BTN_CLOSE = 0      # SpaceMouse button index that slowly CLOSES the grasp
SM_BTN_OPEN = 1       # SpaceMouse button index that slowly OPENS the grasp (swap if reversed)
# Map each robot control axis to a (spacemouse axis, sign). Edit here to fix any
# inverted/swapped axis. NOTE: "correct" X/Y directions depend on where the operator stands relative
# to the robot — calibrate from one fixed viewpoint. On this device (256f:c652): x inverted;
# y direct; z inverted; roll direct; pitch inverted; yaw direct.
SM_MAP = {
    "x": ("x", -1),
    "y": ("y", +1),
    "z": ("z", -1),
    "roll": ("yaw", -1),
    "pitch": ("roll", -1),
    "yaw": ("pitch", -1),
}

# Xbox gamepad mapping: robot axis -> (gamepad source, sign). Sources: lx/ly (left stick),
# rx/ry (right stick), "triggers" (rt - lt). Defaults follow the requested layout (left stick -> x,y ;
# right stick -> roll,pitch ; triggers -> z, RT up/LT down). gp_calibrate.py overwrites this from your
# own gestures (writes outputs/gp_calib.json). Bumpers LB/RB drive grasp (fixed). evdev stick-Y is
# up-negative, hence y/pitch default to -1.
GP_MAP = {
    "x": ("lx", +1),
    "y": ("ly", -1),
    "z": ("triggers", +1),
    "roll": ("rx", +1),
    "pitch": ("ry", -1),
}

# Per grasp mode: the MAX curl each finger reaches at grasp=1, ordered (index, middle, ring,
# thumb). The grasp DOF (0..1) scales ALL of them, so every finger opens/closes with the grasp.
# In "pinch" the thumb + pointer are capped at 0.8 because that's where their tips MEET (past
# that they cross); the middle+ring close fully. "full" closes all four fully.
GRASP_MODES = {
    "pinch": (0.8, 1.0, 1.0, 0.8),
    "full": (1.0, 1.0, 1.0, 1.0),
}

# Stream Deck bottom row (the finger-CLOSE keys, logical 10-14) is temporarily repurposed as a 1-5
# star episode rating after Save. Left->right = 1..5 stars.
RATING_KEYS = (10, 11, 12, 13, 14)

# Box-color choices (RGBA, 0..1) for the panel dropdown — ROYGBIV, in order.
ROYGBIV = [
    ("Red", (0.85, 0.10, 0.10, 1.0)),
    ("Orange", (0.95, 0.50, 0.05, 1.0)),
    ("Yellow", (0.93, 0.86, 0.10, 1.0)),
    ("Green", (0.15, 0.75, 0.20, 1.0)),
    ("Blue", (0.12, 0.35, 0.90, 1.0)),
    ("Indigo", (0.29, 0.00, 0.51, 1.0)),
    ("Violet", (0.56, 0.10, 0.90, 1.0)),
]


class ManualController:
    """Target pose around the arm's home pose -> 10-D action.

    Position is absolute (x,y,z). Orientation is roll/pitch/yaw applied intrinsically on
    top of the home wrist orientation, each full range [-pi, pi], then converted to the
    axis-angle rotation vector the action schema uses.
    """

    def __init__(self, home: list[float], grasp_mode: str = "pinch", home_grasp: float = 0.0):
        self.home = [float(v) for v in home]  # [x,y,z, rx,ry,rz] (rx,ry,rz = home rotvec)
        self._home_rot = Rotation.from_rotvec(self.home[3:6])
        self._max_curls = GRASP_MODES[grasp_mode]  # per-finger max curl at grasp=1
        self._home_grasp = float(home_grasp)
        self.reset()

    def reset(self) -> None:
        self.x, self.y, self.z = self.home[:3]
        self.roll = self.pitch = self.yaw = 0.0  # offsets from the home orientation
        self.curls = [self._home_grasp] * 4  # per-finger grasp [0,1] (index, middle, ring, thumb)

    @property
    def grasp(self) -> float:
        """Mean per-finger grasp (read). Assigning sets ALL four fingers — the all-finger open/close
        used by the slider / SpaceMouse buttons / keyboard. Per-finger control uses nudge_finger()."""
        return sum(self.curls) / len(self.curls)

    @grasp.setter
    def grasp(self, v: float) -> None:
        v = 0.0 if v < 0 else 1.0 if v > 1 else float(v)
        self.curls = [v] * 4

    def nudge_finger(self, i: int, delta: float) -> None:
        c = self.curls[i] + delta
        self.curls[i] = 0.0 if c < 0 else 1.0 if c > 1 else c

    def ranges(self) -> dict[str, tuple[float, float]]:
        r = {k: (self.home[i] - POS_DELTAS[k], self.home[i] + POS_DELTAS[k])
             for i, k in enumerate(("x", "y", "z"))}
        for k in ("roll", "pitch", "yaw"):
            r[k] = (-math.pi, math.pi)  # full range
        return r

    def grasp_curls(self) -> tuple:
        """Per-finger curls (each finger's [0,1] scaled by its grasp-mode max)."""
        return tuple(c * mc for c, mc in zip(self.curls, self._max_curls))

    def action_vector(self) -> list[float]:
        delta = Rotation.from_euler("xyz", [self.roll, self.pitch, self.yaw])
        rotvec = (self._home_rot * delta).as_rotvec()
        return Action((self.x, self.y, self.z, *rotvec), self.grasp_curls()).to_vector()


def _unique_root(base: str) -> str:
    if not os.path.exists(base):
        return base
    i = 1
    while os.path.exists(f"{base}_{i}"):
        i += 1
    return f"{base}_{i}"


class SimSession:
    """Owns the engine (robot + render) and the dataset; turns actions into frames."""

    def __init__(self, engine: str, repo_id: str, root: str, fps=20, width=320, height=240,
                 use_videos=False, resume=False, hw=None, tool_voltage: int = 12):
        hw = dict(hw or {})
        squares_req = {k: v for k, v in (hw.pop("squares", None) or {}).items() if v is not None}
        self.robot, self._render_fn, self._wrist_fn, self._side_fn = make_engine(engine, **hw)
        self.has_wrist = self._wrist_fn is not None
        self.has_side = self._side_fn is not None
        # Launch-time exposure flow (hardware only): auto -> operator trims -> lock. `_grab` hangs the
        # UsbCamera off the render fn so we can drive it; sim renderers have no `.cam`.
        self.exposure = None
        cams = {n: getattr(f, "cam", None) for n, f in
                (("scene", self._render_fn), ("wrist", self._wrist_fn), ("side", self._side_fn)) if f}
        if any(cams.values()):
            from ..sensors.cameras import ExposureSession
            # for_cams: local UsbCameras get the direct session; cameras served by the camera service
            # (--cameras udp) get a RemoteExposureSession with the same interface, run server-side
            self.exposure = ExposureSession.for_cams(cams)
        # Live camera objects (UsbCamera or CameraProxy), kept for the square-crop flow: a NEW dataset
        # takes --scene-square/--side-square (applied to the cameras now, written to meta/capture.json
        # on create); a RESUMED dataset re-applies its own capture.json in _ensure_dataset.
        self.cams = {k: c for k, c in cams.items() if c is not None}
        if engine == "hardware" and not self.has_side:
            print("[cam] WARNING: no side camera in this session — the dataset will have scene+wrist only, "
                  "which the 3-camera eval cannot drive. Pass --side-cam (local) or check the camera service "
                  "(udp).", flush=True)
        for k in [k for k in squares_req if k not in self.cams]:
            print(f"[crop] WARNING: --{k}-square {squares_req[k]} ignored — there is no '{k}' camera in this "
                  f"session", flush=True)
        squares_req = {k: v for k, v in squares_req.items() if k in self.cams}
        if squares_req and self.cams:
            from ..sensors.capture_meta import apply as _apply_squares, parse_square, squares_of
            before = squares_of(self.cams)
            applied = _apply_squares(self.cams, {k: parse_square(v) for k, v in squares_req.items()})
            print(f"[crop] cameras {before} -> {applied} (from --scene-square/--side-square)")
        self._engine = engine
        self._robot_ip = (hw or {}).get("robot_ip")
        self._tool_voltage = tool_voltage
        self.hand_ok = True  # cleared if a hand send fails mid-teleop (triggers auto-recovery)
        self._connect_robot()
        self.home = self.robot.arm.get_ee_pose()
        self.repo_id, self.root = repo_id, root
        self.fps, self.width, self.height = fps, width, height  # width/height = DATASET image size
        self.use_videos = use_videos  # True (box, has ffmpeg) -> video dataset = fast training-time loading
        self.resume = resume  # append into an existing dataset at `root` instead of creating it
        # Render >= the dataset size so the recorded frame is always a DOWNSAMPLE of the native
        # capture, never an upsample. Sim renders crisp at >=640x480; hardware records at the dataset
        # size (cameras capture native 1080p, so 960x540 keeps real detail).
        self.disp_w, self.disp_h = max(640, width), max(480, height)
        self.dataset = None
        self.recording = False
        self.episodes = 0
        self.frames = 0
        self._save_thread = None   # background finalize of the last episode (save_episode_async)
        self.regrasp_ranges: list = []    # [start, end] frame ranges flagged during this episode
        self._regrasp_open = None         # open range's start frame, or None

    def _connect_robot(self) -> None:
        try:
            self.robot.connect()
        except Exception as e:  # on hardware the hand is tool-powered — power-cycle it + retry once
            if self._engine != "hardware" or not self._robot_ip:
                raise
            import time
            from ..hand.amazing_hand_client import ping_reachable, wait_reachable
            print(f"[connect] {e}")
            try:
                self.robot.disconnect()
            except Exception:  # noqa: BLE001
                pass
            hand_host = getattr(getattr(self.robot, "hand", None), "host", None)
            # Same gate as eval_hw._connect_with_hand_power: a hand that answers ping is powered and on
            # WiFi — the UDP probe missed a loss burst. Power-cycling it reboots a working hand (~20 s
            # off the network) and preempts the arm script; retry the connect instead.
            if hand_host and ping_reachable(hand_host):
                print(f"[connect] hand {hand_host} answers ping — not power-cycling; retrying the connect")
                for attempt in range(3):
                    try:
                        self.robot.connect()
                        break
                    except Exception as e2:  # noqa: BLE001
                        print(f"[connect] retry {attempt + 1}/3 failed: {e2}")
                        try:
                            self.robot.disconnect()
                        except Exception:  # noqa: BLE001
                            pass
                        time.sleep(2.0)
                else:
                    raise RuntimeError(f"hand {hand_host} pings but never answered the UDP probe")
            else:
                print(f"[connect] hand {hand_host or '?'} does not answer ping -> power-cycling the UR tool")
                _power_cycle_tool(self._robot_ip, self._tool_voltage)
                if hand_host:
                    print(f"[connect] hand {'is back' if wait_reachable(hand_host, 45.0) else 'STILL down after 45 s'}")
                else:
                    time.sleep(20)
                self.robot.connect()
        self.hand_ok = True

    def recover_hand(self) -> str:
        """Power-cycle the UR tool to reboot the AmazingHand, then reconnect. The tool URScript
        preempts the arm control script, so the arm is reconnected too. Returns a status message."""
        import time
        if not self._robot_ip:
            return "recover-hand needs the hardware engine"
        _power_cycle_tool(self._robot_ip, self._tool_voltage)
        time.sleep(6)
        arm_note = ""
        try:
            self.robot.arm.reconnect()
        except Exception as e:  # noqa: BLE001
            arm_note = f" (arm reconnect failed: {e}; hit Reconnect UR)"
        try:
            self.robot.hand.close()
            self.robot.hand.connect()
            self.hand_ok = True
            return "hand recovered" + arm_note
        except Exception as e:  # noqa: BLE001
            return f"hand still down after power-cycle: {e}"

    def _ensure_dataset(self) -> None:
        if self.dataset is not None:
            return
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if self.resume and os.path.exists(os.path.join(self.root, "meta", "info.json")):
            # append into the existing dataset; adopt its fps + image size so frames stay compatible
            self.dataset = LeRobotDataset(repo_id=self.repo_id, root=self.root)
            self.fps = self.dataset.fps
            scene = self.dataset.meta.features.get("observation.images.scene")
            if scene is not None:
                self.height, self.width = int(scene["shape"][0]), int(scene["shape"][1])
            # --use-videos only applies when a root is CREATED; a resumed root keeps its storage type.
            # Say so loudly: tigger_bed_1 became a 3 GB image dataset because its first launch lacked
            # the flag and every later launch with it was silently ignored (2026-10-24).
            _stored_video = (scene or {}).get("dtype") == "video"
            if _stored_video != bool(self.use_videos):
                print(f"[resume] WARNING: this dataset stores {'VIDEO' if _stored_video else 'PNG IMAGES'} but you "
                      f"launched with{'' if self.use_videos else 'out'} --use-videos — the flag is ignored on an "
                      f"existing root. Start a new --root to change storage.", flush=True)
            print(f"[resume] appending to {self.root} ({self.dataset.num_episodes} episodes, "
                  f"{self.width}x{self.height} @ {self.fps}fps)")
            # The dataset is the authority on camera geometry too: re-apply ITS crop so appended
            # episodes match the existing ones, whatever cameras.py or the service currently say.
            from ..sensors import capture_meta as _cm
            cap = _cm.read(self.root)
            if cap and cap.get("squares") and getattr(self, "cams", None):
                before = _cm.squares_of(self.cams)
                moved = _cm.differs(before, {k: v for k, v in cap["squares"].items() if k in self.cams})
                applied = _cm.apply(self.cams, {k: v for k, v in cap["squares"].items() if k in self.cams})
                print(f"[resume] crop from capture.json: {applied}"
                      + (f"  (moved {moved} from {before})" if moved else "  (cameras already matched)"))
            elif getattr(self, "cams", None):
                print("[resume] dataset has no capture.json (recorded before it existed) — cameras left "
                      f"at {_cm.squares_of(self.cams)}; make sure that matches how it was recorded")
        else:
            from ..sim.record_sim import build_features, camera_shapes

            # Hardware cams have per-camera shapes now (sideways mounts yield portrait frames, a
            # square crop yields a square), so the dataset declares a shape per camera.
            shapes = camera_shapes(self.width, self.height) if self._engine == "hardware" else None
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                features=build_features(self.width, self.height, use_videos=self.use_videos,
                                        with_wrist=self.has_wrist, with_side=self.has_side,
                                        shapes=shapes),
                root=self.root,
                robot_type=self.robot.name,
                use_videos=self.use_videos,
            )
            if self._engine == "hardware" and getattr(self, "cams", None):
                # Record the geometry this dataset is captured with — the crop in effect on the live
                # cameras, the mount rotations, the fit box and the per-camera stored shapes — so the
                # manager can show it, training can carry it, and eval can re-apply it.
                from ..sensors import capture_meta as _cm
                from ..sensors.cameras import SCENE_ROTATE, SIDE_ROTATE, WRIST_ROTATE
                cap = _cm.write(self.root, _cm.squares_of(self.cams),
                                rotate={"scene": SCENE_ROTATE, "wrist": WRIST_ROTATE, "side": SIDE_ROTATE},
                                capture_box=(self.width, self.height),
                                shapes={n: (h, w) for n, (w, h) in (shapes or {}).items()})
                print(f"[capture] {self.root}/meta/capture.json: {_cm.describe(cap)}")
        self._start_writer()  # async image writer -> add_frame doesn't block the teleop loop

    def _to_dataset_size(self, img: np.ndarray, cam: str) -> np.ndarray:
        """Resize `img` to exactly the shape this camera's dataset feature declares.

        The dataset is the authority — that also covers `--resume`, where the shape comes from the
        existing dataset rather than the current cameras.py constants.
        """
        import cv2

        ft = self.dataset.meta.features.get(f"observation.images.{cam}") if self.dataset else None
        if ft is None or not ft.get("shape"):
            # no declaration to follow (sim engines / a lerobot whose meta differs): the session size
            h, w = int(self.height), int(self.width)
        else:
            h, w = int(ft["shape"][0]), int(ft["shape"][1])
        return img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h))

    def render(self) -> np.ndarray:
        return self._render_fn(self.disp_w, self.disp_h)  # crisp scene for display

    def render_wrist(self):
        return self._wrist_fn(self.width, self.height) if self.has_wrist else None

    def render_side(self):
        return self._side_fn(self.width, self.height) if self.has_side else None

    def step(self, action_vec, task: str, record: bool, do_wrist: bool = True, command_arm: bool = True):
        """observe (scene + side + wrist) -> record -> act. Returns (scene_img, wrist_img|None, side_img|None).

        Scene + side (both 3rd-person) are rendered crisp (disp size) for the display; the recorded
        frame is downsampled to the dataset size. do_wrist=False skips the wrist render this tick for a
        snappier loop; when recording we always render it so every frame has all cameras.
        command_arm=False (freedrive): the arm is hand-guided, so only the hand (grasp) is driven —
        the recorded action's arm pose is the actual pose the human moved through.
        """
        scene = self.render()  # disp_w x disp_h (crisp)
        side = self._side_fn(self.disp_w, self.disp_h) if self.has_side else None  # 2nd scene, crisp for display
        wrist = self.render_wrist() if (self.has_wrist and (record or do_wrist)) else None
        if record:
            import cv2

            self._ensure_dataset()
            obs = self.robot.get_observation()
            state = np.array([obs[n] for n in STATE_NAMES], dtype=np.float32)
            # Resize to each camera's DECLARED shape, not one global size: shapes are per-camera now
            # (portrait for sideways mounts, square when cropped), so forcing (width, height) here
            # would squish them back into landscape and disagree with the dataset schema.
            frame = {
                "observation.state": state,
                "observation.images.scene": self._to_dataset_size(scene, "scene"),
                "action": np.asarray(action_vec, dtype=np.float32),
                "task": task,
            }
            if wrist is not None:
                frame["observation.images.wrist"] = self._to_dataset_size(wrist, "wrist")
            if side is not None:
                frame["observation.images.side"] = self._to_dataset_size(side, "side")
            self.dataset.add_frame(frame)
            self.frames += 1
        try:
            if command_arm:
                self.robot.send_action(dict(zip(ACTION_NAMES, action_vec)))  # advance arm + hand
            else:  # freedrive: arm is hand-guided; drive only the hand from the action's curls
                # via robot.send_curls, NOT robot.hand.send_curls: get_observation reports the cached
                # last command, and the direct call leaves it stale — frames recorded during freedrive
                # would otherwise store the hand curls from before freedrive began
                self.robot.send_curls(list(action_vec[len(ARM_POSE_NAMES):]))
        except Exception as e:  # noqa: BLE001 — a flaky (tool-powered) hand drop mustn't crash the loop
            self.hand_ok = False
            self._send_err = str(e)
        return scene, wrist, side

    def start_episode(self) -> None:
        if self.recording:   # already recording: this used to silently reset frames mid-episode
            return
        self.wait_save()     # the previous episode's background save owns the dataset until it is done
        self._ensure_dataset()
        self.recording = True
        self.frames = 0
        self.regrasp_ranges = []   # completed [start, end] frame ranges for this episode
        self._regrasp_open = None  # frame a range was opened at, or None

    @property
    def grasps(self) -> int:
        """Grasp ATTEMPTS so far: the initial one plus every closed re-grasp range. Never 0 — an
        episode always contains one grasp, so a clean take reports 1 (see webui/annotations.FIELDS)."""
        return 1 + len(self.regrasp_ranges)

    def regrasp_mark(self) -> str:
        """Toggle a re-grasp range at the current frame. Returns a short status for the log.

        The operator flags the stretch where the grasp missed and had to be retried, so training can
        weight or condition on recovery instead of it being ~5% of frames the model treats as noise.
        Closing a range bumps `grasps`; the ranges themselves are what the sampler weights.
        """
        if not self.recording:
            return "not recording"
        if self._regrasp_open is None:
            self._regrasp_open = self.frames
            return f"grasp {self.grasps + 1} START @ frame {self.frames}"
        start = self._regrasp_open
        self._regrasp_open = None
        if self.frames <= start:       # double-tap / zero-length -> discard rather than store junk
            return "re-grasp cancelled (zero length)"
        self.regrasp_ranges.append([start, self.frames])
        return (f"grasp {self.grasps}: frames {start}-{self.frames} "
                f"({self.frames - start} frames), {self.grasps} grasps total")

    def save_episode(self) -> None:
        """Synchronous save (selftest / fallback). The panel uses save_episode_async()."""
        if self.recording and self.dataset is not None and self.frames > 0:
            self.recording = False
            self._save_episode_body()
        self.recording = False

    def _save_episode_body(self) -> None:
        """The slow part: drain the image writer, encode videos, compute stats, write parquet."""
        self.dataset.save_episode()
        self.episodes += 1
        try:  # keep capture.json's recorded.last current (shown on the dataset card)
            from ..sensors import capture_meta as _cm
            _cm.touch(self.root)
        except Exception:  # noqa: BLE001 — a sidecar hiccup must never break a save
            pass
        try:  # every save leaves a full Arrow copy of the dataset in ~/.cache/huggingface — prune
            from .. import hf_cache
            print("[save] " + hf_cache.prune(keep=2), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[save] hf cache prune skipped: {e}", flush=True)

    def save_episode_async(self, on_done=None) -> "int | None":
        """Close the episode NOW and finalize it on a background thread.

        Recording stops synchronously (no more frames land in the buffer), then a thread runs the
        dataset save + checkpoint (finalize/re-open so the on-disk copy is loadable). The caller can
        start the arm's return home and ask for the rating immediately instead of after the 5-15 s
        those take (the re-open re-materialises the dataset's Arrow cache, which grows with it).
        Returns the 0-based index of the episode being saved, or None if nothing was recorded.
        `on_done(ep, err)` runs ON THE THREAD when it finishes (err is None on success) — it must
        not touch Tk. start_episode()/discard_episode()/finalize() wait for an in-flight save.
        """
        if not (self.recording and self.dataset is not None and self.frames > 0):
            self.recording = False
            return None
        self.wait_save()
        self.recording = False
        ep = int(self.dataset.num_episodes)

        def run() -> None:
            err = None
            try:
                self._save_episode_body()
                try:
                    self.checkpoint()   # finalize + re-open: the uploaded copy must be loadable
                except Exception as e:  # noqa: BLE001 — recording can continue on the live dataset
                    err = f"checkpoint warning (episode {ep + 1}): {e}"
            except Exception as e:  # noqa: BLE001
                err = f"SAVE FAILED (episode {ep + 1}): {e}"
            if on_done is not None:
                try:
                    on_done(ep, err)
                except Exception:  # noqa: BLE001
                    pass

        self._save_thread = _threading_mod.Thread(target=run, daemon=True, name="episode-save")
        self._save_thread.start()
        return ep

    @property
    def save_in_flight(self) -> bool:
        t = self._save_thread
        return t is not None and t.is_alive()

    def wait_save(self, timeout: "float | None" = None) -> None:
        t = self._save_thread
        if t is not None and t.is_alive():
            t.join(timeout)

    def discard_episode(self) -> None:
        if self.save_in_flight:   # the episode is already closed and being written — nothing to drop
            return
        if self.dataset is not None:
            self.dataset.episode_buffer = self.dataset.create_episode_buffer()
        self.recording = False
        self.frames = 0

    def _start_writer(self) -> None:
        """Offload per-frame image (PNG) writes to background threads so add_frame doesn't block the
        teleop loop. Without this, LeRobot encodes 3x 960x540 PNGs synchronously on the record tick,
        stalling teleop (idle movement stays smooth because it never adds frames)."""
        if self.dataset is None:
            return
        try:
            # Processes, not only threads: 8 PNG writer threads shared the interpreter lock with the
            # teleop tick (panel at 450% CPU, loop down to 6-7 ticks/s, 1.8 GB of frames queued per
            # 20 s episode because the writers could not keep up). Each process has its own lock.
            self.dataset.start_image_writer(num_processes=2, num_threads=4)
        except Exception:  # noqa: BLE001 — recording still works (synchronously) if this fails
            pass

    def _stop_writer(self) -> None:
        if self.dataset is not None and getattr(self.dataset, "image_writer", None) is not None:
            try:
                self.dataset.stop_image_writer()  # flush pending writes + join threads
            except Exception:  # noqa: BLE001
                pass

    def finalize(self) -> None:
        """Flush episode metadata + write the parquet footers. MUST run before exit, or the last
        data/episodes parquet is left open (no footer) -> the dataset is corrupt and won't load.
        LeRobot only does this in finalize(); its __del__ closes the data writer but NOT the
        metadata writer, and a hard window close (X) may skip __del__ entirely."""
        self.wait_save()
        if self.dataset is not None:
            self._stop_writer()
            self.dataset.finalize()

    def checkpoint(self) -> bool:
        """Finalize the dataset (write meta/episodes footers so it's LOADABLE) then re-open it for
        continued appending — the same round-trip `--resume` does across launches, but in-process, so
        a loadable copy exists after every episode without closing the panel. Returns True on success;
        on failure the live dataset is left recording-capable (best-effort; never crashes the loop)."""
        if self.dataset is None:
            return False
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self._stop_writer()      # flush pending image writes before finalizing
        self.dataset.finalize()  # writes meta/episodes; also closes the data/meta writers
        self.dataset = LeRobotDataset(repo_id=self.repo_id, root=self.root)  # re-open to keep appending
        self._start_writer()     # fresh async writer for the next episode
        return True

    def reset_scene(self) -> None:
        """Reset the whole sim to its start state (arm to home, block back to its spot)."""
        self.robot.connect()

    def _cell(self):
        """The MujocoCell behind the robot (None on the kinematic engine, which has no scene)."""
        return getattr(getattr(self.robot, "arm", None), "cell", None)

    def randomize_box(self):
        c = self._cell()
        return c.randomize_block() if c is not None else None

    def randomize_goal(self):
        c = self._cell()
        return c.randomize_goal() if c is not None else None

    def set_box_color(self, rgba) -> bool:
        c = self._cell()
        if c is None:
            return False
        c.set_block_color(rgba)
        return True


def _apply_dark_theme(win) -> None:
    """Recolor every Tk widget to a dark palette. tk_setPalette applies to existing + future widgets,
    so calling it right after Tk() themes the whole app (buttons, sliders, entries, menus, dialogs)."""
    win.tk_setPalette(
        background="#1e1e1e", foreground="#e0e0e0",
        activeBackground="#3a3a3a", activeForeground="#ffffff",
        disabledForeground="#777777", highlightBackground="#1e1e1e",
        highlightColor="#5a5a5a", insertBackground="#e0e0e0",
        selectBackground="#3a5f8a", selectForeground="#ffffff",
        troughColor="#333333",
    )


def _tune_fonts(win) -> None:
    """Tk's stock fonts on Linux are jagged bitmap fonts; repoint the named fonts at a clean
    scalable (anti-aliased) family. macOS already uses a nice system font, so leave it alone."""
    import sys

    if sys.platform == "darwin":
        return
    import tkinter.font as tkfont

    families = set(tkfont.families(win))
    family = next((f for f in ("Ubuntu", "DejaVu Sans", "Noto Sans", "Liberation Sans")
                   if f in families), None)
    if family is None:
        return
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        try:
            tkfont.nametofont(name).configure(family=family, size=11)
        except Exception:  # noqa: BLE001 — skip any font Tk doesn't define
            pass


def _load_sm_calib(path: str = "outputs/sm_calib.json") -> None:
    """Override SM_MAP from a calibration file written by sm_calibrate.py, if present."""
    import json
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            data = json.load(f)
        SM_MAP.update({k: (v[0], int(v[1])) for k, v in data.items()})
        print(f"[sm] loaded axis calibration from {path}")
    except Exception as e:  # noqa: BLE001
        print(f"[sm] failed to load {path}: {e}")


def _load_gp_calib(path: str = "outputs/gp_calib.json") -> None:
    """Override GP_MAP from a calibration file written by gp_calibrate.py, if present."""
    import json
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            data = json.load(f)
        GP_MAP.update({k: (v[0], int(v[1])) for k, v in data.items()})
        print(f"[gp] loaded gamepad calibration from {path}")
    except Exception as e:  # noqa: BLE001
        print(f"[gp] failed to load {path}: {e}")


TOOL_VOLTAGE = 12  # UR tool output voltage powering the AmazingHand ESP32 (0/12/24 V; 12 per the build)


def _power_cycle_tool(robot_ip: str, voltage: int = TOOL_VOLTAGE) -> None:
    """Toggle the UR tool output voltage OFF->ON to power-cycle the AmazingHand's ESP32 (fed from the
    tool port) so it reboots + rejoins WiFi. Sent as URScript to the UR secondary interface (:30002).
    This preempts any running control script, so the caller must reconnect the ARM afterwards."""
    import socket
    import time

    def send(v: int) -> None:
        try:
            s = socket.create_connection((robot_ip, 30002), timeout=3)
            s.sendall(f"set_tool_voltage({int(v)})\n".encode())
            time.sleep(0.3)
            s.close()
        except Exception:  # noqa: BLE001
            pass

    send(0)
    time.sleep(1.5)
    send(voltage)


def _maximize(win) -> None:
    """Best-effort maximize across platforms: -zoomed (X11/Linux — the robot PC), state('zoomed')
    (Windows), else size to the full screen (macOS has no Tk maximize attribute)."""
    for attempt in (lambda: win.attributes("-zoomed", True), lambda: win.state("zoomed")):
        try:
            attempt()
            return
        except Exception:  # noqa: BLE001
            pass
    try:
        win.update_idletasks()
        win.geometry(f"{win.winfo_screenwidth()}x{win.winfo_screenheight()}+0+0")
    except Exception:  # noqa: BLE001
        pass


def _crop_black(img: np.ndarray, thr: int = 12) -> np.ndarray:
    """Trim near-black letterbox/vignette borders so the fisheye circle fills the preview cell.

    Display-only cosmetic crop (recorded frames are untouched). Guards against over-cropping a dark
    frame: if the content box shrinks below 30% in either axis, keep the original.
    """
    g = img.max(axis=2)
    cols = np.where(g.max(axis=0) > thr)[0]
    rows = np.where(g.max(axis=1) > thr)[0]
    if len(cols) and len(rows):
        c = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        if c.shape[0] >= 0.3 * img.shape[0] and c.shape[1] >= 0.3 * img.shape[1]:
            return c
    return img


def _compose_row(imgs) -> "np.ndarray | None":
    """Crop black borders off each frame, match their heights, lay them side by side for the preview."""
    import cv2

    imgs = [_crop_black(i) for i in imgs if i is not None]
    if not imgs:
        return None
    hh = min(i.shape[0] for i in imgs)
    imgs = [cv2.resize(i, (max(1, round(i.shape[1] * hh / i.shape[0])), hh)) for i in imgs]
    return imgs[0] if len(imgs) == 1 else np.hstack(imgs)


def run_gui(engine, repo_id, root, task_default, fps, width, height, input_mode="sliders", use_videos=False, grasp_mode="pinch", resume=False, hw=None, tool_voltage=TOOL_VOLTAGE, gpu_host="bryan@GPU", gpu_datasets_dir="~/VLA/outputs/datasets", gpu_sync=True, home_on_save=True, operator="", expose_manual=False, ur_power=True, ur_shutdown=False, ui="web", web_port=8090) -> None:
    import tkinter as tk

    from PIL import Image, ImageTk

    from ..robot.ur_power import power_down, power_up

    # Hardware: bring the UR up before connecting (clears a C161A0 protective stop, releases the
    # brakes) and park it on quit. A failed power-up is not fatal — the connect below fails loudly.
    ur_ip = (hw or {}).get("robot_ip") if engine == "hardware" and ur_power else None
    if ur_ip:
        try:
            print(f"[ur] {power_up(ur_ip)}", flush=True)
        except RuntimeError as e:
            print(f"[ur] power-up failed: {e}", flush=True)

    # resume -> record into the exact root (create it once, append every run after); else a fresh folder
    dataset_root = root if resume else _unique_root(root)
    try:
        session = SimSession(engine, repo_id, dataset_root, fps, width, height, use_videos=use_videos, resume=resume, hw=hw, tool_voltage=tool_voltage)
    except BaseException:
        if ur_ip:
            print(f"[ur] {power_down(ur_ip, shutdown=ur_shutdown)}", flush=True)
        raise
    ctrl = ManualController(session.home, grasp_mode=grasp_mode)  # arm's natural keyframe home, grasp 0
    ranges = ctrl.ranges()

    # Captured home JOINT config for the deck 'Reset' key (hold-to-home). Persisted across runs.
    import json as _json
    HOME_PATH, HOME_DQ = "outputs/home_pose.json", 0.03  # rad/tick step (~0.36 rad/s at 12 fps)
    HOME_REACHED_TOL = 0.03    # rad — joints within this of home => auto-return-on-Save is complete
    HOME_SAVE_MAX_STEPS = 150  # safety cap (~12s @12fps) so a stuck return still saves the episode
    hp = {"q": None}  # mutable holder; Set Home updates it
    home_save = {"active": False, "steps": 0}  # auto-return-to-home-on-Save state (driven by tick)
    if os.path.exists(HOME_PATH):
        try:
            hp["q"] = _json.load(open(HOME_PATH)).get("q")
            print(f"[home] loaded home pose from {HOME_PATH}")
        except Exception as e:  # noqa: BLE001
            print(f"[home] could not load {HOME_PATH}: {e}")

    # Live input state — the mode is switchable at runtime via the dropdown, so keep the device
    # handles + current mode in a mutable dict (closures below read/mutate it).
    io = {"mode": input_mode, "sm": None, "gp": None, "gp_ok": False}

    def _teardown_input() -> None:
        for key in ("sm", "gp"):
            if io[key] is not None:
                try:
                    io[key].close()
                except Exception:  # noqa: BLE001
                    pass
                io[key] = None
        try:
            session.robot.arm.stop_freedrive()  # leaves teach mode if we were in freedrive
        except Exception:  # noqa: BLE001
            pass

    def _setup_input(mode) -> str:
        # SpaceMouse drives the arm in 'spacemouse'; in 'freedrive' it's opened only for grasp buttons.
        if mode in ("spacemouse", "freedrive"):
            from .spacemouse import SpaceMouse

            if mode == "spacemouse":
                _load_sm_calib()  # override SM_MAP from outputs/sm_calib.json if it exists
            try:
                io["sm"] = SpaceMouse(deadband=SM_DEADBAND).open()
            except Exception as e:  # noqa: BLE001
                if mode == "spacemouse":
                    return f"SpaceMouse unavailable ({e}); using sliders"
        elif mode == "gamepad":
            from .gamepad import Gamepad

            _load_gp_calib()  # override GP_MAP from outputs/gp_calib.json if it exists
            io["gp_ok"] = os.path.exists("outputs/gp_calib.json")  # gate motion on a calibration
            try:
                io["gp"] = Gamepad().open()
            except Exception as e:  # noqa: BLE001
                return f"gamepad unavailable ({e}); using sliders"
            if not io["gp_ok"]:
                return f"gamepad {io['gp'].name}: NOT CALIBRATED — click 'Calibrate GP' before it will move the arm."
            return f"gamepad: {io['gp'].name} (calibrated)"
        if mode == "freedrive":
            ok = session.robot.arm.start_freedrive()
            return (("FREEDRIVE — hand-guide the arm; grasp: c=close o=open"
                     + (" + SpaceMouse buttons" if io["sm"] else "")) if ok
                    else "freedrive not supported by this arm (needs the real UR5e)")
        return f"input: {mode}"

    sm_msg = _setup_input(input_mode)  # bring up the initial input device

    win = tk.Tk()
    _apply_dark_theme(win)  # global dark palette (buttons/sliders/entries/menus/dialogs)
    _tune_fonts(win)
    win.title(f"UR5e + AmazingHand — teleop ({engine}, {input_mode})")
    win.geometry("760x1020")
    win.grid_columnconfigure(0, weight=1)  # single column: buttons / top strip / cameras / controls / log
    win.grid_rowconfigure(2, weight=1)  # the camera view (row 2) expands; the other rows stay compact

    # Log panel pinned to the very bottom (event history).
    from tkinter.scrolledtext import ScrolledText
    log_text = ScrolledText(win, height=7, bg="#111", fg="#ddd", state="disabled",
                            font=("TkFixedFont", 9), wrap="word")
    log_text.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 6))

    # Render view (left): scene OVER wrist in EQUAL cells, each letterboxed + resize-reactive.
    view = tk.Frame(win, bg="black")
    view.grid(row=2, column=0, sticky="nsew")
    view.grid_columnconfigure(0, weight=1)
    view.grid_rowconfigure(0, weight=1, uniform="cam")
    scene_cell = tk.Frame(view, bg="black")
    scene_cell.grid(row=0, column=0, sticky="nsew")
    img_label = tk.Label(scene_cell, bg="black", borderwidth=0)
    img_label.place(relx=0.5, rely=0.5, anchor="center")
    tk.Label(scene_cell, text="scene · side" if session.has_side else "scene",
             fg="#888", bg="black").place(x=4, y=2)
    wrist_cell = None
    wrist_label = None
    if session.has_wrist:  # equal-weight second row (uniform) -> both feeds render the same size
        view.grid_rowconfigure(1, weight=1, uniform="cam")
        wrist_cell = tk.Frame(view, bg="black")
        wrist_cell.grid(row=1, column=0, sticky="nsew")
        wrist_label = tk.Label(wrist_cell, bg="black", borderwidth=0)
        wrist_label.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(wrist_cell, text="wrist", fg="#888", bg="black").place(x=4, y=2)

    # Controls — under the camera feeds (single column layout).
    controls = tk.Frame(win)
    controls.grid(row=3, column=0, sticky="ew", padx=8, pady=8)

    # Collapsible pose/grasp sliders.
    sliders_frame = tk.Frame(controls)
    sliders: dict[str, tk.DoubleVar] = {}
    for r, key in enumerate(SLIDER_ORDER):
        lo, hi = ranges[key]
        tk.Label(sliders_frame, text=key).grid(row=r, column=0, sticky="e")
        var = tk.DoubleVar(value=getattr(ctrl, key))
        tk.Scale(sliders_frame, from_=lo, to=hi, resolution=0.005, orient="horizontal",
                 length=220, variable=var,
                 command=lambda v, k=key: setattr(ctrl, k, float(v))).grid(row=r, column=1)
        sliders[key] = var
    tk.Label(sliders_frame, text="grasp").grid(row=len(SLIDER_ORDER), column=0, sticky="e")
    grasp_var = tk.DoubleVar(value=ctrl.grasp)
    tk.Scale(sliders_frame, from_=0.0, to=1.0, resolution=0.01, orient="horizontal",
             length=220, variable=grasp_var,
             command=lambda v: setattr(ctrl, "grasp", float(v))).grid(row=len(SLIDER_ORDER), column=1)

    # Task + status vars (their widgets live on the top strip built below; define here so log() works).
    task_var = tk.StringVar(value=task_default)
    status = tk.StringVar(value=(sm_msg or "ready — Start rec to record"))

    webh = {"w": None, "frames": False}   # browser UI handle (ur5e_lerobot/ui/webpanel.WebPanel) + flags

    def log(msg: str) -> None:
        """Set the live status line + append a timestamped line to the bottom log panel."""
        import time
        status.set(msg)
        if webh["w"] is not None:
            webh["w"].log(msg)
        log_text.configure(state="normal")
        log_text.insert("end", f"{time.strftime('%H:%M:%S')}  {msg}\n")
        log_text.see("end")
        log_text.configure(state="disabled")
        print(f"[panel] {msg}", flush=True)  # also to stdout -> log/manual_panel.log

    log(sm_msg or "ready — Start rec to record")

    def do_reset() -> None:
        ctrl.reset()
        session.reset_scene()  # sim: block+arm to home. hardware: reconnect arm/hand.
        if io["mode"] == "freedrive":
            session.robot.arm.start_freedrive()  # reconnect dropped teach mode -> re-enter it
        for k, var in sliders.items():
            var.set(getattr(ctrl, k))
        grasp_var.set(ctrl.grasp)

    def do_set_home() -> None:
        """Capture the arm's CURRENT joint config as the home pose (freedrive there first). Persisted;
        the deck 'Reset' key hold-returns to it."""
        try:
            q = [float(v) for v in session.robot.arm.get_joint_positions()]
        except Exception as e:  # noqa: BLE001
            log(f"set-home failed (no joint feedback): {e}")
            return
        hp["q"] = q
        try:
            os.makedirs("outputs", exist_ok=True)
            _json.dump({"q": q, "tcp": [float(v) for v in session.robot.arm.get_ee_pose()]},
                       open(HOME_PATH, "w"), indent=2)
            log("home pose set ✓ — deck Reset (hold) returns here")
        except Exception as e:  # noqa: BLE001
            log(f"home set in memory; save failed: {e}")

    def _rebase_ctrl() -> None:
        """Snap the teleop target onto the arm's actual pose (no _home_rot change) so resuming teleop
        after a hold-to-home / freedrive doesn't jump."""
        try:
            pose = session.robot.arm.get_ee_pose()
        except Exception:  # noqa: BLE001
            return
        ctrl.x = _clamp(pose[0], *ranges["x"]); ctrl.y = _clamp(pose[1], *ranges["y"]); ctrl.z = _clamp(pose[2], *ranges["z"])
        d = ctrl._home_rot.inv() * Rotation.from_rotvec(pose[3:6])
        ctrl.roll, ctrl.pitch, ctrl.yaw = (float(a) for a in d.as_euler("xyz"))

    def _finalize_save() -> None:
        """Close the episode and hand the slow part (dataset save + checkpoint) to a background
        thread, so the return home and the rating prompt start NOW. The GPU upload runs after that
        thread finishes (it must copy a loadable dataset), in parallel with homing / rating."""
        import time
        if session.save_in_flight:
            log("previous episode still finalizing — waiting…")
            session.wait_save()
        # Re-grasp ranges marked during the episode (deck key 0 while recording). Captured BEFORE the
        # episode is closed, and written alongside the operator so the sidecar is complete in one pass.
        ranges = list(session.regrasp_ranges or [])
        if session._regrasp_open is not None and session.frames > session._regrasp_open:
            # range left open when Save was pressed — close it at the final frame rather than drop it
            ranges.append([session._regrasp_open, session.frames])
            log(f"re-grasp range was still open — closed at final frame {session.frames}")
        grasps = 1 + len(ranges)   # the initial grasp always counts
        # the locked exposure is part of how this episode LOOKS, so it belongs in the record —
        # otherwise a dataset recorded across two lighting setups is indistinguishable after the fact
        expo = (dict(session.exposure.values)
                if session.exposure is not None and session.exposure.locked else None)
        def _done(ep, err) -> None:   # runs on the save thread: no Tk here, post to the tick's queue
            if err:
                upload_q.put(err)
            upload_q.put(f"episode {ep + 1} finalized on disk -> {session.root}")
            _kick_upload("save")       # the dataset is loadable now -> push it to the GPU (background)

        ep_idx = session.save_episode_async(on_done=_done)
        if ep_idx is None:             # nothing was recorded -> no save, no upload, no rating
            return
        # Written unconditionally: `grasps` is always >= 1, so every recorded episode gets a record.
        # A missing key downstream therefore means "recorded before the flag existed", not "clean".
        # The sidecar is independent of the parquet, so this is safe while the save thread runs.
        try:
            _write_annotation(session.root, ep_idx, operator=operator or None,
                              grasps=grasps,
                              regrasp_ranges=ranges or None,
                              exposure=expo)
            log(f"episode {ep_idx + 1}: {grasps} grasp(s)"
                + (f", re-grasp range(s) {ranges}" if ranges else ""))
        except Exception as e:  # noqa: BLE001
            log(f"annotation write failed: {e}")
        sd["save_flash_until"] = time.time() + 1.5  # flash the deck Save key to confirm
        last_saved.update(ep=ep_idx, rating=None, notes="", upload="finalizing…")
        log(f"saved episode {ep_idx + 1} (finalizing in the background) -> {session.root}")
        _enter_rating(ep_idx)          # light up the deck bottom row for a 1-5 star rating NOW

    def do_save() -> None:
        # Save FIRST, then return home with recording already stopped. The return used to be part of
        # the episode ("recorded, grasp held") so the policy would learn to retract to the transport
        # pose; it is now excluded from training on purpose, so the demos end at the grasp and the
        # retract is just rig housekeeping. tick() drives the joint-space return afterwards.
        if not session.recording or session.frames == 0:
            return  # nothing recording to save
        if home_save["active"]:
            return  # a return is already in progress
        _finalize_save()                      # episode closes here — nothing below is recorded
        if hp["q"] is not None and home_on_save:
            home_save["active"] = True
            home_save["steps"] = 0
            log("saved; returning to home (NOT recorded)")

    def do_discard() -> None:
        home_save["active"] = False  # cancel any in-progress auto-home return
        session.discard_episode()
        log("episode discarded")

    def do_rand_box() -> None:
        p = session.randomize_box()
        status.set(f"box -> ({p[0]:.2f}, {p[1]:.2f}) yaw {p[2]:.0f}°" if p else "randomize needs the mujoco engine")

    def do_rand_goal() -> None:
        p = session.randomize_goal()
        status.set(f"goal -> ({p[0]:.2f}, {p[1]:.2f})" if p else "randomize needs the mujoco engine")

    def do_set_color(name) -> None:
        if not session.set_box_color(dict(ROYGBIV)[name]):
            status.set("box color needs the mujoco engine")

    def on_close() -> None:
        """Clean shutdown for the Quit button, the window's X, the page's ⏻ and `supervisorctl stop`
        (SIGTERM): drop an unsaved episode, finalize the dataset (parquet footers), final sync, park
        the arm. Every step is logged, and a watchdog ends the process after CLOSE_DEADLINE_S no
        matter what — supervisor's SIGKILL would do the same, only later and silently."""
        import os as _os
        import threading as _th
        CLOSE_DEADLINE_S = 60.0

        def _deadline() -> None:
            _th.Event().wait(CLOSE_DEADLINE_S)
            print(f"[close] still shutting down after {CLOSE_DEADLINE_S:.0f} s — exiting hard", flush=True)
            _os._exit(0)
        _th.Thread(target=_deadline, daemon=True, name="close-deadline").start()

        def step(name, fn) -> None:
            t0 = _ptime.time()
            try:
                fn()
                print(f"[close] {name} ({(_ptime.time() - t0) * 1000:.0f} ms)", flush=True)
            except Exception as e:  # noqa: BLE001 — never let one step block the rest
                print(f"[close] {name} warning: {e}", flush=True)

        step("input devices released", _teardown_input)          # also leaves freedrive (re-locks the arm)
        step("page server stopped", lambda: webh["w"].stop() if webh["w"] is not None else None)
        step("stream deck released", lambda: sd["pad"].close() if sd["pad"] is not None else None)
        if session.recording:
            step("unsaved episode dropped", session.discard_episode)
        step("dataset finalized", session.finalize)              # footers -> loadable
        if gpu_sync and session.dataset is not None:
            step("final GPU sync", lambda: print("[close] " + (_rsync_dataset("finalized") or "sync off"), flush=True))
        if ur_ip:  # release the RTDE control script first, then park the arm
            step("robot disconnected", session.robot.disconnect)
            step("UR parked", lambda: print(f"[ur] {power_down(ur_ip, shutdown=ur_shutdown)}", flush=True))
        win.destroy()

    shown = {"sliders": True}

    def _set_sliders(show) -> None:
        shown["sliders"] = show
        if show:
            sliders_frame.grid()
            toggle_btn.config(text="Hide sliders")
        else:
            sliders_frame.grid_remove()
            toggle_btn.config(text="Show sliders")

    def toggle_sliders() -> None:
        _set_sliders(not shown["sliders"])

    def switch_mode(mode) -> None:
        """Live-switch the input source from the dropdown (tears down the old device, opens the new)."""
        _teardown_input()
        io["mode"] = mode
        log(_setup_input(mode))
        _set_sliders(mode == "sliders")
        win.title(f"UR5e + AmazingHand — teleop ({engine}, {mode})")

    def do_reconnect() -> None:
        """Re-establish the UR connection after a fault / stopped script — no need to close the app."""
        log("reconnecting UR…")
        win.update_idletasks()
        try:
            msg = session.robot.arm.reconnect()
            if io["mode"] == "freedrive":
                session.robot.arm.start_freedrive()  # re-enter teach mode after the fresh connect
        except Exception as e:  # noqa: BLE001
            msg = f"reconnect failed: {e}"
        log(msg)

    def do_recover_hand() -> None:
        """Toggle the UR tool power to reboot the AmazingHand + reconnect (auto-called on a drop too)."""
        log("recovering hand: power-cycling the UR tool… (~7s)")
        win.update_idletasks()
        log(session.recover_hand())

    def do_calibrate_gamepad() -> None:
        """In-GUI gamepad calibration: guided captures -> outputs/gp_calib.json -> reload GP_MAP live."""
        if io["mode"] != "gamepad" or io["gp"] is None:
            log("set input = gamepad first, then Calibrate GP")
            return
        gp = io["gp"]
        cal_axes = [
            ("x", ["lx", "ly"], "LEFT stick: push the way you want +X (gripper RIGHT)"),
            ("y", ["lx", "ly"], "LEFT stick: push the way you want +Y (AWAY / forward)"),
            ("roll", ["rx", "ry"], "RIGHT stick: the way you want +ROLL"),
            ("pitch", ["rx", "ry"], "RIGHT stick: the way you want +PITCH (nose up)"),
            ("z", ["triggers"], "squeeze the TRIGGER you want for +Z (UP)"),
        ]
        dlg = tk.Toplevel(win)
        dlg.title("Gamepad calibration")
        dlg.geometry("390x190")
        dlg.grab_set()
        st = {"i": 0, "map": {}}
        lbl = tk.Label(dlg, text="", wraplength=360, justify="left", font=("TkDefaultFont", 12))
        lbl.pack(padx=14, pady=14)

        def prompt():
            axis, _, desc = cal_axes[st["i"]]
            lbl.config(text=f"[{st['i'] + 1}/{len(cal_axes)}]  {axis}\n\nHold: {desc}\n\nthen click Capture.")

        def capture():
            import time
            axis, srcs, _ = cal_axes[st["i"]]
            sums = {k: 0.0 for k in srcs}
            n = 0
            t0 = time.time()
            while time.time() - t0 < 1.2:  # sample while the user holds the input
                gp.read()
                s = gp.state()
                for src in srcs:
                    sums[src] += (s["rt"] - s["lt"]) if src == "triggers" else s[src]
                n += 1
                dlg.update()
                time.sleep(0.02)
            avg = {k: v / max(n, 1) for k, v in sums.items()}
            dom = max(srcs, key=lambda k: abs(avg[k]))
            if abs(avg[dom]) < 0.2:
                lbl.config(text=f"weak signal on {axis} — push firmly, then Capture again.")
                return
            st["map"][axis] = [dom, 1 if avg[dom] > 0 else -1]
            st["i"] += 1
            if st["i"] >= len(cal_axes):
                import json
                os.makedirs("outputs", exist_ok=True)
                with open("outputs/gp_calib.json", "w") as f:
                    json.dump(st["map"], f, indent=2)
                _load_gp_calib()
                io["gp_ok"] = True
                log("gamepad calibrated ✓ — it will now drive the arm")
                dlg.destroy()
            else:
                prompt()

        tk.Button(dlg, text="Capture (hold the input)", command=capture).pack(pady=8)
        prompt()

    # Action buttons in a single row along the bottom of the window (spans view + controls).
    btns = tk.Frame(win)
    actions = [
        ("Reset scene", do_reset),
        ("🏠 Set Home", do_set_home),
        ("● Start rec", session.start_episode),
        ("■ Save ep", do_save),
        ("Discard", do_discard),
        ("🎮 Calibrate GP", do_calibrate_gamepad),
        ("⟳ Reconnect UR", do_reconnect),
        ("🖐 Recover Hand", do_recover_hand),
        # (no Quit button: it fell off the bottom of the screen; the window's X and the Stream Deck's
        #  Quit key both run on_close)
    ]
    for i, (text, cmd) in enumerate(actions):
        tk.Button(btns, text=text, command=cmd, width=12).grid(row=0, column=i, padx=3, pady=2)

    # Scene controls (SIM ONLY): randomize box/goal + box color — meaningless on real hardware, so hide.
    scene_frame = None
    if engine != "hardware":
        scene_frame = tk.Frame(controls)
        tk.Button(scene_frame, text="Randomize Box", command=do_rand_box, width=11).grid(row=0, column=0, padx=2, pady=2)
        tk.Button(scene_frame, text="Randomize Goal", command=do_rand_goal, width=11).grid(row=0, column=1, padx=2, pady=2)
        tk.Label(scene_frame, text="box color").grid(row=1, column=0, sticky="e", pady=(4, 0))
        color_var = tk.StringVar(value=ROYGBIV[0][0])
        tk.OptionMenu(scene_frame, color_var, *[n for n, _ in ROYGBIV], command=do_set_color).grid(
            row=1, column=1, sticky="w", pady=(4, 0))
        session.set_box_color(dict(ROYGBIV)[color_var.get()])  # apply the default color on startup

    # Top strip (the 2nd line, directly under the button bar): input + task + status + the sliders
    # toggle, all on ONE line. Live-switch the action source from the input dropdown.
    topinfo = tk.Frame(win)
    tk.Label(topinfo, text="input").grid(row=0, column=0, sticky="e", padx=(0, 4))
    mode_var = tk.StringVar(value=io["mode"])
    tk.OptionMenu(topinfo, mode_var, "sliders", "spacemouse", "gamepad", "freedrive",
                  command=switch_mode).grid(row=0, column=1, sticky="w")
    tk.Label(topinfo, text="task").grid(row=0, column=2, sticky="e", padx=(12, 4))
    task_entry = tk.Entry(topinfo, textvariable=task_var, width=44)
    task_entry.grid(row=0, column=3, sticky="w")
    tk.Label(topinfo, textvariable=status, fg="white", anchor="w", width=18).grid(
        row=0, column=4, sticky="ew", padx=(12, 8))
    toggle_btn = tk.Button(topinfo, text="Hide sliders", command=toggle_sliders, width=12)
    toggle_btn.grid(row=0, column=5, sticky="e")
    topinfo.grid_columnconfigure(4, weight=1)  # status stretches; keeps the toggle pinned right

    # Control sections under the cameras (input/task/status moved up to the top strip).
    sliders_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    if scene_frame is not None:
        scene_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))

    # Window rows: buttons(0) / top strip(1) / cameras(2, expands) / controls(3) / log(4).
    btns.grid(row=0, column=0, sticky="ew", padx=8, pady=(4, 6))
    topinfo.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))

    # With any live input (SpaceMouse / gamepad / freedrive) the sliders are just feedback — collapse.
    _set_sliders(io["mode"] == "sliders")

    # --- Stream Deck (optional): 5 action keys + per-finger open/close (2 keys each) ---
    import queue as _queue
    # Deck layout (5x3, deck mounted upside down -> flip=True). Row 1: actions. Row 2: per-finger
    # OPEN + Open-All. Row 3: per-finger CLOSE + Close-All. In sd_fingers: finger index -1 == all
    # fingers; sign -1 == open, +1 == close.
    # Keys 0/3/4 are fixed actions. Keys 1/2 are recording-state-dependent (handled in _pump_streamdeck):
    #   recording -> 1=Save, 2=Discard   |   idle -> 1=Reset (hold->home), 2=Freedrive (toggle).
    sd_actions = {0: session.start_episode, 3: do_reconnect, 4: on_close}
    sd_action_labels = {0: "●", 3: "⟳", 4: "✖"}          # REC · reconnect UR · quit (icons only)
    sd_action_fg = {0: (225, 45, 45), 4: (225, 45, 45)}    # red REC dot and red quit cross
    # Keys are numbered by PHYSICAL finger, thumb first: 1 thumb · 2 index · 3 middle · 4 ring/pinky.
    # The curl channels (schema order index, middle, ring, thumb = 0..3) do not drive the fingers their
    # names suggest on this hand — measured 2026-10-26: channel 0 moves the middle finger, 1 ring/pinky,
    # 2 index, 3 thumb. The channel names are left alone (they are the dataset's action columns); the
    # deck maps number -> channel so the key does what its label says.
    # Row 2 = OPEN, row 3 = CLOSE: the first column is the whole hand (🖐 / ✊), then fingers 1-4.
    _ch = {1: 3, 2: 2, 3: 0, 4: 1}   # finger number -> curl channel
    sd_fingers = {5: (-1, -1), 6: (_ch[1], -1), 7: (_ch[2], -1), 8: (_ch[3], -1), 9: (_ch[4], -1),
                  10: (-1, +1), 11: (_ch[1], +1), 12: (_ch[2], +1), 13: (_ch[3], +1), 14: (_ch[4], +1)}
    sd_labels = {5: "🖐", 6: "1", 7: "2", 8: "3", 9: "4",
                 10: "✊", 11: "1", 12: "2", 13: "3", 14: "4"}
    sd = {"pad": None, "queue": _queue.Queue(), "held": {}, "reset_held": False, "prev_mode": "spacemouse",
          "rec_bg": None, "k1": None, "k2": None, "save_flash_until": 0.0,
          "rating_until": 0.0, "rating_ep": None,
          "expo": False}   # True while the launch-time exposure keys own row 1
    expo_started = {"done": False}   # one-shot guard for entering exposure mode after first render
    sm_chord = {"down": False}       # both-SpaceMouse-buttons chord: edge-trigger for REC/SAVE
    try:
        from .streamdeck import StreamDeckPad

        _pad = StreamDeckPad(flip=True).open()  # deck is mounted upside down
        sd["pad"] = _pad
        for _k, _lab in {**sd_action_labels, **sd_labels}.items():
            if _k < _pad.key_count:
                _pad.set_label(_k, _lab, fg=sd_action_fg.get(_k, (240, 240, 240)))
        _pad.on_key(lambda key, down: sd["queue"].put((key, down)))  # marshal to the Tk loop
        log(f"Stream Deck ready ({_pad.key_count} keys, flipped): actions | open row | close row")
    except Exception as e:  # noqa: BLE001 — the panel runs fine without a deck
        log(f"Stream Deck not available ({e})")

    # --- GPU auto-sync + Stream Deck star rating ------------------------------------------------
    # On Save: rsync the dataset to the GPU in the background AND light up the deck's bottom row as a
    # 1-5 star rating for the just-saved episode. The rating is written to meta/annotations.json (the
    # same sidecar the web UI reads) then re-synced, so it lands on the GPU automatically. The stars
    # clear on a press or after 30 s. Worker threads NEVER touch Tk — they post to upload_q, drained
    # in tick() -> log() on the main thread.
    import subprocess as _subprocess
    import threading as _threading
    upload_q = _queue.Queue()
    _upload_lock = _threading.Lock()

    def _rsync_dataset(reason: str = "") -> "str | None":
        """Blocking rsync of the current dataset to the GPU; returns a status line (None if disabled)."""
        if not gpu_sync:
            return None
        root_ = session.root
        name = os.path.basename(os.path.normpath(root_))
        dst = f"{gpu_host}:{gpu_datasets_dir.rstrip('/')}/{name}/"
        cmd = ["rsync", "-az", "--partial", "--exclude=*.tmp",
               "-e", "ssh -o ConnectTimeout=15", root_.rstrip("/") + "/", dst]
        try:
            r = _subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if r.returncode == 0:
                return f"GPU sync ✓ {name}" + (f" ({reason})" if reason else "")
            tail = (r.stderr.strip().splitlines() or [f"rc={r.returncode}"])[-1]
            return f"GPU sync FAILED [{name}]: {tail[:70]}"
        except Exception as e:  # noqa: BLE001
            return f"GPU sync error [{name}]: {e}"

    def _kick_upload(reason: str = "") -> None:
        if not gpu_sync:
            return

        def worker():
            with _upload_lock:  # serialize so uploads never race
                session.wait_save()   # never copy a half-written episode: wait for the finalize thread
                msg = _rsync_dataset(reason)
                if msg:
                    upload_q.put(msg)

        _threading.Thread(target=worker, daemon=True).start()

    def _write_annotation(root_: str, ep: int, **fields) -> None:
        """Merge fields (rating / operator / notes) into one episode's record in
        meta/annotations.json — the exact sidecar the web UI reads. Empty values clear the field."""
        import json
        p = os.path.join(root_, "meta", "annotations.json")
        data = {}
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    data = json.load(f)
            except Exception:  # noqa: BLE001
                data = {}
        if not isinstance(data, dict):
            data = {}
        rec = data.get(str(ep), {})
        for k, v in fields.items():
            if v is None or v == "":
                rec.pop(k, None)
            else:
                rec[k] = v
        if rec:
            data[str(ep)] = rec
        else:
            data.pop(str(ep), None)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, p)  # atomic on POSIX

    # --- launch-time exposure: auto-expose, trim with 3 keys, lock -----------------------------
    # Row 1 is taken over at startup: 0 = brighter, 1 = darker, 2 = SET. The live feeds are already
    # rendering, so the operator sees each nudge immediately. SET locks manual exposure and hands row
    # 1 back to the normal actions. Locking (rather than leaving auto on) also stops the driver
    # re-metering as the arm sweeps through frame, which would otherwise make image brightness a
    # function of arm position — a spurious correlate this policy has already been shown to prefer
    # over actually looking at the object.
    EXPO_KEYS = {0: "＋ brighter", 1: "－ darker", 2: "✓ SET"}

    def _render_expo_keys() -> None:
        pad = sd["pad"]
        if pad is None:
            return
        for k, lab in EXPO_KEYS.items():
            if k < pad.key_count:
                pad.set_label(k, lab, bg=(40, 70, 120) if k < 2 else (20, 120, 40))
        for k in range(3, pad.key_count):   # everything else dark until exposure is set
            pad.set_label(k, "", bg=(12, 12, 12))

    def _restore_normal_keys() -> None:
        pad = sd["pad"]
        sd["expo"] = False
        sd["rec_bg"] = sd["k1"] = sd["k2"] = None   # force the normal renderer to repaint
        if pad is None:
            return
        for k, lab in {**sd_action_labels, **sd_labels}.items():
            if k < pad.key_count:
                pad.set_label(k, lab, fg=sd_action_fg.get(k, (240, 240, 240)))   # keep the red REC / quit

    def _enter_exposure() -> None:
        if session.exposure is None:
            return                      # sim engine: no real cameras to expose
        if expose_manual:               # take over row 1 only when the operator wants to trim
            sd["expo"] = True
            _render_expo_keys()
        log("exposure: searching for a non-clipping setting…")
        win.update_idletasks()
        session.exposure.begin()                              # driver auto, to settle the sensor
        vals = session.exposure.auto_expose(session.width, session.height)
        if not expose_manual:
            # Default: lock the metered result straight away — auto has been reliable on this rig,
            # and a locked value is what stops the driver re-metering as the arm sweeps through
            # frame. --expose-manual keeps the trim step for when the room lighting defeats it.
            _expo_set()
            return
        log(f"exposure: auto -> {session.exposure.summary()}  "
            f"— ＋/－ to trim, ✓ SET to lock (row 1)")

    def _expo_nudge(factor: float) -> None:
        vals = session.exposure.nudge(factor)
        log(f"exposure {'+' if factor > 1 else '-'} -> {session.exposure.summary()}")
        return vals

    def _expo_set() -> None:
        vals = session.exposure.lock()
        log(f"exposure LOCKED: {session.exposure.summary()} (saved with each episode)")
        _restore_normal_keys()
        return vals

    def _enter_rating(ep) -> None:
        pad = sd["pad"]
        if pad is None or ep is None or ep < 0:
            return
        import time
        sd["rating_ep"] = ep
        sd["rating_until"] = time.time() + 30.0
        for i, k in enumerate(RATING_KEYS):  # bottom row: 1..5 stars, left->right
            pad.set_label(k, f"{i + 1}\n★", bg=(70, 55, 10), fg=(245, 210, 60))
        log(f"rate episode {ep + 1}: press 1-5 on the deck bottom row (30s)…")

    def _exit_rating() -> None:
        sd["rating_until"] = 0.0
        sd["rating_ep"] = None
        if sd["pad"] is not None:
            for k in RATING_KEYS:  # restore the finger-close labels
                sd["pad"].set_label(k, sd_labels[k])

    def _apply_rating(stars: int) -> None:
        ep = sd["rating_ep"]
        _exit_rating()
        if ep is None:
            return
        try:
            _write_annotation(session.root, ep, rating=int(stars))
            last_saved["rating"] = int(stars)
            log(f"rated episode {ep + 1}: {stars}★")
            _kick_upload(f"rating {stars}★")  # push the rating to the GPU
        except Exception as e:  # noqa: BLE001
            log(f"rating write failed: {e}")

    def _toggle_freedrive_key() -> None:
        # deck key 2 when idle: flip between teach mode and the prior driving input
        if io["mode"] == "freedrive":
            switch_mode(sd["prev_mode"])
        else:
            sd["prev_mode"] = io["mode"] if io["mode"] != "freedrive" else "spacemouse"
            switch_mode("freedrive")

    def _pump_streamdeck() -> None:
        moved = False
        while True:  # drain events pushed from the deck's callback thread
            try:
                key, down = sd["queue"].get_nowait()
            except _queue.Empty:
                break
            if sd["expo"]:                      # exposure mode owns row 1 until ✓ SET
                if down:
                    if key == 0:
                        _expo_nudge(session.exposure.STEP)
                    elif key == 1:
                        _expo_nudge(1.0 / session.exposure.STEP)
                    elif key == 2:
                        _expo_set()
                continue
            if sd["rating_until"] and key in RATING_KEYS:  # rating active -> bottom row = stars
                if down:
                    _apply_rating(RATING_KEYS.index(key) + 1)
                continue
            if key == 0:                       # ⚑ grasp mark (recording) | ● REC (idle)
                if down:
                    if session.recording:
                        log(session.regrasp_mark())
                    else:
                        session.start_episode()
            elif key in sd_actions:
                if down:
                    sd_actions[key]()          # fixed actions (Recon / Quit)
            elif key == 1:                     # Save (recording) | Reset-hold (idle)
                if session.recording:
                    if down:
                        do_save()
                else:
                    sd["reset_held"] = down
            elif key == 2:                     # Discard (recording) | Freedrive-toggle (idle)
                if session.recording:
                    if down:
                        do_discard()
                elif down:
                    _toggle_freedrive_key()
            elif key in sd_fingers:
                sd["held"][key] = down          # hold-to-ramp
        for key, held in sd["held"].items():
            if held:
                fi, sign = sd_fingers[key]
                delta = SM_GRASP_CLOSE_RATE if sign > 0 else -SM_GRASP_OPEN_RATE
                for f in (range(4) if fi < 0 else (fi,)):  # fi < 0 == all fingers (Open/Close All)
                    ctrl.nudge_finger(f, delta)
                moved = True
        if moved:
            grasp_var.set(ctrl.grasp)

    def _update_deck_status() -> None:
        # Keys 0/1/2 are dynamic: REC blinks red while recording; 1=Save(recording, flashes green on
        # save)/Reset(idle); 2=Discard(recording)/Freedrive(idle, blue when on). Re-render only on change.
        pad = sd["pad"]
        if pad is None:
            return
        import time
        if sd["rating_until"] and time.time() > sd["rating_until"]:  # rating window elapsed
            _exit_rating()
            log("rating timed out (30s) — deck restored")
        if sd["expo"]:
            return                     # exposure keys own row 1; don't fight them
        rec = session.recording
        on = int(time.time() * 1.5) % 2 == 0  # ~1.5 Hz blink phase
        # key 0 doubles as the re-grasp marker while recording (it cannot start an episode that is
        # already running); amber once a range is open so the operator can see it must be closed.
        # The label carries the running grasp count so the operator can confirm the mark registered.
        rec_bg = ((200, 30, 30) if on else (45, 0, 0)) if rec else (25, 25, 25)
        if rec and session._regrasp_open is not None:
            rec_bg = (200, 130, 0) if on else (60, 38, 0)
        k0 = ((f"END\n{session.grasps + 1}" if session._regrasp_open is not None
               else f"+\n{session.grasps}") if rec else "●")
        if (rec_bg, k0) != sd["rec_bg"]:
            pad.set_label(0, k0, bg=rec_bg, fg=(240, 240, 240) if rec else (225, 45, 45))
            sd["rec_bg"] = (rec_bg, k0)
        if rec:
            sbg = ((20, 170, 20) if on else (0, 55, 0)) if time.time() < sd["save_flash_until"] else (25, 25, 25)
            k1 = ("💾", sbg)
        else:
            k1 = ("⌂", (150, 112, 22))
        if k1 != sd["k1"]:
            pad.set_label(1, k1[0], bg=k1[1])
            sd["k1"] = k1
        if rec:
            k2 = ("🗑", (25, 25, 25))
        else:
            k2 = ("🆓", (30, 90, 175) if io["mode"] == "freedrive" else (60, 60, 64))   # freedrive toggle
        if k2 != sd["k2"]:
            pad.set_label(2, k2[0], bg=k2[1])
            sd["k2"] = k2

    period = max(1, int(1000 / fps))
    dt = 1.0 / fps

    def _clamp(v, lo, hi):
        return lo if v < lo else hi if v > hi else v

    def _apply_spacemouse() -> None:
        io["sm"].read()
        s = io["sm"].state()
        for axis in ("x", "y", "z"):
            src, sign = SM_MAP[axis]
            val = getattr(ctrl, axis) + SM_POS_GAIN * sign * s[src] * dt
            setattr(ctrl, axis, _clamp(val, *ranges[axis]))
        for axis in ("roll", "pitch", "yaw"):
            src, sign = SM_MAP[axis]
            val = getattr(ctrl, axis) + SM_ROT_GAIN * sign * s[src] * dt
            setattr(ctrl, axis, _clamp(val, -math.pi, math.pi))
        btn = s["buttons"]
        # BOTH buttons together = start / save the episode, so a whole take can be run from the
        # SpaceMouse without reaching for the deck. Edge-triggered, and while the chord is held the
        # per-button grasp actions are suppressed — otherwise open and close would fight each other.
        both = (len(btn) > max(SM_BTN_CLOSE, SM_BTN_OPEN)
                and btn[SM_BTN_CLOSE] and btn[SM_BTN_OPEN])
        if both and not sm_chord["down"]:
            if session.recording:
                log("spacemouse chord -> SAVE")
                do_save()
            else:
                log("spacemouse chord -> REC")
                session.start_episode()
        sm_chord["down"] = both
        if not both:
            if len(btn) > SM_BTN_CLOSE and btn[SM_BTN_CLOSE]:
                ctrl.grasp = min(1.0, ctrl.grasp + SM_GRASP_CLOSE_RATE)  # hold to close
            if len(btn) > SM_BTN_OPEN and btn[SM_BTN_OPEN]:
                ctrl.grasp = max(0.0, ctrl.grasp - SM_GRASP_OPEN_RATE)  # hold to open
        for k, var in sliders.items():
            var.set(getattr(ctrl, k))
        grasp_var.set(ctrl.grasp)

    def _apply_gamepad() -> None:
        io["gp"].read()
        s = io["gp"].state()

        def gv(src):  # gamepad value for a mapped source ("triggers" = RT up / LT down)
            return (s["rt"] - s["lt"]) if src == "triggers" else s[src]

        for axis in ("x", "y", "z"):
            src, sign = GP_MAP[axis]
            setattr(ctrl, axis, _clamp(getattr(ctrl, axis) + SM_POS_GAIN * sign * gv(src) * dt, *ranges[axis]))
        for axis in ("roll", "pitch"):
            src, sign = GP_MAP[axis]
            setattr(ctrl, axis, _clamp(getattr(ctrl, axis) + SM_ROT_GAIN * sign * gv(src) * dt, -math.pi, math.pi))
        if s["rb"]:
            ctrl.grasp = min(1.0, ctrl.grasp + SM_GRASP_CLOSE_RATE)  # RB closes
        if s["lb"]:
            ctrl.grasp = max(0.0, ctrl.grasp - SM_GRASP_OPEN_RATE)   # LB opens
        for k, var in sliders.items():
            var.set(getattr(ctrl, k))
        grasp_var.set(ctrl.grasp)

    def _apply_freedrive_grasp() -> None:
        # arm is hand-guided; only the grasp is commanded — from the SpaceMouse buttons (+ keyboard).
        if io["sm"] is not None:
            io["sm"].read()
            btn = io["sm"].state()["buttons"]
            if len(btn) > SM_BTN_CLOSE and btn[SM_BTN_CLOSE]:
                ctrl.grasp = min(1.0, ctrl.grasp + SM_GRASP_CLOSE_RATE)
            if len(btn) > SM_BTN_OPEN and btn[SM_BTN_OPEN]:
                ctrl.grasp = max(0.0, ctrl.grasp - SM_GRASP_OPEN_RATE)
        grasp_var.set(ctrl.grasp)

    def _grasp_key(delta):  # keyboard grasp (works in any mode); ignored while typing the task
        def handler(_event):
            if win.focus_get() is task_entry:
                return
            ctrl.grasp = _clamp(ctrl.grasp + delta, 0.0, 1.0)
            grasp_var.set(ctrl.grasp)
        return handler

    win.bind("<KeyPress-c>", _grasp_key(+0.08))            # c / ] : close a step (hold to repeat)
    win.bind("<KeyPress-bracketright>", _grasp_key(+0.08))
    win.bind("<KeyPress-o>", _grasp_key(-0.08))            # o / [ : open a step
    win.bind("<KeyPress-bracketleft>", _grasp_key(-0.08))

    # ---- browser UI (ur5e_lerobot/ui/webpanel.py): a view + command channel; this loop stays the
    # controller. With --cameras udp the page streams straight from the camera service; otherwise
    # (sim / local cameras) we hand it our own frames. Commands are drained once per tick, like the
    # Stream Deck's, so the robot is only ever touched from here.
    show_tk = ui in ("tk", "both")
    home_go = {"active": False}          # press-and-go return home from the page (deck Reset is hold-to-home)
    web_hold = {"dir": 0, "t": 0.0}      # page open/close held: ramp per tick until released or 0.6 s silent
    last_saved = {"ep": None, "rating": None, "notes": "", "upload": ""}
    if ui in ("web", "both"):
        try:
            from ..ui.webpanel import WebPanel, service_cam_urls
            cam_names = ["scene"] + (["side"] if session.has_side else []) + (["wrist"] if session.has_wrist else [])
            if (hw or {}).get("cameras") == "udp":
                cams = service_cam_urls(cam_names, (hw or {}).get("camera_host") or "127.0.0.1")
            else:
                cams = {n: None for n in cam_names}
                webh["frames"] = True
            w = WebPanel("teleop", os.path.basename(os.path.normpath(session.root)), port=web_port, cams=cams,
                         on_ready=lambda u: upload_q.put(f"browser UI: {u}"))   # late bind -> logged by tick()
            url = w.start()
            webh["w"] = w
            if not url:
                log(f"browser UI: port {web_port} is busy (another robot app still running?) — will keep "
                    f"trying every 3 s and announce the URL when it is free")
            try:
                from ..sensors.capture_meta import squares_of
                webh["crop"] = {k: v for k, v in squares_of(session.cams).items() if k in ("scene", "side")}
            except Exception:  # noqa: BLE001
                webh["crop"] = {}
            if engine == "hardware":
                try:  # passive controller event stream -> fault banner with the decoded C-code
                    from ..robot.ur_events import EventListener
                    webh["events"] = EventListener((hw or {}).get("robot_ip") or "192.168.11.21")
                    webh["events"].start()
                except Exception:  # noqa: BLE001
                    pass
            if url:
                log(f"browser UI: {url}")
                print(f"[web] open {url}", flush=True)
        except OSError as e:
            log(f"browser UI unavailable ({e}) — is another robot app already serving port {web_port}?")
        except Exception as e:  # noqa: BLE001
            log(f"browser UI failed: {e}")

    def _web_expo(action, cam=None) -> None:
        ex = session.exposure
        if ex is None:
            log("no exposure session")
        elif action in ("plus", "minus") and cam:
            if cam not in session.cams:
                log(f"no '{cam}' camera")
                return
            ex.nudge(1.25 if action == "plus" else 1 / 1.25, cam)
            log(f"exposure {cam} {'+' if action == 'plus' else '-'} -> {ex.summary()}")
        elif action == "plus":
            _expo_nudge(1.25)
        elif action == "minus":
            _expo_nudge(1 / 1.25)
        elif action == "lock":
            _expo_set()
        elif action == "auto":
            ex.auto_expose(session.width, session.height)
            log(f"exposure auto -> {ex.summary()}")

    def _web_crop(cam, at) -> None:
        if session.dataset is not None:
            log("crop is fixed once the dataset exists — start a new --root to change it")
            return
        from ..sensors.capture_meta import apply as _apply_sq, parse_square
        if cam not in session.cams:
            log(f"no '{cam}' camera in this session")
            return
        applied = _apply_sq(session.cams, {cam: parse_square(at)})
        webh["crop"] = {k: v for k, v in applied.items() if k in ("scene", "side")}
        log(f"crop {cam} -> {applied.get(cam)}")

    def _pump_web() -> None:
        w = webh["w"]
        if w is None:
            return
        for c in w.commands():
            name, a = c.get("name"), c.get("args") or {}
            try:
                if name == "rec":
                    session.start_episode()
                    log("web -> REC")
                elif name == "save":
                    do_save()
                elif name == "discard":
                    do_discard()
                elif name == "home":
                    if home_go["active"]:
                        home_go["active"] = False
                        log("home cancelled")
                    elif hp["q"] is None:
                        log("no home pose captured yet (Set Home first)")
                    elif session.recording or io["mode"] == "freedrive":
                        log("home: stop recording / leave freedrive first")
                    else:
                        home_go["active"] = True
                        log("returning to home…")
                elif name == "freedrive":
                    _toggle_freedrive_key()
                elif name == "reconnect":
                    do_reconnect()
                elif name == "recover_hand":
                    do_recover_hand()
                elif name == "grasp":
                    ctrl.grasp = _clamp(ctrl.grasp + float(a.get("delta", 0.1)), 0.0, 1.0)
                    grasp_var.set(ctrl.grasp)
                elif name == "grasp_hold":
                    web_hold["dir"] = max(-1, min(1, int(a.get("dir", 0))))
                    web_hold["t"] = _ptime.time()
                elif name == "grasp_mark":
                    log(session.regrasp_mark())
                elif name == "expo":
                    _web_expo(a.get("action"), a.get("cam"))
                elif name == "crop":
                    _web_crop(a.get("cam"), a.get("at"))
                elif name == "rating":
                    if sd["rating_ep"] is None and last_saved.get("ep") is not None:
                        sd["rating_ep"] = last_saved["ep"]
                    _apply_rating(int(a.get("stars", 0)))
                elif name == "notes":
                    ep = last_saved.get("ep")
                    text = (a.get("text") or "").strip()
                    if ep is None:
                        log("no saved episode to annotate yet")
                    else:
                        _write_annotation(session.root, ep, notes=text or None)
                        last_saved["notes"] = text
                        log(f"notes saved for episode {ep + 1}")
                        _kick_upload("notes")
                elif name == "task":
                    text = str(a.get("text", "")).strip()
                    if text:
                        task_var.set(text)
                        log(f"task: {text}")
                elif name == "dataset":
                    _web_switch_dataset(str(a.get("name", "")).strip())
                elif name == "input_mode":
                    m = a.get("mode")
                    if m in ("spacemouse", "gamepad", "freedrive", "sliders") and m != io["mode"]:
                        try:
                            mode_var.set(m)
                        except Exception:  # noqa: BLE001
                            pass
                        switch_mode(m)
                elif name == "quit":
                    on_close()
                else:
                    log(f"web: unknown command {name!r}")
            except Exception as e:  # noqa: BLE001 — a bad command must never stop the loop
                log(f"web command {name} failed: {e}")

    def _web_switch_dataset(name: str) -> None:
        """Point recording at another dataset root (same parent dir): an existing one is appended to,
        a new name is created on the next save. Refused while recording or while a save finalizes."""
        import re as _re
        if not name or not _re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]*", name):
            log(f"dataset: invalid name {name!r}")
            return
        if session.recording or session.save_in_flight:
            log("dataset: stop recording / wait for the save to finish first")
            return
        new_root = os.path.join(os.path.dirname(os.path.normpath(session.root)), name)
        if os.path.normpath(new_root) == os.path.normpath(session.root):
            return
        try:
            session.wait_save()
            session._stop_writer()
            if session.dataset is not None:
                try:
                    session.dataset.finalize()   # footers for the one we leave, so it stays loadable
                except Exception:  # noqa: BLE001
                    pass
            session.dataset = None
            session.root, session.repo_id = new_root, f"local/{name}"
            session.resume = True                # append if it exists, create if not
            session.episodes = 0
            webh["w"].title = name
            exists = os.path.exists(os.path.join(new_root, "meta", "info.json"))
            log(f"dataset -> {name} ({'existing, appending' if exists else 'new, created on first save'})")
        except Exception as e:  # noqa: BLE001
            log(f"dataset switch failed: {e}")

    def _servo_short(summary: str) -> str:
        import re as _re
        slow = _re.search(r"slow cycles (\d+)", summary or "")
        mx = _re.search(r"max cycle (\d+) ms", summary or "")
        if not slow:
            return "servo —"
        return f"servo {slow.group(1)} slow · max {mx.group(1) if mx else '?'} ms"

    def _web_state() -> None:
        w = webh["w"]
        if w is None:
            return
        arm = session.robot.arm
        try:
            pose = [round(float(v), 4) for v in arm.get_ee_pose()]
        except Exception:  # noqa: BLE001
            pose = []
        if tickn["i"] % 12 == 0:   # the heavier reads once a second
            try:
                webh["base"] = round(math.degrees(float(arm.get_joint_positions()[0])), 1)
            except Exception:  # noqa: BLE001
                pass
            try:
                webh["ready"] = bool(arm.is_ready()) if hasattr(arm, "is_ready") else True
            except Exception:  # noqa: BLE001
                webh["ready"] = False
            try:
                webh["servo"] = arm.stream_summary() if hasattr(arm, "stream_summary") else ""
            except Exception:  # noqa: BLE001
                webh["servo"] = ""
        fault = None
        ev = webh.get("events")
        if ev is not None and ev.safety_mode not in (None, "NORMAL", "REDUCED"):
            e = ev.last_safety_event()
            fault = e.text if e else ev.safety_mode
        ex = session.exposure
        servo = webh.get("servo", "")
        w.set_state({
            "mode": "teleop", "engine": engine, "input": io["mode"], "task": task_var.get(),
            "recording": session.recording, "frames": session.frames, "episodes": session.episodes,
            "grasps": session.grasps, "regrasp_open": session._regrasp_open is not None,
            "rec_s": session.frames / max(session.fps, 1), "save_in_flight": session.save_in_flight,
            "dataset": {"name": os.path.basename(os.path.normpath(session.root)), "root": session.root,
                        "created": session.dataset is not None,
                        "storage": "video" if session.use_videos else "png"},
            "ur": {"connected": True, "ready": webh.get("ready", True), "pose": pose,
                   "reach": round(math.hypot(pose[0], pose[1]), 3) if len(pose) >= 2 else None,
                   "base_deg": webh.get("base"), "fault": fault, "freedrive": io["mode"] == "freedrive",
                   "homing": bool(home_save["active"] or home_go["active"] or sd["reset_held"])},
            "hand": {"ok": session.hand_ok, "grasp": round(float(ctrl.grasp), 3)},
            "tick": {"ms": tickn.get("last_ms", 0.0), "max_ms": tickn.get("max_ms", 0.0), "hz": tickn.get("hz", 0.0)},
            "servo": {"short": _servo_short(servo), "slow": bool(servo) and "slow cycles 0" not in servo},
            "exposure": {"values": dict(ex.values) if ex is not None else {},
                         "locked": bool(ex.locked) if ex is not None else False, "phase": bool(sd["expo"]),
                         "medians": dict(webh.get("medians", {}))},
            "crop": webh.get("crop", {}), "crop_locked": session.dataset is not None,
            "last": dict(last_saved), "status": status.get(),
        })

    tickn = {"i": 0}

    def tick() -> None:
        # An exception anywhere in a tick used to end the loop silently: Tk never re-armed `after`, the
        # window stayed up and the arm just held. Log it and keep ticking instead; the operator sees
        # the message in the log pane / browser page and can act.
        _t_tick = _ptime.time()
        try:
            _tick_body(_t_tick)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            try:
                log(f"tick error: {e}")
            except Exception:  # noqa: BLE001
                pass
            win.after(max(5, period - int((_ptime.time() - _t_tick) * 1000)), tick)

    def _tick_body(_t_tick: float) -> None:
        while True:  # surface background GPU-sync results (worker threads can't touch Tk)
            try:
                _m = upload_q.get_nowait()
            except _queue.Empty:
                break
            log(_m)
            if "GPU sync" in _m or "finalized" in _m:
                last_saved["upload"] = _m
        # One-shot: start the exposure flow after the FIRST render, not at construction — the cameras
        # connect lazily on first read, so before that there is nothing to meter or read back.
        if not expo_started["done"]:
            expo_started["done"] = True
            try:
                _enter_exposure()
            except Exception as e:  # noqa: BLE001 — never block startup on this
                log(f"exposure setup skipped: {e}")
        _pump_streamdeck()  # drain deck key events (actions + per-finger ramp + reset/freedrive keys)
        _pump_web()         # ...and the browser page's commands, the same way
        if web_hold["dir"] and _ptime.time() - web_hold["t"] < 0.6:   # page open/close held -> ramp
            ctrl.grasp = _clamp(ctrl.grasp + (SM_GRASP_CLOSE_RATE if web_hold["dir"] > 0 else -SM_GRASP_OPEN_RATE), 0.0, 1.0)
            grasp_var.set(ctrl.grasp)
        elif web_hold["dir"]:
            web_hold["dir"] = 0
        _update_deck_status()  # dynamic deck labels: REC blink, Save/Reset, Discard/Freedrive
        mode = io["mode"]
        saving_home = home_save["active"]  # post-save return to home — NOT recorded (episode is closed)
        # Hold-to-home: deck Reset held while idle -> slow joint-space return to the captured home. Drives
        # the arm directly (servoJ) and re-bases the teleop target so releasing Reset doesn't jump.
        if home_go["active"] and (session.recording or mode == "freedrive" or hp["q"] is None):
            home_go["active"] = False   # conditions changed under a press-and-go return: drop it
        resetting = (not saving_home and (sd["reset_held"] or home_go["active"]) and not session.recording
                     and mode != "freedrive" and hp["q"] is not None)
        if saving_home:
            try:
                session.robot.arm.home_step(hp["q"], HOME_DQ)  # joint-space -> no Cartesian singularity
                _rebase_ctrl()  # track the teleop target to the arm so resuming stays at home, no jump-back
            except Exception as e:  # noqa: BLE001 — a sim arm may lack home_step
                log(f"auto-home unavailable ({e})")
                home_save["active"] = saving_home = False
        elif resetting:
            _done = session.robot.arm.home_step(hp["q"], HOME_DQ)
            _rebase_ctrl()
            if _done and home_go["active"]:
                home_go["active"] = False
                log("home reached")
        elif mode == "gamepad" and io["gp"] is not None:
            if io["gp_ok"]:
                _apply_gamepad()  # else: uncalibrated -> don't move the arm until 'Calibrate GP' is run
        elif mode == "spacemouse" and io["sm"] is not None:
            _apply_spacemouse()
        elif mode == "freedrive":
            _apply_freedrive_grasp()
        tickn["i"] += 1
        do_wrist = (tickn["i"] % 2 == 0) or saving_home  # render wrist preview at half rate when idle
        if saving_home or mode == "freedrive":  # track the arm's ACTUAL pose; grasp held through it
            pose = session.robot.arm.get_ee_pose()
            action_vec = Action(tuple(pose[:6]), ctrl.grasp_curls()).to_vector()
        else:
            action_vec = ctrl.action_vector()
        _t_step = _ptime.time()
        scene_img, wrist_img, side_img = session.step(
            action_vec, task_var.get(), record=session.recording, do_wrist=do_wrist,
            command_arm=(mode != "freedrive") and not resetting and not saving_home,
        )
        if saving_home:  # drive the return to completion; the episode is already saved
            home_save["steps"] += 1
            done = home_save["steps"] >= HOME_SAVE_MAX_STEPS
            try:
                q_now = session.robot.arm.get_joint_positions()
                done = done or max(abs(a - b) for a, b in zip(q_now, hp["q"])) < HOME_REACHED_TOL
            except Exception:  # noqa: BLE001
                done = True
            if done:
                home_save["active"] = False
                log("home reached")
        def _fit(img, cell, label):  # letterbox into the cell; identical logic -> equal display sizes
            # cv2.resize (GIL released, ~2 ms) instead of PIL's bicubic resize of a 1920x540 row
            # (~20 ms on the main thread every tick — the biggest single piece of the tick's work).
            import cv2
            cw, ch = cell.winfo_width(), cell.winfo_height()
            if cw > 20 and ch > 20:
                ih, iw = img.shape[:2]
                scale = min(cw / iw, ch / ih)
                size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            photo = ImageTk.PhotoImage(Image.fromarray(img))
            label.configure(image=photo)
            label.image = photo

        # both 3rd-person scene cams side by side in the top cell (scene left, side right); wrist below.
        # Crop the black letterbox/vignette borders so the fisheye circles fill the cells (preview only).
        _t_render = _ptime.time()
        if show_tk:
            top_img = _compose_row([scene_img, side_img])
            _fit(top_img, scene_cell, img_label)
            if wrist_label is not None and wrist_img is not None:
                _fit(_crop_black(wrist_img), wrist_cell, wrist_label)
        if webh["w"] is not None and tickn["i"] % 12 == 0:   # measured brightness per camera, once a second
            try:
                _np = np
                _m = {}
                for _n, _im in (("scene", scene_img), ("side", side_img), ("wrist", wrist_img)):
                    if _im is not None:
                        _g = _im[::4, ::4]
                        _nz = _g.max(axis=2) > 12
                        _m[_n] = int(_np.median(_g[_nz])) if _nz.any() else int(_np.median(_g))
                webh["medians"] = {**webh.get("medians", {}), **_m}
            except Exception:  # noqa: BLE001
                pass
        if webh["frames"] and webh["w"] is not None:   # sim / local cameras: the page streams from us
            webh["w"].set_frame("scene", scene_img)
            if side_img is not None:
                webh["w"].set_frame("side", side_img)
            if wrist_img is not None:
                webh["w"].set_frame("wrist", wrist_img)
        if session.recording:
            status.set(f"● REC  episode {session.episodes + 1}  frames={session.frames}")
        if not session.hand_ok:  # a hand send failed mid-loop -> power-cycle the tool + reconnect
            log("hand disconnected — auto-recovering (tool power-cycle)…")
            win.update_idletasks()
            log(session.recover_hand())
            session.hand_ok = True  # don't re-trigger every tick if it's still down
        # Tick timing: a tick over ~2 periods is a stutter the operator can feel. Log it (rate-limited)
        # with where the time went, plus the servo streamer's counters, so log/manual_panel.log explains
        # a slow stretch after the fact. A summary line also goes out every 60 s while driving.
        _now = _ptime.time()
        _tick_ms = (_now - _t_tick) * 1000
        tickn["max_ms"] = max(tickn.get("max_ms", 0.0), _tick_ms)
        tickn["last_ms"] = _tick_ms
        if "t_prev" in tickn:
            _dt = max(_now - tickn["t_prev"], 1e-3)
            tickn["hz"] = 0.9 * tickn.get("hz", 1.0 / _dt) + 0.1 / _dt
        tickn["t_prev"] = _now
        _web_state()
        if _tick_ms > 2.0 * period and _now - tickn.get("slow_warned", 0.0) > 2.0:
            tickn["slow_warned"] = _now
            tickn["slow"] = tickn.get("slow", 0) + 1
            log(f"slow tick {_tick_ms:.0f} ms (pre-step {(_t_step - _t_tick) * 1000:.0f}, step "
                f"{(_t_render - _t_step) * 1000:.0f}, render {(_now - _t_render) * 1000:.0f}) — "
                f"{getattr(session.robot.arm, 'stream_summary', lambda: '')()}")
        if _now - tickn.get("summary_at", _now) > 60.0 or "summary_at" not in tickn:
            if "summary_at" in tickn and mode != "freedrive":
                print(f"[tick] 60 s: max tick {tickn['max_ms']:.0f} ms, slow ticks {tickn.get('slow', 0)}; "
                      f"{getattr(session.robot.arm, 'stream_summary', lambda reset=False: '')(reset=True)}", flush=True)
            tickn["summary_at"] = _now
            tickn["max_ms"] = 0.0
            tickn["slow"] = 0
        # Schedule relative to this tick's START, not its end: `after(period)` at the end made the real
        # period = 83 ms + the tick's work (~30-40 ms) -> 8 ticks/s against a 12 fps dataset clock.
        win.after(max(5, period - int((_now - _t_tick) * 1000)), tick)

    win.protocol("WM_DELETE_WINDOW", on_close)  # window X button -> same clean shutdown as Quit
    # `supervisorctl stop vla_teleop` (SIGTERM) -> the same clean shutdown, on the Tk thread
    import signal as _signal

    def _on_sigterm(*_):
        win.after(0, on_close)
    try:
        _signal.signal(_signal.SIGTERM, _on_sigterm)
    except Exception:  # noqa: BLE001 — not the main thread / unsupported
        pass
    _maximize(win)  # start maximized (falls back to full-screen geometry where -zoomed is unsupported)
    if not show_tk:
        win.withdraw()  # --ui web: the browser page is the UI; Tk only schedules the loop
    win.after(period, tick)
    win.mainloop()


def selftest(root: str = "outputs/datasets/manual_selftest") -> None:
    """Headless: exercise the control + recording logic (kinematic engine for speed)."""
    import glob
    import math
    import shutil

    shutil.rmtree(root, ignore_errors=True)
    session = SimSession("kinematic", "local/manual_selftest", root, width=160, height=120)
    ctrl = ManualController(session.home)
    session.start_episode()
    for f in range(12):
        ctrl.x = ctrl.home[0] + 0.03 * math.sin(f / 3.0)
        ctrl.grasp = f / 11.0
        session.step(ctrl.action_vector(), "selftest grasp sweep", record=True)
    session.save_episode()
    session.finalize()  # write parquet footers (the X-button corruption fix) -> dataset is loadable

    meta = session.dataset.meta
    assert meta.total_episodes == 1 and meta.total_frames == 12, (meta.total_episodes, meta.total_frames)
    assert glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True), "no data parquet"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    rl = LeRobotDataset(repo_id="local/manual_selftest", root=root)  # must load from disk (footers written)
    assert rl.num_episodes == 1 and rl.num_frames == 12, (rl.num_episodes, rl.num_frames)
    print(f"selftest OK: {meta.total_episodes} ep, {meta.total_frames} frames, reloads from disk, "
          f"action_dim={len(ctrl.action_vector())}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Teleop control panel for the UR5e + AmazingHand sim.")
    ap.add_argument("--engine", choices=["mujoco", "kinematic", "hardware"], default="mujoco")
    ap.add_argument("--input", choices=["spacemouse", "sliders", "gamepad", "freedrive"],
                    default="spacemouse",
                    help="action source: spacemouse | gamepad (Xbox) | freedrive (hand-guide the real "
                         "arm; grasp via keys/SpaceMouse buttons) | sliders")
    ap.add_argument("--selftest", action="store_true", help="headless logic check, no GUI")
    ap.add_argument("--repo_id", default="local/ur5e_amazinghand_manual")
    ap.add_argument("--root", default="outputs/datasets/manual")
    ap.add_argument("--task", default="put the block on the green pad")
    ap.add_argument("--fps", type=int, default=12)  # ~27ms/render on Mac; 12fps fits 2 cams
    ap.add_argument("--width", type=int, default=960)   # 16:9; cameras capture native 1080p
    ap.add_argument("--height", type=int, default=540)  # SmolVLA resizes to 512² so 960x540 is ample
    ap.add_argument("--use-videos", dest="use_videos", action="store_true", default=True,
                    help="store episodes as MP4 (default since 2026-10-26; ~25x smaller than PNG frames, what every "
                         "dataset uses). Only applies when a root is CREATED")
    ap.add_argument("--png", dest="use_videos", action="store_false",
                    help="store episodes as PNG frames instead (not recommended: 374 MB per 20 s episode)")
    ap.add_argument("--grasp", choices=["pinch", "full"], default="pinch",
                    help="pinch = thumb + pointer only (default); full = whole-hand grasp")
    ap.add_argument("--resume", action="store_true",
                    help="append episodes into the dataset at --root (created on first run) instead "
                         "of a fresh folder per run; adopts that dataset's fps/image size")
    # engine=hardware only:
    ap.add_argument("--robot-ip", default="192.168.11.21", help="UR5e IP (engine=hardware)")
    ap.add_argument("--hand-host", default="192.168.11.117", help="AmazingHand controller IP (engine=hardware)")
    ap.add_argument("--scene-cam", default=None, help="scene camera: device index or /dev path "
                    "(engine=hardware; default = stable by-id path for the 4K cam)")
    ap.add_argument("--wrist-cam", default=None, help="wrist camera: device index or /dev path "
                    "(engine=hardware; default = stable by-id path for the ARC cam)")
    ap.add_argument("--side-cam", nargs="?", const=True, default=None,
                    help="record a 2nd scene camera (occlusion coverage): pass --side-cam alone to use "
                         "the default port (cameras.SIDE_CAM), or --side-cam <device/path> to override")
    ap.add_argument("--cameras", choices=["local", "udp"], default="local",
                    help="local: open the USB cameras here (exclusive). udp: subscribe to the camera "
                         "service (supervisor unit vla_cameras) so other consumers can watch too and "
                         "the locked exposure survives between launches")
    ap.add_argument("--camera-host", default="127.0.0.1",
                    help="host running the camera service's control port (--cameras udp)")
    ap.add_argument("--hand-transport", choices=["udp", "tcp"], default="udp",
                    help="udp (default): fire-and-forget setpoints, a lost packet costs one step; "
                         "needs the UDP-capable hand firmware (:8766). tcp: legacy :8765")
    ap.add_argument("--scene-square", default=None, metavar="AT",
                    help="scene camera square crop for a NEW dataset (0.0 top .. 1.0 bottom, 'none' = off); "
                         "applied to the camera service at launch and written to the dataset's "
                         "meta/capture.json. --resume ignores this and applies the dataset's own value")
    ap.add_argument("--side-square", default=None, metavar="AT", help="same for the side camera")
    ap.add_argument("--tool-voltage", type=int, default=12, choices=[0, 12, 24],
                    help="UR tool output voltage powering the AmazingHand ESP32 (engine=hardware)")
    ap.add_argument("--no-ur-power", action="store_true",
                    help="don't power the UR up on launch / off on quit (dashboard :29999; engine=hardware)")
    ap.add_argument("--ur-shutdown", action="store_true",
                    help="on quit also shut the UR controller down (it then needs switching on by hand)")
    ap.add_argument("--gpu-host", default="bryan@GPU",
                    help="rsync target (user@host) for auto-uploading saved episodes to the GPU. 'GPU' is an "
                         "ssh alias (~/.ssh/config on this machine -> the GPU box's robot-subnet address), so "
                         "the address lives in one place; the old default was a DHCP hostname that moved and "
                         "silently broke auto-upload.")
    ap.add_argument("--gpu-datasets-dir", default="~/VLA/outputs/datasets",
                    help="datasets dir on the GPU host (the dataset folder syncs under it)")
    ap.add_argument("--no-gpu-sync", action="store_true",
                    help="disable auto-upload of saved episodes / ratings to the GPU")
    ap.add_argument("--no-home-on-save", action="store_true",
                    help="disable the joint-space auto-return-to-home on Save (save immediately instead)")
    ap.add_argument("--expose-manual", action="store_true",
                    help="after auto-exposing, hand deck row 1 to + / - / SET for a manual trim instead "
                         "of locking the metered value straight away (the default)")
    ap.add_argument("--operator", default="",
                    help="operator name tagged on every recorded episode (shows in the dataset manager)")
    ap.add_argument("--ui", choices=["web", "tk", "both"], default="web",
                    help="web: browser page (http://<this host>:8090) with the Tk window hidden; tk: the "
                         "classic Tk window; both: Tk window + browser page")
    ap.add_argument("--web-port", type=int, default=8090, help="port for the browser page (--ui web/both)")
    args = ap.parse_args()

    # Everything printed also goes to log/manual_panel.log (append): the terminal scrolls away and
    # diagnosing a stutter / fault after the fact needs the [servo], [crop], [cam] lines.
    try:
        import sys as _sys
        import time as _t
        os.makedirs("log", exist_ok=True)
        _lf = open(os.path.join("log", "manual_panel.log"), "a", buffering=1)
        _lf.write(f"\n=== {_t.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()} {' '.join(_sys.argv)} ===\n")

        class _Tee:
            def __init__(self, *streams):
                self._streams = streams

            def write(self, data):
                for st in self._streams:
                    try:
                        st.write(data)
                    except Exception:  # noqa: BLE001
                        pass
                return len(data)

            def flush(self):
                for st in self._streams:
                    try:
                        st.flush()
                    except Exception:  # noqa: BLE001
                        pass

            def isatty(self):
                return False

        _sys.stdout = _Tee(_sys.__stdout__, _lf)
        _sys.stderr = _Tee(_sys.__stderr__, _lf)
    except Exception as e:  # noqa: BLE001 — never block the panel on logging
        print(f"[log] file logging unavailable: {e}")

    if args.selftest:
        selftest()
    else:
        hw = {"robot_ip": args.robot_ip, "hand_host": args.hand_host,
              "scene_cam": args.scene_cam, "wrist_cam": args.wrist_cam, "side_cam": args.side_cam,
              "cameras": args.cameras, "camera_host": args.camera_host,
              "hand_transport": args.hand_transport,
              "squares": {"scene": args.scene_square, "side": args.side_square}}
        run_gui(args.engine, args.repo_id, args.root, args.task, args.fps, args.width, args.height,
                input_mode=args.input, use_videos=args.use_videos, grasp_mode=args.grasp,
                resume=args.resume, hw=hw, tool_voltage=args.tool_voltage,
                gpu_host=args.gpu_host, gpu_datasets_dir=args.gpu_datasets_dir,
                gpu_sync=not args.no_gpu_sync, home_on_save=not args.no_home_on_save,
                operator=args.operator, expose_manual=args.expose_manual,
                ur_power=not args.no_ur_power, ur_shutdown=args.ur_shutdown,
                ui=args.ui, web_port=args.web_port)


if __name__ == "__main__":
    main()
