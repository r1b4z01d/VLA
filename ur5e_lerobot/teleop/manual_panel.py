"""Tkinter manual control panel — collect sim demos WITHOUT a SpaceMouse.

Sliders set the UR5e EE-pose target (x,y,z,roll,pitch,yaw) and a hand grasp [0,1];
a live MuJoCo render shows the arm + hand. Buttons record episodes into a LeRobotDataset.
A task text field is captured per frame (used later by the language-conditioned VLA).

    .venv/bin/python -m ur5e_lerobot.teleop.manual_panel                 # GUI, MuJoCo arm+hand
    .venv/bin/python -m ur5e_lerobot.teleop.manual_panel --engine kinematic
    .venv/bin/python -m ur5e_lerobot.teleop.manual_panel --selftest      # headless check

When the SpaceMouse arrives it replaces these sliders as the action source — same robot
adapter, same schema, same dataset.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
from scipy.spatial.transform import Rotation

from ..schema import ACTION_NAMES, ARM_POSE_NAMES, STATE_NAMES, Action
from ..sim.record_sim import make_engine

SLIDER_ORDER = ("x", "y", "z", "roll", "pitch", "yaw")
# Position sliders span home ± delta (z wide so the EE can reach the floor).
POS_DELTAS = {"x": 0.40, "y": 0.30, "z": 0.45}

# SpaceMouse teleop gains (per second at full deflection); tune to taste.
SM_POS_GAIN = 0.131  # m/s  (was 0.175; -25%)
SM_ROT_GAIN = 0.45   # rad/s (was 0.6;   -25%)
SM_GRASP_CLOSE_RATE = 0.052  # grasp close per tick while the close button is held (2x faster)
SM_GRASP_OPEN_RATE = 0.030   # grasp open per tick while the open button is held (2x faster)
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
    "roll": ("pitch", +1),
    "pitch": ("roll", -1),
    "yaw": ("yaw", +1),
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
        self.robot, self._render_fn, self._wrist_fn = make_engine(engine, **(hw or {}))
        self.has_wrist = self._wrist_fn is not None
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
        self.disp_w, self.disp_h = 640, 480  # render the scene crisp (render cost is size-independent)
        self.dataset = None
        self.recording = False
        self.episodes = 0
        self.frames = 0

    def _connect_robot(self) -> None:
        try:
            self.robot.connect()
        except Exception as e:  # on hardware the hand is tool-powered — power-cycle it + retry once
            if self._engine != "hardware" or not self._robot_ip:
                raise
            print(f"[connect] {e} -> power-cycling the UR tool + retrying the hand")
            try:
                self.robot.disconnect()
            except Exception:  # noqa: BLE001
                pass
            _power_cycle_tool(self._robot_ip, self._tool_voltage)
            import time
            time.sleep(6)  # ESP32 reboot + WiFi rejoin
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
            print(f"[resume] appending to {self.root} ({self.dataset.num_episodes} episodes, "
                  f"{self.width}x{self.height} @ {self.fps}fps)")
        else:
            from ..sim.record_sim import build_features

            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                features=build_features(self.width, self.height, use_videos=self.use_videos, with_wrist=self.has_wrist),
                root=self.root,
                robot_type=self.robot.name,
                use_videos=self.use_videos,
            )

    def render(self) -> np.ndarray:
        return self._render_fn(self.disp_w, self.disp_h)  # crisp scene for display

    def render_wrist(self):
        return self._wrist_fn(self.width, self.height) if self.has_wrist else None

    def step(self, action_vec, task: str, record: bool, do_wrist: bool = True, command_arm: bool = True):
        """observe (scene + wrist) -> record -> act. Returns (scene_img, wrist_img|None).

        Scene is rendered crisp (disp size) for the display; the recorded frame is
        downsampled to the dataset size. do_wrist=False skips the wrist render this tick for a
        snappier loop; when recording we always render it so every frame has both cameras.
        command_arm=False (freedrive): the arm is hand-guided, so only the hand (grasp) is driven —
        the recorded action's arm pose is the actual pose the human moved through.
        """
        scene = self.render()  # disp_w x disp_h (crisp)
        wrist = self.render_wrist() if (self.has_wrist and (record or do_wrist)) else None
        if record:
            import cv2

            self._ensure_dataset()
            obs = self.robot.get_observation()
            state = np.array([obs[n] for n in STATE_NAMES], dtype=np.float32)
            scene_rec = cv2.resize(scene, (self.width, self.height))  # downsample to dataset size
            frame = {
                "observation.state": state,
                "observation.images.scene": scene_rec,
                "action": np.asarray(action_vec, dtype=np.float32),
                "task": task,
            }
            if wrist is not None:
                frame["observation.images.wrist"] = wrist
            self.dataset.add_frame(frame)
            self.frames += 1
        try:
            if command_arm:
                self.robot.send_action(dict(zip(ACTION_NAMES, action_vec)))  # advance arm + hand
            else:  # freedrive: arm is hand-guided; drive only the hand from the action's curls
                self.robot.hand.send_curls(list(action_vec[len(ARM_POSE_NAMES):]))
        except Exception as e:  # noqa: BLE001 — a flaky (tool-powered) hand drop mustn't crash the loop
            self.hand_ok = False
            self._send_err = str(e)
        return scene, wrist

    def start_episode(self) -> None:
        self._ensure_dataset()
        self.recording = True
        self.frames = 0

    def save_episode(self) -> None:
        if self.recording and self.dataset is not None and self.frames > 0:
            self.dataset.save_episode()
            self.episodes += 1
        self.recording = False

    def discard_episode(self) -> None:
        if self.dataset is not None:
            self.dataset.episode_buffer = self.dataset.create_episode_buffer()
        self.recording = False
        self.frames = 0

    def finalize(self) -> None:
        """Flush episode metadata + write the parquet footers. MUST run before exit, or the last
        data/episodes parquet is left open (no footer) -> the dataset is corrupt and won't load.
        LeRobot only does this in finalize(); its __del__ closes the data writer but NOT the
        metadata writer, and a hard window close (X) may skip __del__ entirely."""
        if self.dataset is not None:
            self.dataset.finalize()

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


def run_gui(engine, repo_id, root, task_default, fps, width, height, input_mode="sliders", use_videos=False, grasp_mode="pinch", resume=False, hw=None, tool_voltage=TOOL_VOLTAGE) -> None:
    import tkinter as tk

    from PIL import Image, ImageTk

    # resume -> record into the exact root (create it once, append every run after); else a fresh folder
    dataset_root = root if resume else _unique_root(root)
    session = SimSession(engine, repo_id, dataset_root, fps, width, height, use_videos=use_videos, resume=resume, hw=hw, tool_voltage=tool_voltage)
    ctrl = ManualController(session.home, grasp_mode=grasp_mode)  # arm's natural keyframe home, grasp 0
    ranges = ctrl.ranges()

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
    tk.Label(scene_cell, text="scene", fg="#888", bg="black").place(x=4, y=2)
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

    def log(msg: str) -> None:
        """Set the live status line + append a timestamped line to the bottom log panel."""
        import time
        status.set(msg)
        log_text.configure(state="normal")
        log_text.insert("end", f"{time.strftime('%H:%M:%S')}  {msg}\n")
        log_text.see("end")
        log_text.configure(state="disabled")

    log(sm_msg or "ready — Start rec to record")

    def do_reset() -> None:
        ctrl.reset()
        session.reset_scene()  # sim: block+arm to home. hardware: reconnect arm/hand.
        if io["mode"] == "freedrive":
            session.robot.arm.start_freedrive()  # reconnect dropped teach mode -> re-enter it
        for k, var in sliders.items():
            var.set(getattr(ctrl, k))
        grasp_var.set(ctrl.grasp)

    def do_save() -> None:
        import time
        session.save_episode()
        sd["save_flash_until"] = time.time() + 1.5  # flash the deck Save key to confirm
        log(f"saved episode {session.episodes} -> {session.root}")

    def do_discard() -> None:
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
        """Clean shutdown for BOTH the Quit button and the window's X — drop any unsaved in-progress
        episode, then finalize the dataset so parquet footers are written (otherwise it corrupts)."""
        try:
            _teardown_input()  # close input devices + leave freedrive (re-lock the arm)
            if sd["pad"] is not None:
                sd["pad"].close()  # blank + release the Stream Deck
            if session.recording:
                session.discard_episode()  # an unfinished (un-Saved) demo -> drop it
            session.finalize()
        except Exception as e:  # noqa: BLE001 — never let cleanup block the window from closing
            print(f"[close] finalize warning: {e}")
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
        ("● Start rec", session.start_episode),
        ("■ Save ep", do_save),
        ("Discard", do_discard),
        ("🎮 Calibrate GP", do_calibrate_gamepad),
        ("⟳ Reconnect UR", do_reconnect),
        ("🖐 Recover Hand", do_recover_hand),
        ("Quit", on_close),
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
    task_entry = tk.Entry(topinfo, textvariable=task_var, width=22)
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
    sd_actions = {0: session.start_episode, 1: do_save, 2: do_discard, 3: do_reconnect, 4: on_close}
    sd_action_labels = {0: "● REC", 1: "■ Save", 2: "Discard", 3: "⟳ Recon", 4: "Quit"}
    sd_fingers = {5: (0, -1), 6: (1, -1), 7: (2, -1), 8: (3, -1), 9: (-1, -1),
                  10: (0, +1), 11: (1, +1), 12: (2, +1), 13: (3, +1), 14: (-1, +1)}
    sd_labels = {5: "idx\nopen", 6: "mid\nopen", 7: "rng\nopen", 8: "thb\nopen", 9: "OPEN\nALL",
                 10: "idx\nclose", 11: "mid\nclose", 12: "rng\nclose", 13: "thb\nclose", 14: "CLOSE\nALL"}
    sd = {"pad": None, "queue": _queue.Queue(), "held": {},
          "rec_bg": None, "save_bg": None, "save_flash_until": 0.0}  # deck key-blink state
    try:
        from .streamdeck import StreamDeckPad

        _pad = StreamDeckPad(flip=True).open()  # deck is mounted upside down
        sd["pad"] = _pad
        for _k, _lab in {**sd_action_labels, **sd_labels}.items():
            if _k < _pad.key_count:
                _pad.set_label(_k, _lab)
        _pad.on_key(lambda key, down: sd["queue"].put((key, down)))  # marshal to the Tk loop
        log(f"Stream Deck ready ({_pad.key_count} keys, flipped): actions | open row | close row")
    except Exception as e:  # noqa: BLE001 — the panel runs fine without a deck
        log(f"Stream Deck not available ({e})")

    def _pump_streamdeck() -> None:
        moved = False
        while True:  # drain events pushed from the deck's callback thread
            try:
                key, down = sd["queue"].get_nowait()
            except _queue.Empty:
                break
            if key in sd_actions:
                if down:
                    sd_actions[key]()          # fire the action on press
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
        # Blink REC (key 0) while recording; flash Save (key 1) briefly after a save. Re-render a key
        # only when its background actually changes (cheap).
        pad = sd["pad"]
        if pad is None:
            return
        import time
        on = int(time.time() * 1.5) % 2 == 0  # ~1.5 Hz blink phase
        rec_bg = ((200, 30, 30) if on else (45, 0, 0)) if session.recording else (25, 25, 25)
        if rec_bg != sd["rec_bg"]:
            pad.set_label(0, "● REC", bg=rec_bg)
            sd["rec_bg"] = rec_bg
        save_bg = ((20, 170, 20) if on else (0, 55, 0)) if time.time() < sd["save_flash_until"] else (25, 25, 25)
        if save_bg != sd["save_bg"]:
            pad.set_label(1, "■ Save", bg=save_bg)
            sd["save_bg"] = save_bg

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

    tickn = {"i": 0}

    def tick() -> None:
        _pump_streamdeck()  # drain deck key events (actions + per-finger ramp)
        _update_deck_status()  # blink REC while recording / flash Save after a save
        mode = io["mode"]
        if mode == "gamepad" and io["gp"] is not None:
            if io["gp_ok"]:
                _apply_gamepad()  # else: uncalibrated -> don't move the arm until 'Calibrate GP' is run
        elif mode == "spacemouse" and io["sm"] is not None:
            _apply_spacemouse()
        elif mode == "freedrive":
            _apply_freedrive_grasp()
        tickn["i"] += 1
        do_wrist = (tickn["i"] % 2 == 0)  # render wrist preview at half rate when idle
        if mode == "freedrive":  # arm is hand-guided: the recorded arm action IS the actual pose
            pose = session.robot.arm.get_ee_pose()
            action_vec = Action(tuple(pose[:6]), ctrl.grasp_curls()).to_vector()
        else:
            action_vec = ctrl.action_vector()
        scene_img, wrist_img = session.step(
            action_vec, task_var.get(), record=session.recording, do_wrist=do_wrist,
            command_arm=(mode != "freedrive"),
        )
        def _fit(img, cell, label):  # letterbox into the cell; identical logic -> equal display sizes
            pil = Image.fromarray(img)
            cw, ch = cell.winfo_width(), cell.winfo_height()
            if cw > 20 and ch > 20:
                iw, ih = pil.size
                scale = min(cw / iw, ch / ih)
                pil = pil.resize((max(1, int(iw * scale)), max(1, int(ih * scale))))
            photo = ImageTk.PhotoImage(pil)
            label.configure(image=photo)
            label.image = photo

        _fit(scene_img, scene_cell, img_label)
        if wrist_label is not None and wrist_img is not None:
            _fit(wrist_img, wrist_cell, wrist_label)
        if session.recording:
            status.set(f"● REC  episode {session.episodes + 1}  frames={session.frames}")
        if not session.hand_ok:  # a hand send failed mid-loop -> power-cycle the tool + reconnect
            log("hand disconnected — auto-recovering (tool power-cycle)…")
            win.update_idletasks()
            log(session.recover_hand())
            session.hand_ok = True  # don't re-trigger every tick if it's still down
        win.after(period, tick)

    win.protocol("WM_DELETE_WINDOW", on_close)  # window X button -> same clean shutdown as Quit
    _maximize(win)  # start maximized (falls back to full-screen geometry where -zoomed is unsupported)
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
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--use-videos", action="store_true",
                    help="encode cameras as video, not images (needs system ffmpeg; "
                         "use on the Linux box so training-time loading is fast)")
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
    ap.add_argument("--tool-voltage", type=int, default=12, choices=[0, 12, 24],
                    help="UR tool output voltage powering the AmazingHand ESP32 (engine=hardware)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
    else:
        hw = {"robot_ip": args.robot_ip, "hand_host": args.hand_host,
              "scene_cam": args.scene_cam, "wrist_cam": args.wrist_cam}
        run_gui(args.engine, args.repo_id, args.root, args.task, args.fps, args.width, args.height,
                input_mode=args.input, use_videos=args.use_videos, grasp_mode=args.grasp,
                resume=args.resume, hw=hw, tool_voltage=args.tool_voltage)


if __name__ == "__main__":
    main()
