"""AmazingHand MuJoCo sim — drive the hand by 4 per-finger curls and render offscreen.

Model vendored from r1b4z01d/rd_ws (AHSimulation, right hand) under
``assets/amazing_hand/AH_Right``. It exposes 8 position actuators
``finger{1..4}_motor{1,2}`` (ctrlrange ±pi/2). Finger order matches the project schema:
finger1=index, finger2=middle, finger3=ring, finger4=thumb.

Curl in [0,1]: 0 = open (keyframe "zero"), 1 = closed. The fingers are a parallel
2-motor linkage; **flexion = opposite-sign motor pair**, so we drive
``motor1 = +curl*amp``, ``motor2 = -curl*amp``.

NOTE: this maps curl→motors using the MuJoCo model's own zero (open) convention, which
is NOT the same zero as the hardware `J:` TCP protocol (see amazing_hand_client). Both
are valid "curl" abstractions; aligning the two numerically is a later calibration step.
"""
from __future__ import annotations

import math
import os

import mujoco

_HERE = os.path.dirname(__file__)
DEFAULT_MODEL = os.path.normpath(
    os.path.join(_HERE, "..", "..", "assets", "amazing_hand", "AH_Right", "scene.xml")
)

FLEX_AMP = 1.4  # kept for back-compat; mapping now uses the open/close tables
N_FINGERS = 4

# Per-servo offsets (deg) at open (curl 0) and closed (curl 1), ordered
# finger{idx//2+1}_motor{idx%2+1} (index, middle, ring, thumb).
#
# Each finger is a 2-motor parallel mechanism. Firmware (HAND_CTRL.h) convention:
#     flexion f, abduction a  ->  motor1 = f + a,  motor2 = -f + a
# (open = flex -35/abd 0 -> (-35,+35); close = flex +90/abd 0 -> (+90,-90)). We close ALL four
# fingers with pure flexion (abduction 0, within the real +/-20 deg limit): fingers 1-3 reach
# x~0.056 and the thumb reaches x~0.011 (hand frame), straddling the grasp pocket — a physically
# correct envelope around the block. (The block is HELD by the grasp aid, not finger friction; the
# hand can't form a stable physics grasp — see combined_model._add_grasp_weld. An earlier hack
# closed the thumb to (90,30) = 60 deg abduction, 3x past the real limit; removed.)
SIM_OPEN_OFFSETS = (-35.0, 35.0, -35.0, 35.0, -35.0, 35.0, -35.0, 35.0)
SIM_CLOSE_OFFSETS = (90.0, -90.0, 90.0, -90.0, 90.0, -90.0, 90.0, -90.0)


def curls_to_motor_radians(curls) -> list[float]:
    """8 motor targets (rad) from 4 per-finger curls, using the sim-calibrated open/close table.

    curl 0 = open (fingers splayed to -35/+35 deg), curl 1 = closed. The thumb closes to the
    opposing pose (90, 30) so the pinch actually meets the fingertips (see SIM_CLOSE_OFFSETS).
    """
    seq = list(curls)
    if len(seq) != N_FINGERS:
        raise ValueError(f"expected {N_FINGERS} curls, got {len(seq)}")
    out = []
    for fi in range(N_FINGERS):
        c = 0.0 if seq[fi] < 0.0 else 1.0 if seq[fi] > 1.0 else float(seq[fi])
        for k in (0, 1):
            o, cl = SIM_OPEN_OFFSETS[2 * fi + k], SIM_CLOSE_OFFSETS[2 * fi + k]
            out.append(math.radians(o + (cl - o) * c))
    return out


class AmazingHandMujoco:
    def __init__(self, model_path: str = DEFAULT_MODEL, flex_amp: float = FLEX_AMP):
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(model_path))
        self.data = mujoco.MjData(self.model)
        self.flex_amp = flex_amp
        self._act = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(self.model.nu)
        }
        self._renderer: mujoco.Renderer | None = None
        self.reset()

    def reset(self) -> None:
        """Open hand (keyframe 'zero')."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

    def set_curls(self, curls) -> None:
        if len(list(curls)) != N_FINGERS:
            raise ValueError(f"expected {N_FINGERS} curls, got {len(list(curls))}")
        for idx, rad in enumerate(curls_to_motor_radians(curls)):
            fi, mj = idx // 2 + 1, idx % 2 + 1
            self.data.ctrl[self._act[f"finger{fi}_motor{mj}"]] = rad

    def settle(self, steps: int = 600) -> None:
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def _default_camera(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 150.0, -20.0, 0.32
        cam.lookat[:] = [0.03, 0.0, 0.10]
        return cam

    def render(self, width: int = 640, height: int = 480, cam: mujoco.MjvCamera | None = None):
        width = max(16, min(int(width), int(self.model.vis.global_.offwidth)))
        height = max(16, min(int(height), int(self.model.vis.global_.offheight)))
        if self._renderer is None or self._renderer.width != width or self._renderer.height != height:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=cam or self._default_camera())
        return self._renderer.render()

    def save_frame(self, path: str, **render_kw) -> None:
        from PIL import Image

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        Image.fromarray(self.render(**render_kw)).save(path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Render AmazingHand open & closed poses.")
    p.add_argument("--outdir", default="outputs/sim")
    p.add_argument("--amp", type=float, default=FLEX_AMP)
    args = p.parse_args()

    hand = AmazingHandMujoco(flex_amp=args.amp)

    hand.reset()
    hand.settle(300)
    hand.save_frame(f"{args.outdir}/hand_open.png")

    hand.set_curls([0.5, 0.5, 0.5, 0.5])
    hand.settle(600)
    hand.save_frame(f"{args.outdir}/hand_half.png")

    hand.set_curls([1.0, 1.0, 1.0, 1.0])
    hand.settle(600)
    hand.save_frame(f"{args.outdir}/hand_closed.png")

    print(f"saved open/half/closed frames to {args.outdir}/")
