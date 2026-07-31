"""Combined UR5e + AmazingHand MuJoCo cell — the real sim backend for the adapter.

One MuJoCo scene with the arm (driven to EE-pose targets via mink IK) and the hand
(driven by 4 curls). `CellArm` / `CellHand` expose it through the same ArmInterface /
hand-client contracts URAmazingHand expects, so the adapter records through the real sim.

EE-pose state (`get_ee_pose`) is read from the *physical* flange site after stepping, so
recorded (state, action) pairs are not the trivial identity the kinematic SimArm gives.
"""
from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..robot.arm_interface import ArmInterface
from ..robot.workspace import clamp_out_of_nogo
from .amazing_hand_mujoco import curls_to_motor_radians
from .combined_model import (BLOCK_POS, BLOCK_SIZE, EE_SITE, GRASP_CENTER_LOCAL, GRASP_WELD,
                             HAND_PREFIX, HAND_ROOT_BODY, TARGET_POS, WRIST_CAM, build_combined_model)
from .mujoco_arm import MujocoArm

ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")

# Grasp-aid (attach-on-close) thresholds. The hand can't form a stable physics grasp (see
# combined_model._add_grasp_weld), so when the commanded grasp closes (mean curl past CLOSE) with
# the block inside CAPTURE_R of the grasp center, weld the block to the hand in place; release it
# when the grasp opens (mean curl below OPEN). Hysteresis (CLOSE > OPEN) avoids chatter.
GRASP_CLOSE = 0.55
GRASP_OPEN = 0.35
GRASP_CAPTURE_R = 0.09  # m: how close the block must be to the grasp center to be picked up

# Scene (free) camera used by render() — single source of truth so FOV tests match what's drawn.
SCENE_CAM = {"azimuth": 135.0, "elevation": -25.0, "distance": 1.3, "lookat": (0.0, 0.35, 0.2)}

# Randomization region for the box/goal: a floor rectangle near the known-reachable pick/place,
# further clamped to the scene camera's FOV so both stay in frame (see _in_scene_fov / _sample_xy).
RAND_X = (0.14, 0.34)
RAND_Y = (0.20, 0.70)
RAND_MIN_SEP = 0.12   # keep box and goal apart so the task stays non-trivial
FOV_MARGIN = 0.82     # stay within this fraction of the half-FOV (leaves a visible border)


class MujocoCell:
    def __init__(self, ik_iters: int = 80, settle_steps: int = 8):
        self.model = build_combined_model()
        self.data = mujoco.MjData(self.model)
        # ik_iters high + early-break => the IK converges to the target each tick (tight
        # tracking, no coast); it stops early once converged so it's still cheap.
        self._ik = MujocoArm(ik_iters=ik_iters)  # arm-only model, for IK + EE target
        self.settle_steps = settle_steps

        def aid(name: str) -> int:
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if i < 0:
                raise ValueError(f"actuator {name!r} not found in combined model")
            return i

        self._arm_aid = [aid(n) for n in ARM_ACTUATORS]
        self._hand_aid = {
            (fi, mj): aid(f"{HAND_PREFIX}finger{fi}_motor{mj}") for fi in (1, 2, 3, 4) for mj in (1, 2)
        }
        self._site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

        # grasp aid: weld id + the bodies/pocket it uses (all may be absent if the model has no block)
        self._weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, GRASP_WELD)
        self._hand_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, HAND_PREFIX + HAND_ROOT_BODY)
        self._block_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self._grasp_center = np.asarray(GRASP_CENTER_LOCAL, dtype=float)
        self._grasp_level = 0.0

        # scene randomization: block freejoint + block/goal geoms, the FOV camera basis, and the
        # intended box/goal/color config (restored on reset so randomization survives a scene reset)
        self._block_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "block")
        self._goal_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target")
        self._block_qadr = self._block_vadr = -1
        if self._block_body >= 0:
            bj = self.model.body_jntadr[self._block_body]
            self._block_qadr = int(self.model.jnt_qposadr[bj])
            self._block_vadr = int(self.model.jnt_dofadr[bj])
        self._rng = np.random.default_rng()
        self._cam_pos, self._cam_fwd, self._cam_right, self._cam_up = self._scene_cam_pose()
        self._block_z, self._goal_z = float(BLOCK_SIZE[2]), float(TARGET_POS[2])
        self._block_xy = (float(BLOCK_POS[0]), float(BLOCK_POS[1]))
        self._block_yaw = 0.0  # box yaw about vertical (rad); randomized by randomize_block
        self._goal_xy = (float(TARGET_POS[0]), float(TARGET_POS[1]))
        self._block_rgba = None  # None -> keep the model's default block color
        self.reset()

    def reset(self) -> None:
        self._ik.reset()
        mujoco.mj_resetData(self.model, self.data)
        q6 = np.asarray(self._ik.get_joint_positions())
        self.data.qpos[:6] = q6
        self._write_arm_ctrl(q6)
        self._grasp_level = 0.0
        if self._weld_id >= 0:
            self.data.eq_active[self._weld_id] = 0  # start ungrasped
        self._apply_scene()  # restore the intended box/goal/color (survives a scene reset)
        mujoco.mj_forward(self.model, self.data)

    # --- scene randomization (box/goal placement + box color) -----------------------------------
    def _scene_cam_pose(self):
        """Position + orthonormal basis (forward/right/up) of the scene render camera."""
        c = SCENE_CAM
        az, el = np.radians(c["azimuth"]), np.radians(c["elevation"])
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])  # cam -> lookat
        pos = np.asarray(c["lookat"], float) - c["distance"] * fwd
        right = np.cross(fwd, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd); up /= np.linalg.norm(up)
        return pos, fwd, right, up

    def _in_scene_fov(self, xyz, aspect: float = 4.0 / 3.0, margin: float = FOV_MARGIN) -> bool:
        """True if a world point projects inside the scene camera's frame (with a margin)."""
        rel = np.asarray(xyz, float) - self._cam_pos
        depth = float(rel @ self._cam_fwd)
        if depth <= 0:
            return False
        tan_v = np.tan(np.radians(float(self.model.vis.global_.fovy)) / 2.0)
        ny = (rel @ self._cam_up) / (depth * tan_v)
        nx = (rel @ self._cam_right) / (depth * tan_v * aspect)
        return abs(nx) <= margin and abs(ny) <= margin

    def _sample_xy(self, z: float, avoid=None):
        """A random (x,y) in the region that is in-FOV and >= RAND_MIN_SEP from `avoid`."""
        for _ in range(500):
            x, y = self._rng.uniform(*RAND_X), self._rng.uniform(*RAND_Y)
            if not self._in_scene_fov((x, y, z)):
                continue
            if avoid is not None and np.hypot(x - avoid[0], y - avoid[1]) < RAND_MIN_SEP:
                continue
            return (float(x), float(y))
        return (sum(RAND_X) / 2.0, sum(RAND_Y) / 2.0)  # fallback: region center (always in-FOV)

    def _apply_scene(self) -> None:
        """Write the intended box pose, goal position, and box color into model/data."""
        if self._block_qadr >= 0:
            half = self._block_yaw / 2.0  # quaternion for a yaw (z-axis) rotation
            self.data.qpos[self._block_qadr:self._block_qadr + 3] = (*self._block_xy, self._block_z)
            self.data.qpos[self._block_qadr + 3:self._block_qadr + 7] = (np.cos(half), 0.0, 0.0, np.sin(half))
            self.data.qvel[self._block_vadr:self._block_vadr + 6] = 0.0
        if self._goal_geom >= 0:
            self.model.geom_pos[self._goal_geom] = (*self._goal_xy, self._goal_z)
        if self._block_rgba is not None and self._block_geom >= 0:
            self.model.geom_rgba[self._block_geom] = self._block_rgba

    def randomize_block(self):
        """Respawn the box at a random in-FOV, reachable spot + random yaw (kept clear of the goal)."""
        self._block_xy = self._sample_xy(self._block_z, avoid=self._goal_xy)
        self._block_yaw = float(self._rng.uniform(-np.pi, np.pi))
        if self._weld_id >= 0:
            self.data.eq_active[self._weld_id] = 0  # drop any held weld before teleporting
        self._grasp_level = 0.0
        self._apply_scene()
        mujoco.mj_forward(self.model, self.data)
        return (*self._block_xy, float(np.degrees(self._block_yaw)))

    def randomize_goal(self):
        """Move the place target to a random in-FOV spot (kept clear of the box)."""
        self._goal_xy = self._sample_xy(self._goal_z, avoid=self._block_xy)
        self._apply_scene()
        mujoco.mj_forward(self.model, self.data)
        return self._goal_xy

    def set_block_color(self, rgba) -> None:
        """Set the box color (RGBA in 0..1); persists across resets."""
        self._block_rgba = tuple(float(c) for c in rgba)
        self._apply_scene()

    def _write_arm_ctrl(self, q6) -> None:
        for aid, v in zip(self._arm_aid, q6):
            self.data.ctrl[aid] = float(v)

    def set_ee_pose(self, pose: Sequence[float]) -> None:
        pose, _ = clamp_out_of_nogo(pose)  # keep the EE out of the robot-body no-go zone
        self._ik.send_ee_pose(pose)  # IK on the arm-only model (iterates to the target)
        self._write_arm_ctrl(self._ik.get_joint_positions())  # actuators track (contact lifts the box)

    def set_curls(self, curls: Sequence[float]) -> None:
        seq = list(curls)
        self._grasp_level = float(np.mean(seq)) if seq else 0.0  # drives the grasp aid
        for idx, rad in enumerate(curls_to_motor_radians(seq)):
            fi, mj = idx // 2 + 1, idx % 2 + 1
            self.data.ctrl[self._hand_aid[(fi, mj)]] = rad

    def _update_grasp_aid(self) -> None:
        """Weld the block to the hand when the grasp closes near it; release on open."""
        if self._weld_id < 0 or self._block_body < 0:
            return
        active = bool(self.data.eq_active[self._weld_id])
        if not active and self._grasp_level >= GRASP_CLOSE:
            hand_pos = self.data.xpos[self._hand_body]
            hand_mat = self.data.xmat[self._hand_body].reshape(3, 3)
            center = hand_pos + hand_mat @ self._grasp_center
            if np.linalg.norm(self.data.xpos[self._block_body] - center) <= GRASP_CAPTURE_R:
                self._weld_in_place()
                self.data.eq_active[self._weld_id] = 1
        elif active and self._grasp_level <= GRASP_OPEN:
            self.data.eq_active[self._weld_id] = 0

    def _weld_in_place(self) -> None:
        """Set the weld's relative pose to the block's CURRENT pose in the hand frame (grab in place)."""
        hq, bq = self.data.xquat[self._hand_body], self.data.xquat[self._block_body]
        neg = np.zeros(4)
        mujoco.mju_negQuat(neg, hq)
        relq = np.zeros(4)
        mujoco.mju_mulQuat(relq, neg, bq)
        relp = np.zeros(3)
        mujoco.mju_rotVecQuat(relp, self.data.xpos[self._block_body] - self.data.xpos[self._hand_body], neg)
        self.model.eq_data[self._weld_id, 3:6] = relp
        self.model.eq_data[self._weld_id, 6:10] = relq
        self.model.eq_data[self._weld_id, 10] = 1.0

    def step(self, n: int | None = None) -> None:
        self._update_grasp_aid()
        for _ in range(n if n is not None else self.settle_steps):
            mujoco.mj_step(self.model, self.data)

    def get_ee_pose(self) -> list[float]:
        pos = self.data.site_xpos[self._site_id].copy()
        rotvec = Rotation.from_matrix(self.data.site_xmat[self._site_id].reshape(3, 3)).as_rotvec()
        return [*pos, *rotvec]

    def get_joint_positions(self) -> list[float]:
        return [float(v) for v in self.data.qpos[:6]]

    def _default_camera(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = SCENE_CAM["azimuth"], SCENE_CAM["elevation"], SCENE_CAM["distance"]
        cam.lookat[:] = SCENE_CAM["lookat"]  # frame the arm reach + the block on the floor
        return cam

    def render(self, width: int = 640, height: int = 480, camera=None):
        """Render the scene cam (camera=None) or a named/id camera (e.g. the wrist cam).

        Renderers are cached per (width, height) so the scene and wrist views don't thrash.
        """
        fw, fh = int(self.model.vis.global_.offwidth), int(self.model.vis.global_.offheight)
        width = max(16, min(int(width), fw))
        height = max(16, min(int(height), fh))
        r = self._renderers.get((width, height))
        if r is None:
            r = mujoco.Renderer(self.model, height=height, width=width)
            self._renderers[(width, height)] = r
        r.update_scene(self.data, camera=(camera if camera is not None else self._default_camera()))
        return r.render()

    def render_wrist(self, width: int = 640, height: int = 480):
        return self.render(width, height, camera=WRIST_CAM)


class CellArm(ArmInterface):
    """ArmInterface over a shared MujocoCell (sets arm ctrl; cell is stepped by CellHand)."""

    def __init__(self, cell: MujocoCell):
        self.cell = cell
        self._connected = False

    def connect(self) -> None:
        self.cell.reset()
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_ee_pose(self) -> list[float]:
        return self.cell.get_ee_pose()

    def get_joint_positions(self) -> list[float]:
        return self.cell.get_joint_positions()

    def send_ee_pose(self, pose: Sequence[float]) -> list[float]:
        self.cell.set_ee_pose(pose)
        return self.cell.get_ee_pose()


class CellHand:
    """AmazingHandClient-compatible backend over the shared MujocoCell.

    `send_curls` sets the hand ctrl AND steps the cell — since URAmazingHand.send_action
    calls arm.send_ee_pose() then hand.send_curls(), one step advances arm + hand together.
    """

    def __init__(self, cell: MujocoCell):
        self.cell = cell
        self._connected = False

    def connect(self) -> "CellHand":
        self._connected = True
        return self

    def close(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def send_curls(self, curls, speed=None) -> None:
        self.cell.set_curls(curls)
        self.cell.step()

    def render(self, **kw):
        return self.cell.render(**kw)
