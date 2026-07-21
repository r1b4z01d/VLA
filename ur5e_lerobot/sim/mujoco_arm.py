"""MuJoCo UR5e arm with mink IK — a real ArmInterface backend that runs on the Mac.

Loads the MuJoCo Menagerie UR5e (lit scene) and uses mink differential IK to turn an
EE-pose command [x,y,z, rx,ry,rz] (position + axis-angle, base frame) into joint targets
— the same Cartesian→joints role MoveIt Servo plays on the real robot. Kinematic IK (sets
qpos from the IK solution); swap to actuator+dynamics later if needed.
"""
from __future__ import annotations

import os
from collections.abc import Sequence

import mink
import mujoco
import numpy as np
from robot_descriptions import ur5e_mj_description

from ..robot.arm_interface import ArmInterface

DEFAULT_SCENE = os.path.join(os.path.dirname(ur5e_mj_description.MJCF_PATH), "scene.xml")
EE_SITE = "attachment_site"  # UR5e flange (and where the AmazingHand will mount)


class MujocoArm(ArmInterface):
    def __init__(
        self,
        scene_path: str = DEFAULT_SCENE,
        ik_iters: int = 6,
        dt: float = 0.02,
        solver: str = "daqp",
        damping: float = 1e-3,
    ):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.config = mink.Configuration(self.model)
        self.task = mink.FrameTask(
            EE_SITE, "site", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0
        )
        self.posture = mink.PostureTask(self.model, cost=1e-2)
        self.ik_iters, self.dt, self.solver, self.damping = ik_iters, dt, solver, damping
        self._renderer: mujoco.Renderer | None = None
        self._connected = False
        self.reset()

    def reset(self) -> None:
        try:
            self.config.update_from_keyframe("home")
        except (KeyError, ValueError):
            self.config.update(np.zeros(self.model.nq))
        self.posture.set_target_from_configuration(self.config)
        self._sync_data()

    def _sync_data(self) -> None:
        self.data.qpos[: self.config.nq] = self.config.q
        mujoco.mj_forward(self.model, self.data)

    # --- ArmInterface ---
    def connect(self) -> None:
        self.reset()
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_ee_pose(self) -> list[float]:
        T = self.config.get_transform_frame_to_world(EE_SITE, "site")
        pos = np.asarray(T.translation(), dtype=float)
        rotvec = np.asarray(T.rotation().log(), dtype=float)
        return [*pos, *rotvec]

    def get_joint_positions(self) -> list[float]:
        return [float(v) for v in np.asarray(self.config.q)[:6]]

    def send_ee_pose(self, pose: Sequence[float]) -> list[float]:
        pos = np.asarray(pose[:3], dtype=float)
        rotvec = np.asarray(pose[3:6], dtype=float)
        target = mink.SE3.from_rotation_and_translation(mink.SO3.exp(rotvec), pos)
        self.task.set_target(target)
        for _ in range(self.ik_iters):
            vel = mink.solve_ik(self.config, [self.task, self.posture], self.dt, self.solver, self.damping)
            self.config.integrate_inplace(vel, self.dt)
            if float(np.linalg.norm(vel)) < 1e-3:  # converged — stop early (tight tracking)
                break
        self._sync_data()
        return self.get_ee_pose()

    # --- rendering ---
    def _default_camera(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 130.0, -20.0, 1.6
        cam.lookat[:] = [0.1, 0.0, 0.4]
        return cam

    def render(self, width: int = 640, height: int = 480, cam: mujoco.MjvCamera | None = None):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=cam or self._default_camera())
        return self._renderer.render()
