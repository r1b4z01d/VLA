"""Verify the grasp aid: close near the block -> weld -> LIFT the arm -> block follows -> open -> drops.

Drives the combined model with the same weld-in-place logic the cell uses, then lifts the arm and
checks the block tracks the hand (held), and that opening releases it.

    PYTHONPATH=. .venv/bin/python scripts/grip_aid_test.py
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
from PIL import Image

from ur5e_lerobot.sim.combined_model import GRASP_CENTER_LOCAL, GRASP_WELD, build_combined_model

HAND = "rh_"
ARM_POSE = {"shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.2, "elbow_joint": 1.4,
            "wrist_1_joint": -1.2, "wrist_2_joint": -1.5708, "wrist_3_joint": 0.0}

m = build_combined_model(with_base=False, with_target=False, grasp_aid=True)
d = mujoco.MjData(m)
act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, GRASP_WELD)
hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HAND + "r_wrist_interface")
bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block")
bjid = m.body_jntadr[bb]
qadr = m.jnt_qposadr[bjid]
lift_aid = act["shoulder_lift"]


def set_fa(fa):
    for fi, (f, a) in enumerate(fa):
        d.ctrl[act[f"{HAND}finger{fi + 1}_motor1"]] = math.radians(f + a)
        d.ctrl[act[f"{HAND}finger{fi + 1}_motor2"]] = math.radians(-f + a)


def weld_in_place():
    neg = np.zeros(4)
    mujoco.mju_negQuat(neg, d.xquat[hb])
    relq = np.zeros(4)
    mujoco.mju_mulQuat(relq, neg, d.xquat[bb])
    relp = np.zeros(3)
    mujoco.mju_rotVecQuat(relp, d.xpos[bb] - d.xpos[hb], neg)
    m.eq_data[wid, 3:6] = relp
    m.eq_data[wid, 6:10] = relq
    m.eq_data[wid, 10] = 1.0


for jn, v in ARM_POSE.items():
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
    d.qpos[m.jnt_qposadr[jid]] = v
    aid = act.get(jn.replace("_joint", ""))
    if aid is not None:
        d.ctrl[aid] = v
set_fa([(-35, 0)] * 4)
mujoco.mj_forward(m, d)

hp, hR = d.xpos[hb].copy(), d.xmat[hb].reshape(3, 3).copy()
pocket = hp + hR @ np.asarray(GRASP_CENTER_LOCAL)
hquat = np.zeros(4)
mujoco.mju_mat2Quat(hquat, hR.flatten())
d.qpos[qadr:qadr + 3] = pocket
d.qpos[qadr + 3:qadr + 7] = hquat
mujoco.mj_forward(m, d)

renderer = mujoco.Renderer(m, 700, 900)
cam = mujoco.MjvCamera()
cam.azimuth, cam.elevation, cam.distance = 120, -10, 0.55
cam.lookat[:] = pocket
shots = []


def shot():
    renderer.update_scene(d, camera=cam)
    shots.append(renderer.render().copy())


shot()
# 1) close the fingers while the block is supported (pinned, as the floor would), then weld in place
for i in range(300):
    c = (i + 1) / 300.0
    set_fa([(-35 + 125 * c, 0)] * 4)
    d.qpos[qadr:qadr + 3] = pocket
    d.qpos[qadr + 3:qadr + 7] = hquat
    d.qvel[m.jnt_dofadr[bjid]:m.jnt_dofadr[bjid] + 6] = 0
    mujoco.mj_step(m, d)
weld_in_place()
d.eq_active[wid] = 1
for _ in range(100):
    mujoco.mj_step(m, d)
shot()


def block_in_hand():  # block position expressed in the hand frame (invariant iff rigidly held)
    rel = np.zeros(3)
    neg = np.zeros(4)
    mujoco.mju_negQuat(neg, d.xquat[hb])
    mujoco.mju_rotVecQuat(rel, d.xpos[bb] - d.xpos[hb], neg)
    return rel


rel0 = block_in_hand()
z_grab = d.xpos[bb][2]

# 2) MOVE the arm (ramp shoulder_lift); a rigidly-held block keeps a constant pose in the hand frame
for i in range(600):
    d.ctrl[lift_aid] = -1.2 - 0.45 * (i + 1) / 600.0
    mujoco.mj_step(m, d)
shot()
slip = np.linalg.norm(block_in_hand() - rel0) * 1000  # drift of the block within the hand
print(f"hand moved {np.round((d.xpos[hb][2] - hp[2]) * 1000, 0)} mm in z; block rose "
      f"{np.round((d.xpos[bb][2] - z_grab) * 1000, 0)} mm")
print(f"block drift in hand frame: {slip:.1f} mm  ->  {'HELD (rigid)' if slip < 10 else 'SLIPPED'}")

# 3) OPEN -> release; block should detach and fall
z_before = d.xpos[bb][2]
d.eq_active[wid] = 0
set_fa([(-35, 0)] * 4)
for _ in range(500):
    mujoco.mj_step(m, d)
shot()
fell = (z_before - d.xpos[bb][2]) * 1000
print(f"after open, block fell {fell:.0f} mm  ->  {'RELEASED' if fell > 30 else 'still attached'}")
Image.fromarray(np.hstack(shots)).save("outputs/grip_aid_test.png")
print("wrote outputs/grip_aid_test.png (placed | grabbed | moved | released)")
