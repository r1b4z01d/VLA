"""Characterize the AmazingHand fingers in-sim to ground the curl->motor mapping.

Each finger is a 2-motor parallel linkage. Firmware convention (HAND_CTRL.h):
    flexion f, abduction a (deg)  ->  motor1 = f + a,  motor2 = -f + a
(open = flex -35/abd 0 -> (-35,35); close = flex 90/abd 0 -> (90,-90)).

We measure fingertip (tip1..tip4) world positions under various (f,a) to verify:
  - pure flexion curls each finger toward the grasp center (an ENVELOPING close), and
  - whether the thumb (finger4) needs a flipped flexion sign / nonzero abduction to oppose
    the other three within the real +/-20 deg abduction limit.

Reports, per pose, each tip and the distance from the thumb tip4 to the centroid of the
index/middle/ring tips (tip1..3) -- small = good opposition / envelope.

    .venv/bin/python scripts/characterize_hand.py
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ur5e_lerobot.sim.amazing_hand_mujoco import DEFAULT_MODEL

m = mujoco.MjModel.from_xml_path(DEFAULT_MODEL)
d = mujoco.MjData(m)

ACT = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
TIPS = ["tip1", "tip2", "tip3", "tip4"]
TIP_ID = {t: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, t) for t in TIPS}


def set_fa(fa):
    """fa: list of 4 (flex_deg, abduct_deg)."""
    for fi, (f, a) in enumerate(fa):
        d.ctrl[ACT[f"finger{fi + 1}_motor1"]] = math.radians(f + a)
        d.ctrl[ACT[f"finger{fi + 1}_motor2"]] = math.radians(-f + a)


def pose(label, fa, settle=900):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    set_fa(fa)
    for _ in range(settle):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    tp = {t: d.site_xpos[TIP_ID[t]].copy() for t in TIPS}
    arr = np.array([tp[t] for t in TIPS])
    spread = np.linalg.norm(arr.max(0) - arr.min(0)) * 1000
    fingers_ctr = np.array([tp["tip1"], tp["tip2"], tp["tip3"]]).mean(0)
    thumb_gap = np.linalg.norm(tp["tip4"] - fingers_ctr) * 1000
    print(f"\n{label}")
    for t in TIPS:
        print(f"  {t}: {np.round(tp[t], 4)}")
    print(f"  spread(all4)={spread:5.1f}mm   thumb_tip->fingers_centroid={thumb_gap:5.1f}mm")
    return tp


print("====================  OPEN  ====================")
pose("open  (flex -35, abd 0) x4", [(-35, 0)] * 4)

print("\n====================  ENVELOPE: all pure flexion  ====================")
pose("close (flex +90, abd 0) x4  <- does the thumb meet the fingers?", [(90, 0)] * 4)

print("\n====================  THUMB (finger4) opposition test  ====================")
print("(index/middle/ring held at flex 90; vary only the thumb)")
held = [(90, 0), (90, 0), (90, 0)]
pose("thumb flex +90 abd   0", held + [(90, 0)])
pose("thumb flex -90 abd   0  (flipped sign)", held + [(-90, 0)])
pose("thumb flex +90 abd +20", held + [(90, 20)])
pose("thumb flex +90 abd -20", held + [(90, -20)])
pose("thumb flex +45 abd +20", held + [(45, 20)])
pose("thumb flex +45 abd -20", held + [(45, -20)])
pose("thumb LEGACY raw(90,30) = flex30/abd60", held + [(30, 60)])
