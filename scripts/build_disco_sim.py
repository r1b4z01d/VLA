"""Assemble the full Robot Disco robot in MuJoCo for visualization:

    robot_disco mobile base  +  UR5e  +  AmazingHand

The UR base is mounted on the back RPLidar (rp_s2e_lidar): 30 mm above it, x/y centered
(matching ur_mount.xacro in the robot_disco description). The arm is parented to the lidar
frame, so it tracks the lidar/platform position.

The current robot_disco URDF places the back lidar at z=0 (a placeholder), so by default the
arm sits low. Pass --platform-z 0.7 to preview with the platform raised to a realistic height.

    .venv/bin/python scripts/build_disco_sim.py --platform-z 0.7 --out outputs/disco_full.png
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

import ur5e_lerobot.sim.combined_model as cm

# robot_disco ROS description dir + xacro binary — override via env; sensible per-user defaults.
DESC = os.environ.get("ROBOT_DISCO_DESC",
                      os.path.expanduser("~/Documents/rd_ws/src/robot_disco/description"))
XACRO = os.environ.get("XACRO") or shutil.which("xacro") or "xacro"
LIDAR_XY = (-0.52, 0.0)  # rp_s2e_lidar (back) origin on base_link


def base_urdf(platform_z: float = 0.0) -> str:
    """Process the robot_disco xacros (minus ros2_control) to a clean URDF; bump the back
    lidar to platform_z so we can preview the arm at a realistic height."""
    top = "/tmp/_disco_top.urdf.xacro"
    with open(top, "w") as f:
        f.write('<?xml version="1.0"?>\n<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="robot_disco">\n')
        f.write('<xacro:arg name="sim_mode" default="false"/>\n')
        for x in ("robot_core", "velodyne", "rp_s1_lidar", "rp_s2e_lidar", "realsense_d435"):
            f.write(f'<xacro:include filename="{DESC}/{x}.xacro"/>\n')
        f.write("</robot>\n")
    subprocess.run([XACRO, top, "-o", "/tmp/_disco.urdf"], check=True)
    tree = ET.parse("/tmp/_disco.urdf")
    root = tree.getroot()
    for tag in ("gazebo", "transmission", "ros2_control"):
        for el in root.findall(tag):
            root.remove(el)
    for j in root.findall("joint"):  # bump the back lidar (and thus the arm) to platform_z
        if j.get("name") == "rp_s2e_lidar_joint":
            o = j.find("origin")
            xyz = o.get("xyz").split()
            xyz[2] = str(platform_z)
            o.set("xyz", " ".join(xyz))
    tree.write("/tmp/_disco_clean.urdf")
    return "/tmp/_disco_clean.urdf"


def build(platform_z: float = 0.0, yaw_deg: float = 0.0) -> mujoco.MjModel:
    """UR5e scene is the ROOT (so its visual meshes survive), with the mobile base attached
    UNDER it at the inverse mount transform. Attaching the base (all primitives) carries
    cleanly; attaching the mesh-based UR *into* the base instead drops the UR's visual meshes.

    yaw_deg yaws the UR about the mount's vertical axis (through the lidar) relative to the base,
    so the arm/hand can be aimed over the robot's center. The arm/hand kinematics are intact
    (fixed base); the mobile base is rigid visual context.
    """
    import math
    base = mujoco.MjSpec.from_file(base_urdf(platform_z))
    spec = mujoco.MjSpec.from_file(cm.ARM_SCENE)
    hand = mujoco.MjSpec.from_file(cm.HAND_XML)
    spec.site("attachment_site").attach_body(hand.body(cm.HAND_ROOT_BODY), "rh_", "")  # hand -> flange
    # UR base mounts 30 mm above the lidar at LIDAR_XY. With the UR at the origin and yawed by
    # psi relative to the base, base_link sits at -Rz(-psi)*(mount offset) and is oriented Rz(-psi).
    psi = math.radians(yaw_deg)
    vx, vy, vz = LIDAR_XY[0], LIDAR_XY[1], platform_z + 0.03
    s = spec.body("base").add_site()
    s.name = "disco_mount"
    s.pos = [-(vx * math.cos(psi) + vy * math.sin(psi)), vx * math.sin(psi) - vy * math.cos(psi), -vz]
    s.quat = [math.cos(psi / 2), 0.0, 0.0, -math.sin(psi / 2)]
    spec.site("disco_mount").attach_body(base.body("base_link"), "disco_", "")
    for g in spec.worldbody.geoms:  # drop the scene floor (the base hangs below the UR base)
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            g.rgba = [0, 0, 0, 0]
    spec.visual.global_.offwidth = 1300
    spec.visual.global_.offheight = 1000
    return spec.compile()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform-z", type=float, default=0.0, help="preview platform/lidar height (m)")
    ap.add_argument("--yaw", type=float, default=-90.0, help="UR yaw about the mount Z (deg, -CW)")
    ap.add_argument("--out", default="outputs/disco_full.png")
    args = ap.parse_args()
    m = build(args.platform_z, args.yaw)
    d = mujoco.MjData(m)
    # arm straight up (candle pose): upper arm + forearm vertical
    import math
    up = {"shoulder_pan_joint": 0.0, "shoulder_lift_joint": -math.pi / 2, "elbow_joint": 0.0,
          "wrist_1_joint": 0.0, "wrist_2_joint": 0.0, "wrist_3_joint": 0.0}
    for jn, val in up.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = val
    mujoco.mj_forward(m, d)
    gp = d.geom_xpos
    ctr = (gp.min(0) + gp.max(0)) / 2
    size = float(np.linalg.norm(gp.max(0) - gp.min(0)))
    r = mujoco.Renderer(m, 1000, 1300)
    views = []
    for az, el in [(135, -15), (180, -10), (90, -10)]:
        cam = mujoco.MjvCamera()
        # compensate for the base's world yaw so the base looks fixed and the UR appears to rotate
        cam.azimuth, cam.elevation, cam.distance = az - args.yaw, el, size * 1.45
        cam.lookat[:] = ctr
        r.update_scene(d, camera=cam)
        views.append(r.render().copy())
    Image.fromarray(np.hstack(views)).save(args.out)
    print(f"bodies={m.nbody}  wrote {args.out} (iso | back | side)")


if __name__ == "__main__":
    main()
