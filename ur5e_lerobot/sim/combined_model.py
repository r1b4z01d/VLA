"""Build a combined UR5e + AmazingHand MuJoCo model (hand attached to the flange).

Uses MuJoCo's MjSpec model-composition API to attach the AmazingHand's root body to the
UR5e's `attachment_site`. Hand actuators/joints get the `rh_` prefix in the merged model.
"""
from __future__ import annotations

import os

import mujoco
from robot_descriptions import ur5e_mj_description

from ..robot.workspace import NO_GO_XY

ARM_SCENE = os.path.join(os.path.dirname(ur5e_mj_description.MJCF_PATH), "scene.xml")
HAND_XML = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "assets", "amazing_hand", "AH_Right", "robot.xml")
)
EE_SITE = "attachment_site"
HAND_PREFIX = "rh_"
HAND_ROOT_BODY = "r_wrist_interface"

# Robot Disco mobile base — the UR is mounted on its back platform. Added as static visual
# context so the teleop/sim model matches the real robot. The UR base sits 30 mm above the back
# lidar (rp_s2e_lidar at base_link -0.52,0), x/y centered, yawed -90 deg (CW) to align with the
# robot's center axis. The processed URDF (primitives only, no meshes) lives in assets/.
DISCO_BASE_URDF = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "assets", "robot_disco_base.urdf"))
DISCO_LIDAR_XY = (-0.52, 0.0)
DISCO_YAW_DEG = -90.0


# Manipulation object: a 4 x 4 x 9 cm box (half-extents below), resting on the floor.
BLOCK_SIZE = (0.02, 0.02, 0.045)
BLOCK_RGBA = (0.90, 0.30, 0.10, 1.0)

PICK_XY = (0.251, 0.252)  # object start position
PLACE_XY = (0.216, 0.697)  # place target position
BLOCK_POS = (PICK_XY[0], PICK_XY[1], BLOCK_SIZE[2])  # resting on the floor (z = half-height)

# Place target: a flat, non-colliding marker disk on the floor.
TARGET_RADIUS = 0.06
TARGET_RGBA = (0.20, 0.80, 0.30, 0.55)
TARGET_POS = (PLACE_XY[0], PLACE_XY[1], 0.0015)


def _add_block(spec: mujoco.MjSpec, pos, size, rgba) -> None:
    """Add a free-floating box (the manipulation target) to the world."""
    body = spec.worldbody.add_body()
    body.name = "block"
    body.pos = list(pos)
    try:
        body.add_freejoint()
    except AttributeError:  # older MjSpec
        j = body.add_joint()
        j.name = "block_free"
        j.type = mujoco.mjtJoint.mjJNT_FREE
    g = body.add_geom()
    g.name = "block"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = list(size)  # half-extents
    g.rgba = list(rgba)
    g.friction = [3.0, 0.1, 0.002]  # keep low: higher tips this tall box over at rest (MuJoCo
    g.condim = 4  # contact quirk). Grip comes from the finger pads (30) anyway — contact uses the
    #               MAX of the two geoms' friction, so the pads govern the pinch regardless.
    g.solref = list(GRIP_SOLREF)  # stiff contact -> shallow penetration, firm grip
    g.solimp = list(GRIP_SOLIMP)
    g.mass = 0.02  # light enough for the hand to lift
    # contype/conaffinity = 3 (bits 1+2): bit 1 -> rests on the floor (1),
    # bit 2 -> collides with the hand geoms (which we set to contype=2).
    g.contype = 3
    g.conaffinity = 3


def _configure_hand_collision(spec: mujoco.MjSpec) -> None:
    """Hybrid hand collision so the whole hand is solid (doesn't pass through objects) while the
    pinch stays crisp:

    - DISTAL phalanges (the bodies holding the tipN sites): mesh collision OFF — the crisp
      fingertip capsules (_add_finger_capsules) handle the grip. Mesh there would make the
      pinch mushy (the CAD hulls penetrate ~1.5-2.7 cm).
    - REST of the hand (proximal segments, palm, links): mesh collision ON, with stiff
      solref/solimp so it makes firm contact instead of letting objects sink in / pass through.
    - Floor collides with the hand (bit 2).
    Hand geoms never self-collide or fight the arm (conaffinity 0).
    """
    tip_sites = {HAND_PREFIX + t for t in TIP_SITES}
    tip_bodies = {b.name for b in spec.bodies if any(s.name in tip_sites for s in b.sites)}
    for body in spec.bodies:
        is_hand = body.name.startswith(HAND_PREFIX)
        for g in body.geoms:
            if is_hand and body.name not in tip_bodies:
                g.contype = 2
                g.conaffinity = 0
                g.solref = list(GRIP_SOLREF)
                g.solimp = list(GRIP_SOLIMP)
            elif is_hand:  # distal phalanx — left to the fingertip capsule
                g.contype = 0
                g.conaffinity = 0
            elif g.type == mujoco.mjtGeom.mjGEOM_PLANE:
                g.conaffinity = 3


def _add_target(spec: mujoco.MjSpec, pos, radius, rgba) -> None:
    """Add a flat, non-colliding marker disk (the place target) to the world."""
    g = spec.worldbody.add_geom()
    g.name = "target"
    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.size = [radius, 0.001, 0.0]  # cylinder: [radius, half-length]
    g.pos = list(pos)
    g.rgba = list(rgba)
    g.contype = 0  # purely visual — no collision
    g.conaffinity = 0


NO_GO_ZONE_RGBA = (0.90, 0.10, 0.10, 0.15)


def _add_nogo_zone(spec: mujoco.MjSpec) -> None:
    """Translucent box over the robot-body no-go rectangle (NO_GO_XY). VISUAL ONLY — the real
    keep-out is enforced on the EE target (robot.workspace.clamp_out_of_nogo). Off by default,
    since it would otherwise show up in recorded camera frames."""
    x0, x1, y0, y1 = NO_GO_XY
    g = spec.worldbody.add_geom()
    g.name = "nogo_zone"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos = [(x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.15]
    g.size = [(x1 - x0) / 2.0, (y1 - y0) / 2.0, 0.15]
    g.rgba = list(NO_GO_ZONE_RGBA)
    g.contype = 0
    g.conaffinity = 0


WRIST_CAM = "wrist_cam"
WRIST_CAM_TARGET = HAND_PREFIX + "rotule_ball_3"  # body at the grasp center (auto-aim)
WRIST_CAM_POS = (0.0, -0.08, 0.0)  # local to the hand root; offset to peek at the grasp
WRIST_CAM_FOVY = 100


def _add_wrist_camera(spec: mujoco.MjSpec) -> None:
    """Add an eye-in-hand camera on the wrist, auto-aimed at the grasp center."""
    body = spec.body(HAND_PREFIX + HAND_ROOT_BODY)  # rh_r_wrist_interface (moves with the hand)
    cam = body.add_camera()
    cam.name = WRIST_CAM
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    cam.targetbody = WRIST_CAM_TARGET
    cam.pos = list(WRIST_CAM_POS)
    cam.fovy = WRIST_CAM_FOVY


# Fingertip markers on the distal phalanges (index, middle, ring, thumb), prefixed in the merge.
TIP_SITES = ("tip1", "tip2", "tip3", "tip4")
FINGER_CAPSULE_RADIUS = 0.009  # crisp capsule collider along each distal phalanx
# Stiff contact (analytic capsule-box) so the fingers barely penetrate and the grip is firm.
GRIP_SOLREF = (0.005, 1.0)
GRIP_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)


def _add_finger_capsules(spec: mujoco.MjSpec) -> None:
    """Add a crisp capsule collider along each distal phalanx (body origin -> fingertip site).

    Capsule-vs-box collision is analytic (not a mesh convex hull), so it makes firm, shallow
    contact — the fingers grip the object instead of sinking through it. High friction + stiff
    solref/solimp so the pinch holds. contype 2 / conaffinity 0: collides the block (via the
    block's affinity bits) and the floor, never the hand itself or the arm.
    """
    targets = {HAND_PREFIX + t for t in TIP_SITES}
    for body in spec.bodies:
        for s in body.sites:
            if s.name in targets:
                tip = list(s.pos)  # distal end (local); body origin is the proximal end
                g = body.add_geom()
                g.name = s.name + "_cap"
                g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
                g.fromto = [0.0, 0.0, 0.0, tip[0], tip[1], tip[2]]
                g.size = [FINGER_CAPSULE_RADIUS, 0.0, 0.0]
                g.rgba = [0.1, 0.8, 0.2, 1.0]
                g.friction = [30.0, 2.0, 0.2]  # very high grip (bumped from 15/1/0.1) for a firm pinch
                g.condim = 4
                g.contype = 2
                g.conaffinity = 0
                g.solref = list(GRIP_SOLREF)
                g.solimp = list(GRIP_SOLIMP)


# --- Grasp aid (attach-on-close) ----------------------------------------------------------------
# This hand's geometry can't form a stable physics grasp of the block: finger 4 contacts high and
# to one side while fingers 1-3 press the other, an UNBALANCED squeeze with no true opposition, so
# any object is ejected the instant it's freed (verified: drops at every size/placement/contact
# stiffness; scripts/characterize_hand.py shows the geometry, scripts/grip_aid_test.py verifies the
# aid). The grasp aid welds the block to the hand when the hand
# closes near it and releases on open -> reliable pick-and-place for demo collection. The recorded
# motion (approach, close, carry, open) is unchanged; only the hold is made to actually hold.
GRASP_WELD = "grasp_weld"
GRASP_CENTER_LOCAL = (0.030, 0.0, 0.082)  # grasp pocket in the hand-root frame (from characterize_hand)


def _add_grasp_weld(spec: mujoco.MjSpec) -> None:
    """Add an initially-inactive WELD equality (hand root <-> block). The cell activates it in
    place when the hand closes near the block and deactivates it on open."""
    eq = spec.add_equality()
    eq.name = GRASP_WELD
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name1 = HAND_PREFIX + HAND_ROOT_BODY  # body the block welds to (moves with the flange)
    eq.name2 = "block"
    eq.active = False
    # data = [anchor(3), relpose pos(3) + quat(4), torquescale(1)]; cell overwrites relpose in place.
    eq.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    eq.solref = [0.002, 1.0]  # stiff weld -> the block tracks the hand rigidly (minimal lag)
    eq.solimp = [0.999, 0.9999, 0.0005, 0.5, 2.0]


def _attach_disco_base(spec: mujoco.MjSpec, yaw_deg: float = DISCO_YAW_DEG) -> None:
    """Attach the Robot Disco mobile base UNDER the UR base as static visual context.

    The UR (root) stays at the origin so arm IK/control and the manipulation task are unchanged;
    the base hangs below at the inverse mount transform (UR 30 mm above the back lidar, yawed).
    Base geoms are made non-colliding (purely visual) so they never perturb the arm or task.
    """
    import math

    base = mujoco.MjSpec.from_file(DISCO_BASE_URDF)
    for b in base.bodies:  # visual only — never collides with the arm, floor, or objects
        for g in b.geoms:
            g.contype = 0
            g.conaffinity = 0
    psi = math.radians(yaw_deg)
    vx, vy, vz = DISCO_LIDAR_XY[0], DISCO_LIDAR_XY[1], 0.03
    s = spec.body("base").add_site()
    s.name = "disco_mount"
    s.pos = [-(vx * math.cos(psi) + vy * math.sin(psi)), vx * math.sin(psi) - vy * math.cos(psi), -vz]
    s.quat = [math.cos(psi / 2), 0.0, 0.0, -math.sin(psi / 2)]
    spec.site("disco_mount").attach_body(base.body("base_link"), "disco_", "")


def build_combined_model(
    arm_scene: str = ARM_SCENE,
    hand_xml: str = HAND_XML,
    with_base: bool = True,
    with_block: bool = True,
    block_pos=BLOCK_POS,
    block_size=BLOCK_SIZE,
    block_rgba=BLOCK_RGBA,
    with_target: bool = True,
    target_pos=TARGET_POS,
    target_radius: float = TARGET_RADIUS,
    target_rgba=TARGET_RGBA,
    grasp_aid: bool = True,
    show_nogo_zone: bool = False,
) -> mujoco.MjModel:
    """Compile a single MjModel of the UR5e + AmazingHand (+ graspable block + place target).

    grasp_aid adds an attach-on-close weld (the cell drives it) so pick-and-place is reliable
    despite the hand being unable to form a stable physics grasp (see _add_grasp_weld).
    """
    spec_arm = mujoco.MjSpec.from_file(arm_scene)
    spec_hand = mujoco.MjSpec.from_file(hand_xml)
    spec_arm.site(EE_SITE).attach_body(spec_hand.body(HAND_ROOT_BODY), HAND_PREFIX, "")
    if with_block:
        _add_block(spec_arm, block_pos, block_size, block_rgba)
    if with_target:
        _add_target(spec_arm, target_pos, target_radius, target_rgba)
    _add_wrist_camera(spec_arm)
    _configure_hand_collision(spec_arm)     # whole hand solid (mesh) except distal phalanges
    _add_finger_capsules(spec_arm)          # crisp capsule colliders on the distal phalanges (grip)
    if with_block and grasp_aid:
        _add_grasp_weld(spec_arm)           # attach-on-close: reliable hold (cell drives eq_active)
    if with_base:
        _attach_disco_base(spec_arm)        # mobile base under the UR (static visual context)
    if show_nogo_zone:
        _add_nogo_zone(spec_arm)            # translucent keep-out box (debug viz; off by default)

    # Allow large offscreen renders (reactive window) than the 640x480 default.
    spec_arm.visual.global_.offwidth = 1920
    spec_arm.visual.global_.offheight = 1200
    return spec_arm.compile()
