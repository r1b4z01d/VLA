"""Single source of truth for the UR5e + AmazingHand action / observation schema.

Decisions (see docs/decisions.md D6 and docs/data_schema.md):
  ARM action  = end-effector pose, 6-DoF Cartesian (position xyz + rotation vector rxyz)
  HAND action = 4 per-finger curls in [0, 1]  (index, middle, ring, thumb)
  => ACTION is 10-D:
     [ee_x, ee_y, ee_z, ee_rx, ee_ry, ee_rz, curl_index, curl_middle, curl_ring, curl_thumb]

These constants are the contract shared by the teleop layer, the LeRobot Robot adapter,
the simulator, and the dataset features. Map to LeRobot `features` once the version is
pinned (see data_schema.md for the sketch).
"""
from __future__ import annotations

from dataclasses import dataclass

# --- Action layout ---------------------------------------------------------
ARM_POSE_DIM = 6   # [x, y, z, rx, ry, rz] — meters; rotation vector (axis-angle), rad
HAND_CURL_DIM = 4  # [index, middle, ring, thumb] in [0, 1]
ACTION_DIM = ARM_POSE_DIM + HAND_CURL_DIM  # 10

ARM_POSE_NAMES = ("ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz")
HAND_CURL_NAMES = ("curl_index", "curl_middle", "curl_ring", "curl_thumb")
ACTION_NAMES = ARM_POSE_NAMES + HAND_CURL_NAMES

ARM_SLICE = slice(0, ARM_POSE_DIM)
HAND_SLICE = slice(ARM_POSE_DIM, ACTION_DIM)

# --- Observation layout ----------------------------------------------------
ARM_JOINT_DIM = 6
ARM_JOINT_NAMES = tuple(f"q{i + 1}" for i in range(ARM_JOINT_DIM))
# proprioceptive state = ee pose (6) + arm joints (6) + hand curls (4) = 16
STATE_NAMES = ARM_POSE_NAMES + ARM_JOINT_NAMES + HAND_CURL_NAMES
STATE_DIM = len(STATE_NAMES)  # 16

# Camera image keys (see docs/camera_setup.md). scene+wrist is the minimum tier.
CAMERA_KEYS = ("scene", "wrist", "side")


@dataclass
class Action:
    """Structured action; `to_vector`/`from_vector` cross the dataset boundary."""

    arm_pose: tuple  # len ARM_POSE_DIM (x,y,z, rx,ry,rz)
    hand_curls: tuple  # len HAND_CURL_DIM (index, middle, ring, thumb), each in [0,1]

    def __post_init__(self) -> None:
        if len(self.arm_pose) != ARM_POSE_DIM:
            raise ValueError(f"arm_pose must be {ARM_POSE_DIM}-D, got {len(self.arm_pose)}")
        if len(self.hand_curls) != HAND_CURL_DIM:
            raise ValueError(f"hand_curls must be {HAND_CURL_DIM}-D, got {len(self.hand_curls)}")

    def to_vector(self) -> list:
        return [float(v) for v in self.arm_pose] + [float(v) for v in self.hand_curls]

    @classmethod
    def from_vector(cls, v) -> "Action":
        v = list(v)
        if len(v) != ACTION_DIM:
            raise ValueError(f"expected {ACTION_DIM}-D action, got {len(v)}")
        return cls(tuple(v[ARM_SLICE]), tuple(v[HAND_SLICE]))


# --- V2 action layout: full 8-DOF hand (flex + abduct per finger) -----------
# Enabled once the hand reports per-finger flex AND abduction (PCF8574 buttons + firmware `S:`
# report, see docs/amazinghand_esp32_buttons.md). Kept SEPARATE from the 10-D v1 above so existing
# 4-curl datasets + the trained ACT policy stay valid; the teleop/record path opts in explicitly.
HAND_FLEX_DIM = 4     # per-finger flexion, [0,1] (0=open, 1=closed)
HAND_ABDUCT_DIM = 4   # per-finger abduction, [-1,1] (0=neutral, ±1 = full spread/adduct)
HAND_DOF_V2 = HAND_FLEX_DIM + HAND_ABDUCT_DIM  # 8
ACTION_DIM_V2 = ARM_POSE_DIM + HAND_DOF_V2  # 14

HAND_FLEX_NAMES = ("flex_index", "flex_middle", "flex_ring", "flex_thumb")
HAND_ABDUCT_NAMES = ("abduct_index", "abduct_middle", "abduct_ring", "abduct_thumb")
ACTION_NAMES_V2 = ARM_POSE_NAMES + HAND_FLEX_NAMES + HAND_ABDUCT_NAMES

STATE_NAMES_V2 = ARM_POSE_NAMES + ARM_JOINT_NAMES + HAND_FLEX_NAMES + HAND_ABDUCT_NAMES
STATE_DIM_V2 = len(STATE_NAMES_V2)  # 20

ARM_SLICE_V2 = slice(0, ARM_POSE_DIM)
FLEX_SLICE_V2 = slice(ARM_POSE_DIM, ARM_POSE_DIM + HAND_FLEX_DIM)
ABDUCT_SLICE_V2 = slice(ARM_POSE_DIM + HAND_FLEX_DIM, ACTION_DIM_V2)


@dataclass
class ActionV2:
    """8-DOF hand action: arm EE pose + per-finger flex[0,1] + abduct[-1,1]."""

    arm_pose: tuple      # len ARM_POSE_DIM
    hand_flex: tuple     # len HAND_FLEX_DIM, each in [0,1]
    hand_abduct: tuple   # len HAND_ABDUCT_DIM, each in [-1,1]

    def __post_init__(self) -> None:
        for label, seq, dim in (("arm_pose", self.arm_pose, ARM_POSE_DIM),
                                 ("hand_flex", self.hand_flex, HAND_FLEX_DIM),
                                 ("hand_abduct", self.hand_abduct, HAND_ABDUCT_DIM)):
            if len(seq) != dim:
                raise ValueError(f"{label} must be {dim}-D, got {len(seq)}")

    def to_vector(self) -> list:
        return ([float(v) for v in self.arm_pose]
                + [float(v) for v in self.hand_flex]
                + [float(v) for v in self.hand_abduct])

    @classmethod
    def from_vector(cls, v) -> "ActionV2":
        v = list(v)
        if len(v) != ACTION_DIM_V2:
            raise ValueError(f"expected {ACTION_DIM_V2}-D action, got {len(v)}")
        return cls(tuple(v[ARM_SLICE_V2]), tuple(v[FLEX_SLICE_V2]), tuple(v[ABDUCT_SLICE_V2]))
