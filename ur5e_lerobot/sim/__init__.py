from .sim_arm import SimArm
from .sim_hand import SimHand

# The MuJoCo sim modules need `mujoco` + `robot_descriptions` (a large model download). The robot
# PC runs --engine hardware without them, so keep these optional — still imported where present.
try:
    from .amazing_hand_mujoco import AmazingHandMujoco
    from .mujoco_arm import MujocoArm
    from .mujoco_cell import CellArm, CellHand, MujocoCell
except ImportError:  # sim stack absent (hardware-only env)
    AmazingHandMujoco = MujocoArm = MujocoCell = CellArm = CellHand = None  # type: ignore

__all__ = [
    "AmazingHandMujoco",
    "MujocoArm",
    "MujocoCell",
    "CellArm",
    "CellHand",
    "SimArm",
    "SimHand",
]
