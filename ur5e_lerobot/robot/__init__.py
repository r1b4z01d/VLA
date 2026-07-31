from .arm_interface import ArmInterface, Ros2ArmInterface

# The LeRobot Robot adapter + its config need `lerobot` installed. The lightweight hardware preflight
# (RtdeArmInterface / cameras) runs on the robot PC without lerobot, so make these optional — they're
# still imported normally wherever lerobot is present (Mac/GPU).
try:
    from .config import URAmazingHandConfig
    from .ur_amazinghand_robot import URAmazingHand
except ImportError:  # preflight-only env without lerobot
    URAmazingHandConfig = URAmazingHand = None  # type: ignore

__all__ = ["URAmazingHand", "URAmazingHandConfig", "ArmInterface", "Ros2ArmInterface"]
