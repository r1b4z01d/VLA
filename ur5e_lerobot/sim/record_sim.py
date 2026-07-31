"""Record a LeRobotDataset entirely in sim on the Mac — no hardware.

Two engines (``--engine``):
  * ``mujoco``    — real UR5e + AmazingHand combined MuJoCo scene (mink IK). Visible,
                    dynamic arm + hand; recorded (state, action) are physically distinct.
  * ``kinematic`` — fast stub (SimArm + standalone hand sim) for quick pipeline checks.

The scene camera is the MuJoCo render; the action source is a scripted trajectory
(EE dip + staggered curl sweep) standing in for teleop until the SpaceMouse is available.
Images are stored (not video) so there's no FFmpeg dependency on macOS.

    .venv/bin/python -m ur5e_lerobot.sim.record_sim --engine mujoco --episodes 2 --frames 60
"""
from __future__ import annotations

import argparse
import math

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ..robot import URAmazingHand, URAmazingHandConfig
from ..schema import ACTION_DIM, ACTION_NAMES, STATE_DIM, STATE_NAMES, Action


def scripted_action(step: int, period: int, home: list[float]) -> list[float]:
    """EE dip/sway around the home pose + staggered per-finger curl sweep."""
    phase = 2.0 * math.pi * step / period
    ee = list(home)
    ee[0] += 0.05 * math.sin(phase)                 # sway x
    ee[1] += 0.04 * math.cos(phase)                 # sway y
    ee[2] -= 0.10 * (0.5 - 0.5 * math.cos(phase))   # dip down and back up
    curls = [0.5 * (1.0 - math.cos(phase + i * 0.4)) for i in range(4)]  # 0..1, staggered
    return Action(tuple(ee[:6]), tuple(curls)).to_vector()


def build_features(width: int, height: int, use_videos: bool, with_wrist: bool = True,
                   with_side: bool = False) -> dict:
    img_dtype = "video" if use_videos else "image"

    def img():
        return {"dtype": img_dtype, "shape": (height, width, 3), "names": ["height", "width", "channel"]}

    feats = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": list(STATE_NAMES)},
        "observation.images.scene": img(),
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": list(ACTION_NAMES)},
    }
    if with_wrist:
        feats["observation.images.wrist"] = img()
    if with_side:
        feats["observation.images.side"] = img()  # 2nd 3rd-person view (occlusion coverage)
    return feats


def make_engine(engine: str, settle_steps: int = 40, *, robot_ip: str = "192.168.11.21",
                hand_host: str = "192.168.11.117", scene_cam=None, wrist_cam=None, side_cam=None):
    """Return (robot, scene_render_fn, wrist_render_fn, side_render_fn). wrist/side are None if absent.

    engine: 'mujoco' (sim), 'kinematic' (fast stub), or 'hardware' (real UR5e via RTDE +
    AmazingHand TCP + USB cameras). The hardware kwargs are ignored by the sim engines.
    side_cam: None disables the 2nd scene cam; True uses cameras.SIDE_CAM; or pass a device path.
    """
    cfg = URAmazingHandConfig(id="sim", cameras={})
    if engine == "mujoco":
        from .mujoco_cell import CellArm, CellHand, MujocoCell

        cell = MujocoCell(settle_steps=settle_steps)
        robot = URAmazingHand(cfg, arm=CellArm(cell), hand=CellHand(cell))
        return (
            robot,
            (lambda w, h: cell.render(width=w, height=h)),
            (lambda w, h: cell.render_wrist(width=w, height=h)),
            None,  # no side cam in sim
        )

    if engine == "hardware":
        from ..hand import AmazingHandClient
        from ..robot.rtde_arm import RtdeArmInterface
        from ..sensors import UsbCamera
        from ..sensors.cameras import (SCENE_CAM, SCENE_GAIN, SCENE_ROTATE, SIDE_CAM,
                                        SIDE_GAIN, SIDE_ROTATE, WRIST_CAM, WRIST_GAIN,
                                        WRIST_ROTATE)

        scene_cam = SCENE_CAM if scene_cam is None else scene_cam  # stable device paths by default
        wrist_cam = WRIST_CAM if wrist_cam is None else wrist_cam
        if side_cam is True:  # enable the 2nd scene cam using the configured default port
            side_cam = SIDE_CAM
        robot = URAmazingHand(cfg, arm=RtdeArmInterface(robot_ip), hand=AmazingHandClient(hand_host))

        def _grab(cam: UsbCamera):
            def read(w, h):  # lazy-connect on first render (after robot.connect brings up arm+hand)
                if not cam.is_connected:
                    cam.connect()
                return cam.read(w, h)
            return read

        # Per-camera mount orientation lives in cameras.py (SCENE/WRIST/SIDE_ROTATE) so it's tweakable in
        # one place while dialing in the physical mounts — watch the live panel and adjust those constants.
        side_fn = (_grab(UsbCamera(side_cam, rotate=SIDE_ROTATE, gain=SIDE_GAIN))
                   if side_cam else None)
        return (robot, _grab(UsbCamera(scene_cam, rotate=SCENE_ROTATE, gain=SCENE_GAIN)),
                _grab(UsbCamera(wrist_cam, rotate=WRIST_ROTATE, gain=WRIST_GAIN)), side_fn)

    from .amazing_hand_mujoco import AmazingHandMujoco
    from .sim_arm import SimArm
    from .sim_hand import SimHand

    sim = AmazingHandMujoco()
    robot = URAmazingHand(cfg, arm=SimArm(), hand=SimHand(sim=sim, settle_steps=8))
    return robot, (lambda w, h: sim.render(width=w, height=h)), None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Record a LeRobotDataset in sim (no hardware).")
    ap.add_argument("--engine", choices=["mujoco", "kinematic"], default="mujoco")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--repo_id", default="local/ur5e_amazinghand_sim")
    ap.add_argument("--root", default="outputs/datasets/sim_hand")
    ap.add_argument("--task", default="dip and grasp")
    args = ap.parse_args()

    robot, render, render_wrist, render_side = make_engine(args.engine)
    robot.connect()

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=build_features(args.width, args.height, use_videos=False,
                                with_wrist=render_wrist is not None, with_side=render_side is not None),
        root=args.root,
        robot_type=robot.name,
        use_videos=False,
    )

    for ep in range(args.episodes):
        robot.connect()  # reset to home
        home = robot.arm.get_ee_pose()
        for f in range(args.frames):
            obs = robot.get_observation()
            state = np.array([obs[n] for n in STATE_NAMES], dtype=np.float32)
            action = scripted_action(f, args.frames, home)
            frame = {
                "observation.state": state,
                "observation.images.scene": render(args.width, args.height),
                "action": np.asarray(action, dtype=np.float32),
                "task": args.task,
            }
            if render_wrist is not None:
                frame["observation.images.wrist"] = render_wrist(args.width, args.height)
            if render_side is not None:
                frame["observation.images.side"] = render_side(args.width, args.height)
            dataset.add_frame(frame)
            robot.send_action(dict(zip(ACTION_NAMES, action)))
        dataset.save_episode()
        print(f"  saved episode {ep + 1}/{args.episodes} ({args.frames} frames)")

    robot.disconnect()
    print(f"\nDONE [{args.engine}]. root={dataset.root}  total_frames={len(dataset)}")


if __name__ == "__main__":
    main()
