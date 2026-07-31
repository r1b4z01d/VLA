"""Run a trained LeRobot policy closed-loop in the MuJoCo sim and record it.

Loads an ACT (or other) checkpoint, drives the UR5e + AmazingHand cell through the
URAmazingHand adapter, and writes a scene|wrist video so you can watch the policy attempt
the task. Runs on the Mac (has both MuJoCo and LeRobot).

    .venv/bin/python -m ur5e_lerobot.sim.deploy \
        --ckpt outputs/train/act_pickplace/checkpoints/last/pretrained_model \
        --episodes 4 --out outputs/playback/policy.mp4
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from ..robot import URAmazingHand, URAmazingHandConfig
from ..schema import ACTION_NAMES, STATE_NAMES
from .mujoco_cell import CellArm, CellHand, MujocoCell

TASK = "put the block on the green pad"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_policy(ckpt: str, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
    policy.eval()
    policy.to(device)
    # the saved processors pin device=cuda; override to this machine's device
    dev_override = {"device_processor": {"device": device}}
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=ckpt,
        preprocessor_overrides=dev_override, postprocessor_overrides=dev_override,
    )
    return policy, pre, post, cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy a trained policy in the sim.")
    ap.add_argument("--ckpt", required=True, help="path to .../checkpoints/<step>/pretrained_model")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400, help="max control steps per episode")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--out", default="outputs/playback/policy.mp4")
    ap.add_argument("--device", default=_pick_device())
    args = ap.parse_args()

    from lerobot.utils.control_utils import predict_action

    device = args.device
    policy, pre, post, cfg = load_policy(args.ckpt, device)
    print(f"loaded {cfg.type} on {device}")

    cell = MujocoCell(settle_steps=8)
    robot = URAmazingHand(URAmazingHandConfig(id="deploy", cameras={}), arm=CellArm(cell), hand=CellHand(cell))
    robot.connect()
    bid = __import__("mujoco").mj_name2id(cell.model, __import__("mujoco").mjtObj.mjOBJ_BODY, "block")

    import cv2
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = imageio.get_writer(args.out, fps=20, codec="libx264", quality=8, macro_block_size=16)

    successes = 0
    for ep in range(args.episodes):
        robot.connect()  # reset scene (block to start, arm home)
        policy.reset()
        pre.reset()
        post.reset()
        z0 = float(cell.data.xpos[bid][2])
        for _ in range(args.steps):
            obs = robot.get_observation()
            state = np.array([obs[n] for n in STATE_NAMES], dtype=np.float32)
            scene = cell.render(args.width, args.height)
            wrist = cell.render_wrist(args.width, args.height)
            observation = {
                "observation.state": state,
                "observation.images.scene": scene,
                "observation.images.wrist": wrist,
            }
            action_t = predict_action(
                observation=observation, policy=policy, device=torch.device(device),
                preprocessor=pre, postprocessor=post,
                use_amp=getattr(cfg, "use_amp", False), task=TASK, robot_type=robot.name,
            )
            action = np.asarray(action_t).reshape(-1)[: len(ACTION_NAMES)]
            robot.send_action(dict(zip(ACTION_NAMES, action.tolist())))
            # video frame (scene | wrist)
            w = cv2.resize(wrist, (scene.shape[1], scene.shape[0])) if wrist.shape != scene.shape else wrist
            frame = np.hstack([scene, w])
            cv2.putText(frame, f"ep {ep}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            writer.append_data(frame)
        lifted = float(cell.data.xpos[bid][2]) - z0
        print(f"  ep {ep}: block dz={lifted:+.3f}")
        successes += lifted > 0.05
    writer.close()
    print(f"\nDONE. {successes}/{args.episodes} episodes lifted the block. video -> {args.out}")


if __name__ == "__main__":
    main()
