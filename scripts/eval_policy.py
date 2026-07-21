"""Evaluate a trained policy in the MuJoCo sim: roll out pick-and-place and measure success.

Loads a policy checkpoint, runs it in the combined UR5e+AmazingHand cell for N episodes
(optionally randomizing box+goal each episode), and scores success = the block ends within the
target radius of the goal. Renders one rollout to a frame strip.

    PYTHONPATH=. .venv/bin/python scripts/eval_policy.py \
        --ckpt outputs/train/merged_act/checkpoints/last/pretrained_model \
        --episodes 10 --steps 300 --device cuda [--randomize]
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from ur5e_lerobot.schema import ACTION_NAMES, STATE_NAMES
from ur5e_lerobot.sim.combined_model import TARGET_RADIUS
from ur5e_lerobot.sim.record_sim import make_engine

W, H = 320, 240  # dataset image size
TASK = "put the block on the green pad"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/train/merged_act/checkpoints/last/pretrained_model")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--randomize", action="store_true", help="random box+goal each episode (else fixed)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/eval_rollout.png", help="frame-strip PNG of episode 0")
    ap.add_argument("--video", action="store_true", help="also write an MP4 of every rollout")
    ap.add_argument("--video-out", default="outputs/eval.mp4")
    args = ap.parse_args()

    policy = ACTPolicy.from_pretrained(args.ckpt)
    policy.eval()
    policy.to(args.device)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt,
                                         preprocessor_overrides={"device_processor": {"device": args.device}})

    robot, render_fn, wrist_fn = make_engine("mujoco")
    robot.connect()
    cell = robot.arm.cell
    bqa, gid = cell._block_qadr, cell._goal_geom

    def chw(img):
        return torch.from_numpy(img.copy()).permute(2, 0, 1).float().div(255)[None]

    def observe():
        o = robot.get_observation()
        state = np.array([o[n] for n in STATE_NAMES], dtype=np.float32)
        scene = render_fn(640, 480)  # full-res scene, reused for the obs (resized) and the video
        obs = {"observation.state": torch.from_numpy(state)[None],
               "observation.images.scene": chw(cv2.resize(scene, (W, H))),
               "observation.images.wrist": chw(cv2.resize(wrist_fn(640, 480), (W, H))),
               "task": [TASK]}
        return obs, scene

    writer = None
    if args.video:
        writer = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (640, 480))
    results, frames = [], []
    for ep in range(args.episodes):
        cell.reset()
        if args.randomize:
            cell.randomize_block()
            cell.randomize_goal()
        policy.reset()
        goal_xy = cell.model.geom_pos[gid][:2].copy()
        start_xy = cell.data.qpos[bqa:bqa + 2].copy()
        max_z = -1.0
        for t in range(args.steps):
            obs, scene = observe()
            with torch.no_grad():
                act = post(policy.select_action(pre(obs)))[0].cpu().numpy()
            robot.send_action(dict(zip(ACTION_NAMES, act)))
            max_z = max(max_z, float(cell.data.qpos[bqa + 2]))
            if writer is not None:
                f = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
                cv2.putText(f, f"ep{ep} t{t}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                writer.write(f)
            if ep == 0 and t % max(1, args.steps // 6) == 0:
                frames.append(cv2.resize(scene, (320, 240)))
        dist = float(np.linalg.norm(cell.data.qpos[bqa:bqa + 2] - goal_xy))
        moved = float(np.linalg.norm(cell.data.qpos[bqa:bqa + 2] - start_xy))
        ok = dist <= TARGET_RADIUS
        results.append(ok)
        print(f"ep {ep:2d}: {'SUCCESS' if ok else 'fail   '}  final_dist={dist * 100:5.1f}cm  "
              f"max_lift={max_z * 100:4.1f}cm  moved={moved * 100:4.1f}cm")

    sr = sum(results) / len(results)
    print(f"\nSUCCESS RATE ({'random' if args.randomize else 'fixed'} pos): "
          f"{sr * 100:.0f}%  ({sum(results)}/{len(results)})")
    if writer is not None:
        writer.release()
        print(f"wrote {args.video_out}")
    if frames:
        Image.fromarray(np.hstack(frames)).save(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
