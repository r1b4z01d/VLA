"""Remote inference SERVER — load a policy (ACT|SmolVLA) on the GPU and serve actions to the robot PC's
`eval_hw.py --remote` client. Run on the GPU box (SmolVLA is ~2.6 s/inference on the robot PC CPU but
~tens of ms on the 4090). See ur5e_lerobot/remote.py for the SSH-relay topology across subnets.

    .venv/bin/python scripts/infer_server.py \
        --ckpt outputs/train/<run>/checkpoints/last/pretrained_model --device cuda --port 8777 \
        [--n-action-steps 8]     # reactivity lives HERE (server-side), not on the client

--mock: skip the policy and return a zero action — for offline protocol/client testing with no GPU/ckpt.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> ur5e_lerobot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                     # scripts/ -> eval_hw

from ur5e_lerobot import remote
from ur5e_lerobot.schema import ACTION_NAMES

W, H = 320, 240  # dataset image size (must match training / the client's resize)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a trained policy's actions to the robot PC over TCP.")
    ap.add_argument("--ckpt", help="path to .../checkpoints/last/pretrained_model (omit with --mock)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="execution horizon / reactivity; on GPU you can afford small values")
    ap.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEFF")
    ap.add_argument("--mock", action="store_true", help="no policy; return a zero action (test only)")
    args = ap.parse_args()

    if args.mock:
        n = len(ACTION_NAMES)
        print(f"[server] MOCK mode — returning zero action (dim {n})", flush=True)
        remote.serve(args.host, args.port, lambda state, scene, wrist, task: [0.0] * n, reset_fn=lambda: None)
        return

    if not args.ckpt:
        ap.error("--ckpt is required (or use --mock)")

    import cv2
    import numpy as np
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    import eval_hw  # reuse the exact policy-load + reactivity logic the local eval uses

    policy, ptype = eval_hw._load_policy(args.ckpt, args.device)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt,
                                         preprocessor_overrides={"device_processor": {"device": args.device}})
    eval_hw._set_reactivity(policy, ptype, args.n_action_steps, args.temporal_ensemble)
    policy.reset()
    print(f"[server] loaded {ptype} on {args.device}", flush=True)

    def chw(img):
        return torch.from_numpy(img.copy()).permute(2, 0, 1).float().div(255)[None]

    def infer(state, scene, wrist, task):
        obs = {"observation.state": torch.from_numpy(np.asarray(state, dtype=np.float32))[None],
               "observation.images.scene": chw(cv2.resize(scene, (W, H))),
               "observation.images.wrist": chw(cv2.resize(wrist, (W, H))),
               "task": [task]}
        with torch.no_grad():
            return post(policy.select_action(pre(obs)))[0].cpu().numpy()

    # Warm up so the FIRST client request isn't a ~100 s cold start (CUDA autotune + first VLM forward).
    import time as _time
    from ur5e_lerobot.schema import STATE_NAMES

    print("[server] warming up (one dummy inference)…", flush=True)
    _t = _time.time()
    _dummy = np.zeros((H, W, 3), dtype=np.uint8)
    infer(np.zeros(len(STATE_NAMES), dtype=np.float32), _dummy, _dummy, "warmup")
    policy.reset()  # clear the queue so the first real request re-plans from the real observation
    print(f"[server] warmup done in {_time.time() - _t:.1f}s — ready", flush=True)

    remote.serve(args.host, args.port, infer, reset_fn=policy.reset)


if __name__ == "__main__":
    main()
