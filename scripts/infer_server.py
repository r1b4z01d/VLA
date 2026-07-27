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
    ap.add_argument("--eval-name", default=None,
                    help="name for the GPU-side auto-captured eval run (default: <model>_<timestamp>)")
    ap.add_argument("--no-eval", action="store_true", help="disable GPU-side eval capture")
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

    # GPU-side eval capture: record this serving session as outputs/evals/<name>/eval.json so it shows
    # up in the manager. This captures the OBJECTIVE inference stats (checkpoint, episodes≈resets,
    # inference count + latency); the operator fills in success/rating/notes from the GUI afterward.
    serve_infer, serve_reset = infer, policy.reset
    if not args.no_eval:
        import json as _json
        import time as _t

        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _parts = os.path.abspath(args.ckpt).split(os.sep)
        _model = _parts[_parts.index("train") + 1] if "train" in _parts else ptype
        _name = args.eval_name or f"{_model}_{_t.strftime('%Y%m%d_%H%M%S')}"
        _run = os.path.join(_root, "outputs", "evals", _name)
        os.makedirs(_run, exist_ok=True)
        _ev = {"created": _t.time(), "source": "auto", "model": _model, "checkpoint": args.ckpt,
               "device": args.device, "n_action_steps": args.n_action_steps,
               "episodes": 0, "inferences": 0, "task": None}
        _lat: list[float] = []

        def _save() -> None:
            s = dict(_ev)
            if _lat:
                s["infer_ms_avg"] = round(sum(_lat) / len(_lat), 1)
            with open(os.path.join(_run, "eval.json"), "w") as f:
                _json.dump(s, f, indent=2, sort_keys=True)

        _save()
        print(f"[server] eval capture -> outputs/evals/{_name}", flush=True)

        def serve_infer(state, scene, wrist, task):  # noqa: F811
            _t0 = _t.time()
            a = infer(state, scene, wrist, task)
            _lat.append((_t.time() - _t0) * 1000)
            _ev["inferences"] += 1
            if _ev["task"] is None and task and task != "warmup":
                _ev["task"] = task
            if _ev["inferences"] % 50 == 0:
                _save()
            return a

        def serve_reset():  # noqa: F811
            _ev["episodes"] += 1
            _save()
            policy.reset()

    remote.serve(args.host, args.port, serve_infer, reset_fn=serve_reset)


if __name__ == "__main__":
    main()
