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

W, H = 960, 540  # dataset image size (must match training); only used to size the warmup frame now


class _ChunkScheduler:
    """Own the action-chunk queue so a re-plan never blocks the robot.

    `policy.select_action()` computes a new chunk INLINE the moment its queue drains, so the client
    waits the whole forward pass on that step. Measured on the 4090 over a wired link: a normal step
    costs ~23 ms and the re-plan step ~124 ms, i.e. a 5x periodic spike past the 83 ms budget at 12 fps
    — the remaining source of stutter once the network was fixed.

    Here the server holds the queue and recomputes in a BACKGROUND thread as soon as it is nearly
    drained, so every client request is answered from memory and every step costs ~23 ms.

    TRADE-OFF, and it is not small: the replacement chunk is computed from an observation a few steps
    old, so its first action targets where the arm WAS. Measured on hardware, the first step of each
    chunk commands ~46 mm against ~10 mm mid-chunk (4.5x), versus 15 mm (2.5x) without prefetch — i.e.
    prefetch buys flat timing at the cost of a rougher seam.

    `--rtc` addresses that directly: SmolVLA in this lerobot ships Real-Time Chunking (`policies/rtc`,
    wired into the flow-matching denoise loop and reachable only via `predict_action_chunk`), which
    guides the incoming chunk toward the outgoing chunk's unplayed tail so the seam blends instead of
    stepping. Off by default so prefetch and RTC can be A/B'd separately.
    """

    def __init__(self, policy, pre, post, horizon: int, prefetch_at: int = 2, use_rtc: bool = False,
                 inference_delay: "int | None" = None):
        import threading

        self.policy, self.pre, self.post = policy, pre, post
        self.horizon = max(1, int(horizon))
        # refill this early so the forward pass has time to finish before the queue empties
        self.prefetch_at = max(1, min(int(prefetch_at), self.horizon - 1)) if self.horizon > 1 else 1
        self.use_rtc = bool(use_rtc)
        # how many steps elapse while the chunk is computed; RTC pins that many leading actions to the
        # outgoing chunk. Defaults to prefetch_at, i.e. exactly the steps we expect to consume.
        self.inference_delay = int(inference_delay) if inference_delay is not None else self.prefetch_at
        self._q: list = []           # postprocessed actions still to hand out
        self._raw = None             # SAME chunk pre-postprocessing (normalized) — RTC's reference frame
        self._pending = None         # (raw, actions) computed by the background thread, not installed
        self._thread = None
        self._latest_obs = None
        self._lock = threading.Lock()      # guards _q / _raw / _pending / _thread
        self._gpu = threading.Lock()       # serialises forward passes (one at a time on the device)
        self._threading = threading
        self.stats = {"inline": 0, "prefetched": 0, "waited": 0, "rtc_guided": 0, "chunk_ms": []}

    def _compute(self, obs, prev_left=None):
        """One forward pass -> (normalized chunk, postprocessed actions), trimmed to the horizon.

        `prev_left` is the outgoing chunk's not-yet-executed actions in NORMALIZED space (the units
        predict_action_chunk returns, before `post` un-normalizes). RTC guides the new chunk toward
        those so the seam blends instead of stepping; the processor zero-pads it to x_t's shape itself.
        """
        import time as _t

        import torch

        kwargs = {}
        if self.use_rtc and prev_left is not None and prev_left.shape[1] > 0:
            kwargs = {"prev_chunk_left_over": prev_left,
                      "inference_delay": self.inference_delay,
                      "execution_horizon": self.horizon}
        t0 = _t.time()
        with self._gpu, torch.no_grad():  # RTC re-enables grad internally where it needs autograd
            raw = self.policy.predict_action_chunk(self.pre(obs), **kwargs)
            phys = self.post(raw)
        ms = (_t.time() - t0) * 1000
        self.stats["chunk_ms"].append(ms)
        if kwargs:
            self.stats["rtc_guided"] += 1
        return raw[:, :self.horizon].detach(), [a for a in phys[0, :self.horizon].cpu().numpy()]

    def _spawn_prefetch(self, obs, prev_left) -> None:
        def work():
            try:
                c = self._compute(obs, prev_left)
            except Exception as e:  # noqa: BLE001 — never kill the serving thread
                print(f"[prefetch] failed: {type(e).__name__}: {e}", flush=True)
                c = None
            with self._lock:
                self._pending, self._thread = c, None

        self._thread = self._threading.Thread(target=work, daemon=True)
        self._thread.start()

    def _install(self, computed) -> None:
        """Swap a freshly computed (raw, actions) pair in as the live chunk. Caller holds _lock."""
        if computed is not None:
            self._raw, self._q = computed[0], list(computed[1])

    def next_action(self, obs):
        with self._lock:
            self._latest_obs = obs
            if not self._q:
                if self._pending is not None:
                    self._install(self._pending)
                    self._pending = None
                    self.stats["prefetched"] += 1
                    t = None
                else:
                    t = self._thread
            else:
                t = None

        if t is not None:          # queue dry but a prefetch is in flight — wait for it, don't re-run
            t.join()
            with self._lock:
                if self._pending is not None:
                    self._install(self._pending)
                    self._pending = None
                self.stats["waited"] += 1

        with self._lock:
            need_inline = not self._q
        if need_inline:            # first call, or the prefetch errored: compute on the request path
            c = self._compute(obs)
            with self._lock:
                self._install(c)
                self.stats["inline"] += 1

        with self._lock:
            action = self._q.pop(0)
            if len(self._q) <= self.prefetch_at and self._thread is None and self._pending is None:
                # the outgoing chunk's tail — the actions that will still play while we compute — is
                # RTC's anchor. It is exactly the last len(self._q) entries of the live raw chunk.
                prev_left = None
                if self.use_rtc and self._raw is not None and self._q:
                    prev_left = self._raw[:, self.horizon - len(self._q):]
                self._spawn_prefetch(self._latest_obs, prev_left)
            return action

    def reset(self) -> None:
        """Drop everything in flight — the client is starting a fresh episode from a new pose."""
        with self._lock:
            self._q, self._raw, self._pending, self._thread = [], None, None, None
        self.policy.reset()
        cm = self.stats["chunk_ms"]
        chunk = (f", chunk compute med={sorted(cm)[len(cm) // 2]:.0f}ms max={max(cm):.0f}ms n={len(cm)}"
                 if cm else "")
        rtc = f", {self.stats['rtc_guided']} RTC-guided" if self.use_rtc else ""
        print(f"[prefetch] reset (served: {self.stats['prefetched']} prefetched, "
              f"{self.stats['inline']} inline, {self.stats['waited']} waited{rtc}{chunk})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a trained policy's actions to the robot PC over TCP.")
    ap.add_argument("--ckpt", help="path to .../checkpoints/last/pretrained_model (omit with --mock)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="execution horizon / reactivity; on GPU you can afford small values")
    ap.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEFF")
    ap.add_argument("--prefetch", action="store_true",
                    help="compute the next action chunk in a BACKGROUND thread so a re-plan never "
                         "blocks the client (removes the ~124ms periodic spike; the new chunk is based "
                         "on a slightly stale observation — see _ChunkScheduler)")
    ap.add_argument("--prefetch-at", type=int, default=2,
                    help="refill once this many actions remain in the queue (--prefetch only)")
    ap.add_argument("--rtc", action="store_true",
                    help="Real-Time Chunking: guide the new chunk toward the outgoing chunk's unplayed "
                         "tail so the seam blends instead of jumping (needs --prefetch). Measured "
                         "without it: the first step of each chunk commands ~46mm vs ~10mm mid-chunk.")
    ap.add_argument("--rtc-inference-delay", type=int, default=None,
                    help="steps assumed to elapse during a chunk computation (default: --prefetch-at)")
    ap.add_argument("--rtc-max-guidance", type=float, default=10.0,
                    help="RTCConfig.max_guidance_weight — how hard the seam is pulled toward the old chunk")
    ap.add_argument("--mock", action="store_true", help="no policy; return a zero action (test only)")
    ap.add_argument("--eval-name", default=None,
                    help="name for the GPU-side auto-captured eval run (default: <model>_<timestamp>)")
    ap.add_argument("--no-eval", action="store_true", help="disable GPU-side eval capture")
    args = ap.parse_args()

    if args.mock:
        n = len(ACTION_NAMES)
        print(f"[server] MOCK mode — returning zero action (dim {n})", flush=True)
        remote.serve(args.host, args.port, lambda state, images, task: [0.0] * n, reset_fn=lambda: None)
        return

    if not args.ckpt:
        ap.error("--ckpt is required (or use --mock)")

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

    def build_obs(state, images, task):
        obs = {"observation.state": torch.from_numpy(np.asarray(state, dtype=np.float32))[None],
               "task": [task]}
        for name, img in images.items():  # scene / wrist / side — whatever the client sent
            # Pass the client's frames through UNRESIZED. remote.encode_obs already shrank them to fit
            # the 512x512 box the policy pads into, so forcing them back up to (W, H) here only
            # resampled 512x288 -> 960x540 -> 512x288 and blurred every frame for nothing.
            obs[f"observation.images.{name}"] = chw(img)
        return obs

    if args.rtc and not args.prefetch:
        ap.error("--rtc needs --prefetch (RTC blends the seam that prefetch's stale chunk creates)")
    if args.rtc:
        # SmolVLA ships RTC (policies/rtc) but leaves rtc_config=None, and select_action asserts
        # against it — it only works through predict_action_chunk, which is what the scheduler uses.
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=args.n_action_steps or 8,
                                             max_guidance_weight=args.rtc_max_guidance)
        policy.init_rtc_processor()
        print(f"[server] RTC on — execution_horizon={policy.config.rtc_config.execution_horizon}, "
              f"max_guidance_weight={args.rtc_max_guidance}", flush=True)

    scheduler = _ChunkScheduler(policy, pre, post, horizon=args.n_action_steps or 8,
                                prefetch_at=args.prefetch_at, use_rtc=args.rtc,
                                inference_delay=args.rtc_inference_delay) if args.prefetch else None
    if scheduler is not None:
        print(f"[server] PREFETCH on — horizon={scheduler.horizon}, refill when "
              f"<={scheduler.prefetch_at} actions remain (re-plan runs off the request path)"
              + (f", rtc inference_delay={scheduler.inference_delay}" if scheduler.use_rtc else ""),
              flush=True)

    def infer(state, images, task):
        obs = build_obs(state, images, task)
        if scheduler is not None:
            return scheduler.next_action(obs)
        with torch.no_grad():
            return post(policy.select_action(pre(obs)))[0].cpu().numpy()

    # image keys this policy was trained with (scene/wrist/side) — used to build the warmup obs
    try:
        img_keys = [k.split(".")[-1] for k in policy.config.input_features
                    if k.startswith("observation.images.")]
    except Exception:  # noqa: BLE001
        img_keys = []
    if not img_keys:
        img_keys = ["scene", "wrist", "side"]
    print(f"[server] image inputs: {img_keys}", flush=True)

    # Warm up so the FIRST client request isn't a ~100 s cold start (CUDA autotune + first VLM forward).
    import time as _time
    from ur5e_lerobot.schema import STATE_NAMES

    print("[server] warming up (one dummy inference)…", flush=True)
    _t = _time.time()
    _dummy = np.zeros((H, W, 3), dtype=np.uint8)
    infer(np.zeros(len(STATE_NAMES), dtype=np.float32),
          {k: _dummy for k in img_keys}, "warmup")
    policy.reset()  # clear the queue so the first real request re-plans from the real observation
    print(f"[server] warmup done in {_time.time() - _t:.1f}s — ready", flush=True)

    # GPU-side eval capture: record this serving session as outputs/evals/<name>/eval.json so it shows
    # up in the manager. This captures the OBJECTIVE inference stats (checkpoint, episodes≈resets,
    # inference count + latency); the operator fills in success/rating/notes from the GUI afterward.
    # with --prefetch the scheduler owns the queue, so a reset must clear ITS state too (otherwise a
    # new episode replays actions planned from the previous episode's pose)
    reset_all = scheduler.reset if scheduler is not None else policy.reset
    serve_infer, serve_reset = infer, reset_all
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

        def serve_infer(state, images, task):  # noqa: F811
            _t0 = _t.time()
            a = infer(state, images, task)
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
            reset_all()  # NOT policy.reset() — with --prefetch the scheduler's queue must clear too

    remote.serve(args.host, args.port, serve_infer, reset_fn=serve_reset)


if __name__ == "__main__":
    main()
