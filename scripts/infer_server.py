"""Remote inference SERVER — load a policy (ACT|SmolVLA) on a GPU and serve actions to the
`eval_hw.py --remote` client. Since 2026-10-10 it runs ON THE ROBOT (the Jetson Orin) as the supervisor
unit vla_infer_server (services/rd/vla_infer_server.conf, loopback :8777) and the eval client on the same
box connects to 127.0.0.1:8777; the GPU box's own unit (services/vla_infer_server.conf) is a fallback for
driving from the GPU box over the LAN. See ur5e_lerobot/remote.py for the topology.

    .venv/bin/python scripts/infer_server.py \
        --ckpt outputs/train/<run>/checkpoints/last/pretrained_model --device cuda --port 8777 \
        [--n-action-steps 8]     # reactivity lives HERE (server-side), not on the client
        [--also <run> ...]       # keep more checkpoints RESIDENT; {'cmd':'select','model':<run>} switches
                                 # between them instantly (a route's grab and place models, ~1 GB each)

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
                 inference_delay: "int | None" = None, execution_horizon: "int | None" = None):
        import threading

        self.policy, self.pre, self.post = policy, pre, post
        self.horizon = max(1, int(horizon))
        # refill this early so the forward pass has time to finish before the queue empties
        self.prefetch_at = max(1, min(int(prefetch_at), self.horizon - 1)) if self.horizon > 1 else 1
        self.use_rtc = bool(use_rtc)
        # how many steps elapse while the chunk is computed; RTC pins that many leading actions to the
        # outgoing chunk. Defaults to prefetch_at, i.e. exactly the steps we expect to consume.
        self.inference_delay = int(inference_delay) if inference_delay is not None else self.prefetch_at
        # How far RTC's guidance reaches. Defaults to the whole served horizon, which anchors EVERY
        # executed action to the previous plan -- smooth, but the policy then can't react to a moved
        # object. Keep it just above inference_delay to leave the tail free.
        self.execution_horizon = int(execution_horizon) if execution_horizon else self.horizon
        self._q: list = []           # postprocessed actions still to hand out
        self._raw = None             # SAME chunk pre-postprocessing (normalized) — RTC's reference frame
        self._pending = None         # (raw, actions) computed by the background thread, not installed
        self._thread = None
        self._latest_obs = None
        # Bumped by reset(). A prefetch carries the epoch it was spawned under and discards its result
        # if that no longer matches, so a chunk planned for a finished episode can never be installed
        # into the next one. See reset() for the failure this prevents.
        self._epoch = 0
        self._lock = threading.Lock()      # guards _q / _raw / _pending / _thread / _epoch
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
                      "execution_horizon": self.execution_horizon}
        t0 = _t.time()
        with self._gpu, torch.no_grad():  # RTC re-enables grad internally where it needs autograd
            raw = self.policy.predict_action_chunk(self.pre(obs), **kwargs)
            phys = self.post(raw)
        ms = (_t.time() - t0) * 1000
        self.stats["chunk_ms"].append(ms)
        if kwargs:
            self.stats["rtc_guided"] += 1
        # Keep prefetch_at extra actions beyond the horizon: a prefetched chunk is planned from an
        # observation taken when prefetch_at actions were still queued, and those play while it
        # computes, so _install skips that many and the horizon that follows is the fresh part.
        keep = min(self.horizon + self.prefetch_at, raw.shape[1])
        return raw[:, :keep].detach(), [a for a in phys[0, :keep].cpu().numpy()]

    def _spawn_prefetch(self, obs, prev_left) -> None:
        epoch = self._epoch          # the episode this chunk is being planned for

        def work():
            try:
                c = self._compute(obs, prev_left)
            except Exception as e:  # noqa: BLE001 — never kill the serving thread
                print(f"[prefetch] failed: {type(e).__name__}: {e}", flush=True)
                c = None
            with self._lock:
                if epoch != self._epoch:
                    # reset() ran while this was computing: the chunk describes a finished episode's
                    # pose. Dropping it on the floor is the whole point of the epoch counter — do NOT
                    # touch _thread here, a newer prefetch may already own it.
                    return
                self._pending, self._thread = c, None

        self._thread = self._threading.Thread(target=work, daemon=True)
        self._thread.start()

    def _install(self, computed, skip: int = 0) -> None:
        """Swap a freshly computed (raw, actions) pair in as the live chunk. Caller holds _lock.

        `skip` is how many of the chunk's leading actions already elapsed while it was computing
        (prefetch_at for a prefetched chunk, 0 for an inline one). Before 2026-10-11 every chunk was
        installed from action 0, so a prefetched chunk re-played the stretch the arm had just driven:
        a seam of prefetch_at x (per-step motion) — 46 mm on the GPU box at prefetch_at=2, and a
        ~200 mm snap back every chunk on the Jetson at prefetch_at=21 (the arm "rocking"). RTC makes
        the skipped prefix consistent with the old tail, so starting at `skip` is seamless.
        """
        if computed is None:
            return
        raw, acts = computed
        if skip and len(acts) - skip < 1:
            print(f"[prefetch] chunk too short to skip {skip} elapsed actions (len {len(acts)}); "
                  "installing from 0 — raise the policy chunk_size or lower --prefetch-at", flush=True)
            skip = 0
        self._raw = raw[:, skip:skip + self.horizon]
        self._q = list(acts[skip:skip + self.horizon])

    def next_action(self, obs):
        with self._lock:
            self._latest_obs = obs
            if not self._q:
                if self._pending is not None:
                    self._install(self._pending, skip=self.prefetch_at)
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
                    self._install(self._pending, skip=self.prefetch_at)
                    self._pending = None
                self.stats["waited"] += 1

        with self._lock:
            need_inline = not self._q
        if need_inline:            # first call, or the prefetch errored: compute on the request path
            c = self._compute(obs)
            with self._lock:
                self._install(c, skip=0)      # computed from the current observation: nothing elapsed
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
        """Drop everything in flight — the client is starting a fresh episode from a new pose.

        The epoch bump is load-bearing. Clearing _pending is not enough: a prefetch spawned near the end
        of the previous episode is still running, and it used to write its result into _pending AFTER
        this returned. That chunk was planned from the old episode's final observation (hand closing on
        the object), and the next episode installed it as soon as its first chunk drained — measured on
        hardware as an 8-step window (one horizon) deviating ~50x the repeat-noise floor, starting
        around step 10 of every run after the first. On the robot that looked like the fingers grasping,
        opening and fluttering before the real behaviour resumed. Run 1 was always clean because nothing
        preceded it.
        """
        with self._lock:
            self._epoch += 1
            self._q, self._raw, self._pending, self._thread = [], None, None, None
            self._latest_obs = None
        self.policy.reset()
        cm = self.stats["chunk_ms"]
        chunk = (f", chunk compute med={sorted(cm)[len(cm) // 2]:.0f}ms max={max(cm):.0f}ms n={len(cm)}"
                 if cm else "")
        rtc = f", {self.stats['rtc_guided']} RTC-guided" if self.use_rtc else ""
        print(f"[prefetch] reset (served: {self.stats['prefetched']} prefetched, "
              f"{self.stats['inline']} inline, {self.stats['waited']} waited{rtc}{chunk})", flush=True)


def _capture_for(real: str) -> "dict | None":
    """The camera geometry a checkpoint was trained on (capture_meta): training copies the dataset's
    meta/capture.json into outputs/train/<run>/capture.json; fall back to the dataset root named in
    train_config.json when that copy is missing (older runs). The eval applies it to the cameras, so a
    floor-task model and a shelf-task model each get their own framing."""
    import json as _json

    run_dir = os.path.dirname(os.path.dirname(os.path.dirname(real))) if real else ""
    cands = [os.path.join(run_dir, "capture.json")] if run_dir else []
    try:
        tc = _json.load(open(os.path.join(real, "train_config.json")))
        droot = (tc.get("dataset") or {}).get("root")
        if droot:
            cands.append(os.path.join(droot, "meta", "capture.json"))
    except Exception:  # noqa: BLE001
        pass
    for p in cands:
        if os.path.isfile(p):
            try:
                capture = _json.load(open(p))
                capture["source"] = p
                return capture
            except Exception:  # noqa: BLE001
                continue
    return None


class Served:
    """One resident checkpoint: policy + processors + its own chunk scheduler + the geometry it wants."""

    def __init__(self, ckpt: str, args, eval_hw, make_pre_post_processors):
        import torch

        self.ckpt = ckpt
        self.real = os.path.realpath(ckpt)
        self.tail = self.real.split("/outputs/", 1)[-1] if "/outputs/" in self.real else self.real
        parts = self.real.split(os.sep)
        self.run = parts[parts.index("train") + 1] if "train" in parts else os.path.basename(self.real)
        self.policy, self.ptype = eval_hw._load_policy(ckpt, args.device)
        if args.num_steps and hasattr(self.policy.config, "num_steps"):
            self.policy.config.num_steps = int(args.num_steps)
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": args.device}})
        eval_hw._set_reactivity(self.policy, self.ptype, args.n_action_steps, args.temporal_ensemble)
        self.policy.reset()
        if args.rtc:
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

            eh = args.rtc_execution_horizon or (args.n_action_steps or 8)
            self.policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=eh,
                                                      max_guidance_weight=args.rtc_max_guidance)
            self.policy.init_rtc_processor()
        self.scheduler = _ChunkScheduler(self.policy, self.pre, self.post, horizon=args.n_action_steps or 8,
                                         prefetch_at=args.prefetch_at, use_rtc=args.rtc,
                                         inference_delay=args.rtc_inference_delay,
                                         execution_horizon=args.rtc_execution_horizon) if args.prefetch else None
        try:
            self.img_keys = [k.split(".")[-1] for k in self.policy.config.input_features
                             if k.startswith("observation.images.")]
        except Exception:  # noqa: BLE001
            self.img_keys = []
        if not self.img_keys:
            self.img_keys = ["scene", "wrist", "side"]
        self.capture = _capture_for(self.real)
        self._torch = torch

    def matches(self, key: str) -> bool:
        k = key.strip()
        return k in (self.run, self.tail, self.ckpt, self.real) or (os.path.exists(k) and os.path.realpath(k) == self.real)

    def infer(self, obs):
        if self.scheduler is not None:
            return self.scheduler.next_action(obs)
        with self._torch.no_grad():
            return self.post(self.policy.select_action(self.pre(obs)))[0].cpu().numpy()

    def reset(self) -> None:
        if self.scheduler is not None:
            self.scheduler.reset()      # drops any in-flight prefetch too
        else:
            self.policy.reset()

    def brief(self) -> dict:
        return {"run": self.run, "checkpoint": self.tail, "checkpoint_realpath": self.real,
                "capture": (self.capture or {}).get("squares")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a trained policy's actions to the robot PC over TCP.")
    ap.add_argument("--ckpt", help="path to .../checkpoints/last/pretrained_model (omit with --mock); "
                                   "served at start")
    ap.add_argument("--also", action="append", default=[], metavar="RUN|PATH",
                    help="more checkpoints to keep RESIDENT (a run name under outputs/train, run@step, or a "
                         "path); {'cmd':'select','model':RUN} switches between them without a restart")
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
    ap.add_argument("--rtc-execution-horizon", type=int, default=None,
                    help="how many leading actions RTC guides toward the previous chunk (default: the "
                         "whole served horizon). This is a SMOOTHNESS vs RESPONSIVENESS dial: guidance "
                         "decays linearly from --rtc-inference-delay to here, and anything inside the "
                         "window is anchored to the old plan, so setting it to the full horizon leaves "
                         "NO served action free to react to a new observation (the policy then keeps "
                         "driving to where the object was). Set it just above the inference delay — e.g. "
                         "3 with delay 2 — to pin only the in-flight actions and leave the rest free.")
    ap.add_argument("--num-steps", type=int, default=None,
                    help="override the flow-matching denoising steps per chunk (SmolVLA config num_steps, "
                         "default 10). Fewer = faster chunks at the cost of action fidelity: on the Jetson "
                         "AGX Orin 10 steps ~875 ms, 6 ~600 ms (mean action drift 0.035), 4 ~460 ms (0.054).")
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
    if args.rtc and not args.prefetch:
        ap.error("--rtc needs --prefetch (RTC blends the seam that prefetch's stale chunk creates)")

    import numpy as np
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    import eval_hw  # reuse the exact policy-load + reactivity logic the local eval uses
    from ur5e_lerobot.services import resolve_checkpoint

    # Resolve + dedupe the resident set (serve_ckpt usually points at one of the --also runs).
    wanted = [args.ckpt]
    for a in args.also:
        try:
            wanted.append(resolve_checkpoint(a))
        except FileNotFoundError as e:
            print(f"[server] --also {a}: {e} — skipped", flush=True)
    models: list = []
    for ck in wanted:
        real = os.path.realpath(ck)
        if any(m.real == real for m in models):
            continue
        import time as _time
        _t = _time.time()
        m = Served(ck, args, eval_hw, make_pre_post_processors)
        models.append(m)
        print(f"[server] loaded {m.ptype} {m.tail} on {args.device} in {_time.time() - _t:.1f} s"
              + (f" (num_steps={m.policy.config.num_steps})" if args.num_steps else "")
              + f"; image inputs {m.img_keys}; capture {(m.capture or {}).get('squares')}", flush=True)
    active = {"m": models[0]}
    if args.rtc:
        print(f"[server] RTC on — execution_horizon={models[0].policy.config.rtc_config.execution_horizon}, "
              f"max_guidance_weight={args.rtc_max_guidance}", flush=True)
    if models[0].scheduler is not None:
        sch = models[0].scheduler
        print(f"[server] PREFETCH on — horizon={sch.horizon}, refill when <={sch.prefetch_at} actions remain "
              f"(re-plan runs off the request path)"
              + (f", rtc inference_delay={sch.inference_delay}" if sch.use_rtc else ""), flush=True)
    if len(models) > 1:
        print(f"[server] {len(models)} models resident: {[m.run for m in models]} — active: {models[0].run}", flush=True)

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

    def infer(state, images, task):
        return active["m"].infer(build_obs(state, images, task))

    # Warm up EVERY resident model so the first request (and the first request after a select) isn't
    # a ~100 s cold start (CUDA autotune + first VLM forward).
    import time as _time
    from ur5e_lerobot.schema import STATE_NAMES

    _dummy = np.zeros((H, W, 3), dtype=np.uint8)
    for m in models:
        print(f"[server] warming up {m.run} (one dummy inference)…", flush=True)
        _t = _time.time()
        m.infer(build_obs(np.zeros(len(STATE_NAMES), dtype=np.float32), {k: _dummy for k in m.img_keys}, "warmup"))
        m.reset()  # clear the queue so the first real request re-plans from the real observation
        print(f"[server] warmup done in {_time.time() - _t:.1f}s", flush=True)
    print("[server] ready", flush=True)

    # GPU-side eval capture: record each serving session (per active model) as outputs/evals/<name>/
    # eval.json so it shows up in the manager. This captures the OBJECTIVE inference stats (checkpoint,
    # episodes≈resets, inference count + latency); the operator fills in success/rating/notes from the
    # GUI afterward. Written LAZILY, on the first inference / reset: a server that only answered `info`
    # (a model switch, an eval preflight) used to leave a 0-episode record on every start.
    serve_infer, serve_reset = infer, lambda: active["m"].reset()
    cap = {"ev": None, "run": None, "lat": []}
    if not args.no_eval:
        import json as _json

        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        def _new_capture(m: "Served") -> None:
            name = args.eval_name or f"{m.run}_{_time.strftime('%Y%m%d_%H%M%S')}"
            cap["run"] = os.path.join(_root, "outputs", "evals", name)
            cap["ev"] = {"created": _time.time(), "source": "auto", "model": m.run, "checkpoint": m.tail,
                         "device": args.device, "n_action_steps": args.n_action_steps,
                         "episodes": 0, "inferences": 0, "task": None}
            cap["lat"] = []
            print(f"[server] eval capture -> outputs/evals/{name} (written on the first inference)", flush=True)

        def _save() -> None:
            os.makedirs(cap["run"], exist_ok=True)
            s = dict(cap["ev"])
            if cap["lat"]:
                s["infer_ms_avg"] = round(sum(cap["lat"]) / len(cap["lat"]), 1)
            with open(os.path.join(cap["run"], "eval.json"), "w") as f:
                _json.dump(s, f, indent=2, sort_keys=True)

        _new_capture(models[0])

        def serve_infer(state, images, task):  # noqa: F811
            _t0 = _time.time()
            a = infer(state, images, task)
            cap["lat"].append((_time.time() - _t0) * 1000)
            ev = cap["ev"]
            ev["inferences"] += 1
            if ev["task"] is None and task and task != "warmup":
                ev["task"] = task
            if ev["inferences"] == 1 or ev["inferences"] % 50 == 0:
                _save()
            return a

        def serve_reset():  # noqa: F811
            cap["ev"]["episodes"] += 1
            _save()
            active["m"].reset()  # NOT policy.reset() — with --prefetch the scheduler's queue must clear too

    def serve_info() -> dict:
        """Answer {'cmd':'info'}: what this process serves, RESOLVED. --ckpt is usually the
        outputs/serve_ckpt symlink, which is useless in an eval record (it is repointed between runs);
        the realpath gives train/<run>/checkpoints/<step>, which is what groups evals by model."""
        m = active["m"]
        return {"checkpoint": m.tail, "checkpoint_realpath": m.real, "run": m.run,
                "n_action_steps": args.n_action_steps, "prefetch": bool(args.prefetch),
                "rtc": bool(args.rtc), "rtc_execution_horizon": args.rtc_execution_horizon,
                "num_steps": args.num_steps, "device": args.device, "capture": m.capture,
                "models": [x.brief() for x in models]}

    def serve_cmd(msg: dict) -> dict:
        """{'cmd':'select','model': run|path}: make a resident model the active one (instant)."""
        if msg.get("cmd") != "select":
            return {"ok": False, "error": f"unknown cmd {msg.get('cmd')!r}"}
        key = str(msg.get("model", ""))
        m = next((x for x in models if x.matches(key)), None)
        if m is None:
            return {"ok": False, "error": f"model {key!r} is not resident; resident: {[x.run for x in models]}"}
        if m is not active["m"]:
            _t = _time.time()
            active["m"].reset()         # drop the old model's in-flight chunk
            active["m"] = m
            m.reset()
            if not args.no_eval and cap["ev"] is not None:
                _new_capture(m)
            print(f"[server] selected {m.run} in {(_time.time() - _t) * 1000:.0f} ms", flush=True)
        return {"ok": True, **serve_info()}

    remote.serve(args.host, args.port, serve_infer, reset_fn=serve_reset, info_fn=serve_info, cmd_fn=serve_cmd)


if __name__ == "__main__":
    main()
