"""Deploy a trained policy (ACT or SmolVLA) on the REAL hardware — the policy drives the UR5e +
AmazingHand from live camera + proprioceptive observations. The policy class is auto-detected from the
checkpoint's config.json. **MOTION-PRODUCING; keep the e-stop in hand.** Runs on the robot PC (that's
where the cameras + RTDE + hand live).

    ~/VLA/run.sh scripts/eval_hw.py --ckpt outputs/train/<run>/checkpoints/last/pretrained_model \
        --fps 12 [--video]

**Interactive, Stream Deck driven — LAUNCHES PAUSED.** Four keys (a stdin fallback maps the same
letters if no deck is found):
- **PLAY / PAUSE** (`p`) — toggle the policy. Paused holds the arm; play (re)starts it and resets the
  policy so it re-observes from the current pose.
- **FREEDRIVE** (`f`) — toggle gravity-comp teach mode so you can hand-guide the arm to a *training*
  start pose. The policy is suspended while freedriving; pressing PLAY exits freedrive and starts.
- **RESET** (`r`) — reset the policy's internal state and pause (arm holds). Abort a run and start clean.
- **EXIT** (`x`) — stop and disconnect cleanly (Ctrl-C also aborts).

Typical loop: launch (paused) → FREEDRIVE, position the arm at a training-like start, FREEDRIVE off →
PLAY → watch → PAUSE / reposition / PLAY again → EXIT. There is NO automatic success metric on
hardware — watch it and judge; `--video` records the scene feed for review. Safety: the `rtde_arm`
no-go + `max_step` + IK-reachability clamps stay active, so the arm ramps toward targets and can't jump.

Notes:
- `--device cpu` by default (the robot PC has no CUDA). ACT (~52M) keeps up at 12 fps on CPU. **SmolVLA
  (~450M) is ~2.6 s/inference on this CPU (measured) — unusable for real-time.** Run it via `--remote
  HOST:PORT` (inference on the GPU box, see scripts/infer_server.py + ur5e_lerobot/remote.py); in remote
  mode no policy/checkpoint/deps are needed here. Local SmolVLA only works at a very low query rate.
- Policy class auto-detected from config.json. ACT loads on lerobot 0.4.4; SmolVLA also needs
  `transformers` + `num2words` + the SmolVLM2 HF cache in the eval env (only relevant for LOCAL inference).
- Offline box: seed `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` once (the ACT backbone
  fetches ImageNet weights on build); `run.sh` sets `HF_HUB_OFFLINE=1`. Trained weights overwrite it.
- `--steps N` (optional) auto-pauses after N *played* steps as a safety cap; default 0 = unlimited.
- Reactivity: both ACT (chunk 100) and SmolVLA (chunk 50) default to executing the whole chunk (fully
  OPEN-LOOP). `--n-action-steps 8` (both) or `--temporal-ensemble 0.01` (ACT-only) makes it closed-loop
  WITHOUT retraining.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import time

import cv2
import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors

from ur5e_lerobot.schema import ACTION_NAMES, HAND_CURL_DIM, STATE_NAMES
from ur5e_lerobot.sim.record_sim import make_engine

W, H = 960, 540  # dataset image size (must match training)
TOOL_VOLTAGE = 12  # UR tool output voltage powering the AmazingHand ESP32 (fed from the tool port)

# Stream Deck key assignments (LOGICAL coords; StreamDeckPad flip-remaps to the flipped hardware).
KEY_PLAY = 0       # top-left
KEY_FREEDRIVE = 1  # next to play
KEY_RESET = 2      # reset policy + pause
KEY_EXIT = 4       # top-right corner (blank key 3 between it and the rest -> hard to hit by accident)
KEY_OPEN_HAND = 5  # row 2, start — release the grasp (key 3 stays blank as the EXIT guard)


def _power_cycle_tool(robot_ip: str, voltage: int = TOOL_VOLTAGE) -> None:
    """Toggle UR tool voltage OFF->ON to reboot the tool-powered AmazingHand ESP32. Sent as URScript
    to the UR secondary interface (:30002); this preempts any running control script, so the arm must
    be reconnected afterwards."""
    import socket

    def send(v: int) -> None:
        try:
            s = socket.create_connection((robot_ip, 30002), timeout=3)
            s.sendall(f"set_tool_voltage({int(v)})\n".encode())
            time.sleep(0.3)
            s.close()
        except Exception:  # noqa: BLE001
            pass

    send(0)
    time.sleep(1.5)
    send(voltage)


def _connect_with_hand_power(robot, robot_ip: str, voltage: int) -> None:
    """Connect the robot; if the hand is unreachable because the UR tool is unpowered, power-cycle the
    tool (12 V) and retry once. Mirrors the teleop panel's _connect_robot."""
    try:
        robot.connect()
    except Exception as e:  # noqa: BLE001  — most likely the hand socket (ESP32 off / tool unpowered)
        print(f"[connect] {e} -> power-cycling the UR tool ({voltage} V) + retrying the hand")
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
        _power_cycle_tool(robot_ip, voltage)
        time.sleep(6)  # ESP32 reboot + WiFi rejoin
        robot.connect()


def _load_policy(ckpt: str, device: str):
    """Load whichever policy the checkpoint holds — ACT or SmolVLA — dispatching on config.json 'type'.
    (SmolVLA needs `transformers` in the eval env for its SmolVLM backbone; the robot PC has it.)"""
    ptype = json.load(open(os.path.join(ckpt, "config.json"))).get("type")
    if ptype == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy as Cls
    elif ptype == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as Cls
    else:
        raise SystemExit(f"eval_hw: unsupported policy type {ptype!r} (expected 'act' or 'smolvla')")
    policy = Cls.from_pretrained(ckpt)
    policy.eval()
    policy.to(device)
    return policy, ptype


def _set_reactivity(policy, ptype, n_action_steps, temporal_ensemble) -> None:
    """Override the policy's action-chunk *execution* horizon at inference — NO retraining. Both ACT and
    SmolVLA predict `chunk_size` actions and default to executing all of them (fully OPEN-LOOP); this
    controls how many run before re-observing. Lower = more reactive. `temporal_ensemble` (a coeff, e.g.
    0.01) is ACT-only: forces n_action_steps=1 and blends overlapping chunk predictions every step."""
    cfg = policy.config
    if temporal_ensemble is not None:
        if ptype != "act":
            print(f"[reactivity] --temporal-ensemble is ACT-only; ignoring for {ptype}")
        else:
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

            cfg.temporal_ensemble_coeff = temporal_ensemble
            cfg.n_action_steps = 1
            policy.temporal_ensembler = ACTTemporalEnsembler(temporal_ensemble, cfg.chunk_size)
            print(f"[reactivity] temporal ensemble coeff={temporal_ensemble}, n_action_steps=1 "
                  f"(re-observe every step — most reactive)")
            return
    if n_action_steps is not None:
        if getattr(cfg, "temporal_ensemble_coeff", None) is not None:
            cfg.temporal_ensemble_coeff = None
        cfg.n_action_steps = n_action_steps
        print(f"[reactivity] n_action_steps={n_action_steps} of chunk_size={cfg.chunk_size} "
              f"(re-observe every {n_action_steps} steps)")
    else:
        openloop = cfg.n_action_steps and cfg.n_action_steps >= cfg.chunk_size
        print(f"[reactivity] {ptype} default n_action_steps={cfg.n_action_steps}, "
              f"chunk_size={cfg.chunk_size}"
              + ("  ** FULLY OPEN-LOOP — try --n-action-steps 8 **" if openloop else ""))


def _make_control(evq: "queue.Queue"):
    """Open the Stream Deck if present, else fall back to a stdin reader. Both push a KEY_* int onto
    `evq` on each press. Returns (pad_or_None, description)."""
    try:
        from ur5e_lerobot.teleop.streamdeck import StreamDeckPad

        pad = StreamDeckPad(flip=True).open()  # deck is mounted upside down on the robot (matches panel)
        pad.on_key(lambda key, down: evq.put(key) if down else None)
        return pad, f"stream deck ({pad.key_count} keys)"
    except Exception as e:  # noqa: BLE001 — no deck / busy / no HID access -> keyboard
        print(f"[deck] unavailable ({e}); keyboard: p=play/pause  f=freedrive  r=reset  o=open hand  "
              f"x=exit  <enter>")
        import sys
        import threading

        keymap = {"p": KEY_PLAY, "f": KEY_FREEDRIVE, "r": KEY_RESET, "x": KEY_EXIT, "q": KEY_EXIT,
                  "o": KEY_OPEN_HAND}

        def reader() -> None:
            for line in sys.stdin:
                c = line.strip().lower()[:1]
                if c in keymap:
                    evq.put(keymap[c])

        threading.Thread(target=reader, daemon=True).start()
        return None, "keyboard"


def _render_deck(pad, mode: str) -> None:
    """Paint the three control keys to reflect the current mode (green=paused/safe, red=arm live)."""
    if pad is None:
        return
    if mode == "playing":
        pad.set_label(KEY_PLAY, "RUNNING\n■ pause", bg=(160, 30, 30))
    else:
        pad.set_label(KEY_PLAY, "PAUSED\n▶ play", bg=(25, 120, 40))
    if mode == "freedrive":
        pad.set_label(KEY_FREEDRIVE, "FREEDRIVE\n● ON", bg=(30, 90, 175))
    else:
        pad.set_label(KEY_FREEDRIVE, "freedrive\noff", bg=(45, 45, 45))
    pad.set_label(KEY_RESET, "RESET\n+ pause", bg=(150, 110, 20))
    pad.set_label(KEY_EXIT, "EXIT", bg=(70, 70, 70))
    pad.set_label(KEY_OPEN_HAND, "OPEN\nhand", bg=(20, 110, 110))


def _rotvec_angle_deg(a, b) -> float:
    """Geodesic angle (deg) between two axis-angle (rotvec) orientations — how far the wrist must turn
    from a to b. Uses cv2.Rodrigues (already imported). NaN on any degenerate input."""
    try:
        Ra, _ = cv2.Rodrigues(np.asarray(a, dtype=np.float64))
        Rb, _ = cv2.Rodrigues(np.asarray(b, dtype=np.float64))
        cos = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    except Exception:  # noqa: BLE001
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Roll out a trained policy on the real UR5e + AmazingHand.")
    ap.add_argument("--ckpt", help="path to .../checkpoints/last/pretrained_model (local inference)")
    ap.add_argument("--remote", metavar="HOST:PORT",
                    help="run inference on a remote GPU server (scripts/infer_server.py) instead of loading "
                         "the policy here — for SmolVLA, which is too slow on the robot PC CPU. Reactivity "
                         "is configured on the server. e.g. --remote <GPU_IP>:8777 (same subnet), or the "
                         "Mac relay IP if bridging subnets")
    ap.add_argument("--steps", type=int, default=0, help="auto-pause after N played steps (0 = unlimited)")
    ap.add_argument("--fps", type=int, default=12, help="control rate; match the training data")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--task", default="pick up the Home Depot bucket")
    ap.add_argument("--robot-ip", default="192.168.11.21")
    ap.add_argument("--hand-host", default="192.168.11.117")
    ap.add_argument("--tool-voltage", type=int, default=TOOL_VOLTAGE, choices=[0, 12, 24],
                    help="UR tool output voltage powering the hand; used only if the first connect fails")
    ap.add_argument("--video", action="store_true", help="record the played rollout (scene) to an MP4")
    ap.add_argument("--video-out", default=None,
                    help="default: <run-dir>/rollout.mp4 — see --run-dir")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="override the action-chunk execution horizon (ACT chunk=100, SmolVLA=50; both "
                         "fully open-loop by default). Try 8-16 for closed-loop; lower = more reactive.")
    ap.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEFF",
                    help="ACT-only: temporal ensembling (e.g. 0.01); forces n_action_steps=1 (re-observe "
                         "every step). Takes precedence over --n-action-steps; ignored for SmolVLA.")
    ap.add_argument("--log-csv", default=None,
                    help="default: <run-dir>/steps.csv — see --run-dir")
    ap.add_argument("--run-dir", default=None,
                    help="where this run's artifacts go (default: outputs/eval_runs/<timestamp>_<task>). "
                         "Every run gets its OWN directory so a new run can never overwrite the last "
                         "one's telemetry; pass a path to pin it.")
    ap.add_argument("--log-every", type=int, default=6, help="print a console diagnostic line every N steps")
    ap.add_argument("--obs-warn-ms", type=float, default=50.0,
                    help="print a [obs] SLOW line naming any observation source that took longer than "
                         "this (arm state / scene / wrist / side). Normal is ~4-13ms each.")
    ap.add_argument("--ik-mode", choices=["servoL", "dls"], default="dls",
                    help="arm IK: 'dls' (Python DLS+servoJ, singularity-robust — avoids the "
                         "get_inverse_kin fault) or 'servoL' (proven Cartesian path, can fault near singularities)")
    args = ap.parse_args()

    # Per-run artifact directory. The old defaults were fixed paths opened with "w", so every launch
    # truncated the previous run's telemetry — a run you wanted to analyse was routinely destroyed by
    # the next one starting (and a run that never pressed PLAY left only a header behind).
    if args.run_dir is None:
        slug = "".join(c if c.isalnum() else "-" for c in (args.task or "eval")).strip("-")[:40]
        args.run_dir = os.path.join("outputs", "eval_runs",
                                    f"{time.strftime('%Y%m%d_%H%M%S')}_{slug or 'eval'}")
    os.makedirs(args.run_dir, exist_ok=True)
    if args.log_csv is None:
        args.log_csv = os.path.join(args.run_dir, "steps.csv")
    if args.video_out is None:
        args.video_out = os.path.join(args.run_dir, "rollout.mp4")
    with open(os.path.join(args.run_dir, "run.json"), "w") as _f:
        json.dump({"started": time.time(), "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "args": vars(args)}, _f, indent=2, sort_keys=True, default=str)
    print(f"[run] artifacts -> {args.run_dir}")

    remote_client = None
    policy = pre = post = None
    if args.remote:
        from ur5e_lerobot.remote import RemotePolicyClient

        host, _, port = args.remote.partition(":")
        remote_client = RemotePolicyClient(host, int(port or 8777)).connect()
        print(f"[policy] REMOTE inference @ {args.remote} — reactivity is configured on the server")
    else:
        if not args.ckpt:
            ap.error("--ckpt is required for local inference (or use --remote HOST:PORT)")
        policy, ptype = _load_policy(args.ckpt, args.device)
        print(f"[policy] loaded {ptype} on {args.device}")
        pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt,
                                             preprocessor_overrides={"device_processor": {"device": args.device}})
        _set_reactivity(policy, ptype, args.n_action_steps, args.temporal_ensemble)
        policy.reset()

    # 3rd-person 'side' cam included so a 3-cam policy (scene/wrist/side) gets all its inputs; the
    # bridge (remote.py) + server (infer_server.py) carry whatever image set the policy expects.
    robot, scene_fn, wrist_fn, side_fn = make_engine("hardware", robot_ip=args.robot_ip,
                                                     hand_host=args.hand_host, side_cam=True)
    _connect_with_hand_power(robot, args.robot_ip, args.tool_voltage)
    arm = getattr(robot, "arm", None)  # RtdeArmInterface — has start/stop_freedrive on hardware
    if arm is not None and hasattr(arm, "ik_mode"):
        arm.ik_mode = args.ik_mode  # policy eval defaults to 'dls' (singularity-robust); teleop stays servoL
        print(f"[arm] ik_mode = {arm.ik_mode}")

    def chw(img):  # RGB HWC uint8 -> [1,3,H,W] float in [0,1] (preprocessor handles the rest)
        return torch.from_numpy(img.copy()).permute(2, 0, 1).float().div(255)[None]

    # Cameras are read SEQUENTIALLY, so the slowest one gates the whole observation — and obs_ms used
    # to be a single number covering arm state + all three reads, which hid WHICH source stalled.
    # Runs regularly show obs_ms spiking to ~7 s (against an ~85 ms loop); these per-source timings are
    # what identifies the culprit. `last_obs_ms` is read by the logger after each observe().
    last_obs_ms = {"state": 0.0, "scene": 0.0, "wrist": 0.0, "side": 0.0}

    def observe():  # read each camera at the dataset size (W,H) so it matches how training recorded
        t = time.time()
        o = robot.get_observation()
        state = np.array([o[n] for n in STATE_NAMES], dtype=np.float32)
        last_obs_ms["state"] = (time.time() - t) * 1000

        imgs = {}
        for name, fn in (("scene", scene_fn), ("wrist", wrist_fn), ("side", side_fn)):
            if fn is None:
                continue
            t = time.time()
            imgs[name] = fn(W, H)
            last_obs_ms[name] = (time.time() - t) * 1000
        return state, imgs

    if remote_client is not None:
        def infer(state, imgs):  # ship the obs to the GPU server, get one action back
            return remote_client.infer(state, imgs, args.task)

        def reset_policy():
            remote_client.reset()
    else:
        def infer(state, imgs):
            obs = {"observation.state": torch.from_numpy(state)[None], "task": [args.task]}
            for name, img in imgs.items():
                obs[f"observation.images.{name}"] = chw(img)
            with torch.no_grad():
                return post(policy.select_action(pre(obs)))[0].cpu().numpy()

        def reset_policy():
            policy.reset()

    writer = None
    if args.video:
        writer = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (W, H))

    evq: "queue.Queue[int]" = queue.Queue()
    pad, ctrl = _make_control(evq)
    st = {"mode": "paused", "played": 0, "exit": False}

    def set_mode(m: str) -> None:
        st["mode"] = m
        _render_deck(pad, m)
        print(f"[mode] {m}")

    def stop_freedrive() -> None:
        if arm is not None:
            try:
                arm.stop_freedrive()
            except Exception as e:  # noqa: BLE001
                print(f"[freedrive] stop failed: {e}")

    def enter_playing() -> None:
        if st["mode"] == "freedrive":
            stop_freedrive()
        reset_policy()          # fresh chunk: re-observe from wherever the arm is now
        st["played"] = 0
        set_mode("playing")

    def toggle_freedrive() -> None:
        if st["mode"] == "freedrive":
            stop_freedrive()
            set_mode("paused")
        elif arm is None:
            print("[freedrive] not available on this engine")
        else:
            try:
                arm.start_freedrive()
                set_mode("freedrive")
            except Exception as e:  # noqa: BLE001
                print(f"[freedrive] start failed: {e} (try PAUSE first, or Reconnect the UR)")

    def open_hand() -> None:
        """Release the grasp. PAUSES first if playing — the policy commands curls every step, so an
        open issued mid-run would be overwritten on the very next action and look like a dead key."""
        if st["mode"] == "playing":
            set_mode("paused")
            print("[hand] paused so the open isn't overwritten by the next policy action")
        try:
            robot.hand.send_curls([0.0] * HAND_CURL_DIM)
            print("[hand] opened")
        except Exception as e:  # noqa: BLE001 — a hand fault must not kill the eval loop
            print(f"[hand] open failed: {e}")

    def do_reset() -> None:
        """Reset the policy's internal state and PAUSE (arm holds where it is). Use to abort a run and
        start clean — then FREEDRIVE to reposition and PLAY again."""
        if st["mode"] == "freedrive":
            stop_freedrive()
        reset_policy()
        st["played"] = 0
        set_mode("paused")
        print("[reset] policy reset — paused")

    _render_deck(pad, "paused")
    dt = 1.0 / args.fps
    max_step = getattr(arm, "max_step", 0.05)  # translation clamp (m); NB orientation is NOT clamped
    prev = {"tgt": None, "t_end": None}
    # Connect the cameras NOW, not on the first read. make_engine hands back lazy grabbers, and
    # UsbCamera.connect() costs ~2.25 s each (VideoCapture open + 15 auto-exposure warm-up frames) —
    # ~6.9 s for three. Left lazy, that lands inside the first control step: the operator presses PLAY,
    # the arm sits still for seven seconds, then moves. Pay it here, while paused, instead.
    print("[cam] connecting cameras (~7s: VideoCapture open + auto-exposure warm-up)…", flush=True)
    _t_cam = time.time()
    observe()
    print(f"[cam] ready in {time.time() - _t_cam:.1f}s — " +
          "  ".join(f"{k}={v:.0f}ms" for k, v in last_obs_ms.items() if v), flush=True)

    log_f = open(args.log_csv, "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["t", "loop_ms", "obs_ms", "state_ms", "scene_ms", "wrist_ms", "side_ms",
                    "infer_ms", "send_ms", "d_move_mm", "d_turn_deg",
                    "cmd_dmove_mm", "cmd_dturn_deg", "clamp_trans",
                    "cur_x", "cur_y", "cur_z", "cur_rx", "cur_ry", "cur_rz",
                    "tgt_x", "tgt_y", "tgt_z", "tgt_rx", "tgt_ry", "tgt_rz",
                    "curl_index", "curl_middle", "curl_ring", "curl_thumb"])
    print(f"READY — PAUSED. control: {ctrl}. PLAY/PAUSE · FREEDRIVE · RESET · EXIT.  log -> {args.log_csv}")
    print("**e-STOP IN HAND.**")
    try:
        while not st["exit"]:
            try:  # drain control events (deck callback / stdin thread)
                while True:
                    key = evq.get_nowait()
                    if key == KEY_EXIT:
                        st["exit"] = True
                    elif key == KEY_PLAY:
                        set_mode("paused") if st["mode"] == "playing" else enter_playing()
                    elif key == KEY_FREEDRIVE:
                        toggle_freedrive()
                    elif key == KEY_RESET:
                        do_reset()
                    elif key == KEY_OPEN_HAND:
                        open_hand()
            except queue.Empty:
                pass
            if st["exit"]:
                break

            if st["mode"] == "playing":
                # A protective stop / e-stop halts the UR control script -> servoJ becomes a silent
                # no-op (the arm freezes while the loop spams "control script not running"). Detect it,
                # PAUSE, and reconnect (clears the protective stop + reuploads the script) so the run
                # recovers cleanly instead of driving blind into a dead controller.
                if arm is not None and hasattr(arm, "is_ready") and not arm.is_ready():
                    set_mode("paused")
                    try:
                        print(f"[fault] UR control stopped (protective/e-stop) — reconnecting: {arm.reconnect()}")
                    except Exception as e:  # noqa: BLE001
                        print(f"[fault] UR control stopped; reconnect failed: {e}")
                    print("  paused. clear the e-stop if engaged, FREEDRIVE to a safe pose, then PLAY.")
                    continue
                t0 = time.time()
                state, imgs = observe()
                t_obs = time.time()
                act = infer(state, imgs)
                t_inf = time.time()
                robot.send_action(dict(zip(ACTION_NAMES, act)))
                t_snd = time.time()
                try:  # per-step diagnostics — never let logging break the control loop
                    cur, tgt = state[:6], act[:6]
                    d_move = float(np.linalg.norm(tgt[:3] - cur[:3])) * 1000.0   # mm asked (target vs measured)
                    d_turn = _rotvec_angle_deg(tgt[3:6], cur[3:6])               # deg the wrist is asked to turn
                    d_cmd_m = float(np.linalg.norm(tgt[:3] - prev["tgt"][:3])) * 1000.0 if prev["tgt"] is not None else 0.0
                    d_cmd_r = _rotvec_angle_deg(tgt[3:6], prev["tgt"][3:6]) if prev["tgt"] is not None else 0.0
                    clamp = int(d_move > max_step * 1000.0)  # translation clamp engaged this step?
                    loop_ms = (t_snd - prev["t_end"]) * 1000.0 if prev["t_end"] is not None else 0.0
                    log_w.writerow([st["played"], round(loop_ms, 1), round((t_obs - t0) * 1000, 1),
                                    round(last_obs_ms["state"], 1), round(last_obs_ms["scene"], 1),
                                    round(last_obs_ms["wrist"], 1), round(last_obs_ms["side"], 1),
                                    round((t_inf - t_obs) * 1000, 1), round((t_snd - t_inf) * 1000, 1),
                                    round(d_move, 1), round(d_turn, 1), round(d_cmd_m, 1), round(d_cmd_r, 1), clamp,
                                    *[round(float(v), 4) for v in cur], *[round(float(v), 4) for v in tgt],
                                    *[round(float(v), 3) for v in act[6:10]]])
                    log_f.flush()
                    # Name the culprit the moment a source stalls, instead of only in the CSV later.
                    slow = {k: v for k, v in last_obs_ms.items() if v > args.obs_warn_ms}
                    if slow:
                        print("  [obs] SLOW: " + "  ".join(f"{k}={v:.0f}ms" for k, v in
                                                           sorted(slow.items(), key=lambda kv: -kv[1])))
                    if st["played"] % max(1, args.log_every) == 0:
                        print(f"t{st['played']:<4} loop={loop_ms:4.0f}ms inf={(t_inf - t_obs) * 1000:4.0f}ms "
                              f"Δmove={d_move:5.1f}mm Δturn={d_turn:5.1f}° cmdΔturn={d_cmd_r:5.1f}° "
                              f"{'CLAMP' if clamp else '     '} curls={np.round(act[6:10], 2)}")
                    prev["tgt"], prev["t_end"] = tgt, t_snd
                except Exception as e:  # noqa: BLE001
                    print(f"[log] {e}")
                if writer is not None:
                    f = cv2.cvtColor(imgs["scene"], cv2.COLOR_RGB2BGR)
                    cv2.putText(f, f"t{st['played']}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    writer.write(f)
                st["played"] += 1
                if args.steps and st["played"] >= args.steps:
                    print(f"[cap] reached --steps={args.steps}; pausing")
                    set_mode("paused")
                time.sleep(max(0.0, dt - (time.time() - t0)))
            else:  # paused or freedrive — don't drive; the arm holds (or is hand-guided)
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\naborted")
    finally:
        try:
            log_f.close()
            print(f"wrote {args.log_csv} ({st['played']} steps logged)")
            print(f"[run] artifacts in {args.run_dir}")
        except Exception:  # noqa: BLE001
            pass
        if pad is not None:
            pad.close()
        if remote_client is not None:
            remote_client.close()
        if writer is not None:
            writer.release()
            print(f"wrote {args.video_out}")
        robot.disconnect()
    print("done")


if __name__ == "__main__":
    main()
