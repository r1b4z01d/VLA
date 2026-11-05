"""Deploy a trained policy (ACT or SmolVLA) on the REAL hardware — the policy drives the UR5e +
AmazingHand from live camera + proprioceptive observations. The policy class is auto-detected from the
checkpoint's config.json. **MOTION-PRODUCING; keep the e-stop in hand.** Runs on the robot PC (the
Jetson Orin: cameras + RTDE + hand live there). Inference runs on the same box too: the vla_infer_server
unit serves outputs/serve_ckpt on loopback and this client talks to it with --remote (SmolVLA in-process
is the fallback, ~0.9 s a chunk on the Orin with no prefetch; the GPU box over the LAN is the other fallback).

    ~/rd_ws/VLA/run.sh scripts/eval_hw.py --remote 127.0.0.1:8777 --cameras udp --fps 12 [--video]
    ~/rd_ws/VLA/run.sh scripts/eval_hw.py --ckpt outputs/train/<run>/checkpoints/last/pretrained_model --fps 12   # in-process

**Launch-time exposure first.** Before the control loop starts, the cameras auto-expose (median-
targeted ladder search, ~1 s) and the result is LOCKED — auto has been reliable, so that is the
default. `--expose-manual` instead opens the live preview and hands row 1 to ＋ brighter / － darker /
✓ SET (keyboard: +/-/s) for a manual trim, exactly as the teleop panel's manual mode. Either way the
locked value is written into the run's `eval.json` so a rollout can be compared against the exposure
its dataset was recorded at. `--no-expose` skips the phase and keeps whatever is already set. Locking also stops the driver re-metering as the arm sweeps through
frame, which would make brightness a function of arm pose — a shortcut this policy prefers to vision.

**Interactive, Stream Deck driven — LAUNCHES PAUSED.** Four keys (a stdin fallback maps the same
letters if no deck is found):
- **PLAY / PAUSE** (`p`) — toggle the policy. Paused holds the arm. Play first walks the arm to the
  captured home pose (the same press-and-go return as HOME; any key cancels it and leaves the run
  paused) and then (re)starts the policy, reset so it re-observes from home — every demo starts there,
  so a run that starts anywhere else begins out of distribution. `--no-home-start` plays from wherever
  the arm is.
- **FREEDRIVE** (`f`) — toggle gravity-comp teach mode so you can hand-guide the arm to a *training*
  start pose. The policy is suspended while freedriving; pressing PLAY exits freedrive and starts.
- **RESET** (`r`) — reset the policy's internal state and pause (arm holds). Abort a run and start clean.
- **HOME** (`h`) — press-and-go: one press walks the arm back to the captured home pose on its own
  (joint-space servoJ, one increment per tick, so every other key stays live). Press HOME again, or any
  other key, to stop it. Nothing is being held, so those are the only outs — **e-stop in hand.**
- **EXIT** (`x`) — stop and disconnect cleanly (Ctrl-C also aborts).

**Multiple runs per launch.** NEW RUN (`n`) finalises the current rollout and opens a fresh one in the
same process — cameras stay connected and the exposure stays locked, so back-to-back attempts are
actually comparable (restarting would re-expose and move the one variable that must not move). Each
run lights the bottom row for a 1-5 star rating; the stars, step count, duration, locked exposure and
served checkpoint go into that run's `eval.json` in webui/evals.py's registry format, so runs show up
in the manager's Evals tab (each run is also rsynced to the GPU box, the registry's home). Any key other than 1-5 skips the rating.

Typical loop: launch (paused) → PLAY (the arm goes home, then the policy runs) → watch → PAUSE → NEW RUN →
rate 1-5 → PLAY … → EXIT. FREEDRIVE still hand-guides the arm (to clear a jam, or with `--no-home-start`
to try a different start pose). There is NO automatic success metric on
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
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors

from ur5e_lerobot.robot.ur_power import power_down, power_up
from ur5e_lerobot.schema import ACTION_NAMES, HAND_CURL_DIM, HAND_CURL_NAMES, STATE_NAMES
from ur5e_lerobot.sim.record_sim import make_engine

W, H = 960, 540  # dataset image size (must match training)
TOOL_VOLTAGE = 12  # UR tool output voltage powering the AmazingHand ESP32 (fed from the tool port)

# Stream Deck key assignments (LOGICAL coords; StreamDeckPad flip-remaps to the flipped hardware).
KEY_PLAY = 0       # top-left
KEY_FREEDRIVE = 1  # next to play
KEY_RESET = 2      # reset policy + pause
KEY_EXIT = 4       # top-right corner (blank key 3 between it and the rest -> hard to hit by accident)
KEY_OPEN_HAND = 5  # row 2, first column — 🖐 open, where the teleop panel has it (key 3 stays blank as the EXIT guard)
KEY_HOME = 6       # next to it — slow joint-space return to the captured home pose
KEY_NEWRUN = 7     # finish this run (rate it) and roll to a fresh one WITHOUT restarting
KEY_CLOSE_HAND = 10  # row 3, first column — ✊ close, where the teleop panel has it; doubles as 1★ while rating

# Bottom row = 1-5 stars when a run is being rated, same keys and order as the teleop panel so the
# rating gesture is identical whether you are recording demos or evaluating a policy.
RATING_KEYS = (10, 11, 12, 13, 14)

CSV_HEADER = ["t", "loop_ms", "obs_ms", "state_ms", "scene_ms", "wrist_ms", "side_ms",
              "infer_ms", "send_ms", "d_move_mm", "d_turn_deg",
              "cmd_dmove_mm", "cmd_dturn_deg", "clamp_trans",
              "cur_x", "cur_y", "cur_z", "cur_rx", "cur_ry", "cur_rz",
              "tgt_x", "tgt_y", "tgt_z", "tgt_rx", "tgt_ry", "tgt_rz",
              "curl_index", "curl_middle", "curl_ring", "curl_thumb"]

# Launch-time exposure phase reuses row 1 (same keys/labels as the teleop panel, so the muscle memory
# carries over): 0 = brighter, 1 = darker, 2 = SET. They revert to PLAY/FREEDRIVE/RESET once locked.
EXPO_KEYS = {KEY_PLAY: "＋ brighter", KEY_FREEDRIVE: "－ darker", KEY_RESET: "✓ SET"}
HOME_PATH = "outputs/home_pose.json"  # written by the teleop panel's 'Set Home'
HOME_DQ = 0.03         # rad per control tick (~0.36 rad/s at 12 fps) — same rate the panel uses
HOME_MAX_STEPS = 200   # safety cap (~17 s at 12 fps) so a blocked return still terminates
# --auto --still: a step counts as "nothing commanded to move" below all three (per-step command deltas)
AUTO_STILL_MM, AUTO_STILL_DEG, AUTO_STILL_CURL = 1.5, 0.8, 0.02
AUTO_GRASP_CURL = 0.3   # --still-mode fingers: the countdown arms once the curls have moved this far from the run's first action
# --done-when: the run ends by the GRASP LEVEL (mean commanded curl averaged over --still seconds), not by
# motion. Measured 2026-10-26 on the grab model: curl commands jitter up to 0.25/step while holding and the
# policy re-grasps repeatedly, so a motion test never settled; the level test fires ~2 s after the close.
AUTO_CLOSED_CURL, AUTO_OPEN_CURL, AUTO_ARMED_CURL = 0.7, 0.2, 0.6
HAND_STATE_PATH = "outputs/hand_state.json"   # last commanded curls, so a stage that exits carrying hands over to the next


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
    from ur5e_lerobot.hand.amazing_hand_client import ping_reachable, wait_reachable

    hand_host = getattr(getattr(robot, "hand", None), "host", None)
    try:
        robot.connect()
        return
    except Exception as e:  # noqa: BLE001  — most likely the hand (probe missed / ESP32 off / tool unpowered)
        print(f"[connect] {e}")
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
    # Decide WHY the hand failed before doing anything drastic. A power cycle is the right fix for a
    # hand with no power; it is the WRONG fix for a hand that is up but on a lossy link: it reboots a
    # working hand, preempts the arm's control script (the UR goes dead), and the ESP32 needs ~20 s
    # to rejoin WiFi — so the old fixed 6 s retry failed too and every launch rebooted the hand again.
    if hand_host and ping_reachable(hand_host):
        print(f"[connect] hand {hand_host} answers ping — NOT power-cycling; retrying the connect "
              "(a loss burst at launch, not a dead hand)")
        for attempt in range(3):
            try:
                robot.connect()
                return
            except Exception as e2:  # noqa: BLE001
                print(f"[connect] retry {attempt + 1}/3 failed: {e2}")
                try:
                    robot.disconnect()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(2.0)
        raise RuntimeError(f"hand {hand_host} is reachable by ping but never answered the UDP probe — "
                           "wrong firmware, or the link is dropping nearly everything")
    print(f"[connect] hand {hand_host or '?'} does not answer ping -> power-cycling the UR tool ({voltage} V)")
    _power_cycle_tool(robot_ip, voltage)
    if hand_host:
        back = wait_reachable(hand_host, timeout_s=45.0)
        print(f"[connect] hand {'is back on the network' if back else 'STILL not pinging after 45 s'}")
    else:
        time.sleep(20)
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


def _make_control(evq: "queue.Queue", held: "dict | None" = None):
    """Open the Stream Deck if present, else fall back to a stdin reader. Both push a KEY_* int onto
    `evq` on each press. Returns (pad_or_None, description).

    `held` (optional) is kept up to date with each key's up/down state. Nothing reads it now that HOME
    is press-and-go rather than hold-to-home; it is left wired because the deck is the only input that
    can report a key-up at all, and any future dead-man key would need it. NOTE only PRESSES reach
    `evq` — key-up is recorded in `held` and never enqueued, which is what lets HOME toggle safely."""
    try:
        from ur5e_lerobot.teleop.streamdeck import StreamDeckPad

        pad = StreamDeckPad(flip=True).open()  # deck is mounted upside down on the robot (matches panel)

        def _on_key(key: int, down: bool) -> None:
            if held is not None:
                held[key] = down
            if down:
                evq.put(key)

        pad.on_key(_on_key)
        return pad, f"stream deck ({pad.key_count} keys)"
    except Exception as e:  # noqa: BLE001 — no deck / busy / no HID access -> keyboard
        print(f"[deck] unavailable ({e}); keyboard: p=play/pause  f=freedrive  r=reset  o=open hand  c=close hand  "
              "n=new run  1-5=rate  (exposure phase: +=brighter  -=darker  s=SET)  "
              f"h=home  x=exit  <enter>")
        import sys
        import threading

        # +/-/s duplicate row 1 during the exposure phase; they map to the same KEY_* the deck sends,
        # so the phase loop needs no separate keyboard path.
        keymap = {"p": KEY_PLAY, "f": KEY_FREEDRIVE, "r": KEY_RESET, "x": KEY_EXIT, "q": KEY_EXIT,
                  "o": KEY_OPEN_HAND, "c": KEY_CLOSE_HAND, "h": KEY_HOME, "n": KEY_NEWRUN,
                  "1": RATING_KEYS[0], "2": RATING_KEYS[1], "3": RATING_KEYS[2],
                  "4": RATING_KEYS[3], "5": RATING_KEYS[4],
                  "+": KEY_PLAY, "=": KEY_PLAY, "-": KEY_FREEDRIVE, "_": KEY_FREEDRIVE, "s": KEY_RESET}

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
    pad.set_label(KEY_OPEN_HAND, "🖐")      # same keys and icons as the teleop panel's whole-hand column
    pad.set_label(KEY_CLOSE_HAND, "✊")
    pad.set_label(KEY_HOME, "HOME\n⌂ go", bg=(90, 60, 140))
    pad.set_label(KEY_NEWRUN, "NEW RUN\n＋rate", bg=(120, 80, 20))


def _parse_stage(spec: str) -> dict:
    """--stage "name=grab model=<run> task='Grab and hold tigger' done=closed keep=1" -> dict."""
    import shlex

    d: dict = {}
    for tok in shlex.split(spec):
        k, eq, v = tok.partition("=")
        if not eq:
            raise ValueError(f"--stage: expected k=v, got {tok!r} in {spec!r}")
        d[k.strip()] = v
    if "model" not in d or "task" not in d:
        raise ValueError(f"--stage needs model= and task=: {spec!r}")
    d.setdefault("name", d["model"])
    d["done"] = d.get("done", "none")
    if d["done"] not in ("none", "closed", "open"):
        raise ValueError(f"--stage done= must be none|closed|open: {spec!r}")
    d["keep"] = str(d.get("keep", "0")).lower() in ("1", "true", "yes")
    d["home"] = str(d.get("home", "1")).lower() in ("1", "true", "yes")
    d["seconds"] = float(d["seconds"]) if d.get("seconds") else None
    d["still"] = float(d["still"]) if d.get("still") else None
    return d


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
    ap.add_argument("--no-home-start", action="store_true",
                    help="PLAY from wherever the arm is instead of walking it to the captured home pose first")
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
    ap.add_argument("--operator", default="", help="name recorded on each run's eval.json")
    ap.add_argument("--landmark", default="",
                    help="the landmark (named place, manager Landmarks tab) this run is done at; recorded on eval.json")
    ap.add_argument("--gpu-host", default="bryan@GPU",
                    help="push each finished run to this host's eval registry ('' disables)")
    ap.add_argument("--gpu-evals-dir", default="~/VLA/outputs/evals",
                    help="registry dir on --gpu-host that the web UI lists")
    ap.add_argument("--cameras", choices=["local", "udp"], default="local",
                    help="local: open the USB cameras here (exclusive). udp: subscribe to the camera "
                         "service (supervisor unit vla_cameras); other consumers can watch at once and "
                         "the exposure teleop locked is already in effect")
    ap.add_argument("--camera-host", default="127.0.0.1",
                    help="host running the camera service's control port (--cameras udp)")
    ap.add_argument("--hand-transport", choices=["udp", "tcp"], default="udp",
                    help="udp (default): fire-and-forget setpoints, a lost packet costs one step; "
                         "needs the UDP-capable hand firmware (:8766). tcp: legacy :8765 — on a "
                         "lossy link it retransmits stale setpoints seconds late")
    ap.add_argument("--scene-square", default=None, metavar="AT",
                    help="override the scene camera's square crop (0.0 top .. 1.0 bottom, or 'none'). "
                         "Default: whatever the served checkpoint was trained with (its capture.json)")
    ap.add_argument("--side-square", default=None, metavar="AT", help="same for the side camera")
    ap.add_argument("--expose-manual", action="store_true",
                    help="after auto-exposing, open the trim loop (deck row 1: + / - / SET) instead of "
                         "locking the metered value straight away")
    ap.add_argument("--no-expose", action="store_true",
                    help="skip the exposure phase entirely: keep whatever the cameras/service already have")
    ap.add_argument("--obs-warn-ms", type=float, default=50.0,
                    help="print a [obs] SLOW line naming any observation source that took longer than "
                         "this (arm state / scene / wrist / side). Normal is ~4-13ms each.")
    ap.add_argument("--ik-mode", choices=["servoL", "dls"], default="dls",
                    help="arm IK: 'dls' (Python DLS+servoJ, singularity-robust — avoids the "
                         "get_inverse_kin fault) or 'servoL' (proven Cartesian path, can fault near singularities)")
    ap.add_argument("--no-ur-power", action="store_true",
                    help="don't power the UR up on launch / off on exit (dashboard :29999)")
    ap.add_argument("--ui", choices=["web", "none"], default="web",
                    help="web: serve the browser page (http://<this host>:8090) next to the deck/keyboard")
    ap.add_argument("--web-port", type=int, default=8090, help="port for the browser page")
    ap.add_argument("--ur-shutdown", action="store_true",
                    help="on exit also shut the UR controller down (it then needs switching on by hand)")
    # --- launch preflight (ur5e_lerobot/services.py) ---------------------------------------------------
    ap.add_argument("--no-preflight", action="store_true",
                    help="skip the supervisor preflight (stop vla_teleop / the ROS UR driver if they run, make "
                         "sure vla_infer_server answers on --remote); a no-op where there is no supervisor")
    ap.add_argument("--no-auto-infer", action="store_true",
                    help="preflight only verifies the inference server instead of starting it")
    # --- unattended runs (the route runner's vla: actions) -------------------------------------------
    ap.add_argument("--auto", action="store_true",
                    help="unattended run: PLAY as soon as READY, stop on --steps / --max-seconds / --still, "
                         "open the hand (unless --keep-grasp), return home (unless --no-home), exit. "
                         "Exit code 0 done, 3 aborted on a UR fault, 4 preflight failure")
    ap.add_argument("--max-seconds", type=float, default=120.0, help="--auto: wall-clock cap on the playing phase")
    ap.add_argument("--still", type=float, default=0.0,
                    help="--auto: finish once the commanded motion has stayed below the stillness thresholds "
                         "for this many seconds (0 = off)")
    ap.add_argument("--min-seconds", type=float, default=10.0,
                    help="--auto: stillness is not checked before this many seconds of play")
    ap.add_argument("--keep-grasp", action="store_true",
                    help="--auto: do NOT open the hand at the end (carry the object), and leave the UR powered "
                         "on at exit so the tool voltage keeps the hand's servos holding")
    ap.add_argument("--no-home", action="store_true", help="--auto: skip the return to home at the end")
    ap.add_argument("--wait-go", action="store_true",
                    help="--auto: launch PAUSED and take PLAY (deck / page / p) as the go, instead of playing at READY")
    ap.add_argument("--still-mode", choices=["all", "fingers"], default="all",
                    help="--auto --still: what must be still. 'all' = the arm and the fingers; 'fingers' = only the "
                         "commanded finger curls, and only once they have moved (a grasp or a release happened), "
                         "so the run ends when the hand has settled whatever the arm does")
    ap.add_argument("--leave-powered", action="store_true",
                    help="do not power the UR down at exit (back-to-back stages); implied by --keep-grasp")
    ap.add_argument("--done-when", choices=["none", "closed", "open"], default="none",
                    help="--auto: end the run by the GRASP LEVEL. 'closed': the mean commanded curl averaged "
                         "over the last --still seconds is >= 0.7 (a pick: the hand closed and stays closed); "
                         "'open': that average is <= 0.2 once the hand has been closed (a place: the release "
                         "happened). Robust to the policy's curl jitter and to its re-grasp attempts")
    ap.add_argument("--stage", action="append", default=[], metavar="'k=v ...'",
                    help="--auto stage list (repeatable), e.g. \"name=grab model=tigger_1_merged_smolvla_1 "
                         "task='Grab and hold tigger' done=closed keep=1\": ONE process runs the stages in "
                         "order — switch the server's model, apply its crop, wait for PLAY as the go, run, "
                         "keep/open the hand, home, next stage. Keys: name model task done keep home seconds still")
    ap.add_argument("--cycles", type=int, default=1, help="--stage: run the list this many times (0 = until EXIT)")
    ap.add_argument("--result-file", default=None,
                    help="--auto: write {exit_code, run_dir, steps, stop…} here at exit (for the caller)")
    args = ap.parse_args()
    try:
        stages = [_parse_stage(x) for x in args.stage]
    except ValueError as e:
        ap.error(str(e))
    if stages:
        args.auto, args.wait_go = True, True
        if not args.remote:
            ap.error("--stage needs --remote (the server switches models)")

    if not args.no_preflight:
        # Nothing else may drive the UR (the recording panel, the ROS UR driver), and a loopback
        # --remote needs vla_infer_server answering — bring it up rather than failing on the first infer.
        try:
            from ur5e_lerobot import services as _services

            _services.eval_preflight(args.remote, auto_infer=not args.no_auto_infer,
                                     log=lambda m: print(f"[preflight] {m}", flush=True))
        except Exception as e:  # noqa: BLE001
            print(f"[preflight] FAILED: {e}", flush=True)
            sys.exit(4)

    def _ur_power_down() -> None:
        if not args.no_ur_power:
            print(f"[ur] {power_down(args.robot_ip, shutdown=args.ur_shutdown)}", flush=True)

    # Per-run artifact directory. The old defaults were fixed paths opened with "w", so every launch
    # truncated the previous run's telemetry — a run you wanted to analyse was routinely destroyed by
    # the next one starting (and a run that never pressed PLAY left only a header behind).
    # Runs are ROLLABLE: the deck's NEW RUN key finalises the current one and opens the next in the
    # same process. Evals used to require a full restart per attempt (~7 s of camera connect plus a
    # fresh exposure phase each time), which made back-to-back comparisons impractical and meant the
    # exposure differed between the runs being compared — the one variable that must not move.
    def _slug(task: str) -> str:
        return "".join(c if c.isalnum() else "-" for c in (task or "eval")).strip("-")[:40] or "eval"

    _fixed_dir = args.run_dir          # an explicit --run-dir pins run 1 only; later runs get their own
    auto_meta: dict = {"enabled": bool(args.auto)}   # --auto stop reason etc.; stamped on every eval.json
    exit_code = [0]
    # Per-run settings. A stage list rewrites these between runs, so one process can run "grab" and then
    # "place" with different tasks / end criteria; everything below reads them instead of args.*.
    run_cfg = {"task": args.task, "done_when": args.done_when, "keep": args.keep_grasp, "home": not args.no_home,
           "max_seconds": args.max_seconds, "still": args.still}

    class Run:
        """One rollout's artifacts + metadata. `roll()` closes this one and returns the next."""

        def __init__(self, n: int, run_dir: "str | None" = None):
            self.n = n
            # The run INDEX is part of the name, and a collision still gets a suffix. Naming on the
            # timestamp alone meant two runs rolled inside the same second landed in one directory —
            # the second silently truncated the first's steps.csv and replaced its rating. Rapid
            # back-to-back runs are exactly what this feature is for, so that is the common case.
            if run_dir is None:
                base = os.path.join("outputs", "eval_runs",
                                    f"{time.strftime('%Y%m%d_%H%M%S')}_{_slug(run_cfg['task'])}_r{n:02d}")
                cand, i = base, 1
                while os.path.exists(cand):
                    cand = f"{base}-{i}"
                    i += 1
                run_dir = cand
            self.dir = run_dir
            os.makedirs(self.dir, exist_ok=True)
            self.csv_path = os.path.join(self.dir, "steps.csv")
            self.video_path = os.path.join(self.dir, "rollout.mp4")
            self.started = time.time()
            self.rating = None
            self.played = 0
            self.log_f = open(self.csv_path, "w", newline="")
            self.log_w = csv.writer(self.log_f)
            self.log_w.writerow(CSV_HEADER)
            self.writer = None
            if args.video:
                from ur5e_lerobot.sensors.cameras import SCENE_ROTATE, SCENE_SQUARE, out_size

                self.writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                              float(args.fps),
                                              out_size(SCENE_ROTATE, SCENE_SQUARE, W, H))
            self.write_meta()
            # Print the hand state each run STARTS from. observation.state carries the curls, so a run
            # that begins clamped is already out of distribution before the first action — and this line
            # makes that visible in the console instead of only in the csv after the fact.
            try:
                o = robot.get_observation()
                c = [float(o[n]) for n in HAND_CURL_NAMES if n in o]
                tag = "OPEN" if c and max(c) < 0.08 else "CLOSED — run starts out of distribution"
                print(f"[run {self.n}] start hand curls {[round(x, 3) for x in c]} -> {tag}", flush=True)
            except Exception:  # noqa: BLE001
                pass
            print(f"[run {self.n}] artifacts -> {self.dir}", flush=True)

        def write_meta(self) -> None:
            """Registry-format eval.json (webui/evals.py schema) so the run shows up in the manager's
            Evals tab next to the GPU-side captures, rather than only as loose files on the robot PC."""
            meta = {
                "created": self.started,
                "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started)),
                "source": "eval_hw",
                # The checkpoint the server RESOLVED (train/<run>/checkpoints/<step>), not the address
                # it was reached at — "127.0.0.1:8777" told nobody which model ran. `model` is the run
                # name for grouping; `server` keeps the address.
                "model": (served.get("checkpoint") or args.ckpt or "").split("/checkpoints/")[0].split("train/")[-1] or "smolvla",
                "checkpoint": served.get("checkpoint") or args.ckpt or args.remote or "",
                "server": args.remote or "local",
                "served": {k: served.get(k) for k in ("n_action_steps", "prefetch", "rtc", "rtc_execution_horizon", "num_steps") if k in served},
                "capture": {"squares": dict(crop_applied),
                            "from_checkpoint": bool((served.get("capture") or {}).get("squares"))},
                "task": run_cfg["task"],
                "episodes": 1,
                "steps": self.played,
                "duration_s": round(time.time() - self.started, 1),
                "operator": args.operator,
                "landmark": args.landmark,
                "exposure": expo_values,
                "fps": args.fps,
                "n_action_steps": args.n_action_steps,
                "remote": args.remote,
                "args": vars(args),
                "auto": dict(auto_meta),
            }
            if self.rating is not None:
                meta["rating"] = self.rating
            for fn in (os.path.join(self.dir, "run.json"), os.path.join(self.dir, "eval.json")):
                with open(fn, "w") as f:
                    json.dump(meta, f, indent=2, sort_keys=True, default=str)

        def close(self) -> None:
            self.write_meta()
            try:
                self.log_f.close()
            except Exception:  # noqa: BLE001
                pass
            if self.writer is not None:
                self.writer.release()   # must be released BEFORE upload or the mp4 has no moov atom
                self.writer = None
            print(f"[run {self.n}] closed — {self.played} steps, "
                  f"rating {self.rating if self.rating is not None else '—'} -> {self.dir}", flush=True)
            self.upload()

        def upload(self) -> None:
            """Push this run into the GPU's outputs/evals/<name>/ — the directory webui/evals.py lists.

            Runs are written on the ROBOT PC, but the manager reads the registry on the GPU box, so a
            rating pressed on the deck was previously only ever visible on the machine nobody looks at.
            rsync in the background: a rollout must never wait on the network, and a failed push is a
            logged warning, not a lost run (the local copy is authoritative and re-syncable).
            """
            if not args.gpu_host or not self.played:
                return
            dst = f"{args.gpu_host}:{args.gpu_evals_dir.rstrip('/')}/{os.path.basename(self.dir)}/"
            cmd = ["rsync", "-a", "--timeout=20", "-e",
                   "ssh -o BatchMode=yes -o ConnectTimeout=10", f"{self.dir}/", dst]

            def _push() -> None:
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                    if r.returncode:
                        print(f"[run {self.n}] upload failed ({r.returncode}): "
                              f"{(r.stderr or '').strip()[:200]}", flush=True)
                    else:
                        print(f"[run {self.n}] -> {dst}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[run {self.n}] upload error: {type(e).__name__}: {e}", flush=True)

            threading.Thread(target=_push, daemon=True).start()

        def roll(self) -> "Run":
            self.close()
            return Run(self.n + 1)

    remote_client = None
    policy = pre = post = None
    served: dict = {}       # what the inference server reports it is serving (see below)
    crop_applied: dict = {} # camera crops in effect for this run (recorded into eval.json)
    if args.remote:
        from ur5e_lerobot.remote import RemotePolicyClient

        host, _, port = args.remote.partition(":")
        remote_client = RemotePolicyClient(host, int(port or 8777)).connect()
        print(f"[policy] REMOTE inference @ {args.remote} — reactivity is configured on the server")
        # Ask the server WHAT it serves and record that, not the address. 146 eval records carried
        # "127.0.0.1:8777" / "192.168.11.131:8777" as their checkpoint, which cannot be grouped by
        # model after the fact. An old server answers {} and the record falls back to the address.
        try:
            served = remote_client.info() or {}
        except Exception as e:  # noqa: BLE001
            print(f"[policy] server did not answer info ({e}); recording the address only")
            served = {}
        if served.get("checkpoint"):
            print(f"[policy] serving {served['checkpoint']}  n_action_steps={served.get('n_action_steps')} "
                  f"prefetch={served.get('prefetch')} rtc={served.get('rtc')}", flush=True)
        else:
            print("[policy] WARNING: server reports no checkpoint — this run will be hard to attribute to a model",
                  flush=True)
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
                                                     hand_host=args.hand_host, side_cam=True,
                                                     cameras=args.cameras, camera_host=args.camera_host,
                                                     hand_transport=args.hand_transport)
    # Launch-time exposure, same flow as the teleop panel. `_grab` hangs the UsbCamera off each render
    # fn so the session can drive the drivers directly. Locking manual exposure (rather than leaving
    # the driver on auto) matters more at eval than at record time: with auto on, the driver re-meters
    # as the arm sweeps through frame, so image brightness becomes a function of arm pose — exactly the
    # kind of proprioceptive shortcut this policy was already measured to prefer over looking.
    exposure = None
    _cams = {n: getattr(f, "cam", None) for n, f in
             (("scene", scene_fn), ("wrist", wrist_fn), ("side", side_fn)) if f}
    if any(_cams.values()):
        from ur5e_lerobot.sensors.cameras import ExposureSession

        # for_cams: with --cameras udp this is a RemoteExposureSession — same flow, run inside the
        # camera service, and it STARTS at whatever the last operator locked (e.g. the recording's)
        exposure = ExposureSession.for_cams(_cams)

    # --- camera GEOMETRY: make the cameras match what the checkpoint was trained on ----------------
    # The square crop is part of the observation; a shelf-task model and a floor-task model were
    # recorded with different ones. The server reports the checkpoint's capture.json with its
    # `info`; apply it before the first observation. Flags override (for experiments); with neither,
    # the cameras stay as the service has them — loudly, because that is the silent-failure case.
    from ur5e_lerobot.sensors.capture_meta import apply as _apply_squares, parse_square, squares_of

    live_cams = {n: c for n, c in _cams.items() if c is not None}
    want = {}
    cap = served.get("capture") if isinstance(served, dict) else None
    if cap and cap.get("squares"):
        want.update({k: v for k, v in cap["squares"].items() if k in live_cams})
    forced = {n: parse_square(v) for n, v in (("scene", args.scene_square), ("side", args.side_square))
              if v is not None}
    want.update(forced)
    if live_cams and want:
        before = squares_of(live_cams)
        crop_applied = _apply_squares(live_cams, want)
        src = "flags" if forced else f"checkpoint capture.json ({os.path.basename(os.path.dirname(cap.get('source', '')))})"
        print(f"[crop] {before} -> {crop_applied}   (from {src})", flush=True)
    elif live_cams:
        crop_applied = squares_of(live_cams)
        print(f"[crop] WARNING: server reported no capture geometry and no --scene-square/--side-square "
              f"given — cameras left at {crop_applied}; if that is not what the model trained on, "
              f"it is looking at the wrong part of the room", flush=True)
    if not args.no_ur_power:
        try:
            print(f"[ur] {power_up(args.robot_ip)}", flush=True)
        except RuntimeError as e:  # carry on: the connect below fails loudly if the arm really is down
            print(f"[ur] power-up failed: {e}", flush=True)
    try:
        _connect_with_hand_power(robot, args.robot_ip, args.tool_voltage)
    except BaseException:
        _ur_power_down()
        raise
    arm = getattr(robot, "arm", None)  # RtdeArmInterface — has start/stop_freedrive on hardware

    # A previous session that exited carrying something (--keep-grasp) left the hand closed, but a fresh
    # robot object assumes open curls, so observation.state would lie to the policy (a place model
    # expects to START holding). Re-send what that session last commanded, so hand and state agree.
    if os.path.exists(HAND_STATE_PATH):
        try:
            _hs = json.load(open(HAND_STATE_PATH))
            if _hs.get("keep_grasp") and len(_hs.get("curls") or []) == HAND_CURL_DIM:
                robot.send_curls([float(c) for c in _hs["curls"]])
                print(f"[hand] resuming the grasp the previous session left: curls "
                      f"{[round(float(c), 2) for c in _hs['curls']]} ({HAND_STATE_PATH})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[hand] could not read {HAND_STATE_PATH}: {e}")

    # Captured home JOINT config for the deck HOME key. Written by the teleop panel's 'Set Home' and
    # shared with it, so eval returns to exactly the pose the demos started/ended at.
    home_q = None
    if os.path.exists(HOME_PATH):
        try:
            home_q = json.load(open(HOME_PATH)).get("q")
            print(f"[home] loaded home pose from {HOME_PATH}")
        except Exception as e:  # noqa: BLE001
            print(f"[home] could not read {HOME_PATH}: {e}")
    else:
        print(f"[home] no {HOME_PATH} — the HOME key will be inert until 'Set Home' in the panel")
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
            return remote_client.infer(state, imgs, run_cfg["task"])

        def reset_policy():
            remote_client.reset()
    else:
        def infer(state, imgs):
            obs = {"observation.state": torch.from_numpy(state)[None], "task": [run_cfg["task"]]}
            for name, img in imgs.items():
                obs[f"observation.images.{name}"] = chw(img)
            with torch.no_grad():
                return post(policy.select_action(pre(obs)))[0].cpu().numpy()

        def reset_policy():
            policy.reset()

    evq: "queue.Queue[int]" = queue.Queue()

    # `supervisorctl stop vla_eval` (SIGTERM) -> EXIT: the run is closed + uploaded, the UR powered down

    import signal as _signal

    try:

        _signal.signal(_signal.SIGTERM, lambda *_: evq.put(KEY_EXIT))

    except Exception:  # noqa: BLE001

        pass
    held: dict = {}                      # live key up/down state (unused: HOME is press-and-go)
    pad, ctrl = _make_control(evq, held)
    st = {"mode": "paused", "played": 0, "exit": False, "homing": False, "home_steps": 0, "home_result": None,
          "rating": False}

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

    def _start_play() -> None:
        reset_policy()          # fresh chunk: re-observe from wherever the arm is now
        st["played"] = 0
        set_mode("playing")

    def enter_playing() -> None:
        """PLAY. The arm walks to the captured home pose first and the policy starts from there: every
        demo starts at home, so a run started anywhere else begins out of distribution. The return is the
        HOME key's press-and-go homing (one servoJ increment per tick, any key cancels) plus a latch that
        starts the policy the moment home is reached (see the homing branch of the loop). Already at home
        = one tick. --no-home-start, or no captured home pose, plays from the current pose."""
        if st["mode"] == "freedrive":
            stop_freedrive()
        if args.no_home_start or home_q is None or arm is None or not hasattr(arm, "home_step"):
            if home_q is None and not args.no_home_start:
                print(f"[home] no {HOME_PATH} — playing from the current pose", flush=True)
            _start_play()
            return
        st["play_after_home"] = True
        start_homing()
        print("[home] going home first — the policy starts when the arm gets there", flush=True)

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
            robot.send_curls([0.0] * HAND_CURL_DIM)
            print("[hand] opened")
        except Exception as e:  # noqa: BLE001 — a hand fault must not kill the eval loop
            print(f"[hand] open failed: {e}")

    def close_hand() -> None:
        """Close the grasp by hand (all four fingers to full curl). PAUSES first for the same reason as
        open_hand: the policy commands curls every step and would overwrite the close at once."""
        if st["mode"] == "playing":
            set_mode("paused")
            print("[hand] paused so the close isn't overwritten by the next policy action")
        try:
            robot.send_curls([1.0] * HAND_CURL_DIM)
            print("[hand] closed")
        except Exception as e:  # noqa: BLE001
            print(f"[hand] close failed: {e}")

    def start_homing() -> None:
        """Latch the return home. PRESS AND GO: one press runs the arm to home on its own; the key does
        NOT have to be held. The motion happens one servoJ increment per control tick in the idle branch
        below — never as a blocking loop — so PAUSE/EXIT and every other key stay live while it moves.

        It stops on any of: reaching home, the HOME_MAX_STEPS cap, pressing HOME again, or pressing any
        other key. Since the operator is no longer holding a dead-man key, those outs are the whole
        safety story for an arm moving unattended — keep the e-stop in hand.
        """
        if home_q is None:
            print(f"[home] no home pose — run the teleop panel's 'Set Home' to write {HOME_PATH}")
            return
        if arm is None or not hasattr(arm, "home_step"):
            print("[home] this engine has no home_step (hardware only)")
            return
        if st["mode"] == "freedrive":
            stop_freedrive()      # home_step is a no-op while the arm is in teach mode
        if st["mode"] == "playing":
            set_mode("paused")    # the policy would fight the return
        st["homing"], st["home_steps"] = True, 0
        print("[home] homing — press HOME again (or any key) to stop", flush=True)

    def stop_homing(why: str) -> None:
        if st["homing"]:
            st["homing"] = False
            st["home_result"] = why
            print(f"[home] {why} after {st['home_steps']} steps")

    def _cancelled_prehome() -> bool:
        """A key stopped the return that PLAY had started: drop the latch so the policy does not start
        when a later HOME press reaches home, and stay paused."""
        if st.pop("play_after_home", False):
            print("[home] pre-play homing cancelled — paused; PLAY again to retry", flush=True)
            return True
        return False

    def homing_tick() -> None:
        """One increment toward home. Joint space (servoJ), so no IK/singularity fault."""
        try:
            reached = arm.home_step(home_q, HOME_DQ)
        except Exception as e:  # noqa: BLE001
            stop_homing(f"failed ({e})")
            return
        st["home_steps"] += 1
        if reached:
            stop_homing("reached home")
        elif st["home_steps"] >= HOME_MAX_STEPS:
            stop_homing(f"gave up (cap {HOME_MAX_STEPS})")

    def do_reset() -> None:
        """Reset the policy's internal state and PAUSE (arm holds where it is). Use to abort a run and
        start clean — then FREEDRIVE to reposition and PLAY again. Also opens the hand, for the same
        reason new_run does: a clamped hand is an out-of-distribution start state (see release_hand)."""
        if st["mode"] == "freedrive":
            stop_freedrive()
        release_hand()
        reset_policy()
        st["played"] = 0
        set_mode("paused")
        print("[reset] policy reset, hand opened — paused")

    _render_deck(pad, "paused")
    dt = 1.0 / args.fps
    max_step = getattr(arm, "max_step", 0.05)  # translation clamp (m); NB orientation is NOT clamped
    prev = {"tgt": None, "t_end": None}
    # Connect the cameras NOW, not on the first read. make_engine hands back lazy grabbers, and
    # UsbCamera.connect() costs ~2.25 s each (VideoCapture open + 15 auto-exposure warm-up frames) —
    # ~6.9 s for three. Left lazy, that lands inside the first control step: the operator presses PLAY,
    # the arm sits still for seven seconds, then moves. Pay it here, while paused, instead.
    # Startup (cameras + exposure phase) runs BEFORE the main try/finally, so an exception here
    # used to skip every cleanup path: the Stream Deck HID handle stayed open and libusb aborted
    # the process with a pthread assertion, leaving the deck wedged until it was unplugged.
    try:
        print("[cam] connecting cameras (~7s: VideoCapture open + auto-exposure warm-up)…", flush=True)
        _t_cam = time.time()
        observe()
        print(f"[cam] ready in {time.time() - _t_cam:.1f}s — " +
              "  ".join(f"{k}={v:.0f}ms" for k, v in last_obs_ms.items() if v), flush=True)

        # --- launch-time exposure: auto-expose, operator trims on row 1, lock ---------------------
        # Runs AFTER the cameras are connected (they must be streaming to meter) and BEFORE the loop, so
        # the policy never sees a frame at a different exposure than the one the operator signed off on.
        # The locked values go into run.json: without them a rollout cannot be compared against the
        # dataset it was trained on, and "it looked different" stays unfalsifiable.
        expo_values = None
        if exposure is not None and not args.no_expose:
            def _render_expo_keys() -> None:
                if pad is None:
                    return
                for k, lab in EXPO_KEYS.items():
                    if k < pad.key_count:
                        pad.set_label(k, lab, bg=(40, 70, 120) if k != KEY_RESET else (20, 120, 40))
                for k in range(3, pad.key_count):   # everything else dark until exposure is set
                    pad.set_label(k, "", bg=(12, 12, 12))

            class _Preview:
                """Live feeds during the exposure phase, via Tk — NOT cv2.imshow.

                The robot PC has opencv-python-headless (`GUI: NONE`), so cv2.imshow raises and takes the
                process down mid-phase. The teleop panel already draws its feeds with Tk + PIL on this box,
                so this uses the same path. Over a plain SSH session there is no DISPLAY at all, so the
                preview degrades to the numeric readout below rather than failing the run.
                """

                def __init__(self):
                    self.ok = False
                    try:
                        import tkinter as tk
                        from PIL import Image, ImageTk  # noqa: F401

                        self.tk, self.Image, self.ImageTk = tk, Image, ImageTk
                        self.root = tk.Tk()
                        self.root.title("exposure — scene / wrist / side, top to bottom   (+/- trim, s = SET)")
                        self.label = tk.Label(self.root, bg="black")
                        self.label.pack()
                        self.root.bind("<Key>", self._key)
                        self.ok = True
                    except Exception as e:  # noqa: BLE001 — no DISPLAY / no Tk -> numbers only
                        print(f"[expo] no live preview ({type(e).__name__}); "
                              "trimming by the per-camera readout below", flush=True)

                def _key(self, ev):
                    k = (ev.char or "").lower()
                    if k in ("+", "="):
                        evq.put(KEY_PLAY)
                    elif k in ("-", "_"):
                        evq.put(KEY_FREEDRIVE)
                    elif k in ("s", "\r"):
                        evq.put(KEY_RESET)
                    elif k in ("x", "q"):
                        st["exit"] = True

                MAX_H = 960      # column height cap for the robot PC's display; width follows

                def show(self, imgs):
                    if not self.ok:
                        return
                    tiles = [imgs[n] for n in ("scene", "wrist", "side") if n in imgs]
                    if not tiles:
                        return
                    # Stacked VERTICALLY — scene / wrist / side, top to bottom — at a common width,
                    # so the column sits beside the terminal instead of spanning the screen.
                    w = min(t.shape[1] for t in tiles)
                    col = np.vstack([cv2.resize(t, (w, max(1, round(t.shape[0] * w / t.shape[1]))))
                                     for t in tiles])
                    if col.shape[0] > self.MAX_H:
                        col = cv2.resize(col, (max(1, round(col.shape[1] * self.MAX_H / col.shape[0])),
                                               self.MAX_H))
                    try:
                        photo = self.ImageTk.PhotoImage(self.Image.fromarray(col))
                        self.label.configure(image=photo)
                        self.label.image = photo          # keep a ref or Tk garbage-collects it
                        self.root.update()
                    except Exception:  # noqa: BLE001 — window closed mid-phase
                        self.ok = False

                def close(self):
                    if self.ok:
                        try:
                            self.root.destroy()
                        except Exception:  # noqa: BLE001
                            pass
                        self.ok = False

            # The sensor does not reflect a new exposure until ~2 frames later: measured on the scene cam,
            # a +56% step read median 21 (the OLD value) on frame 0, 179 on frame 1, then settled at ~235
            # from frame 2 on. Reading once after a nudge therefore reports the PREVIOUS setting, which had
            # the readout claiming "darker" made the image brighter. Throw the stale frames away first.
            SETTLE_FRAMES = 3

            def _settled_obs():
                for _ in range(SETTLE_FRAMES - 1):
                    observe()
                return observe()

            def _readout(imgs) -> str:
                """Per-camera median + clip%, so the phase is still usable with no preview window."""
                out = []
                for n in ("scene", "wrist", "side"):
                    if n in imgs:
                        out.append(f"{n}={exposure.values.get(n, 0):.0f} "
                                   f"med:{np.median(imgs[n]):.0f} clip:{exposure.clip_pct(imgs[n]):.1f}%")
                return "  |  ".join(out)

            print("[expo] auto-exposing (searching for a non-clipping setting)…", flush=True)
            exposure.begin()
            exposure.auto_expose(W, H)
            print(f"[expo] auto -> {exposure.summary()}", flush=True)
            # Default: take the metered result and lock it — auto has been reliable on this rig.
            # --expose-manual keeps the trim loop (deck row 1 / +/-/s) for when it is not.
            if args.expose_manual:
                _render_expo_keys()
                print("[expo] ＋/－ to trim, ✓ SET to lock   (deck row 1, or +/-/s on the keyboard)", flush=True)
                preview = _Preview()
                _, _imgs0 = _settled_obs()
                print(f"[expo] {_readout(_imgs0)}", flush=True)
                while not st["exit"]:
                    _, imgs = observe()
                    preview.show(imgs)
                    done = False
                    try:
                        while True:
                            key = evq.get_nowait()
                            if key in (KEY_PLAY, KEY_FREEDRIVE):
                                up = key == KEY_PLAY
                                exposure.nudge(exposure.STEP if up else 1.0 / exposure.STEP)
                                _, imgs = _settled_obs()     # settled re-read, not the stale in-flight frame
                                print(f"[expo] {'+' if up else '-'} -> {_readout(imgs)}", flush=True)
                            elif key == KEY_RESET:
                                done = True
                            elif key == KEY_EXIT:
                                st["exit"] = True
                    except queue.Empty:
                        pass
                    if done:
                        break
                    if not preview.ok:
                        time.sleep(0.05)     # no Tk pump to pace the loop; don't spin the cameras flat out
                preview.close()
            expo_values = exposure.lock()
            print(f"[expo] LOCKED: {exposure.summary()}", flush=True)
            _render_deck(pad, "paused")   # hand row 1 back to PLAY / FREEDRIVE / RESET
            # NB the locked value is NOT written here — the run directory does not exist yet. Run.write_meta
            # picks `expo_values` up from this scope and stamps it on every run's eval.json.

        run = Run(1, _fixed_dir)
    except BaseException:
        if pad is not None:
            pad.close()
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass
        if remote_client is not None:
            remote_client.close()
        _ur_power_down()
        raise

    def _enter_rating() -> None:
        """Light the bottom row 1-5 for the run just finished. Non-blocking: the loop keeps servicing
        the arm, and any other key simply leaves the run unrated rather than trapping the operator."""
        st["rating"] = True
        if pad is None:
            print("[rate] press 1-5 to rate this run (any other key skips)", flush=True)
            return
        for i, k in enumerate(RATING_KEYS):
            if k < pad.key_count:
                pad.set_label(k, f"{i + 1}\n{'★' * (i + 1)}", bg=(70, 55, 10), fg=(245, 210, 60))

    def _exit_rating() -> None:
        st["rating"] = False
        if pad is not None:
            for k in RATING_KEYS:
                if k < pad.key_count:
                    pad.set_label(k, "", bg=(12, 12, 12))
            if KEY_CLOSE_HAND < pad.key_count:
                pad.set_label(KEY_CLOSE_HAND, "✊")   # the stars borrowed its key

    def release_hand(settle: float = 0.6) -> None:
        """Open the hand and WAIT for it to physically get there before the next run observes.

        observation.state carries HAND_CURL_NAMES, so the curls are a policy input, not just an output.
        A run that ends holding the object leaves the hand clamped, and the next run then starts from
        (arm at home, hand closed) — a combination that appears nowhere in the demos, which all start
        open. Measured across two consecutive runs: run 1 started at curls ~0.007 and was rated 5; run 2
        started at ~0.825 and spent the run oscillating its way back open (curl churn 3.6x run 1's).
        This policy leans on proprioception far more than vision, so a wrong hand state at step 0 is the
        worst possible initial condition.
        """
        try:
            robot.send_curls([0.0] * HAND_CURL_DIM)
        except Exception as e:  # noqa: BLE001 — a hand fault must not block the next run
            print(f"[hand] release failed: {e}")
            return
        # No finger feedback exists on this firmware path: observation.state's curls ARE the cache
        # `robot.send_curls` just set to zero, so polling the observation "to confirm the release"
        # only ever read that cache back. (Worse, before send_curls existed the direct hand call left
        # the cache at the previous grasp, the poll never confirmed, and run 2 started "closed".)
        # What is left to wait for is the physical open — a fixed settle covers it.
        time.sleep(settle)
        print(f"[hand] released (cache zeroed, {settle:.1f}s settle)")

    def new_run(rate: bool = True) -> None:
        """Finish this rollout and open the next — no process restart, cameras and exposure untouched."""
        nonlocal run
        if st["mode"] == "freedrive":
            stop_freedrive()
        set_mode("paused")
        release_hand()      # restore the open-hand initial condition every demo starts from
        reset_policy()
        if rate and run.played:
            _enter_rating()
            print(f"[run {run.n}] rate it 1-5, then the next run starts", flush=True)
            st["pending_roll"] = True
        else:
            run = _web_roll(run)
            st["played"] = 0

    def apply_rating(stars: int) -> None:
        nonlocal run
        run.rating = int(stars)
        run.write_meta()
        print(f"[run {run.n}] rated {stars}★", flush=True)
        _exit_rating()
        if st.pop("pending_roll", False):
            run = _web_roll(run)
            st["played"] = 0

    # ---- browser UI (ur5e_lerobot/ui/webpanel.py): same page as the teleop panel, in eval mode. Its
    # commands become the KEY_* events the deck and keyboard produce, so one dispatch handles all.
    web = None
    web_frames = False
    web_t: dict = {}
    web_runs: list = []
    web_last_push = [0.0]
    web_events = None
    if args.ui == "web":
        try:
            from ur5e_lerobot.ui.webpanel import WebPanel, service_cam_urls
            _names = [n for n, f in (("scene", scene_fn), ("wrist", wrist_fn), ("side", side_fn)) if f is not None]
            if args.cameras == "udp":
                _cams = service_cam_urls(_names, args.camera_host)
            else:
                _cams = {n: None for n in _names}
                web_frames = True
            web = WebPanel("eval", f"{args.task} · {args.remote or os.path.basename(args.ckpt or '')}",
                           port=args.web_port, cams=_cams, on_ready=lambda u: print(f"[web] open {u}", flush=True))
            _u = web.start()
            print(f"[web] open {_u}" if _u else f"[web] port {args.web_port} busy — retrying every 3 s", flush=True)
            import builtins as _bi
            _orig_print = _bi.print

            def _tee_print(*a, **k):   # every console line also reaches the page's log pane
                _orig_print(*a, **k)
                try:
                    web.log(" ".join(str(x) for x in a))
                except Exception:  # noqa: BLE001
                    pass
            _bi.print = _tee_print
            try:
                from ur5e_lerobot.robot.ur_events import EventListener
                web_events = EventListener(args.robot_ip)
                web_events.start()
            except Exception:  # noqa: BLE001
                web_events = None
        except OSError as e:
            print(f"[web] unavailable ({e}) — is the teleop panel serving port {args.web_port}?")
            web = None
        except Exception as e:  # noqa: BLE001
            print(f"[web] failed: {e}")
            web = None

    _WEB_KEYS = {"play": KEY_PLAY, "freedrive": KEY_FREEDRIVE, "reset": KEY_RESET, "home": KEY_HOME,
                 "newrun": KEY_NEWRUN, "open_hand": KEY_OPEN_HAND, "close_hand": KEY_CLOSE_HAND, "exit": KEY_EXIT}

    def _pump_web_eval() -> None:
        if web is None:
            return
        for c in web.commands():
            name, a = c.get("name"), c.get("args") or {}
            if name in _WEB_KEYS:
                evq.put(_WEB_KEYS[name])
            elif name == "rating":
                n = int(a.get("stars", 0))
                if 1 <= n <= 5:
                    if not st["rating"]:
                        _enter_rating()   # rate the current run even without NEW RUN
                    evq.put(RATING_KEYS[n - 1])
            elif name == "expo":
                print("[web] exposure trim is deck/keyboard-only during the manual phase (--expose-manual)")
            else:
                print(f"[web] unknown command {name!r}")

    def _web_frames(imgs: dict) -> None:
        if web is not None and web_frames:
            for n, im in imgs.items():
                web.set_frame(n, im)

    def _web_roll(r):
        web_runs.append({"name": os.path.basename(r.dir), "n": r.n, "steps": r.played, "rating": r.rating})
        return r.roll()

    def _web_state(force: bool = False) -> None:
        if web is None:
            return
        now = time.time()
        if not force and now - web_last_push[0] < 0.1:
            return
        web_last_push[0] = now
        fault = None
        if web_events is not None and web_events.safety_mode not in (None, "NORMAL", "REDUCED"):
            e = web_events.last_safety_event()
            fault = e.text if e else web_events.safety_mode
        flags = {k: served.get(k) for k in ("n_action_steps", "prefetch", "prefetch_at", "rtc",
                                             "rtc_execution_horizon", "num_steps") if served.get(k) is not None}
        cap = (served.get("capture") or {}).get("squares") if isinstance(served.get("capture"), dict) else None
        ck = str(served.get("checkpoint") or args.ckpt or "")
        if "/checkpoints/" in ck:
            _run_name = ck.split("/checkpoints/")[0].rstrip("/").split("/")[-1]
            _step = ck.split("/checkpoints/")[1].split("/")[0]
            model_key = f"{_run_name}/{_step}"
        else:
            model_key = ck[-40:] or None
        web.set_state({
            "mode": "eval", "play_mode": st["mode"], "rating_pending": bool(st["rating"]), "task": run_cfg["task"],
            "stage": ({"name": stage["list"][stage["i"]]["name"], "index": stage["i"] + 1,
                       "count": len(stage["list"]), "phase": stage["phase"]}
                      if stage["list"] and stage["i"] >= 0 else None),
            "run": {"name": os.path.basename(run.dir), "n": run.n, "steps": st["played"],
                    "duration_s": now - run.started, "rating": run.rating},
            "runs": web_runs[-20:],
            "timing": dict(web_t),
            "model": {"model_key": model_key, "checkpoint": ck[-60:] or None, "server": args.remote or "local",
                      "flags": " ".join(f"{k}={v}" for k, v in flags.items()) or None,
                      "capture": " ".join(f"{k} {v}" for k, v in cap.items()) if cap else None},
            "ur": {"connected": True, "freedrive": st["mode"] == "freedrive", "homing": bool(st["homing"]),
                   "fault": fault},
            "hand": {"ok": True},
            "tick": {"ms": web_t.get("step_ms", 0.0), "hz": (1000.0 / web_t["step_ms"]) if web_t.get("step_ms") else 0.0},
            "exposure": {"values": dict(getattr(exposure, "values", {}) or {}),
                         "locked": bool(getattr(exposure, "locked", False))},
        })

    # ---- --auto: the run drives itself (the route runner's vla: actions) --------------------------
    auto = {"t_play": None, "still_since": None, "phase": None, "curl0": None, "grasp_event": False,
            "curl_hist": [], "was_closed": False, "ever_played": False}

    def auto_finish(reason: str) -> None:
        """The run is over: pause, open the hand (unless carrying), go home (unless told not to), exit."""
        auto_meta.update(stop=reason, played=st["played"],
                         play_s=round(time.time() - (auto["t_play"] or time.time()), 1))
        print(f"[auto] finished: {reason} after {st['played']} steps", flush=True)
        set_mode("paused")
        if run_cfg["keep"]:
            print("[auto] keeping the grasp (hand stays closed; UR stays powered at exit)", flush=True)
        else:
            release_hand()
        if run_cfg["home"] and home_q is not None and arm is not None and hasattr(arm, "home_step"):
            auto["phase"] = "homing"
            start_homing()
        else:
            after_run()

    def auto_abort(reason: str, code: int = 3) -> None:
        auto_meta.update(stop=reason, played=st["played"], aborted=True)
        exit_code[0] = code
        print(f"[auto] ABORT: {reason} — exit code {code}", flush=True)
        st["exit"] = True

    def grasp_done() -> "str | None":
        """--done-when: the mean commanded curl over the last `still` seconds, judged as a LEVEL."""
        win = run_cfg["still"] or 2.0
        now = time.time()
        auto["curl_hist"] = [(t, c) for t, c in auto["curl_hist"] if t >= now - win]
        h = auto["curl_hist"]
        if len(h) < 2 or h[0][0] > now - win * 0.9:
            return None                                   # the window is not spanned yet
        avg = sum(c for _, c in h) / len(h)
        if run_cfg["done_when"] == "closed" and avg >= AUTO_CLOSED_CURL:
            return f"hand closed (mean curl {avg:.2f} over {win:.0f} s)"
        if run_cfg["done_when"] == "open" and auto["was_closed"] and avg <= AUTO_OPEN_CURL:
            return f"hand open (mean curl {avg:.2f} over {win:.0f} s, after a grasp)"
        return None

    # ---- stages: one process, several runs — grab, then place, …; PLAY is the go for each ------------
    stage = {"list": stages, "i": -1, "cycle": 0, "phase": None}

    def _select_model(model: str) -> dict:
        """Make the server serve `model`: instant when it is resident (select); else repoint + restart
        it through services and reconnect this client (~16 s)."""
        nonlocal remote_client
        from ur5e_lerobot import services as _svc

        rep = {}
        try:
            rep = remote_client.select(model)
        except Exception as e:  # noqa: BLE001
            print(f"[stage] select failed ({e}); restarting the server", flush=True)
        if rep.get("ok"):
            return rep
        if rep.get("error"):
            print(f"[stage] {rep['error']} — repoint + restart (~16 s)", flush=True)
        host, _, port = args.remote.partition(":")
        host, port = host or "127.0.0.1", int(port or 8777)
        try:
            remote_client.close()
        except Exception:  # noqa: BLE001
            pass
        info = _svc.serve_model(model, host, port, log=lambda m: print(f"[stage] {m}", flush=True))
        remote_client = RemotePolicyClient(host, port).connect()
        return info

    def prepare_stage(i: int) -> None:
        nonlocal run
        s = stage["list"][i]
        n = len(stage["list"])
        stage["i"], stage["phase"] = i, "prepare"
        run_cfg.update(task=s["task"], done_when=s["done"], keep=s["keep"], home=s["home"],
                   max_seconds=s["seconds"] or args.max_seconds, still=s["still"] or args.still)
        print(f"[stage {i + 1}/{n}] {s['name']}: model {s['model']}, task {s['task']!r}, done when "
              f"{s['done']}, {'keep grasp' if s['keep'] else 'open hand'}, {'home' if s['home'] else 'no home'}",
              flush=True)
        t_prep = time.time()
        info = _select_model(s["model"])
        served.clear()
        served.update(info)
        want = {}
        cap_ = info.get("capture") if isinstance(info, dict) else None
        if cap_ and cap_.get("squares"):
            want = {k: v for k, v in cap_["squares"].items() if k in live_cams}
        want.update(forced)
        if live_cams and want:
            before = squares_of(live_cams)
            crop_applied.clear()
            crop_applied.update(_apply_squares(live_cams, want))
            if before != crop_applied:
                print(f"[stage] crop {before} -> {crop_applied}", flush=True)
        reset_policy()
        if run.played:
            run = _web_roll(run)      # the previous stage's run is closed + uploaded; this one gets its own
        else:
            run.write_meta()
        st["played"] = 0
        auto.update(t_play=None, still_since=None, phase=None, curl0=None, grasp_event=False,
                    curl_hist=[], was_closed=False)
        stage["phase"] = "wait_go"
        set_mode("paused")
        if pad is not None:
            pad.set_label(KEY_PLAY, f"GO\n▶ {s['name']}", bg=(25, 120, 40))
        print(f"[stage {i + 1}/{n}] ready in {time.time() - t_prep:.1f} s — PLAY = go · RESET = redo the "
              f"previous stage · EXIT = quit", flush=True)

    def next_stage() -> None:
        i = stage["i"] + 1
        if i >= len(stage["list"]):
            stage["cycle"] += 1
            if args.cycles and stage["cycle"] >= args.cycles:
                print(f"[stage] all {len(stage['list'])} stages done, {stage['cycle']} cycle(s) — exiting", flush=True)
                st["exit"] = True
                return
            i = 0
        prepare_stage(i)

    def after_run() -> None:
        """What follows a finished --auto run: the next stage, or exit."""
        if stage["list"]:
            next_stage()
        else:
            st["exit"] = True

    print(f"READY — PAUSED. control: {ctrl}. PLAY/PAUSE · FREEDRIVE · RESET · NEW RUN · EXIT.")
    print("**e-STOP IN HAND.**")
    if args.auto:
        print(f"[auto] unattended run: {'waiting for PLAY (deck / page / p) as the go' if args.wait_go else 'PLAY now'}; "
              f"stop on steps={args.steps or 'off'} / {args.max_seconds:.0f} s / still={args.still or 'off'} s "
              f"({args.still_mode}); then {'keep grasp' if args.keep_grasp else 'open hand'}, "
              f"{'no home' if args.no_home else 'home'}, exit", flush=True)
        if stage["list"]:
            prepare_stage(0)
        elif not args.wait_go:
            evq.put(KEY_PLAY)
    try:
        while not st["exit"]:
            _pump_web_eval()   # browser page -> the same KEY_* events as the deck / keyboard
            try:  # drain control events (deck callback / stdin thread)
                while True:
                    key = evq.get_nowait()
                    # ANY other key cancels an in-progress return first — checked before the dispatch
                    # below, because those branches would otherwise consume the key and leave the latch
                    # armed (e.g. PLAY would start the policy with homing still pending for the pause).
                    if st["rating"]:
                        # bottom row owns 1-5 while a run is awaiting its rating; anything else skips
                        if key in RATING_KEYS:
                            apply_rating(RATING_KEYS.index(key) + 1)
                            continue
                        _exit_rating()
                        if st.pop("pending_roll", False):
                            run = _web_roll(run)
                            st["played"] = 0
                            print("[rate] skipped — next run started", flush=True)
                    if stage["list"] and stage["phase"] == "wait_go":
                        if key == KEY_RESET:               # redo the previous stage
                            prepare_stage(max(0, stage["i"] - 1))
                            continue
                        if key == KEY_PLAY:
                            stage["phase"] = "playing"
                    if key == KEY_NEWRUN:
                        new_run()
                        continue
                    if key != KEY_HOME and st["homing"]:
                        stop_homing("interrupted")
                        if _cancelled_prehome() and key == KEY_PLAY:
                            continue        # that PLAY was the cancel, not a new go
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
                    elif key == KEY_CLOSE_HAND:
                        close_hand()
                    elif key == KEY_HOME:
                        # press-and-go: a second press is the cancel, since no key is being held
                        if st["homing"]:
                            stop_homing("cancelled")
                            _cancelled_prehome()
                        else:
                            start_homing()
            except queue.Empty:
                pass
            if st["exit"]:
                break
            _web_state()

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
                    if args.auto:
                        auto_abort("UR control stopped (protective stop / e-stop) during the run")
                        continue
                    print("  paused. clear the e-stop if engaged, FREEDRIVE to a safe pose, then PLAY.")
                    continue
                t0 = time.time()
                state, imgs = observe()
                _web_frames(imgs)
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
                    web_t.update(step_ms=loop_ms, infer_ms=(t_inf - t_obs) * 1000.0, obs_ms=(t_obs - t0) * 1000.0)
                    run.log_w.writerow([st["played"], round(loop_ms, 1), round((t_obs - t0) * 1000, 1),
                                    round(last_obs_ms["state"], 1), round(last_obs_ms["scene"], 1),
                                    round(last_obs_ms["wrist"], 1), round(last_obs_ms["side"], 1),
                                    round((t_inf - t_obs) * 1000, 1), round((t_snd - t_inf) * 1000, 1),
                                    round(d_move, 1), round(d_turn, 1), round(d_cmd_m, 1), round(d_cmd_r, 1), clamp,
                                    *[round(float(v), 4) for v in cur], *[round(float(v), 4) for v in tgt],
                                    *[round(float(v), 3) for v in act[6:10]]])
                    run.log_f.flush()
                    # Name the culprit the moment a source stalls, instead of only in the CSV later.
                    slow = {k: v for k, v in last_obs_ms.items() if v > args.obs_warn_ms}
                    if slow:
                        print("  [obs] SLOW: " + "  ".join(f"{k}={v:.0f}ms" for k, v in
                                                           sorted(slow.items(), key=lambda kv: -kv[1])))
                    if st["played"] % max(1, args.log_every) == 0:
                        print(f"t{st['played']:<4} loop={loop_ms:4.0f}ms inf={(t_inf - t_obs) * 1000:4.0f}ms "
                              f"Δmove={d_move:5.1f}mm Δturn={d_turn:5.1f}° cmdΔturn={d_cmd_r:5.1f}° "
                              f"{'CLAMP' if clamp else '     '} curls={np.round(act[6:10], 2)}")
                    if args.auto:
                        pa = prev.get("act")
                        curl_d = float(np.max(np.abs(act[6:10] - pa[6:10]))) if pa is not None else 0.0
                        if auto["curl0"] is None:
                            auto["curl0"] = act[6:10].copy()
                        elif not auto["grasp_event"] and float(np.max(np.abs(act[6:10] - auto["curl0"]))) > AUTO_GRASP_CURL:
                            auto["grasp_event"] = True
                            if args.still_mode == "fingers":
                                print(f"[auto] fingers moved {np.round(auto['curl0'], 2)} -> {np.round(act[6:10], 2)}"
                                      f" — the {args.still:.0f} s stillness countdown is armed", flush=True)
                        fingers_moving = curl_d > AUTO_STILL_CURL
                        if args.still_mode == "fingers":
                            moving = fingers_moving or not auto["grasp_event"]
                        else:
                            moving = d_cmd_m > AUTO_STILL_MM or d_cmd_r > AUTO_STILL_DEG or fingers_moving
                        auto["still_since"] = None if moving else (auto["still_since"] or t_snd)
                        mc = float(np.mean(act[6:10]))
                        auto["curl_hist"].append((t_snd, mc))
                        if mc >= AUTO_ARMED_CURL:
                            auto["was_closed"] = True
                        prev["act"] = act
                    prev["tgt"], prev["t_end"] = tgt, t_snd
                except Exception as e:  # noqa: BLE001
                    print(f"[log] {e}")
                if run.writer is not None:
                    f = cv2.cvtColor(imgs["scene"], cv2.COLOR_RGB2BGR)
                    cv2.putText(f, f"t{st['played']}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    run.writer.write(f)
                st["played"] += 1
                run.played = st["played"]
                if args.auto:
                    if auto["t_play"] is None:
                        auto["t_play"], auto["ever_played"] = t0, True
                        if float(np.mean(state[-HAND_CURL_DIM:])) >= AUTO_ARMED_CURL:
                            auto["was_closed"] = True      # started holding (a hand-over from a grab)
                    play_s = time.time() - auto["t_play"]
                    why = None
                    if args.steps and st["played"] >= args.steps:
                        why = f"steps ({args.steps})"
                    elif play_s >= run_cfg["max_seconds"]:
                        why = f"max_seconds ({run_cfg['max_seconds']:.0f} s)"
                    elif run_cfg["done_when"] != "none":
                        if play_s >= args.min_seconds:
                            why = grasp_done()
                    elif (run_cfg["still"] and play_s >= args.min_seconds and auto["still_since"] is not None
                          and time.time() - auto["still_since"] >= run_cfg["still"]):
                        why = (f"still ({run_cfg['still']:.0f} s without "
                               f"{'finger' if args.still_mode == 'fingers' else 'commanded'} motion)")
                    if why:
                        auto_finish(why)
                elif args.steps and st["played"] >= args.steps:
                    print(f"[cap] reached --steps={args.steps}; pausing")
                    set_mode("paused")
                time.sleep(max(0.0, dt - (time.time() - t0)))
            elif st["homing"]:
                # press-and-go: the latch runs to completion on its own. It used to also require the
                # deck key to still be down, which made this hold-to-home; the key-up check is gone.
                homing_tick()
                if not st["homing"] and st.pop("play_after_home", False):
                    # PLAY's return finished: start the policy from home, or not at all
                    if st["home_result"] == "reached home":
                        _start_play()
                    elif args.auto:
                        auto_abort(f"could not reach home before the run: {st['home_result']}")
                    else:
                        print(f"[home] {st['home_result']} — not starting; PLAY again to retry", flush=True)
                if auto["phase"] == "homing" and not st["homing"]:
                    auto["phase"] = None
                    print("[auto] home", flush=True)
                    after_run()
                time.sleep(1.0 / args.fps)
            else:  # paused or freedrive — don't drive; the arm holds (or is hand-guided)
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\naborted")
    finally:
        try:
            run.close()
        except Exception:  # noqa: BLE001
            pass
        if pad is not None:
            pad.close()
        if web is not None:
            web.stop()
        if remote_client is not None:
            remote_client.close()
        try:   # what the hand was last told, for the next stage (see HAND_STATE_PATH above)
            with open(HAND_STATE_PATH, "w") as f:
                json.dump({"curls": [float(c) for c in getattr(robot, "_hand_curls", [])],
                           "keep_grasp": bool(args.auto and run_cfg["keep"]), "time": time.time(),
                           "run": run.dir}, f)
        except Exception:  # noqa: BLE001
            pass
        robot.disconnect()
        if run_cfg["keep"] or args.leave_powered:
            print("[ur] left powered on (" + ("keep-grasp: the tool voltage keeps the hand's servos holding"
                                             if run_cfg["keep"] else "--leave-powered") + ")", flush=True)
        else:
            _ur_power_down()
    print("done")
    if args.auto and exit_code[0] == 0 and not auto.get("ever_played"):
        exit_code[0] = 5
        print("[auto] exited before any step was played (operator EXIT) — exit code 5", flush=True)
    if args.result_file:
        try:
            with open(args.result_file, "w") as f:
                json.dump({"exit_code": exit_code[0], "run_dir": run.dir, "steps": run.played,
                           "played": st["played"], **auto_meta}, f, indent=2, default=str)
        except Exception as e:  # noqa: BLE001
            print(f"[auto] could not write {args.result_file}: {e}")
    if exit_code[0]:
        sys.exit(exit_code[0])


if __name__ == "__main__":
    main()
