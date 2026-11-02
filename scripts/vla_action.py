#!/usr/bin/env python3
"""One unattended policy run on the arm, as a single command that returns when the run is over.

This is what the robot's route runner (rd_ws/Scripts/path.py, a `vla:` post_action) calls at a waypoint:

    ~/rd_ws/VLA/run.sh scripts/vla_action.py --model pickup_tigger_smolvla_ft --task "pick up tigger" \
        --seconds 90 --still 3 --keep-grasp

In order: (1) preflight — stop vla_teleop, vla_eval and the ROS UR driver if any of them runs (they own
the UR control script, the deck and page port 8090), (2) make vla_infer_server serve --model (repoint
outputs/serve_ckpt + restart only if it serves something else), (3) run scripts/eval_hw.py --auto, which
plays immediately, stops on --steps / --seconds / stillness, opens the hand unless --keep-grasp, returns
the arm to its home (transport) pose and exits. The exit code is the eval's: 0 done, 3 aborted on a UR
fault, 4 preflight/setup failure, 5 the operator pressed EXIT before the go (--wait-go). The run lands in outputs/eval_runs/ and the GPU's eval registry like any
other eval, tagged "auto" with the stop reason.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="training run name, run@step, or a checkpoint path")
    ap.add_argument("--task", required=True, help="the task string the model was trained with")
    ap.add_argument("--seconds", type=float, default=120.0, help="cap on the playing phase (wall clock)")
    ap.add_argument("--steps", type=int, default=0, help="cap in control steps (0 = none)")
    ap.add_argument("--still", type=float, default=0.0,
                    help="finish once nothing has been commanded to move for this many seconds (0 = off)")
    ap.add_argument("--min-seconds", type=float, default=10.0, help="no stillness check before this")
    ap.add_argument("--keep-grasp", action="store_true",
                    help="carry the object: hand stays closed and the UR stays powered for the drive")
    ap.add_argument("--no-home", action="store_true", help="do not return to the home pose at the end")
    ap.add_argument("--wait-go", action="store_true",
                    help="launch paused and take PLAY (deck / page / p) as the go instead of playing at once")
    ap.add_argument("--still-mode", choices=["all", "fingers"], default="all",
                    help="'fingers': the run ends once the fingers have moved (grasp / release) and then stayed still --still s")
    ap.add_argument("--leave-powered", action="store_true", help="do not power the UR down at the end (next stage follows)")
    ap.add_argument("--done-when", choices=["none", "closed", "open"], default="none",
                    help="end the run by the grasp level: 'closed' (a pick: hand closed and staying closed over --still s), "
                         "'open' (a place: released after having been closed)")
    ap.add_argument("--remote", default="127.0.0.1:8777", help="the robot's inference server")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--web-port", type=int, default=8090)
    ap.add_argument("--landmark", default="", help="the landmark this run is done at (recorded on the run's eval.json)")
    ap.add_argument("--extra", default="", help="extra eval_hw.py arguments, one string")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the eval command; touch nothing")
    a = ap.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    from ur5e_lerobot import services

    host, _, port = a.remote.partition(":")
    host, port = host or "127.0.0.1", int(port or 8777)
    result_file = os.path.join(tempfile.gettempdir(), f"vla_action_{os.getpid()}.json")
    cmd = [sys.executable, "scripts/eval_hw.py", "--remote", a.remote, "--cameras", "udp", "--ui", "web",
           "--web-port", str(a.web_port), "--task", a.task, "--fps", str(a.fps),
           "--auto", "--max-seconds", str(a.seconds), "--steps", str(a.steps), "--still", str(a.still),
           "--min-seconds", str(a.min_seconds), "--no-preflight", "--result-file", result_file]
    if not a.no_video:
        cmd.append("--video")
    if a.keep_grasp:
        cmd.append("--keep-grasp")
    if a.no_home:
        cmd.append("--no-home")
    if a.wait_go:
        cmd.append("--wait-go")
    if a.still_mode != "all":
        cmd += ["--still-mode", a.still_mode]
    if a.leave_powered:
        cmd.append("--leave-powered")
    if a.done_when != "none":
        cmd += ["--done-when", a.done_when]
    if a.landmark:
        cmd += ["--landmark", a.landmark]
    cmd += shlex.split(a.extra)

    if a.dry_run:
        try:
            ck = services.resolve_checkpoint(a.model)
        except FileNotFoundError as e:
            print(f"[action] {e}")
            return 4
        cur = services.served_checkpoint()
        print(f"[action] would stop: {', '.join(('vla_eval',) + services.EXCLUSIVE_WITH_EVAL)} (if running)")
        print(f"[action] would serve: {ck}" + ("  (already the serve_ckpt target)" if cur == ck else f"  (serve_ckpt now -> {cur})"))
        print(f"[action] would run: {shlex.join(cmd)}")
        return 0

    t0 = time.time()
    try:
        print(f"[action] preflight: {a.model!r} / {a.task!r}", flush=True)
        services.ensure_stopped(("vla_eval", *services.EXCLUSIVE_WITH_EVAL))
        info = services.serve_model(a.model, host, port)
    except Exception as e:  # noqa: BLE001
        print(f"[action] setup failed: {e}", flush=True)
        return 4
    print(f"[action] serving {info.get('checkpoint')} — starting the run after {time.time() - t0:.0f} s of setup",
          flush=True)
    print(f"[action] {shlex.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=ROOT).returncode

    res: dict = {}
    try:
        with open(result_file) as f:
            res = json.load(f)
        os.unlink(result_file)
    except Exception:  # noqa: BLE001
        pass
    reason = res.get("stop") or {5: "exited before go (operator EXIT)", 3: "aborted on a UR fault",
                                 4: "setup failed"}.get(rc, "exit %d" % rc)
    print(f"[action] {'done' if rc == 0 else 'FAILED'}: {reason} — {res.get('played', res.get('steps', '?'))} steps, "
          f"run {res.get('run_dir', '?')}, {time.time() - t0:.0f} s total, exit code {rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
