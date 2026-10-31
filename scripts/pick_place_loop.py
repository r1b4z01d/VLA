#!/usr/bin/env python3
"""Grab → place → repeat on the arm, each stage waiting for the operator's go — ONE eval process.

    ~/VLA/run.sh scripts/pick_place_loop.py                       # defaults below; --dry-run prints the command
    ~/VLA/run.sh scripts/pick_place_loop.py --pick-model <run> --place-model <run> --cycles 3

This is a thin wrapper over `scripts/eval_hw.py --stage … --stage … --cycles N`: the eval connects the
cameras, meters the exposure and powers the UR once, then for each stage switches the server to the
stage's model (instant when vla_infer_server has it resident — the unit file lists the route models —
else a ~16 s restart), applies the model's crop, and waits for **PLAY** (deck top-left, the page's ▶ PLAY,
or `p`) as the go. A stage ends by the GRASP LEVEL: the grab when the mean commanded curl has averaged
>= 0.7 over --still seconds (hand closed and staying closed; the grasp is kept, the arm goes home), the
place when it has averaged <= 0.2 after having been closed (the release happened; hand open, arm home).
RESET while a stage waits = redo the previous stage; EXIT ends the loop (exit 0, or 5 if nothing ran);
a UR fault aborts (3). The UR is powered down at the end unless the last stage kept a grasp.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pick-model", default="tigger_1_merged_smolvla_1")
    ap.add_argument("--pick-task", default="Grab and hold tigger")
    ap.add_argument("--place-model", default="tigger_bed_1_smolvla")
    ap.add_argument("--place-task", default="place tigger in the banana bed")
    ap.add_argument("--still", type=float, default=2.0, help="averaging window (s) for the grasp-level end test")
    ap.add_argument("--min-seconds", type=float, default=3.0, help="no end test before this much play")
    ap.add_argument("--seconds", type=float, default=120.0, help="cap on a stage's playing phase")
    ap.add_argument("--cycles", type=int, default=0, help="grab+place cycles to run (0 = until EXIT)")
    ap.add_argument("--remote", default="127.0.0.1:8777")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--extra", default="", help="extra eval_hw.py arguments")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)

    grab = (f"name=grab model={a.pick_model} task={shlex.quote(a.pick_task)} done=closed keep=1 "
            f"still={a.still} seconds={a.seconds}")
    place = (f"name=place model={a.place_model} task={shlex.quote(a.place_task)} done=open "
             f"still={a.still} seconds={a.seconds}")
    cmd = [sys.executable, "scripts/eval_hw.py", "--remote", a.remote, "--cameras", "udp", "--ui", "web",
           "--fps", str(a.fps), "--task", a.pick_task, "--min-seconds", str(a.min_seconds),
           "--stage", grab, "--stage", place, "--cycles", str(a.cycles), *shlex.split(a.extra)]
    if not a.no_video:
        cmd.append("--video")
    print(f"[loop] grab: {a.pick_model} {a.pick_task!r}  ->  place: {a.place_model} {a.place_task!r}  "
          f"({'until EXIT' if a.cycles == 0 else f'{a.cycles} cycle(s)'})", flush=True)
    print("[loop] " + shlex.join(cmd), flush=True)
    if a.dry_run:
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
