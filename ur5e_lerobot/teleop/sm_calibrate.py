"""One-shot SpaceMouse axis calibration — run AT the robot PC, from your normal operating position.

For each robot motion it asks you to push/twist the puck the way you want that motion; it captures
which raw SpaceMouse axis (and sign) you used and writes the mapping to ``outputs/sm_calib.json``.
The teleop panel loads that file on startup and overrides its built-in SM_MAP — so you never hand-
tune signs again, and you can re-run this anytime your setup/viewpoint changes.

    ~/VLA/run.sh -m ur5e_lerobot.teleop.sm_calibrate
"""
from __future__ import annotations

import json
import os
import time

from .spacemouse import SpaceMouse

RAW = ["x", "y", "z", "roll", "pitch", "yaw"]
# (robot axis, prompt) — do the gesture the way you want THAT arm motion, from where you stand.
AXES = [
    ("x", "SLIDE the puck the way you want the gripper to move RIGHT"),
    ("y", "SLIDE the puck the way you want the gripper to move AWAY from you (forward)"),
    ("z", "LIFT the puck the way you want the gripper to move UP"),
    ("roll", "TILT/TWIST the puck the way you want the gripper to ROLL"),
    ("pitch", "TILT the puck the way you want the gripper to PITCH (nose up)"),
    ("yaw", "TWIST the puck the way you want the gripper to YAW (spin)"),
]
CALIB_PATH = "outputs/sm_calib.json"


def _capture(sm: SpaceMouse, seconds: float = 1.5):
    sums = {k: 0.0 for k in RAW}
    n = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        sm.read()
        s = sm.state()
        for k in RAW:
            sums[k] += s[k]
        n += 1
        time.sleep(0.02)
    avg = {k: sums[k] / max(n, 1) for k in RAW}
    dom = max(RAW, key=lambda k: abs(avg[k]))
    return dom, avg[dom]


def main() -> None:
    sm = SpaceMouse().open()
    print(f"opened {sm.info.get('product_string')!r}\n")
    print("For each prompt: press ENTER, then immediately PUSH AND HOLD the puck that way until it")
    print("says captured (~1.5 s). Do it all from your normal operating position.\n")
    mapping: dict[str, tuple[str, int]] = {}
    for axis, desc in AXES:
        while True:
            input(f"[{axis}] ENTER, then: {desc} ...")
            dom, val = _capture(sm)
            if abs(val) < 0.15:
                print(f"   weak/no signal ({dom}={val:+.2f}) — push firmly, let's retry.")
                continue
            sign = 1 if val > 0 else -1
            mapping[axis] = (dom, sign)
            print(f"   captured -> {axis}: ({dom!r}, {sign:+d})\n")
            break
    sm.close()

    dups = [k for k in RAW if [m[0] for m in mapping.values()].count(k) > 1]
    if dups:
        print(f"WARNING: raw axis {dups} was used for more than one motion — re-run and make each "
              "gesture cleaner (isolate the motion).\n")

    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
    with open(CALIB_PATH, "w") as f:
        json.dump({k: list(v) for k, v in mapping.items()}, f, indent=2)
    print(f"wrote {CALIB_PATH}:")
    print(json.dumps({k: list(v) for k, v in mapping.items()}, indent=2))
    print("\nRelaunch the teleop panel — it will load this automatically.")


if __name__ == "__main__":
    main()
