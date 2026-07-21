"""One-shot Xbox-gamepad axis calibration — run AT the robot PC, from your normal operating spot.

For each arm motion it asks you to push the stick / trigger the way you want that motion; it records
which gamepad input (and direction) you used and writes ``outputs/gp_calib.json``. The teleop panel
loads that file when the gamepad opens and overrides its built-in GP_MAP — so you never hand-tune
signs, and stick axes can't end up swapped from your viewpoint. Bumpers LB/RB always drive grasp.

    ~/VLA/run.sh -m ur5e_lerobot.teleop.gp_calibrate
"""
from __future__ import annotations

import json
import os
import time

from .gamepad import Gamepad

# (robot axis, candidate sources, prompt). x/y pick between the LEFT stick's two axes; roll/pitch
# between the RIGHT stick's; z is the trigger pair (rt - lt).
AXES = [
    ("x", ["lx", "ly"], "LEFT stick — push the way you want the gripper to move RIGHT (+X)"),
    ("y", ["lx", "ly"], "LEFT stick — push the way you want it to move AWAY/forward (+Y)"),
    ("roll", ["rx", "ry"], "RIGHT stick — the way you want +ROLL"),
    ("pitch", ["rx", "ry"], "RIGHT stick — the way you want +PITCH (nose up)"),
    ("z", ["triggers"], "squeeze the TRIGGER you want for UP (+Z)"),
]
CALIB_PATH = "outputs/gp_calib.json"


def _val(s, src):
    return (s["rt"] - s["lt"]) if src == "triggers" else s[src]


def _capture(gp, sources, seconds: float = 1.5):
    sums = {k: 0.0 for k in sources}
    n = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        gp.read()
        s = gp.state()
        for src in sources:
            sums[src] += _val(s, src)
        n += 1
        time.sleep(0.02)
    avg = {k: v / max(n, 1) for k, v in sums.items()}
    dom = max(sources, key=lambda k: abs(avg[k]))
    return dom, avg[dom]


def main() -> None:
    gp = Gamepad().open()
    print(f"opened {gp.name!r}\n")
    print("For each prompt: press ENTER, then push/hold that input (~1.5 s). Bumpers LB/RB = grasp.\n")
    mapping: dict[str, list] = {}
    for axis, sources, desc in AXES:
        while True:
            input(f"[{axis}] ENTER, then: {desc} ...")
            dom, val = _capture(gp, sources)
            if abs(val) < 0.2:
                print(f"   weak signal ({dom}={val:+.2f}) — push firmly, retry.")
                continue
            sign = 1 if val > 0 else -1
            mapping[axis] = [dom, sign]
            print(f"   captured -> {axis}: ({dom}, {sign:+d})\n")
            break
    gp.close()

    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
    with open(CALIB_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"wrote {CALIB_PATH}:")
    print(json.dumps(mapping, indent=2))
    print("\nRelaunch the panel (or re-select gamepad in the dropdown) — it loads this automatically.")


if __name__ == "__main__":
    main()
