#!/usr/bin/env python3
"""Give a dataset recorded BEFORE meta/capture.json existed the camera geometry it was recorded with.

    ./run.sh scripts/capture_backfill.py --root "outputs/datasets/OLD - pickup_tigger" \\
        --scene none --side none --wrist none --run outputs/train/pickup_tigger_smolvla_ft \\
        --note "recorded with no square crop: the upright portrait frame fit-and-padded (centered) into 960x540"

Why: the inference server reports the checkpoint's outputs/train/<run>/capture.json and the eval applies
it to the camera service before the first observation, so a model trained on an old dataset is otherwise
evaluated through whatever crop the service happens to have — a shelf model through the floor crop, a
no-crop (pillarboxed) model through a 0.75 square. The shapes and the capture box come from the dataset's
info.json; the squares are what the operator determines (frame comparison), `none` = no crop, i.e. the
fit-and-pad geometry. Refuses to overwrite an existing capture.json without --force. With --run the
sidecar is also copied into each run directory (ship those to the robot afterwards).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ur5e_lerobot.sensors import capture_meta  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the dataset directory (holds meta/info.json)")
    ap.add_argument("--scene", required=True, help="crop fraction along the long axis, or 'none'")
    ap.add_argument("--side", required=True)
    ap.add_argument("--wrist", default="none")
    ap.add_argument("--rotate", default="scene=270,side=90,wrist=0", help="mount rotations at record time")
    ap.add_argument("--host", default="robotdisco", help="the host that recorded it")
    ap.add_argument("--note", default="", help="how the geometry was determined (kept in the file)")
    ap.add_argument("--run", action="append", default=[], help="outputs/train/<run> to copy the sidecar into (repeatable)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing meta/capture.json")
    a = ap.parse_args()

    if capture_meta.read(a.root) is not None and not a.force:
        print(f"{capture_meta.path(a.root)} exists — --force to overwrite", file=sys.stderr)
        return 1
    info = json.load(open(os.path.join(a.root, "meta", "info.json")))
    shapes = {k.split(".")[-1]: tuple(v["shape"][:2]) for k, v in info["features"].items()
              if k.startswith("observation.images.")}
    if not shapes:
        print("no image features in info.json", file=sys.stderr)
        return 1
    box = (max(w for _, w in shapes.values()), max(h for h, _ in shapes.values()))
    squares = {"scene": capture_meta.parse_square(a.scene), "side": capture_meta.parse_square(a.side),
               "wrist": capture_meta.parse_square(a.wrist)}
    squares = {k: v for k, v in squares.items() if k in shapes}
    rotate = {k: int(v) for k, v in (kv.split("=") for kv in a.rotate.split(",")) if k in shapes}
    times = [os.path.getmtime(p) for p in glob.glob(os.path.join(a.root, "data", "**", "*.parquet"), recursive=True)]
    recorded = ({"first": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(min(times))),
                 "last": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(max(times))),
                 "approximate": "parquet mtimes; backfilled"} if times else None)
    note = (a.note + " " if a.note else "") + f"[backfilled {time.strftime('%Y-%m-%d')} by capture_backfill.py]"
    d = capture_meta.write(a.root, squares, rotate=rotate, capture_box=box, shapes=shapes, host=a.host,
                           recorded=recorded, note=note.strip())
    print(f"{capture_meta.path(a.root)}:\n{json.dumps(d, indent=2, sort_keys=True)}")
    for r in a.run:
        if not os.path.isdir(r):
            print(f"  run dir {r} missing — skipped", file=sys.stderr)
            continue
        shutil.copy2(capture_meta.path(a.root), os.path.join(r, "capture.json"))
        print(f"  -> {r}/capture.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
