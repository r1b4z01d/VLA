"""Sweep camera exposure and report clipping, so exposure can be picked by number instead of by eye.

Why: auto-exposure meters over the WHOLE sensor frame, which here is dominated by white cabinets,
shelving and floor — and the square crop discards most of that. Measured on captured frames, 4-6% of
content pixels sit at >=250 with p99 = 255: the highlights are clipped and that detail is gone before
the policy ever sees it. Locking manual exposure exposes for the workspace instead of the room.

    ~/rd_ws/VLA/run.sh scripts/tune_exposure.py                      # all three cameras
    ~/rd_ws/VLA/run.sh scripts/tune_exposure.py --cams scene side    # just these
    ~/rd_ws/VLA/run.sh scripts/tune_exposure.py --values 20 39 78 156 --out /tmp/expo

Reports per exposure value, measured on the frame AS RECORDED (rotation + square crop applied, black
bars excluded so the letterbox can't skew the stats):
  clip%  - share of pixels at >=250, i.e. blown highlights. This is the number to minimise.
  mean   - overall brightness; too low and you have traded clipping for a dark, noisy image.
  p1/p99 - the tails, showing whether you are losing the shadow end instead.

UVC exposure support is inconsistent and `UsbCamera` already warns it is "flaky on some cams", so this
VERIFIES the control actually does something: if the frames do not change across the sweep, the camera
is ignoring the setting and it says so rather than leaving you to tune a no-op.

Writes a PNG per value so you can confirm by eye, then set the winners in cameras.py:
    SCENE_AUTO_EXPOSURE, SCENE_EXPOSURE = 1.0, <value>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5e_lerobot.sensors.cameras import (  # noqa: E402
    SCENE_CAM, SCENE_GAIN, SCENE_ROTATE, SCENE_SQUARE,
    SIDE_CAM, SIDE_GAIN, SIDE_ROTATE, SIDE_SQUARE,
    WRIST_CAM, WRIST_GAIN, WRIST_ROTATE, WRIST_SQUARE,
    UsbCamera,
)

CAMS = {
    "scene": (SCENE_CAM, SCENE_ROTATE, SCENE_GAIN, SCENE_SQUARE),
    "wrist": (WRIST_CAM, WRIST_ROTATE, WRIST_GAIN, WRIST_SQUARE),
    "side": (SIDE_CAM, SIDE_ROTATE, SIDE_GAIN, SIDE_SQUARE),
}
# 100us units. Roughly log-spaced across the range indoor UVC cams usually accept.
DEFAULT_VALUES = [5, 10, 20, 39, 78, 156, 312, 625]


def stats(img: np.ndarray) -> dict:
    """Brightness/clipping over the CONTENT pixels only (letterbox bars would skew the mean)."""
    nz = img.max(axis=2) > 12
    px = img[nz] if nz.any() else img.reshape(-1, 3)
    return {"mean": float(px.mean()),
            "clip": 100.0 * float((px.max(axis=-1) >= 250).mean()),
            "p1": float(np.percentile(px, 1)),
            "p99": float(np.percentile(px, 99)),
            "shape": (img.shape[1], img.shape[0])}


def measure(cam: UsbCamera, w: int, h: int, settle: int) -> "tuple[np.ndarray, dict]":
    for _ in range(settle):        # let auto-exposure/the new setting take effect before measuring
        cam.read(w, h)
    img = cam.read(w, h)
    return img, stats(img)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep camera exposure and report clipping.")
    ap.add_argument("--cams", nargs="*", default=list(CAMS), choices=list(CAMS))
    ap.add_argument("--values", nargs="*", type=int, default=DEFAULT_VALUES,
                    help="manual exposure values to try, in 100us units")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--settle", type=int, default=8, help="frames to discard after each change")
    ap.add_argument("--out", default="outputs/exposure", help="where sample PNGs are written")
    ap.add_argument("--max-clip", type=float, default=0.5,
                    help="clip%% considered acceptable when recommending a value")
    args = ap.parse_args()

    import cv2

    os.makedirs(args.out, exist_ok=True)
    for name in args.cams:
        dev, rot, gain, square = CAMS[name]
        print(f"\n=== {name}  (rotate={rot}, square={square}, gain={gain}) ===", flush=True)
        try:
            cam = UsbCamera(dev, rotate=rot, gain=gain, square=square).connect()
        except Exception as e:  # noqa: BLE001
            print(f"  could not open: {e}")
            continue
        try:
            img, base = measure(cam, args.width, args.height, args.settle)
            print(f"  {'AUTO':>8}  clip={base['clip']:5.1f}%  mean={base['mean']:6.1f}  "
                  f"p1={base['p1']:5.0f}  p99={base['p99']:5.0f}   <- current behaviour "
                  f"({base['shape'][0]}x{base['shape'][1]})")
            cv2.imwrite(os.path.join(args.out, f"{name}_auto.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            cam._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)   # manual
            rows, frames = [], []
            for v in args.values:
                cam._cap.set(cv2.CAP_PROP_EXPOSURE, float(v))
                readback = cam._cap.get(cv2.CAP_PROP_EXPOSURE)
                img, s = measure(cam, args.width, args.height, args.settle)
                rows.append((v, readback, s))
                frames.append(img)
                cv2.imwrite(os.path.join(args.out, f"{name}_exp{v}.png"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"  {v:>8}  clip={s['clip']:5.1f}%  mean={s['mean']:6.1f}  "
                      f"p1={s['p1']:5.0f}  p99={s['p99']:5.0f}   (driver reports {readback:g})")

            # Did the control do anything? If every frame is ~identical the camera ignored it.
            spread = max(r[2]["mean"] for r in rows) - min(r[2]["mean"] for r in rows)
            if spread < 2.0:
                print(f"  !! mean varied by only {spread:.1f} across the whole sweep — this camera is "
                      f"IGNORING manual exposure. Leave it on AUTO and fix the lighting instead.")
                continue

            ok = [r for r in rows if r[2]["clip"] <= args.max_clip]
            if ok:
                best = max(ok, key=lambda r: r[2]["mean"])   # brightest that still doesn't clip
                print(f"  -> suggested: {name.upper()}_AUTO_EXPOSURE, {name.upper()}_EXPOSURE = "
                      f"1.0, {best[0]}   (clip={best[2]['clip']:.1f}%, mean={best[2]['mean']:.1f})")
            else:
                low = min(rows, key=lambda r: r[2]["clip"])
                print(f"  -> nothing got under {args.max_clip}% clipping; lowest was {low[0]} at "
                      f"{low[2]['clip']:.1f}%. Try smaller --values, or reduce the lighting.")
        finally:
            cam.disconnect()
    print(f"\nsample frames in {args.out}/ — compare by eye before committing a value", flush=True)


if __name__ == "__main__":
    main()
