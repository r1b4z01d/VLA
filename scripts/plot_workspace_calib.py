"""Render workspace limits (outputs/workspace_calib.json, or the code defaults) as a quick sanity-check
PNG: a top-down x-y plan + a y-z side view showing the robot-body no-go rectangle, the teleop box, the
home pose, the z-floor, and the UR5e reach circle. Uses only cv2/numpy — runs anywhere, no matplotlib.

    .venv/bin/python scripts/plot_workspace_calib.py [--calib outputs/workspace_calib.json] [--out outputs/workspace_calib.png]
"""
import argparse
import json
import os

import cv2
import numpy as np

UR5E_REACH = 0.85  # m, base -> TCP max
S = 250            # pixels per metre
Wp, Hp = 660, 640  # per-panel size
RED, BLUE, GREEN, BROWN, GRAY, DGRAY = (60, 60, 220), (200, 120, 20), (40, 170, 40), (30, 90, 140), (210, 210, 210), (120, 120, 120)


def _grid(img, ox, oy, xr, yr):
    for m in np.arange(np.ceil(xr[0]), xr[1] + 1e-6, 0.25):
        x = int(ox + m * S)
        cv2.line(img, (x, int(oy - yr[1] * S)), (x, int(oy - yr[0] * S)), GRAY, 1)
    for m in np.arange(np.ceil(yr[0]), yr[1] + 1e-6, 0.25):
        y = int(oy - m * S)
        cv2.line(img, (int(ox + xr[0] * S), y), (int(ox + xr[1] * S), y), GRAY, 1)
    cv2.line(img, (ox, int(oy - yr[1] * S)), (ox, int(oy - yr[0] * S)), DGRAY, 1)   # x=0 / y=0 axis
    cv2.line(img, (int(ox + xr[0] * S), oy), (int(ox + xr[1] * S), oy), DGRAY, 1)


def _rect(img, ox, oy, x0, x1, y0, y1, color, thick, alpha=None):
    p0 = (int(ox + x0 * S), int(oy - y1 * S))
    p1 = (int(ox + x1 * S), int(oy - y0 * S))
    if alpha is not None:
        ov = img.copy()
        cv2.rectangle(ov, p0, p1, color, -1)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, p0, p1, color, thick)


def _star(img, ox, oy, x, y, color):
    c = (int(ox + x * S), int(oy - y * S))
    cv2.drawMarker(img, c, color, cv2.MARKER_STAR, 20, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="outputs/workspace_calib.json")
    ap.add_argument("--out", default="outputs/workspace_calib.png")
    args = ap.parse_args()

    calib = {}
    if os.path.isfile(args.calib):
        try:
            with open(args.calib) as f:
                calib = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    from ur5e_lerobot.robot.workspace import NO_GO_XY
    ngo = calib.get("no_go_xy") or list(NO_GO_XY)
    pd = calib.get("pos_deltas")
    home = calib.get("home_tcp")
    zf = calib.get("z_floor")
    src = args.calib if calib else "code defaults (no calib file)"

    img = np.full((Hp, Wp * 2, 3), 255, np.uint8)

    # ---- panel 1: top-down x-y (base at center, y up) ----
    ox1, oy1 = Wp // 2, Hp // 2 + 30
    _grid(img, ox1, oy1, (-1, 1), (-1, 1))
    cv2.circle(img, (ox1, oy1), int(UR5E_REACH * S), DGRAY, 1)               # reach
    cv2.rectangle(img, (ox1 - 6, oy1 - 6), (ox1 + 6, oy1 + 6), (0, 0, 0), -1)  # base
    _rect(img, ox1, oy1, ngo[0], ngo[1], ngo[2], ngo[3], RED, 2, alpha=0.22)
    if home and pd:
        _rect(img, ox1, oy1, home[0] - pd["x"], home[0] + pd["x"], home[1] - pd["y"], home[1] + pd["y"], BLUE, 2)
        _star(img, ox1, oy1, home[0], home[1], GREEN)
    cv2.putText(img, "TOP-DOWN  x-y (base frame)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "+x right, +y up (robot body at +y)", (20, Hp - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DGRAY, 1)

    # ---- panel 2: side y-z (y horizontal, z up; z=0 near bottom) ----
    ox2, oy2 = Wp + Wp // 2, Hp - 130
    _grid(img, ox2, oy2, (-1, 1), (-0.15, 0.85))
    cv2.rectangle(img, (ox2 - 6, oy2 - 6), (ox2 + 6, oy2 + 6), (0, 0, 0), -1)  # base (y=0,z=0)
    if zf is not None:
        y = int(oy2 - zf * S)
        cv2.line(img, (ox2 - int(1 * S), y), (ox2 + int(1 * S), y), BROWN, 3)
    if home and pd:
        _rect(img, ox2, oy2, home[1] - pd["y"], home[1] + pd["y"], home[2] - pd["z"], home[2] + pd["z"], BLUE, 2)
        _star(img, ox2, oy2, home[1], home[2], GREEN)
    cv2.putText(img, "SIDE  y-z (floor + box height)", (Wp + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # ---- legend ----
    leg = [("no-go (robot body)", RED), ("teleop box", BLUE), ("home", GREEN), ("z-floor", BROWN), ("UR reach 0.85m", DGRAY)]
    for i, (lab, col) in enumerate(leg):
        y = 60 + i * 24
        cv2.line(img, (Wp + 20, y), (Wp + 50, y), col, 3)
        cv2.putText(img, lab, (Wp + 58, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    ng = tuple(round(v, 3) for v in ngo)
    sub = f"z_floor={zf}   pos_deltas={pd}   no_go={ng}"
    cv2.putText(img, sub[:120], (20, Hp - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"wrote {args.out}  ({src})")


if __name__ == "__main__":
    main()
