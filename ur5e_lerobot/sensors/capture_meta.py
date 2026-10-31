"""meta/capture.json — the camera GEOMETRY a dataset was recorded with, carried with the data.

A square crop (cameras.SCENE_SQUARE / SIDE_SQUARE) is part of the observation geometry: every stored
frame is already cropped, and a checkpoint expects the framing it trained on. Until this file existed
that value lived only in cameras.py on the robot PC, so switching tasks (shelf -> floor) meant editing
a constant by hand and remembering to put it back before evaluating the old model — and nothing
recorded which value a dataset had actually used. Now:

  * the teleop panel writes this when it CREATES a dataset (and re-applies it when it RESUMES one),
  * the web UI shows it on the dataset card,
  * training copies it into outputs/train/<run>/capture.json,
  * the inference server reports it with its checkpoint ({'cmd':'info'}),
  * the eval applies it to the camera service before the first observation.

The crop can be moved live through the camera service (ControlServer set_square), which is what the
apply() path uses with --cameras udp; with local cameras it sets UsbCamera.square directly.

Schema (version 1):
  squares      {scene: 0.65, side: 0.60, wrist: null}   fraction along the long axis, null = no crop
  rotate       {scene: 270, wrist: 0, side: 90}          cameras.py mount rotations at record time
  capture_box  [960, 540]                                  the box frames are fit into (dataset size)
  shapes       {scene: [540, 540], ...}                    per-camera stored (h, w)
  recorded     {first: iso, last: iso}                     wall-clock span of recording
  host         hostname that recorded it
"""
from __future__ import annotations

import json
import os
import socket
import time

FILE = ("meta", "capture.json")
CAMERAS = ("scene", "wrist", "side")


def path(root: str) -> str:
    return os.path.join(root, *FILE)


def read(root: str) -> "dict | None":
    p = path(root)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _save(root: str, d: dict) -> dict:
    p = path(root)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2, sort_keys=True)
    os.replace(tmp, p)
    return d


def write(root: str, squares: dict, rotate: "dict | None" = None, capture_box=None,
          shapes: "dict | None" = None, host: "str | None" = None, recorded: "dict | None" = None,
          note: "str | None" = None) -> dict:
    """Create or update the sidecar. Keeps `recorded.first` from an existing file; bumps `last`.

    `host`, `recorded` and `note` are for a BACKFILL (scripts/capture_backfill.py): a dataset recorded
    before this file existed gets the geometry it was recorded with, written after the fact from
    another machine — so the recording host and span are passed in rather than taken from here."""
    d = read(root) or {}
    d["version"] = 1
    d["squares"] = {k: (None if v is None else float(v)) for k, v in squares.items()}
    if rotate is not None:
        d["rotate"] = {k: int(v) for k, v in rotate.items()}
    if capture_box is not None:
        d["capture_box"] = [int(capture_box[0]), int(capture_box[1])]
    if shapes is not None:
        d["shapes"] = {k: [int(v[0]), int(v[1])] for k, v in shapes.items()}
    d["host"] = host or socket.gethostname()
    if recorded is not None:
        d["recorded"] = dict(recorded)
    rec = d.setdefault("recorded", {})
    rec.setdefault("first", _now())
    if recorded is None:
        rec["last"] = _now()
    else:
        rec.setdefault("last", rec["first"])
    if note:
        d["note"] = note
    return _save(root, d)


def touch(root: str) -> None:
    """Advance `recorded.last` (called after each saved episode). No-op without a sidecar."""
    d = read(root)
    if d is not None:
        d.setdefault("recorded", {}).setdefault("first", _now())
        d["recorded"]["last"] = _now()
        _save(root, d)


def parse_square(s) -> "float | None":
    """CLI value -> crop fraction. 'none'/'off'/'' -> None (no crop)."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return min(max(float(s), 0.0), 1.0)
    t = str(s).strip().lower()
    if t in ("none", "off", "null", ""):
        return None
    return min(max(float(t), 0.0), 1.0)


def squares_of(cams: dict) -> dict:
    """{name: current square} for live camera objects — UsbCamera (.square attr) or CameraProxy
    (.square property, which asks the service)."""
    out = {}
    for name, c in cams.items():
        if c is None:
            continue
        try:
            out[name] = getattr(c, "square", None)
        except Exception:  # noqa: BLE001 — a proxy whose service is gone; report unknown
            out[name] = None
    return out


def apply(cams: dict, squares: dict) -> dict:
    """Move the live cameras' crops to `squares`. Proxies go through the service (set_square);
    local UsbCameras take the attribute (read() consults it on every grab). Returns squares_of()."""
    for name, at in squares.items():
        c = cams.get(name)
        if c is None:
            continue
        if hasattr(c, "set_square"):
            c.set_square(at)
        else:
            c.square = at
    return squares_of(cams)


def differs(a: "dict | None", b: "dict | None", tol: float = 1e-6) -> "list[str]":
    """Camera names whose crop differs between two squares dicts (None vs number counts)."""
    a, b = a or {}, b or {}
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if (x is None) != (y is None) or (x is not None and abs(float(x) - float(y)) > tol):
            out.append(k)
    return out


def describe(d: "dict | None") -> str:
    """One line for logs / UI: 'scene=0.65 side=0.60 wrist=none · recorded 2026-10-24 .. 2026-10-25'."""
    if not d:
        return "(no capture.json)"
    sq = d.get("squares") or {}
    parts = [f"{k}={'none' if sq.get(k) is None else f'{sq[k]:.2f}'}" for k in CAMERAS if k in sq]
    rec = d.get("recorded") or {}
    span = ""
    if rec.get("first"):
        f, l = rec["first"][:10], (rec.get("last") or rec["first"])[:10]
        span = f" · recorded {f}" + (f" .. {l}" if l != f else "")
    return " ".join(parts) + span
