"""Per-episode annotations (rating / notes / operator), stored as a sidecar OUTSIDE the LeRobot
schema so they never touch the dataset format or break training loads.

Layout: <dataset_root>/meta/annotations.json  ==  {"<episode_index>": {rating, notes, operator}}
Episode indices are the keys; a rebuild op (delete/trim/merge/move) renumbers episodes, so callers
must remap() the sidecar through the old->new index mapping right after the rebuild.
"""
from __future__ import annotations

import json
import os
from typing import Any

_FILE = ("meta", "annotations.json")

# keys the UI may set on an episode; anything else is dropped on write
FIELDS = ("rating", "notes", "operator")


def _path(dataset_root: str) -> str:
    return os.path.join(dataset_root, *_FILE)


def load(dataset_root: str) -> dict[str, dict[str, Any]]:
    p = _path(dataset_root)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save(dataset_root: str, data: dict[str, dict[str, Any]]) -> None:
    p = _path(dataset_root)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, p)  # atomic on POSIX


def set_episode(dataset_root: str, episode: int, fields: dict[str, Any]) -> dict[str, Any]:
    """Merge fields (rating/notes/operator) into one episode's annotation; returns the new record."""
    data = load(dataset_root)
    rec = data.get(str(episode), {})
    for k in FIELDS:
        if k in fields:
            v = fields[k]
            if v is None or v == "":
                rec.pop(k, None)
            else:
                rec[k] = v
    if rec:
        data[str(episode)] = rec
    else:
        data.pop(str(episode), None)
    save(dataset_root, data)
    return rec


def remap(src_root: str, dst_root: str, mapping: dict[int, int]) -> None:
    """Carry annotations from src_root into dst_root under a {old_ep: new_ep} mapping.

    Used after a rebuild: episodes that survived move to their new indices; dropped episodes
    (absent from `mapping`) are discarded. Writes dst even if empty so stale files don't linger.
    """
    old = load(src_root)
    new: dict[str, dict[str, Any]] = {}
    for old_ep, rec in old.items():
        try:
            oe = int(old_ep)
        except ValueError:
            continue
        if oe in mapping:
            new[str(mapping[oe])] = rec
    save(dst_root, new)
