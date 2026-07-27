"""Fast, read-only dataset + episode listing for the web UI.

Reads only parquet/JSON metadata — never decodes frames — so listing a dataset is instant even
for thousands of frames. Episode boundaries come straight from meta/episodes/**/*.parquet
(episode_index, tasks, length, dataset_from_index/to_index), which is authoritative in v3.0;
this avoids the O(num_frames) __getitem__ scan that sim/playback.py falls back to.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any

import pyarrow.parquet as pq

from . import annotations as anno


def _size_mb(root: str) -> float:
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return round(total / 1e6, 1)


def is_dataset(path: str) -> bool:
    """A LeRobot dataset root has meta/info.json."""
    return os.path.isfile(os.path.join(path, "meta", "info.json"))


def _read_info(root: str) -> dict:
    with open(os.path.join(root, "meta", "info.json")) as f:
        return json.load(f)


def _tasks(root: str) -> list[str]:
    """Task strings from meta/tasks.parquet (the string is the parquet index)."""
    tpath = os.path.join(root, "meta", "tasks.parquet")
    if not os.path.isfile(tpath):
        return []
    df = pq.read_table(tpath).to_pandas()
    # task_index is a column; the task string is the (named) index -> list in task_index order
    return [str(t) for t in df.sort_values("task_index").index.tolist()]


def summarize(root: str, name: str) -> dict[str, Any]:
    """One dataset's summary card (no frame decode)."""
    info = _read_info(root)
    feats = info.get("features", {})
    cams = sorted(k.split(".")[-1] for k in feats if k.startswith("observation.images."))
    is_video = bool(info.get("video_path")) or any(
        feats.get(k, {}).get("dtype") == "video" for k in feats if k.startswith("observation.images.")
    )
    return {
        "name": name,
        "episodes": info.get("total_episodes", 0),
        "frames": info.get("total_frames", 0),
        "tasks": _tasks(root),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "cameras": cams,
        "media": "video" if is_video else "image",
        "size_mb": _size_mb(root),
        "has_backup": os.path.isdir(root + ".bak"),
    }


def list_datasets(datasets_dir: str) -> list[dict[str, Any]]:
    """Every LeRobot dataset under datasets_dir (skips *.bak and *.tmp working copies)."""
    out = []
    for entry in sorted(os.listdir(datasets_dir)):
        if entry.endswith((".bak", ".tmp")):
            continue
        root = os.path.join(datasets_dir, entry)
        if os.path.isdir(root) and is_dataset(root):
            try:
                out.append(summarize(root, entry))
            except Exception as e:  # noqa: BLE001 — a half-written dataset shouldn't break the list
                out.append({"name": entry, "error": f"{type(e).__name__}: {e}"})
    return out


def _episodes_df(root: str):
    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "chunk-*", "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"{root}: no meta/episodes parquet")
    import pandas as pd

    frames = [pq.read_table(f).to_pandas() for f in files]
    df = pd.concat(frames, ignore_index=True).sort_values("episode_index")
    return df


def list_episodes(datasets_dir: str, name: str) -> list[dict[str, Any]]:
    """Per-episode rows: index, task(s), length, frame range, merged with the annotations sidecar."""
    root = os.path.join(datasets_dir, name)
    if not is_dataset(root):
        raise FileNotFoundError(f"{name}: not a dataset")
    df = _episodes_df(root)
    notes = anno.load(root)
    fps = _read_info(root).get("fps") or 1
    rows = []
    for _, r in df.iterrows():
        ep = int(r["episode_index"])
        tasks = r.get("tasks")
        task = tasks[0] if hasattr(tasks, "__len__") and len(tasks) else ""
        length = int(r["length"])
        a = notes.get(str(ep), {})
        rows.append({
            "episode": ep,
            "task": str(task),
            "length": length,
            "duration_s": round(length / fps, 1),
            "from_index": int(r.get("dataset_from_index", 0)),
            "to_index": int(r.get("dataset_to_index", 0)),
            "rating": a.get("rating"),
            "notes": a.get("notes", ""),
            "operator": a.get("operator", ""),
        })
    return rows
