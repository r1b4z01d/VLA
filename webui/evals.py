"""Eval registry — eval runs under outputs/evals/<name>/.

Each run is a folder with an eval.json (model/checkpoint, task, episodes, successes, success_rate,
rating, notes, operator, source, created) plus optional artifacts (a video .mp4, a per-step .csv).
Auto runs are written GPU-side by scripts/infer_server.py while it serves a bridge eval; manual runs
are created from the GUI. The operator fills in rating/notes/success from the UI either way.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import time
from typing import Any

_META_KEYS = ("model", "checkpoint", "task", "episodes", "successes", "success_rate",
              "rating", "notes", "operator")


def _dir(repo_root: str) -> str:
    return os.path.join(repo_root, "outputs", "evals")


def _guard(name: str) -> None:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"bad eval name: {name!r}")


def _read(run: str) -> dict:
    p = os.path.join(run, "eval.json")
    if os.path.isfile(p):
        try:
            return json.load(open(p))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _write(run: str, data: dict) -> None:
    os.makedirs(run, exist_ok=True)
    tmp = os.path.join(run, "eval.json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, os.path.join(run, "eval.json"))


def _artifacts(run: str) -> dict:
    return {
        "videos": sorted(os.path.basename(x) for x in glob.glob(os.path.join(run, "*.mp4"))),
        "csvs": sorted(os.path.basename(x) for x in glob.glob(os.path.join(run, "*.csv"))),
    }


def list_evals(repo_root: str) -> list[dict[str, Any]]:
    ed = _dir(repo_root)
    if not os.path.isdir(ed):
        return []
    out = []
    for name in sorted(os.listdir(ed)):
        run = os.path.join(ed, name)
        if not os.path.isdir(run):
            continue
        meta = _read(run)
        out.append({
            "name": name,
            "created": meta.get("created") or os.path.getmtime(run),
            "source": meta.get("source", "manual"),
            **{k: meta.get(k) for k in _META_KEYS},
            "notes": meta.get("notes", ""),
            "operator": meta.get("operator", ""),
            **_artifacts(run),
        })
    out.sort(key=lambda e: -(e["created"] or 0))
    return out


def create_eval(repo_root: str, name: str, fields: dict) -> dict[str, Any]:
    _guard(name)
    run = os.path.join(_dir(repo_root), name)
    if os.path.exists(run):
        raise ValueError(f"{name} already exists")
    meta = {"created": time.time(), "source": "manual"}
    for k in _META_KEYS:
        if fields.get(k) not in (None, ""):
            meta[k] = fields[k]
    _write(run, meta)
    return {"ok": True, "name": name}


def annotate_eval(repo_root: str, name: str, fields: dict) -> dict[str, Any]:
    run = os.path.join(_dir(repo_root), name)
    if not os.path.isdir(run):
        raise ValueError(f"{name} not found")
    meta = _read(run)
    for k in _META_KEYS:
        if k in fields:
            v = fields[k]
            if v in (None, ""):
                meta.pop(k, None)
            else:
                meta[k] = v
    _write(run, meta)
    return {"ok": True, "name": name, "annotation": {k: meta.get(k) for k in _META_KEYS}}


def rename_eval(repo_root: str, name: str, new_name: str) -> dict[str, Any]:
    _guard(new_name)
    ed = _dir(repo_root)
    src, dst = os.path.join(ed, name), os.path.join(ed, new_name)
    if not os.path.isdir(src):
        raise ValueError(f"{name} not found")
    if os.path.exists(dst):
        raise ValueError(f"{new_name} already exists")
    os.rename(src, dst)
    return {"ok": True, "old": name, "new": new_name}


def delete_eval(repo_root: str, name: str) -> dict[str, Any]:
    run = os.path.join(_dir(repo_root), name)
    if not os.path.isdir(run):
        raise ValueError(f"{name} not found")
    shutil.rmtree(run, ignore_errors=True)
    return {"ok": True, "deleted": name}


def eval_dir(repo_root: str, name: str) -> str:
    _guard(name)
    run = os.path.join(_dir(repo_root), name)
    if not os.path.isdir(run):
        raise ValueError(f"{name} not found")
    return run
