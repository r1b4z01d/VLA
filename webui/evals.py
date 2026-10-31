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
              "rating", "notes", "operator", "checkpoint_inferred")

# Written by scripts/eval_hw.py on the robot PC and carried through read-only — the UI shows them but
# never edits them, unlike _META_KEYS. `exposure` is here because a rollout can only be compared with
# the dataset it was trained on if you know the lighting it actually ran at.
_RUN_KEYS = ("steps", "duration_s", "exposure", "fps", "n_action_steps", "remote")


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
            **{k: meta.get(k) for k in _RUN_KEYS},
            "notes": meta.get("notes", ""),
            "operator": meta.get("operator", ""),
            "server": meta.get("server"),
            "served": meta.get("served"),
            "checkpoint_inferred": bool(meta.get("checkpoint_inferred")),
            "model_key": model_key(meta),
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


UNASSIGNED = "unassigned"


def model_key(meta: dict) -> str:
    """The group an eval belongs to: `<run>/<step>` from a resolved checkpoint path such as
    train/tigger_1_ft/checkpoints/020000/pretrained_model. A record whose `checkpoint` is only a
    server address (127.0.0.1:8777 — every RD run before 2026-10-16) or the serve_ckpt symlink is
    `unassigned` until someone sets it (per card, or the group's "assign" action)."""
    ck = str(meta.get("checkpoint") or "")
    if "/checkpoints/" in ck:
        run, _, rest = ck.partition("/checkpoints/")
        run = run.split("train/")[-1].strip("/")
        step = rest.split("/")[0]
        return f"{run}/{step}" if run else UNASSIGNED
    m = meta.get("model")
    # a manual eval created from the GUI names the model run directly
    if m and m not in ("smolvla", "act") and ":" not in m and "serve_ckpt" not in m:
        return str(m)
    return UNASSIGNED


def group_summary(repo_root: str) -> list[dict[str, Any]]:
    """Per model_key: runs, rated runs, combined rating (mean of 1-5 stars), success rate when any
    run records successes, span of dates. Sorted newest-activity first, `unassigned` last."""
    groups: dict[str, dict[str, Any]] = {}
    for e in list_evals(repo_root):
        g = groups.setdefault(e["model_key"], {"model_key": e["model_key"], "runs": 0, "rated": 0,
                                               "rating_sum": 0.0, "successes": 0, "episodes": 0,
                                               "first": None, "last": None, "inferred": 0})
        g["runs"] += 1
        if e.get("rating"):
            g["rated"] += 1
            g["rating_sum"] += float(e["rating"])
        if e.get("successes") is not None and e.get("episodes"):
            g["successes"] += int(e["successes"])
            g["episodes"] += int(e["episodes"])
        if e.get("checkpoint_inferred"):
            g["inferred"] += 1
        c = e.get("created") or 0
        g["first"] = c if g["first"] is None else min(g["first"], c)
        g["last"] = c if g["last"] is None else max(g["last"], c)
    out = []
    for g in groups.values():
        g["rating"] = round(g["rating_sum"] / g["rated"], 2) if g["rated"] else None
        g["success_rate"] = round(g["successes"] / g["episodes"], 3) if g["episodes"] else None
        del g["rating_sum"]
        out.append(g)
    out.sort(key=lambda g: (g["model_key"] == UNASSIGNED, -(g["last"] or 0)))
    return out


def bulk_annotate(repo_root: str, names: list[str], fields: dict) -> dict[str, Any]:
    """annotate_eval over many runs; skips names that don't exist rather than failing the batch."""
    done, missing = [], []
    for n in names:
        try:
            annotate_eval(repo_root, n, fields)
            done.append(n)
        except ValueError:
            missing.append(n)
    return {"ok": True, "updated": done, "missing": missing}


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
