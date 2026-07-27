"""Model registry — trained policies under outputs/train/<run>/.

Each run has checkpoints/<step>/pretrained_model/{model.safetensors, config.json, train_config.json}
and checkpoints/last. train_config.json carries policy.type, steps, and the dataset. The actively
training run is detected from the process table (a running lerobot_train with a matching output_dir);
its live step/loss/ETA are parsed from the run's stdout log in outputs/train/*.log.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
from typing import Any


def _train_dir(repo_root: str) -> str:
    return os.path.join(repo_root, "outputs", "train")


def _running_runs() -> set[str]:
    """Basenames of output_dirs currently being trained (from the process table)."""
    out: set[str] = set()
    try:
        r = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            if "lerobot.scripts.lerobot_train" not in line:
                continue
            m = re.search(r"--output_dir[= ]([^\s]+)", line)
            if m:
                out.add(os.path.basename(m.group(1).rstrip("/")))
    except Exception:  # noqa: BLE001
        pass
    return out


def _train_config(run: str) -> dict:
    cands = [os.path.join(run, "checkpoints", "last", "pretrained_model", "train_config.json")]
    cands += sorted(glob.glob(os.path.join(run, "checkpoints", "*", "pretrained_model", "train_config.json")))
    for p in cands:
        if os.path.isfile(p):
            try:
                return json.load(open(p))
            except Exception:  # noqa: BLE001
                pass
    return {}


def _checkpoints(run: str) -> list[str]:
    d = os.path.join(run, "checkpoints")
    if not os.path.isdir(d):
        return []
    return sorted(x for x in os.listdir(d) if x.isdigit())


def _du_mb(path: str) -> float:
    try:
        r = subprocess.run(["du", "-sm", path], capture_output=True, text=True, timeout=20)
        return float(r.stdout.split()[0])
    except Exception:  # noqa: BLE001
        return 0.0


def _freshest_log(train_dir: str) -> str | None:
    logs = glob.glob(os.path.join(train_dir, "*.log"))
    return max(logs, key=os.path.getmtime) if logs else None


def _parse_log(log_path: str | None) -> dict:
    """Best-effort live stats from a training stdout log: step/total/loss/eta/rate."""
    if not log_path or not os.path.isfile(log_path):
        return {}
    try:
        size = os.path.getsize(log_path)
        with open(log_path, "rb") as f:
            f.seek(max(0, size - 200_000))
            tail = f.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return {}
    info: dict[str, Any] = {}
    losses = re.findall(r"loss:([0-9.]+)", tail)
    if losses:
        info["loss"] = float(losses[-1])
    prog = re.findall(r"(\d+)/(\d+)\s+\[([0-9:]+)<([0-9:,]+),\s+([0-9.]+)step/s\]", tail.replace("\r", "\n"))
    if prog:
        s, t, el, eta, rate = prog[-1]
        info.update(step=int(s), total=int(t), elapsed=el, eta=eta.rstrip(","), rate=float(rate))
    return info


def list_models(repo_root: str) -> list[dict[str, Any]]:
    td = _train_dir(repo_root)
    if not os.path.isdir(td):
        return []
    running = _running_runs()
    fresh = _freshest_log(td)
    out = []
    for name in sorted(os.listdir(td)):
        run = os.path.join(td, name)
        if not os.path.isdir(run):
            continue
        ckpts = _checkpoints(run)
        cfg = _train_config(run)
        if not ckpts and not cfg:
            continue  # not a training run
        steps = cfg.get("steps")
        is_running = name in running
        done = bool(steps) and any(int(c) >= int(steps) for c in ckpts)
        status = "training" if is_running else ("done" if done else "stopped")
        row: dict[str, Any] = {
            "name": name,
            "policy": (cfg.get("policy") or {}).get("type"),
            "dataset": (cfg.get("dataset") or {}).get("repo_id"),
            "steps": steps,
            "checkpoints": ckpts,
            "latest_ckpt": int(ckpts[-1]) if ckpts else 0,
            "status": status,
            "created": os.path.getmtime(run),
            "size_mb": _du_mb(run),
        }
        if is_running:  # live progress (per-run log by convention, else the freshest .log)
            lp = os.path.join(td, name + ".log")
            row.update(_parse_log(lp if os.path.isfile(lp) else fresh))
        out.append(row)
    # training first, then newest
    out.sort(key=lambda m: (m["status"] != "training", -m["created"]))
    return out


# --------------------------------------------------------------------------- ops
def _guard_name(name: str) -> None:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"bad model name: {name!r}")


def rename_model(repo_root: str, name: str, new_name: str) -> dict[str, Any]:
    _guard_name(new_name)
    if name in _running_runs():
        raise ValueError(f"{name} is currently training — stop it before renaming")
    td = _train_dir(repo_root)
    src, dst = os.path.join(td, name), os.path.join(td, new_name)
    if not os.path.isdir(src):
        raise ValueError(f"{name} not found")
    if os.path.exists(dst):
        raise ValueError(f"{new_name} already exists")
    os.rename(src, dst)
    for suf in (".log",):  # carry the stdout log if it follows the convention
        if os.path.isfile(os.path.join(td, name + suf)):
            os.rename(os.path.join(td, name + suf), os.path.join(td, new_name + suf))
    return {"ok": True, "old": name, "new": new_name}


def delete_model(repo_root: str, name: str) -> dict[str, Any]:
    if name in _running_runs():
        raise ValueError(f"{name} is currently training — stop it before deleting")
    td = _train_dir(repo_root)
    run = os.path.join(td, name)
    if not os.path.isdir(run):
        raise ValueError(f"{name} not found")
    shutil.rmtree(run, ignore_errors=True)
    log = os.path.join(td, name + ".log")
    if os.path.isfile(log):
        os.remove(log)
    return {"ok": True, "deleted": name}


def checkpoint_dir(repo_root: str, name: str, step: str | None = None) -> str:
    """Path to a checkpoint's pretrained_model dir (the deployable artifact). Default: latest."""
    run = os.path.join(_train_dir(repo_root), name)
    ckpts = _checkpoints(run)
    if not ckpts:
        raise ValueError(f"{name} has no checkpoints")
    step = step if (step and step in ckpts) else ckpts[-1]
    return os.path.join(run, "checkpoints", step, "pretrained_model")
