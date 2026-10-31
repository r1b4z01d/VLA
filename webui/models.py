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
            # the GUI launches smolvla through scripts/train_regrasp.py (which wraps lerobot_train);
            # act goes through lerobot_train directly — either is "a training run"
            if "lerobot.scripts.lerobot_train" not in line and "train_regrasp.py" not in line:
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
    # runs just launched but not yet on disk (LeRobot hasn't written the dir) -> show from the ps table
    listed = {m["name"] for m in out}
    for r in sorted(running - listed):
        lp = os.path.join(td, r + ".log")
        row = {"name": r, "policy": None, "dataset": None, "steps": None, "checkpoints": [],
               "latest_ckpt": 0, "status": "training", "size_mb": 0.0,
               "created": os.path.getmtime(lp) if os.path.isfile(lp) else 0.0}
        row.update(_parse_log(lp if os.path.isfile(lp) else fresh))
        out.append(row)
    # training first, then newest
    out.sort(key=lambda m: (m["status"] != "training", -m["created"]))
    return out


def _regrasp_share(root: str, info: dict) -> float:
    """Fraction of the dataset's frames inside flagged re-grasp ranges (meta/annotations.json,
    written by the teleop panel's deck key 0). 0.0 when nothing is flagged or the sidecar is absent.
    Same arithmetic scripts/train_regrasp.py reports as "[regrasp] N/M frames (p%)"."""
    total = float(info.get("total_frames") or 0)
    p = os.path.join(root, "meta", "annotations.json")
    if total <= 0 or not os.path.isfile(p):
        return 0.0
    try:
        ann = json.load(open(p))
    except Exception:  # noqa: BLE001
        return 0.0
    hot = 0
    for rec in (ann or {}).values():
        for rng in (rec or {}).get("regrasp_ranges") or []:
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                hot += max(0, int(rng[1]) - int(rng[0]))
    return min(1.0, hot / total)


def start_training(repo_root: str, datasets_dir: str, dataset_name: str,
                   policy_type: str = "smolvla", steps: int = 20000,
                   batch_size: int = 64, save_freq: int = 2500, dry_run: bool = False) -> dict[str, Any]:
    """Launch a training run on a dataset, detached, writing to outputs/train/<run>.log (which
    list_models() reads for live progress). One run at a time (GPU memory). Runs on THIS host's GPU.

    smolvla is a FINE-TUNE of lerobot/smolvla_base (what the UI's "SmolVLA (fine-tune pretrained)"
    option promises). Until 2026-10-16 this passed `--policy.type=smolvla`, which builds a fresh
    policy: the SmolVLM2 backbone still loads, but the flow-matching action expert starts from
    random init — measured on `tigger_1_merged_smolvla` as cosine 0.000 against the base expert
    (a real fine-tune is 0.993), and on hardware as a clearly worse policy than the CLI fine-tune of a
    SMALLER dataset. `dry_run=True` returns the command without launching, so the launcher can be
    checked without burning a 5-hour run.
    """
    import sys

    active = _running_runs()
    if active:
        raise ValueError(f"a training run is already active ({', '.join(sorted(active))}); "
                         "stop it before starting another")
    if policy_type not in ("smolvla", "act"):
        raise ValueError(f"policy must be 'smolvla' or 'act', got {policy_type!r}")
    steps, batch_size, save_freq = int(steps), int(batch_size), int(save_freq)
    if steps <= 0 or batch_size <= 0 or save_freq <= 0:
        raise ValueError("steps, batch_size, save_freq must be positive")

    root = os.path.join(datasets_dir, dataset_name)
    info_path = os.path.join(root, "meta", "info.json")
    if not os.path.isfile(info_path):
        raise ValueError(f"{dataset_name}: not a dataset (no meta/info.json)")
    try:
        info = json.load(open(info_path))
    except Exception:  # noqa: BLE001
        raise ValueError(f"{dataset_name}: unreadable meta/info.json")
    if not info.get("total_episodes"):
        raise ValueError(f"{dataset_name}: 0 episodes (unfinalized?) — nothing to train on")
    repo_id = info.get("repo_id") or f"local/{dataset_name}"

    td = _train_dir(repo_root)
    os.makedirs(td, exist_ok=True)
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_name).strip("_") + f"_{policy_type}"
    run, i = base, 1
    while os.path.exists(os.path.join(td, run)) or os.path.isfile(os.path.join(td, run + ".log")):
        run, i = f"{base}_{i}", i + 1
    output_dir = os.path.join("outputs", "train", run)  # relative to cwd=repo_root; ps parses basename

    common = [f"--dataset.repo_id={repo_id}", f"--dataset.root={os.path.abspath(root)}",
              "--policy.device=cuda", "--policy.push_to_hub=false",
              f"--steps={steps}", f"--batch_size={batch_size}",
              # 4, not 8: each dataloader worker holds prefetch batches of raw 540p frames x 3 cameras
              # in /dev/shm; at 8 the host OOM-killer took the whole GPU box down ~90 s into a run
              # (2026-09-24). 4 measured no throughput loss (1.14 step/s either way).
              "--num_workers=4",
              f"--save_freq={save_freq}", "--eval_freq=0", "--wandb.enable=false",
              f"--output_dir={output_dir}"]
    if policy_type == "smolvla":
        # Fine-tune the published base. smolvla_base was trained on camera1/2/3, so the dataset's
        # camera keys are renamed to those — the same --rename_map the CLI fine-tunes use. Keys are
        # read from the dataset so a 2-camera set maps camera1/2 and a 3-camera set camera1/2/3.
        feats = info.get("features", {})
        img_keys = [k for k in ("observation.images.scene", "observation.images.wrist",
                                "observation.images.side") if k in feats]
        img_keys += sorted(k for k in feats if k.startswith("observation.images.") and k not in img_keys)
        rename = {k: f"observation.images.camera{i + 1}" for i, k in enumerate(img_keys)}
        # scripts/train_regrasp.py wraps lerobot_train and oversamples the re-grasp segments the
        # teleop panel flagged (meta/annotations.json regrasp_ranges). The weight is chosen from the
        # dataset's actual recovery share, because a fixed weight is wrong at both ends: 3 lifts a 6 %
        # share to a useful 16 %, but tigger_1_merged is already 22.9 % recovery and 3 would push that
        # to 47 % — a policy that mostly practises retrying. Aim for roughly a fifth to a third.
        share = _regrasp_share(root, info)
        weight = 3 if share < 0.10 else 2 if share < 0.20 else 1   # 1 = report only, no oversampling
        cmd = [sys.executable, os.path.join("scripts", "train_regrasp.py"), "--regrasp-weight", str(weight),
               "--policy.path=lerobot/smolvla_base", f"--rename_map={json.dumps(rename)}"] + common
        extra = {"regrasp_share": round(share, 3), "regrasp_weight": weight}
    else:
        extra = {}
        cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train", f"--policy.type={policy_type}"] + common

    # Memory cap: a transient systemd user unit with MemoryMax, so a dataloader blow-up kills THIS run
    # and not the login session / web UI / SSH (the 2026-09-24 reboot). systemd-run --user needs the
    # user's session bus, which a supervisor-spawned service may not have — fall back to a plain
    # detached Popen rather than refuse to train.
    unit = "vla-train-" + re.sub(r"[^A-Za-z0-9-]+", "-", run)[:40]
    launcher = ["systemd-run", "--user", f"--unit={unit}", "--collect",
                "-p", "MemoryMax=40G", "-p", "MemorySwapMax=4G", "-p", "OOMPolicy=stop",
                "-p", f"WorkingDirectory={repo_root}", "--"]
    result = {"ok": True, "run": run, "policy": policy_type, "steps": steps,
              "batch_size": batch_size, "save_freq": save_freq, "dataset": repo_id,
              "command": " ".join(cmd), "launcher": "systemd-run", **extra}
    if dry_run:
        result["dry_run"] = True
        return result

    log_path = os.path.join(td, run + ".log")
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_root
    env.pop("HF_HUB_OFFLINE", None)  # allow the pretrained backbone to download; local dataset uses --root
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation / OOM
    logf = open(log_path, "w")
    logf.write("$ " + " ".join(cmd) + "\n\n")
    logf.flush()
    # systemd-run detaches on its own and the unit inherits no stdio, so the log must be redirected
    # inside the unit: wrap the command in a shell that appends to the log file.
    shell = " ".join(subprocess.list2cmdline([c]) for c in cmd) + f" >> {subprocess.list2cmdline([log_path])} 2>&1"
    try:
        r = subprocess.run(launcher + ["bash", "-lc", shell], cwd=repo_root, env=env,
                           capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout).strip()[:200])
    except Exception as e:  # noqa: BLE001 — no user bus / no systemd: run it detached without the cap
        logf.write(f"[launcher] systemd-run unavailable ({e}); running uncapped via Popen\n")
        logf.flush()
        subprocess.Popen(cmd, cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT,
                         env=env, start_new_session=True)  # detached: survives the server process
        result["launcher"] = "popen (no memory cap)"
    return result


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
