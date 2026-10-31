"""Supervisor-level preflight for the robot apps on RD, and model switching for the inference server.

The robot PC runs its apps as supervisor units (services/rd/): vla_teleop (the recording panel), vla_eval
(the policy eval), vla_infer_server (the SmolVLA server on :8777) and the robot_disco ROS stack. Some of
them are mutually exclusive: the panel and the eval both own the SpaceMouse, the Stream Deck, port 8090
and the UR control script, and the ROS UR driver loads its own external-control program onto the same
controller. An eval that starts while the panel is up therefore fails in confusing ways (port busy,
"control script not running", a dead deck). This module makes a launch check and fix that: stop what
must not run, make sure the inference server serves the model we want, and only then let the app start.
It is a no-op where there is no supervisor (the Mac, the sim), so the apps keep working unchanged there.

    python -m ur5e_lerobot.services status
    python -m ur5e_lerobot.services eval-preflight [--remote 127.0.0.1:8777] [--no-auto-infer]
    python -m ur5e_lerobot.services teleop-preflight
    python -m ur5e_lerobot.services ensure-stopped vla_eval [vla_teleop ...]
    python -m ur5e_lerobot.services serve tigger_bed_1_smolvla[@020000]     # switch the served model
    python -m ur5e_lerobot.services info

Everything goes through `sudo -n supervisorctl` (rd has passwordless sudo on RD). Run from the VLA root
(outputs/serve_ckpt and outputs/train are relative), i.e. through ./run.sh.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from typing import Callable, Iterable

INFER_UNIT = "vla_infer_server"
INFER_HOST, INFER_PORT = "127.0.0.1", 8777
SERVE_LINK = os.path.join("outputs", "serve_ckpt")
TRAIN_DIR = os.path.join("outputs", "train")
# Units that own the SpaceMouse / Stream Deck / UR control script / port 8090 and so must not run
# alongside the eval (resp. the panel).
EXCLUSIVE_WITH_EVAL = ("vla_teleop", "robot_disco_ur5e_driver")
EXCLUSIVE_WITH_TELEOP = ("vla_eval", "robot_disco_ur5e_driver")
KNOWN_UNITS = ("vla_teleop", "vla_eval", "vla_infer_server", "vla_cameras", "robot_disco_ur5e_driver")
ACTIVE = {"RUNNING", "STARTING", "BACKOFF"}

Log = Callable[[str], None]


def _log(msg: str) -> None:
    print(f"[services] {msg}", flush=True)


def available() -> bool:
    """True where supervisor exists (RD). On the Mac / in the sim every check degrades to a no-op."""
    return shutil.which("supervisorctl") is not None


def _ctl(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(["sudo", "-n", "supervisorctl", *args], capture_output=True, text=True,
                          timeout=timeout)


def unit_status(name: str) -> str | None:
    """RUNNING / STOPPED / EXITED / FATAL / STARTING / BACKOFF…, or None if supervisor does not know it."""
    if not available():
        return None
    r = _ctl("status", name, timeout=20)
    for line in ((r.stdout or "") + (r.stderr or "")).splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == name:   # "name: ERROR (no such process)" has "name:" -> None
            return parts[1]
    return None


def statuses(names: Iterable[str] = KNOWN_UNITS) -> dict[str, str | None]:
    return {n: unit_status(n) for n in names}


def ensure_stopped(names: Iterable[str], timeout: float = 120.0, log: Log = _log) -> list[str]:
    """Stop every unit in `names` that is active and wait until it is down. Returns the ones stopped.

    `supervisorctl stop` is the unit's clean shutdown (the panel finalizes its dataset and powers the
    UR down, the eval closes and uploads its run), so this can take up to the unit's stopwaitsecs."""
    if not available():
        log("no supervisorctl here — nothing to stop")
        return []
    stopped: list[str] = []
    for n in names:
        s = unit_status(n)
        if s in ACTIVE:
            log(f"{n} is {s} — stopping it (it owns the UR control script / deck / page port)")
            r = _ctl("stop", n, timeout=timeout + 30)
            log(f"  {(r.stdout or r.stderr).strip()}")
            stopped.append(n)
    deadline = time.time() + timeout
    for n in stopped:
        while unit_status(n) in ACTIVE:
            if time.time() > deadline:
                raise RuntimeError(f"{n} did not stop within {timeout:.0f} s")
            time.sleep(1.0)
    return stopped


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_port(host: str, port: int, timeout: float, want: bool = True) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open(host, port) == want:
            return True
        time.sleep(1.0)
    return port_open(host, port) == want


def is_local(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1", "")


def infer_info(host: str = INFER_HOST, port: int = INFER_PORT, timeout: float = 15.0) -> dict:
    """The server's {'cmd':'info'} answer: resolved checkpoint, capture geometry, reactivity flags."""
    from ur5e_lerobot.remote import RemotePolicyClient

    c = RemotePolicyClient(host, port, timeout=timeout).connect()
    try:
        return c.info() or {}
    finally:
        c.close()


def ensure_infer_running(host: str = INFER_HOST, port: int = INFER_PORT, auto_start: bool = True,
                         timeout: float = 240.0, log: Log = _log) -> dict:
    """Return the server's info once it answers on host:port, starting the unit if it is local and allowed."""
    if port_open(host, port):
        info = infer_info(host, port)
        log(f"inference server {host}:{port} serves {info.get('checkpoint')}")
        return info
    if not is_local(host):
        raise RuntimeError(f"inference server {host}:{port} is not listening (remote host — start it there)")
    if not available():
        raise RuntimeError(f"inference server {host}:{port} is not listening and there is no supervisor "
                           f"here to start {INFER_UNIT}")
    s = unit_status(INFER_UNIT)
    if s in ACTIVE:
        log(f"{INFER_UNIT} is {s} but not listening yet — waiting for it to load")
    elif not auto_start:
        raise RuntimeError(f"inference server not listening on {host}:{port} and {INFER_UNIT} is {s} "
                           f"(auto-start disabled: sudo supervisorctl start {INFER_UNIT})")
    else:
        log(f"{INFER_UNIT} is {s} — starting it (~20-30 s to load the checkpoint)")
        r = _ctl("start", INFER_UNIT, timeout=timeout)
        log(f"  {(r.stdout or r.stderr).strip()}")
    if not wait_port(host, port, timeout):
        raise RuntimeError(f"{INFER_UNIT} did not come up on {host}:{port} within {timeout:.0f} s "
                           f"(see log/vla_infer_server.out.log)")
    info = infer_info(host, port)
    log(f"inference server up: {info.get('checkpoint')}")
    return info


def eval_preflight(remote: str | None, auto_infer: bool = True, extra_stop: Iterable[str] = (),
                   log: Log = _log) -> dict:
    """What an eval needs before it touches the robot: nothing else driving the UR, the server up.

    Stops vla_teleop (the recording panel) and the ROS UR driver if they run, plus `extra_stop`; then,
    for a loopback --remote, makes sure vla_infer_server answers (starting it unless auto_infer=False).
    Returns the server's info ({} for local inference)."""
    ensure_stopped((*EXCLUSIVE_WITH_EVAL, *extra_stop), log=log)
    if not remote:
        return {}
    host, _, port = remote.partition(":")
    return ensure_infer_running(host or INFER_HOST, int(port or INFER_PORT), auto_start=auto_infer, log=log)


def teleop_preflight(log: Log = _log) -> list[str]:
    """The panel's counterpart: the eval and the ROS UR driver must not run while recording."""
    return ensure_stopped(EXCLUSIVE_WITH_TELEOP, log=log)


# --------------------------------------------------------------------------- model switching
def _short(path: str) -> str:
    return path.split("outputs/train/")[-1] if "outputs/train/" in path else path


def resolve_checkpoint(spec: str, train_dir: str = TRAIN_DIR) -> str:
    """'run', 'run@STEP', or a path (a run dir, checkpoints/<step>, or pretrained_model) -> the absolute
    pretrained_model directory. 'run' alone takes checkpoints/last, else the highest step."""
    spec = spec.strip()
    name, _, step = spec.partition("@")
    base = name if os.path.isdir(name) else os.path.join(train_dir, name)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"no such model: {spec!r} ({base} is not a directory)")
    if os.path.isfile(os.path.join(base, "config.json")):
        cand = base                                            # already a pretrained_model dir
    elif os.path.isdir(os.path.join(base, "pretrained_model")):
        cand = os.path.join(base, "pretrained_model")          # a checkpoints/<step> dir
    else:
        ck = os.path.join(base, "checkpoints")
        if not os.path.isdir(ck):
            raise FileNotFoundError(f"{base} has no checkpoints/")
        if step:
            d = os.path.join(ck, step.zfill(6) if step.isdigit() else step)
        elif os.path.isdir(os.path.join(ck, "last")):
            d = os.path.join(ck, "last")
        else:
            steps = sorted(x for x in os.listdir(ck) if x.isdigit())
            if not steps:
                raise FileNotFoundError(f"{ck} holds no checkpoints")
            d = os.path.join(ck, steps[-1])
        cand = os.path.join(d, "pretrained_model")
    if not os.path.isfile(os.path.join(cand, "config.json")):
        raise FileNotFoundError(f"{cand} is not a pretrained_model directory (no config.json)")
    return os.path.realpath(cand)


def served_checkpoint(link: str = SERVE_LINK) -> str | None:
    return os.path.realpath(link) if os.path.lexists(link) else None


def serve_model(spec: str, host: str = INFER_HOST, port: int = INFER_PORT, timeout: float = 300.0,
                log: Log = _log) -> dict:
    """Make vla_infer_server serve `spec`: repoint outputs/serve_ckpt and (re)start the unit, unless the
    running server already reports that very checkpoint. Returns the server's info."""
    want = resolve_checkpoint(spec)
    if port_open(host, port):
        try:
            info = infer_info(host, port)
        except Exception as e:  # noqa: BLE001
            log(f"server on {host}:{port} did not answer info ({e}); restarting it")
            info = {}
        cur = info.get("checkpoint_realpath") or ""
        if cur and os.path.realpath(cur) == want:
            log(f"already serving {_short(want)}")
            return info
        # A multi-model server keeps several checkpoints resident: selecting one is instant (a policy
        # reset), against ~16 s for the repoint-and-restart below.
        resident = [m for m in (info.get("models") or [])
                    if os.path.realpath(m.get("checkpoint_realpath") or "") == want]
        if resident:
            from ur5e_lerobot.remote import RemotePolicyClient

            t0 = time.time()
            c = RemotePolicyClient(host, port, timeout=60).connect()
            try:
                rep = c.select(want)
            finally:
                c.close()
            if rep.get("ok") and os.path.realpath(rep.get("checkpoint_realpath") or "") == want:
                log(f"selected the resident model {_short(want)} in {time.time() - t0:.1f} s")
                return rep
            log(f"select failed ({rep.get('error')}); restarting the server instead")
    if not is_local(host):
        raise RuntimeError(f"the model can only be switched on the local server, not {host}:{port}")
    if not available():
        raise RuntimeError(f"no supervisorctl here to restart {INFER_UNIT}")
    if served_checkpoint() != want:
        tmp = SERVE_LINK + ".tmp"
        if os.path.lexists(tmp):
            os.unlink(tmp)
        os.symlink(want, tmp)
        os.replace(tmp, SERVE_LINK)                          # atomic repoint
        log(f"{SERVE_LINK} -> {_short(want)}")
    s = unit_status(INFER_UNIT)
    verb = "restart" if s in ACTIVE else "start"
    log(f"{verb}ing {INFER_UNIT} (~20-30 s to load)")
    r = _ctl(verb, INFER_UNIT, timeout=timeout)
    log(f"  {(r.stdout or r.stderr).strip()}")
    if not wait_port(host, port, timeout):
        raise RuntimeError(f"{INFER_UNIT} did not come up on {host}:{port} within {timeout:.0f} s "
                           f"(see log/vla_infer_server.out.log)")
    info = infer_info(host, port)
    got = os.path.realpath(info.get("checkpoint_realpath") or "")
    if got != want:
        raise RuntimeError(f"server came up serving {got}, wanted {want}")
    cap = (info.get("capture") or {}).get("squares") if isinstance(info.get("capture"), dict) else None
    log(f"serving {_short(want)}  capture={cap}")
    return info


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="the robot app units, the server port and what it serves")
    p = sub.add_parser("eval-preflight", help="stop vla_teleop / the ROS UR driver; make sure the server is up")
    p.add_argument("--remote", default=f"{INFER_HOST}:{INFER_PORT}")
    p.add_argument("--no-auto-infer", action="store_true", help="verify the server instead of starting it")
    sub.add_parser("teleop-preflight", help="stop vla_eval / the ROS UR driver")
    p = sub.add_parser("ensure-stopped", help="stop the given units and wait")
    p.add_argument("units", nargs="+")
    p = sub.add_parser("serve", help="switch the served model (repoint serve_ckpt, restart the server)")
    p.add_argument("model", help="run name, run@step, or a checkpoint path")
    p.add_argument("--remote", default=f"{INFER_HOST}:{INFER_PORT}")
    p = sub.add_parser("info", help="what the server serves")
    p.add_argument("--remote", default=f"{INFER_HOST}:{INFER_PORT}")
    a = ap.parse_args(argv)

    def hp(remote: str) -> tuple[str, int]:
        h, _, pt = remote.partition(":")
        return h or INFER_HOST, int(pt or INFER_PORT)

    try:
        if a.cmd == "status":
            if not available():
                print("no supervisorctl on this machine")
            for n, s in statuses().items():
                print(f"  {n:<26} {s or '—'}")
            up = port_open(INFER_HOST, INFER_PORT)
            print(f"  server {INFER_HOST}:{INFER_PORT}: {'listening' if up else 'down'}   "
                  f"serve_ckpt -> {_short(served_checkpoint() or '(unset)')}")
            if up:
                i = infer_info()
                print(f"  serving {i.get('checkpoint')}  capture={(i.get('capture') or {}).get('squares')}")
                for m in i.get("models") or []:
                    print(f"    resident: {m.get('run')}  ({m.get('checkpoint')})")
        elif a.cmd == "eval-preflight":
            eval_preflight(a.remote, auto_infer=not a.no_auto_infer)
        elif a.cmd == "teleop-preflight":
            teleop_preflight()
        elif a.cmd == "ensure-stopped":
            ensure_stopped(a.units)
        elif a.cmd == "serve":
            serve_model(a.model, *hp(a.remote))
        elif a.cmd == "info":
            i = infer_info(*hp(a.remote))
            for k in sorted(i):
                print(f"  {k}: {i[k]}")
    except Exception as e:  # noqa: BLE001
        print(f"[services] ERROR: {e}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
