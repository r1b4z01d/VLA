"""Log what the UR controller says while something else drives it — read-only diagnostic.

Two sources, both passive and safe to run next to the panel / eval / calibrator:
  * primary interface :30001 (ur5e_lerobot.robot.ur_events) — the controller's event stream: every
    C-code the pendant shows (ROBOT_COMM / SAFETY_MODE), popups, runtime exceptions, program keys;
  * RTDE :30004 (ur_rtde receive) — polled for robot mode / safety mode / runtime state changes, with
    the TCP pose and force at the moment of each change.

    ~/VLA/run.sh scripts/ur_event_log.py --ip 192.168.11.21 --out log/ur_events.log
    tail -f ~/VLA/log/ur_events.log

Why: "the UR faulted" from a script that only prints its own lines leaves no record of WHICH fault
(joint limit, safety violation, script stopped ...). The pendant's log tab has it, but this writes it
next to the run so it can be read remotely and correlated with timestamps.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

from ur5e_lerobot.robot.ur_events import ROBOT_MODE, SAFETY_MODE, EventListener

RUNTIME = {0: "STOPPING", 1: "STOPPED", 2: "PLAYING", 3: "PAUSING", 4: "PAUSED", 5: "RESUMING"}
_lock = threading.Lock()


def _log(out, line: str) -> None:
    ts = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
    with _lock:
        out.write(f"{ts} {line}\n")
        out.flush()


def rtde_loop(ip: str, out, stop: threading.Event) -> None:
    """Poll ur_rtde receive for safety/runtime changes (the primary stream has no safety mode)."""
    try:
        from rtde_receive import RTDEReceiveInterface
    except ImportError:
        _log(out, "[rtde] ur_rtde not installed; safety-mode polling disabled")
        return
    last = None
    r = None
    while not stop.is_set():
        try:
            if r is None:
                r = RTDEReceiveInterface(ip, 50.0)
                _log(out, "[rtde] connected")
            sm, rm, rt = r.getSafetyMode(), r.getRobotMode(), r.getRuntimeState()
            ps, es = r.isProtectiveStopped(), r.isEmergencyStopped()
            cur = (sm, rm, rt, ps, es)
            if cur != last:
                pose = r.getActualTCPPose()
                f = r.getActualTCPForce()
                _log(out, f"[rtde] safety={SAFETY_MODE.get(sm, sm)} robot={ROBOT_MODE.get(rm, rm)} "
                          f"runtime={RUNTIME.get(rt, rt)} protective_stop={ps} e_stop={es} "
                          f"tcp=({pose[0]:+.3f},{pose[1]:+.3f},{pose[2]:+.3f}) "
                          f"force=({f[0]:+.0f},{f[1]:+.0f},{f[2]:+.0f}) N")
                last = cur
            time.sleep(0.1)
        except Exception as e:  # noqa: BLE001
            _log(out, f"[rtde] {e}; reconnecting in 3 s")
            try:
                if r is not None:
                    r.disconnect()
            except Exception:  # noqa: BLE001
                pass
            r = None
            last = None
            time.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Log UR controller events (read-only).")
    ap.add_argument("--ip", default="192.168.11.21")
    ap.add_argument("--out", default="log/ur_events.log", help="append here ('-' = stdout)")
    ap.add_argument("--verbose", action="store_true",
                    help="also log URScript textmsg() output (ur_rtde prints two lines per servo step)")
    args = ap.parse_args()
    out = sys.stdout if args.out == "-" else open(args.out, "a")
    if out is not sys.stdout:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    _log(out, f"=== ur_event_log start pid {os.getpid()} ip {args.ip} ===")

    def on_state(f: dict) -> None:
        _log(out, "[state] " + " ".join(f"{k}={int(v) if isinstance(v, bool) else v}" for k, v in f.items()))

    def on_event(e) -> None:
        if e.kind == "TEXT" and not args.verbose:
            return
        tag = "primary" if e.kind == "LISTENER" else "ur"
        _log(out, f"[{tag}] {e.kind + ' ' if e.kind != 'LISTENER' else ''}{e.text}")

    ev = EventListener(args.ip, on_event=on_event, on_state=on_state)
    ev.start()
    stop = threading.Event()
    threading.Thread(target=rtde_loop, args=(args.ip, out, stop), daemon=True).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
        ev.stop()
        _log(out, "=== stopped ===")


if __name__ == "__main__":
    main()
