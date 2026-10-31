"""UR5e power up / down over the Dashboard Server (:29999) — no pendant or VNC needed.

Used by the eval and recording apps so launching one brings the arm up and quitting parks it, and by
calibrate_workspace.py to recover after a protective stop / safety fault mid-session.

After a power cycle the UR safety system can raise C161A0 ("Large movement of the robot detected
while it was powered off") and sit in PROTECTIVE_STOP until a human accepts it on the pendant. The
dashboard's `unlock protective stop` clears it (verified on PolyScope 5.21.2). The usual cause is
the controller losing power without a shutdown, so the last joint positions never got saved. Every
auto-clear is logged: if it keeps firing after clean exits it is a real encoder/safety fault, not
something to paper over.

Needs the pendant in Remote Control mode. `power_up` releases the brakes without checking that the
arm is clear — keep the workspace clear before launching.
"""
from __future__ import annotations

import socket
import time

from .rtde_arm import DEFAULT_UR5E_IP

DASHBOARD_PORT = 29999


def _dashboard(ip: str, cmd: str) -> str:
    with socket.create_connection((ip, DASHBOARD_PORT), timeout=5) as s:
        s.settimeout(5)
        s.recv(1024)  # "Connected: Universal Robots Dashboard Server"
        s.sendall((cmd + "\n").encode())
        return s.recv(1024).decode().strip()


def _value(reply: str) -> str:
    return reply.split(":", 1)[-1].strip()


def _wait(ip: str, cmd: str, want: set, timeout: float) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if _value(_dashboard(ip, cmd)) in want:
            return True
        time.sleep(1)
    return False


def power_up(ip: str = DEFAULT_UR5E_IP, timeout: float = 60.0) -> str:
    """Bring the arm to RUNNING (brakes released). Returns a status message; raises RuntimeError
    if it can't get there. Already-RUNNING is a no-op."""
    try:
        if _dashboard(ip, "is in remote control") != "true":
            raise RuntimeError("pendant is not in Remote Control mode — switch it on the teach pendant")
        if _value(_dashboard(ip, "robotmode")) == "RUNNING":
            return "UR already RUNNING"
        note = ""
        ss = _value(_dashboard(ip, "safetystatus"))
        if ss in ("FAULT", "VIOLATION"):
            # A safety VIOLATION (e.g. C283A51 tool speed limit while hand-guiding) escalates to a FAULT
            # (C306A1 joint could not stop in time) and the arm powers itself off. The pendant's
            # "Restart Safety" is the dashboard's `restart safety`; it only works while powered off.
            note = f"restarted the safety system after {ss}; "
            print(f"[ur] WARNING: safety status {ss} — a safety limit was violated or a joint could not stop "
                  f"in time (see the pendant log / log/ur_events.log). Restarting safety via the dashboard.",
                  flush=True)
            _dashboard(ip, "close safety popup")
            reply = _dashboard(ip, "restart safety")
            if not _wait(ip, "safetystatus", {"NORMAL"}, 45):
                raise RuntimeError(f"safety did not return to NORMAL after 'restart safety' ({reply}); now "
                                   f"{_dashboard(ip, 'safetystatus')} — restart it on the pendant")
            ss = "NORMAL"
        if ss == "PROTECTIVE_STOP":
            note = "cleared protective stop at startup (C161A0 joint-position mismatch?); "
            print(f"[ur] WARNING: {note.rstrip('; ')}. Repeats after clean exits => encoder/safety fault.",
                  flush=True)
            _dashboard(ip, "close safety popup")
            _dashboard(ip, "unlock protective stop")
            if not _wait(ip, "safetystatus", {"NORMAL"}, 15):
                raise RuntimeError("safety status did not return to NORMAL: "
                                   + _dashboard(ip, "safetystatus"))
        _dashboard(ip, "power on")
        if not _wait(ip, "robotmode", {"IDLE", "RUNNING"}, timeout):
            raise RuntimeError("timed out waiting for IDLE: " + _dashboard(ip, "robotmode"))
        _dashboard(ip, "brake release")
        if not _wait(ip, "robotmode", {"RUNNING"}, timeout):
            raise RuntimeError("timed out waiting for RUNNING: " + _dashboard(ip, "robotmode"))
        return note + "UR powered up (RUNNING)"
    except OSError as e:
        raise RuntimeError(f"UR dashboard unreachable at {ip}:{DASHBOARD_PORT} ({e}) — is the "
                           "controller powered on?") from e


def power_down(ip: str = DEFAULT_UR5E_IP, shutdown: bool = False, timeout: float = 30.0) -> str:
    """Power the arm off (brakes engage). With shutdown=True also shut the controller down, which
    lets the safety system save joint positions — but then the controller has to be switched on by
    hand again, so it is opt-in. Never raises: this runs from exit paths."""
    try:
        _dashboard(ip, "power off")
        _wait(ip, "robotmode", {"POWER_OFF"}, timeout)
        if shutdown:
            _dashboard(ip, "shutdown")
            return "UR powered off and controller shut down"
        return "UR powered off"
    except Exception as e:  # noqa: BLE001
        return f"UR power-down failed: {e}"
