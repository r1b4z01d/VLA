"""Passive listener on the UR primary interface (:30001) — the controller's own event stream.

Every C-code the pendant shows arrives here as a ROBOT_MESSAGE (ROBOT_COMM / SAFETY_MODE), together
with popups, runtime exceptions and the program start/stop keys. Nothing is ever sent, so this is safe
next to ur_rtde (:30002/:30004) and next to the panel / eval / calibrator; the controller allows many
readers on :30001.

    ev = EventListener("192.168.11.21"); ev.start()
    ...
    e = ev.last_safety_event()        # Event(kind='SAFETY_MODE', text='C150A0 ... -> PROTECTIVE_STOP', ...)

Why: a script that only prints its own lines leaves no record of WHICH fault stopped the arm. The
pendant's log tab has it, but it cannot be read remotely and cannot be correlated with the script's
own timeline. scripts/ur_event_log.py writes this stream to a file; calibrate_workspace.py uses it to
tell the operator why freedrive ended.
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from collections import deque, namedtuple

TYPE_ROBOT_STATE = 16
TYPE_ROBOT_MESSAGE = 20
MSG = {0: "TEXT", 1: "ERROR_CODE", 3: "VERSION", 5: "SAFETY_MODE", 6: "ROBOT_COMM", 7: "KEY",
       10: "RUNTIME_EXCEPTION", 12: "PROGRAM_LABEL", 14: "POPUP", 17: "TEXT"}
SAFETY_MODE = {1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY", 5: "SAFEGUARD_STOP",
               6: "SYSTEM_EMERGENCY_STOP", 7: "ROBOT_EMERGENCY_STOP", 8: "VIOLATION", 9: "FAULT",
               10: "VALIDATE_JOINT_ID", 11: "UNDEFINED", 12: "AUTOMATIC_MODE_SAFEGUARD_STOP",
               13: "THREE_POSITION_ENABLING_STOP"}
ROBOT_MODE = {-1: "NO_CONTROLLER", 0: "DISCONNECTED", 1: "CONFIRM_SAFETY", 2: "BOOTING", 3: "POWER_OFF",
              4: "POWER_ON", 5: "IDLE", 6: "BACKDRIVE", 7: "RUNNING", 8: "UPDATING_FIRMWARE"}
JOINTS = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")

# UR error-code guide (universal-robots.com/manuals/.../prod-err-codes/topics/CODE_<n>.html, SW 5.20).
# Entry: (title, A-number is the joint that detected it, {A: sub-code text}). Seen on this robot
# 2026-10-24 while hand-guiding at full stretch: C150A0, then C283A51 -> VIOLATION and C306A1 -> FAULT.
KNOWN_CODES = {
    150: ("position close to joint limits", True, {}),
    153: ("position deviates from path (collision, wrong payload, or pushed faster than the controller "
          "can follow)", True, {}),
    156: ("wrong payload/mounting, or something pushing the robot when entering freedrive", False, {}),
    161: ("large movement detected while the arm was powered off", False, {}),
    283: ("safety system", False, {
        1: "robot is not braking when in stop mode",
        2: "robot is moving when in stop mode",
        7: "a joint exceeded the speed limit of the safety settings",
        26: "force limitation: a joint exceeded the torque window",
        43: "a joint moved too quickly toward a joint position limit",
        44: "the tool moved too fast towards an orientation limit",
        45: "the elbow moved too fast towards a safety plane",
        46: "the tool moved too fast towards a safety plane",
        47: "joint position limit exceeded",
        48: "tool position limit exceeded",
        49: "tool orientation limit exceeded",
        50: "elbow position limit exceeded",
        51: "the TOOL moved faster than the tool speed limit in the safety settings (mm/s)",
        52: "the ELBOW moved faster than the elbow speed limit in the safety settings (mm/s)",
        66: "robot momentum exceeded the momentum limit",
        67: "robot power exceeded the power limit",
    }),
    306: ("joint", False, {
        0: "a critical error occurred in a joint",
        1: "a joint was unable to come to a full stop fast enough (wrong payload, or an external force "
           "still pushing the robot while it stops) -> FAULT, arm powers off",
        9: "a joint moved more than the allowable limit (brakes?)",
    }),
}

Event = namedtuple("Event", "t kind text code arg")


def describe_code(code: int, arg: int, value: "str | None" = None) -> str:
    k = KNOWN_CODES.get(code)
    if k is None:
        return f"C{code}A{arg}" + (f" [{value}]" if value else "")
    title, joint, subs = k
    text = subs.get(arg)
    if text is None:
        text = title + (f" ({JOINTS[arg]} joint)" if joint and 0 <= arg < len(JOINTS) else "")
    else:
        text = f"{title}: {text}"
    if value:
        text += f" [{value}]"
    return f"C{code}A{arg} {text}"


def _comm_value(dtype: int, data: int) -> "str | None":
    """robotMessageDataType: 0 none, 1 unsigned, 2 signed, 3 float, 4 hex, 5 string, 6 joint."""
    try:
        if dtype == 1:
            return str(data)
        if dtype == 2:
            return str(struct.unpack(">i", struct.pack(">I", data))[0])
        if dtype == 3:
            return f"{struct.unpack('>f', struct.pack('>I', data))[0]:.1f}"
        if dtype == 4:
            return hex(data)
        if dtype == 6:
            return JOINTS[data] if 0 <= data < len(JOINTS) else f"joint{data}"
    except struct.error:
        pass
    return None


def _text(b: bytes) -> str:
    return "".join(ch for ch in b.decode("latin-1") if ch.isprintable()).strip()


def decode_message(body: bytes) -> "Event | None":
    """body = a ROBOT_MESSAGE package after its 5-byte header: int64 timestamp, int8 source, int8 type."""
    if len(body) < 10:
        return None
    _ts, source, mtype = struct.unpack(">qbb", body[:10])
    rest = body[10:]
    kind = MSG.get(mtype, f"type{mtype}")
    now = time.time()
    try:
        if mtype == 5 and len(rest) >= 9:  # SAFETY_MODE: code, argument, safetyModeType, dataType, data
            code, arg, smode = struct.unpack(">iiB", rest[:9])
            value = _comm_value(*struct.unpack(">II", rest[9:17])) if len(rest) >= 17 else None
            mode = SAFETY_MODE.get(smode, str(smode))
            return Event(now, "SAFETY_MODE", f"{describe_code(code, arg, value)} -> {mode}", code, arg)
        if mtype == 6 and len(rest) >= 20:  # ROBOT_COMM: code, argument, reportLevel, dataType, data
            code, arg, level, dtype, data = struct.unpack(">iiiiI", rest[:20])
            return Event(now, "ROBOT_COMM",
                         f"{describe_code(code, arg, _comm_value(dtype, data))} level={level} "
                         f"{_text(rest[20:])}".rstrip(), code, arg)
        if mtype == 7 and len(rest) >= 9:  # KEY: code, argument, titleSize, title, text
            code, arg, tsize = struct.unpack(">iiB", rest[:9])
            return Event(now, "KEY", f"C{code}A{arg} [{_text(rest[9:9 + tsize])}] {_text(rest[9 + tsize:])}",
                         code, arg)
        if mtype == 10 and len(rest) >= 8:  # RUNTIME_EXCEPTION: line, column, text
            line, col = struct.unpack(">ii", rest[:8])
            return Event(now, "RUNTIME_EXCEPTION", f"line {line} col {col}: {_text(rest[8:])}", 0, 0)
        if mtype == 12 and len(rest) >= 4:  # PROGRAM_LABEL: id, text
            return Event(now, "PROGRAM_LABEL", _text(rest[4:]), 0, 0)
        if mtype == 14 and len(rest) >= 12:  # POPUP: reqId, reqType, warning, error, blocking, titleSize
            _rid, _rt, warn, err, blocking, tsize = struct.unpack(">IIBBBB", rest[:12])
            flag = "ERROR" if err else ("WARNING" if warn else "info")
            return Event(now, "POPUP", f"{flag}{' blocking' if blocking else ''} "
                                       f"[{_text(rest[12:12 + tsize])}] {_text(rest[12 + tsize:])}", 0, 0)
        if mtype == 3:
            return Event(now, "VERSION", _text(rest), 0, 0)
    except struct.error:
        pass
    if mtype == 0:  # TEXT: int8 source then the textmsg() string
        return Event(now, "TEXT", f"(src {source}) {_text(rest)}", 0, 0)
    return Event(now, kind, f"(src {source}) {_text(rest)!r}", 0, 0)


class EventListener(threading.Thread):
    """Background reader of :30001. Keeps the last `keep` events, the latest robot-mode flags and the
    latest safety mode; optional callbacks fire on each message / each flag change."""

    def __init__(self, ip: str, keep: int = 500, on_event=None, on_state=None, reconnect_s: float = 3.0):
        super().__init__(daemon=True, name="ur-events")
        self.ip = ip
        self.events: "deque[Event]" = deque(maxlen=keep)
        self.flags: dict = {}
        self.safety_mode: "str | None" = None
        self.connected = False
        self._on_event, self._on_state = on_event, on_state
        self._reconnect_s = reconnect_s
        self._stop_ev = threading.Event()   # NOT `_stop`: that shadows Thread._stop(), which CPython calls after fork
        self._lock = threading.Lock()

    def stop(self) -> None:
        self._stop_ev.set()

    def last_safety_event(self, exclude_normal: bool = True) -> "Event | None":
        with self._lock:
            for e in reversed(self.events):
                if e.kind == "SAFETY_MODE" and not (exclude_normal and e.text.endswith("-> NORMAL")):
                    return e
        return None

    def _push(self, e: Event) -> None:
        # TEXT = URScript textmsg() output. ur_rtde's control script prints two per servo step (24/s at
        # 12 Hz), which would flush the safety events out of the deque within seconds — callbacks only.
        if e.kind != "TEXT":
            with self._lock:
                self.events.append(e)
                if e.kind == "SAFETY_MODE":
                    self.safety_mode = e.text.rsplit("-> ", 1)[-1]
        if self._on_event:
            try:
                self._on_event(e)
            except Exception:  # noqa: BLE001 — a logging callback must never kill the reader
                pass

    def _state(self, pkg: bytes) -> None:
        sublen, subtype = struct.unpack(">iB", pkg[:5])
        if subtype != 0 or sublen < 5 + 17:  # ROBOT_MODE_DATA only
            return
        (_t, _conn, _enabled, power, estop, pstop, prog_run, prog_pause, rmode,
         _cmode) = struct.unpack(">Q7Bbb", pkg[5:5 + 17])
        flags = {"robot_mode": ROBOT_MODE.get(rmode, str(rmode)), "power": bool(power), "e_stop": bool(estop),
                 "protective_stop": bool(pstop), "program_running": bool(prog_run), "paused": bool(prog_pause)}
        if flags != self.flags:
            self.flags = flags
            if self._on_state:
                try:
                    self._on_state(dict(flags))
                except Exception:  # noqa: BLE001
                    pass

    def run(self) -> None:
        while not self._stop_ev.is_set():
            s = None
            try:
                s = socket.create_connection((self.ip, 30001), timeout=5)
                s.settimeout(1.0)
                self.connected = True
                buf = b""
                while not self._stop_ev.is_set():
                    try:
                        chunk = s.recv(65536)
                    except socket.timeout:
                        continue
                    if not chunk:
                        raise ConnectionError("primary interface closed")
                    buf += chunk
                    while len(buf) >= 5:
                        plen, ptype = struct.unpack(">iB", buf[:5])
                        if plen < 5 or plen > 1 << 20:
                            buf = b""  # desynced — drop and resync on the next package
                            break
                        if len(buf) < plen:
                            break
                        pkg, buf = buf[5:plen], buf[plen:]
                        if ptype == TYPE_ROBOT_MESSAGE:
                            e = decode_message(pkg)
                            if e is not None:
                                self._push(e)
                        elif ptype == TYPE_ROBOT_STATE and len(pkg) >= 5 + 17:
                            self._state(pkg)
            except Exception as e:  # noqa: BLE001
                self.connected = False
                self._push(Event(time.time(), "LISTENER", f"{e}; retrying in {self._reconnect_s:.0f} s", 0, 0))
                self._stop_ev.wait(self._reconnect_s)
            finally:
                if s is not None:
                    try:
                        s.close()
                    except Exception:  # noqa: BLE001
                        pass
        self.connected = False
