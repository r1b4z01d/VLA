"""UDP (default) / TCP client for the AmazingHand (Pollen Robotics 8-DOF hand).

Speaks the line protocol used by the rd_ws AmazingHand demo
(github.com/r1b4z01d/rd_ws .../HandTracking). The hand controller accepts ASCII command lines on
UDP :8766 (one line per datagram) and on a raw TCP socket :8765:

    J:<a0>,<a1>,<a2>,<a3>,<a4>,<a5>,<a6>,<a7>,<speed>\\n

Why UDP is the default: the hand is an ESP32 on 2.4 GHz WiFi, measured at 12-25% packet loss with
1.5 s holes on the robot network. Over TCP a lost setpoint is retransmitted (the live session showed
28% of bytes resent, rto backed off to 680 ms, cwnd collapsed to 2) and arrives seconds late, so the
fingers freeze and then jump to a stale target. A 12 Hz setpoint stream wants the opposite
guarantee: lose it, forget it, act on the next one. With UDP a hole costs exactly the steps inside
it. Because UDP would otherwise fail silently, connect() probes the firmware ("Q" -> "K:<uptime>")
and refuses to proceed if nothing answers — a hand that is not listening is an error, not a no-op.
TCP remains for firmware without the UDP listener (`transport="tcp"`).

* a0..a7  per-servo angle offsets in DEGREES (8 servos = 4 fingers x 2).
* servo order (FINGER_TO_SERVO): index=(0,1) middle=(2,3) ring=(4,5) thumb=(6,7).
* speed   integer, firmware units 50..2000 (the manager's Hand tab exposes the range).
* default endpoint UDP 192.168.11.117:8766 (the hand's ESP32 on the robot Wi-Fi, powered from the UR
  tool port); TCP :8765 on the same ESP32 is the legacy transport (`transport="tcp"`, TCP_NODELAY).

Per-finger "curl" in [0,1] (0=open, 1=closed) maps to per-servo offsets via:
    offset = open + (close - open) * curl
using the open/close tables below (verbatim from the demo's hand_processing.py).

Design note: the hand stays behind this socket so it never has to share a Python
interpreter with rclpy + lerobot (project decision D5). The arm goes through RTDE
(robot/rtde_arm.py); the LeRobot Robot adapter just writes 8 floats here.
"""
from __future__ import annotations

import errno
import socket
import subprocess
import time
from collections.abc import Mapping

# 4 fingers x 2 servos. Position in the wire payload == servo index.
FINGER_ORDER = ("index", "middle", "ring", "thumb")
FINGER_TO_SERVO = {"index": (0, 1), "middle": (2, 3), "ring": (4, 5), "thumb": (6, 7)}

# Per-servo angle offsets (deg) at fully-open (curl=0) and fully-closed (curl=1).
JOINT_OPEN_OFFSETS = (-35.0, 35.0, -35.0, 35.0, -35.0, 35.0, -35.0, 35.0)
JOINT_CLOSE_OFFSETS = (90.0, -90.0, 90.0, -90.0, 90.0, -90.0, 70.0, -70.0)

DEFAULT_HOST = "192.168.11.117"  # AmazingHand controller (robot subnet 192.168.11.x)
DEFAULT_PORT = 8765          # TCP (legacy / debugging with `nc`)
DEFAULT_UDP_PORT = 8766      # UDP (firmware HAND_UDP_PORT) — the default transport, see module doc
DEFAULT_TRANSPORT = "udp"
DEFAULT_SPEED = 200  # placeholder — confirm the firmware's speed units/range.


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def curls_to_offsets(curls: "Mapping[str, float] | list[float] | tuple[float, ...]") -> list[float]:
    """Map per-finger curl in [0,1] to 8 per-servo angle offsets (deg).

    ``curls`` is either a mapping ``{finger: curl}`` or a length-4 sequence ordered
    ``(index, middle, ring, thumb)``. Values are clamped to [0, 1].
    """
    if isinstance(curls, Mapping):
        per_finger = {f: _clamp01(float(curls.get(f, 0.0))) for f in FINGER_ORDER}
    else:
        seq = list(curls)
        if len(seq) != 4:
            raise ValueError(f"expected 4 curls (index,middle,ring,thumb), got {len(seq)}")
        per_finger = {f: _clamp01(float(v)) for f, v in zip(FINGER_ORDER, seq)}

    offsets = [0.0] * 8
    for finger, servos in FINGER_TO_SERVO.items():
        c = per_finger[finger]
        for idx in servos:
            o, cl = JOINT_OPEN_OFFSETS[idx], JOINT_CLOSE_OFFSETS[idx]
            offsets[idx] = o + (cl - o) * c
    return offsets


def format_joint_command(offsets, speed) -> str:
    """Build the ``J:...\\n`` ASCII command line for 8 servo offsets + speed."""
    vals = list(offsets)
    if len(vals) != 8:
        raise ValueError(f"expected 8 offsets, got {len(vals)}")
    payload = ",".join(f"{float(v):.2f}" for v in vals)
    return f"J:{payload},{int(speed)}\n"


# --- V2: per-finger flex + abduct (full 8-DOF hand) ------------------------------------------------
ABDUCT_MAX_DEG = 20.0  # ± abduction range per finger (mechanism limit, ~±20°)


def flex_abduct_to_offsets(flex, abduct) -> list[float]:
    """Map per-finger flex [0,1] + abduct [-1,1] to the 8 servo offsets (deg).

    Per finger the two servos are A (even index) and B (odd). The mechanism is
    ``flex = (A - B) / 2``, ``abduct = (A + B) / 2`` -> ``A = flex_angle + abd``,
    ``B = -flex_angle + abd``, where ``flex_angle`` interpolates the open/close table (so servo A
    sweeps its JOINT_OPEN->JOINT_CLOSE range) and ``abd = abduct * ABDUCT_MAX_DEG``.
    With ``abduct=0`` this is identical to ``curls_to_offsets`` (pure flexion).
    """
    flex, abduct = list(flex), list(abduct)
    if len(flex) != 4 or len(abduct) != 4:
        raise ValueError(f"expected 4 flex + 4 abduct, got {len(flex)} + {len(abduct)}")
    offsets = [0.0] * 8
    for fi, finger in enumerate(FINGER_ORDER):
        a, b = FINGER_TO_SERVO[finger]  # even (A), odd (B)
        oa, ca = JOINT_OPEN_OFFSETS[a], JOINT_CLOSE_OFFSETS[a]
        flex_angle = oa + (ca - oa) * _clamp01(float(flex[fi]))
        abd = max(-1.0, min(1.0, float(abduct[fi]))) * ABDUCT_MAX_DEG
        offsets[a] = flex_angle + abd
        offsets[b] = -flex_angle + abd
    return offsets


def format_flex_abduct_command(flex, abduct, speed) -> str:
    """Build the ``F:f0,f1,f2,f3,a0,a1,a2,a3,speed\\n`` line (firmware does the servo mapping)."""
    flex, abduct = list(flex), list(abduct)
    if len(flex) != 4 or len(abduct) != 4:
        raise ValueError(f"expected 4 flex + 4 abduct, got {len(flex)} + {len(abduct)}")
    payload = ",".join(f"{float(v):.4f}" for v in flex + abduct)
    return f"F:{payload},{int(speed)}\n"


def parse_state_line(line: str):
    """Parse an ``S:f0,f1,f2,f3,a0,a1,a2,a3`` report -> (flex[4], abduct[4]) or None."""
    s = line.strip()
    if not s.startswith("S:"):
        return None
    try:
        nums = [float(x) for x in s[2:].split(",") if x != ""]
    except ValueError:
        return None
    return (nums[0:4], nums[4:8]) if len(nums) >= 8 else None


def ping_reachable(host: str, count: int = 3, timeout_s: float = 1.0) -> bool:
    """True if the hand answers at least one ICMP echo. Cheap, transport-independent evidence that the
    hand is POWERED and on WiFi — the question a caller must ask before power-cycling the tool."""
    try:
        r = subprocess.run(["ping", "-c", str(count), "-W", str(int(timeout_s)), "-i", "0.3", host],
                           capture_output=True, text=True, timeout=count * (timeout_s + 0.5) + 2)
        return " 0 received" not in r.stdout and "0 packets received" not in r.stdout and r.returncode == 0
    except Exception:  # noqa: BLE001 — no ping binary / odd platform: don't block the caller
        return False


def wait_reachable(host: str, timeout_s: float = 45.0, poll_s: float = 2.0) -> bool:
    """Poll ping until the hand is back (after a power cycle the ESP32 boots + rejoins WiFi + DHCP,
    typically 10-25 s on this network). Returns False if it never comes back within timeout_s."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if ping_reachable(host, count=1, timeout_s=1.0):
            return True
        time.sleep(poll_s)
    return False


class AmazingHandClient:
    """Minimal, dependency-free TCP client for the AmazingHand.

    Example::

        with AmazingHandClient() as hand:   # connects to 192.168.1.194:8765
            hand.send_curls([0, 0, 0, 0])   # fully open
            hand.send_curls({"index": 1.0}) # curl just the index finger
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        speed: int = DEFAULT_SPEED,
        timeout: float = 0.5,
        transport: str = DEFAULT_TRANSPORT,
        udp_port: int = DEFAULT_UDP_PORT,
        probe_timeout: float = 12.0,
    ) -> None:
        if transport not in ("udp", "tcp"):
            raise ValueError(f"transport must be 'udp' or 'tcp', got {transport!r}")
        self.host = host
        self.port = port
        self.udp_port = udp_port
        self.transport = transport
        self.speed = speed
        self.timeout = timeout          # per-datagram wait (also the TCP connect timeout)
        self.probe_timeout = probe_timeout  # total patience for the UDP liveness probe (see connect)
        self._sock: "socket.socket | None" = None
        self._rx = b""  # buffer for streamed S: state reports (read_state)
        # UDP is fire-and-forget: a failed send is counted, not raised (see _send)
        self.send_errors = 0
        self.last_error: "Exception | None" = None
        self.firmware_uptime_ms: "int | None" = None

    def connect(self) -> "AmazingHandClient":
        if self.transport == "tcp":
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = sock
            return self
        # UDP: a *connected* datagram socket — fixes the destination so send() is one syscall, and
        # lets the kernel surface ICMP port-unreachable as an error on the next send instead of
        # swallowing it. Then PROVE the firmware is listening: "Q" must come back as "K:<uptime>".
        # Several tries, because the whole point of this transport is that the link drops ~20%
        # of packets (a probe needs both directions to survive: P(fail/try) ~ 0.36 at 20% loss).
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.udp_port))
        # PATIENT, not strict. The first version gave up after 6 x 0.5 s. On this link losses come in
        # bursts (measured holes of 1.5 s+), so a launch that landed in a hole failed the probe, and the
        # callers' fallback then POWER-CYCLED the tool — rebooting a hand that was fine, which takes
        # ~20 s to rejoin WiFi, so the single retry failed too and the run died with the arm script
        # preempted. A liveness probe only has to establish "firmware listens on UDP"; waiting up to
        # probe_timeout for one answer costs nothing, while a false negative costs a reboot.
        # ECONNREFUSED is different: the kernel got ICMP port-unreachable, the port is CLOSED, this is
        # old firmware — repeated refusals fail fast with the flash message.
        last: "Exception | None" = None
        refused = 0
        deadline = time.monotonic() + max(self.timeout, float(self.probe_timeout))
        while time.monotonic() < deadline:
            try:
                sock.send(b"Q\n")
                data = sock.recv(256)
            except socket.timeout:
                continue
            except OSError as e:
                last = e
                if getattr(e, "errno", None) == errno.ECONNREFUSED:
                    refused += 1
                    if refused >= 3:
                        break
                time.sleep(0.25)
                continue
            if data.startswith(b"K:"):
                try:
                    self.firmware_uptime_ms = int(data[2:].strip() or 0)
                except ValueError:
                    self.firmware_uptime_ms = None
                self._sock = sock
                return self
        sock.close()
        if refused >= 3:
            raise RuntimeError(
                f"AmazingHand at {self.host}:{self.udp_port}/udp refused the liveness probe (port closed) — "
                f"flash the UDP-capable firmware (AmazingHand-main/ArduinoExample/Amazing_Hand_Demo) or use "
                f"transport='tcp'")
        raise RuntimeError(
            f"AmazingHand at {self.host}:{self.udp_port}/udp did not answer the liveness probe within "
            f"{self.probe_timeout:.0f}s{f' ({last})' if last else ''} — hand off, mid-reboot, or the WiFi link "
            f"is dropping everything; check `ping {self.host}` before power-cycling anything")

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> "AmazingHandClient":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    def _send(self, line: str) -> None:
        if self._sock is None:
            raise RuntimeError("not connected — call connect() (or use as a context manager)")
        data = line.encode("ascii")
        if self.transport == "tcp":
            self._sock.sendall(data)
            return
        # UDP: one datagram per line, never block the control loop, never raise. A send that fails
        # (ICMP unreachable from a rebooting hand, interface flap) is counted so a caller CAN notice
        # a dead hand, but the next step's setpoint supersedes this one anyway — raising here would
        # turn a dropped packet into a dropped run, which is exactly what TCP was doing to us.
        try:
            self._sock.send(data)
        except OSError as e:
            self.send_errors += 1
            self.last_error = e

    def send_offsets(self, offsets, speed: "int | None" = None) -> None:
        """Send raw 8 per-servo angle offsets (deg)."""
        self._send(format_joint_command(offsets, self.speed if speed is None else speed))

    def send_curls(self, curls, speed: "int | None" = None) -> None:
        """Send per-finger curl in [0,1] (mapping, or [index,middle,ring,thumb])."""
        self.send_offsets(curls_to_offsets(curls), speed)

    def send_flex_abduct(self, flex, abduct, speed: "int | None" = None) -> None:
        """Send per-finger flex [0,1] + abduct [-1,1] via the ``F:`` command (8-DOF hand).

        Requires firmware that understands ``F:`` (see docs/amazinghand_esp32_buttons.md). For
        firmware that only speaks ``J:``, use ``send_offsets(flex_abduct_to_offsets(flex, abduct))``.
        """
        self._send(format_flex_abduct_command(flex, abduct, self.speed if speed is None else speed))

    def read_state(self):
        """Return the latest streamed ``(flex[4], abduct[4])`` from the hand, or None if none yet.

        Non-blocking: drains whatever the ESP32 has streamed and parses the most recent ``S:`` line.
        Used to RECORD button-driven finger motion (the hand is driven locally; the host logs it)."""
        import select
        if self._sock is None:
            raise RuntimeError("not connected — call connect() first")
        try:
            while select.select([self._sock], [], [], 0)[0]:
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
                self._rx += chunk
        except (BlockingIOError, OSError):
            pass
        if b"\n" not in self._rx:
            return None
        *lines, self._rx = self._rx.split(b"\n")  # keep the trailing partial line buffered
        latest = None
        for ln in lines:
            parsed = parse_state_line(ln.decode("ascii", "ignore"))
            if parsed is not None:
                latest = parsed
        return latest

    def open_hand(self, speed: "int | None" = None) -> None:
        self.send_curls([0.0, 0.0, 0.0, 0.0], speed)

    def close_hand(self, speed: "int | None" = None) -> None:
        self.send_curls([1.0, 1.0, 1.0, 1.0], speed)


if __name__ == "__main__":
    import argparse
    import time

    p = argparse.ArgumentParser(description="AmazingHand smoke test (open/close cycle).")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port (transport=tcp)")
    p.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT)
    p.add_argument("--transport", choices=["udp", "tcp"], default=DEFAULT_TRANSPORT)
    p.add_argument("--speed", type=int, default=DEFAULT_SPEED)
    p.add_argument("--cycles", type=int, default=3, help="0 = probe only, do not move the hand")
    args = p.parse_args()

    with AmazingHandClient(args.host, args.port, speed=args.speed, transport=args.transport,
                           udp_port=args.udp_port) as hand:
        if hand.transport == "udp":
            print(f"udp liveness ok — firmware uptime {hand.firmware_uptime_ms} ms")
        for i in range(args.cycles):
            print(f"[{i}] open");  hand.open_hand();  time.sleep(1.0)
            print(f"[{i}] close"); hand.close_hand(); time.sleep(1.0)
        hand.open_hand()
    print("done")
