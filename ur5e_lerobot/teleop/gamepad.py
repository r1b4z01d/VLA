"""Read an Xbox / XInput gamepad via evdev (Linux). State API mirrors SpaceMouse so the teleop
panel can use it the same way. evdev is Linux-only + needs local-seat access to /dev/input/event*
(run AT the robot PC, like the SpaceMouse).

Panel mapping (see manual_panel):
  left stick  -> x, y        right stick -> roll, pitch
  triggers    -> z (RT up, LT down)      bumpers LB/RB -> grasp open/close

Test:  ~/VLA/run.sh -m ur5e_lerobot.teleop.gamepad   (move sticks / triggers / bumpers)
"""
from __future__ import annotations

import time

DEADZONE = 0.12


class Gamepad:
    """Latest-state reader. Sticks normalized to ~[-1, 1] (deadzoned), triggers [0, 1], buttons 0/1."""

    def __init__(self, deadzone: float = DEADZONE):
        self.deadzone = deadzone
        self.dev = None
        self._ec = None
        self._abs: dict[int, tuple[int, int]] = {}   # code -> (min, max)
        self._raw: dict[int, float] = {}             # code -> latest raw value
        self._keys: dict[int, int] = {}              # code -> 0/1
        self.name = None

    def open(self) -> "Gamepad":
        import os

        import evdev
        from evdev import ecodes

        self._ec = ecodes
        pads = []
        for path in evdev.list_devices():
            d = evdev.InputDevice(path)
            caps = d.capabilities()
            keys = caps.get(ecodes.EV_KEY, [])
            abs_codes = [a[0] if isinstance(a, tuple) else a for a in caps.get(ecodes.EV_ABS, [])]
            if ecodes.ABS_X in abs_codes and (ecodes.BTN_SOUTH in keys or ecodes.BTN_GAMEPAD in keys):
                pads.append(d)
            else:
                d.close()
        if not pads:
            raise RuntimeError("no gamepad found — connect the Xbox controller (Linux/evdev; run at "
                               "the machine, and ensure your user can read /dev/input/event*)")
        self.dev = pads[0]
        self.name = self.dev.name
        for extra in pads[1:]:
            extra.close()
        for code, info in self.dev.capabilities().get(ecodes.EV_ABS, []):
            self._abs[code] = (info.min, info.max)
            self._raw[code] = (info.min + info.max) / 2.0
        os.set_blocking(self.dev.fd, False)  # so read() drains + raises when empty instead of blocking
        return self

    def read(self) -> None:
        """Drain pending events, updating the latest state."""
        if self.dev is None:
            return
        try:
            for e in self.dev.read():
                if e.type == self._ec.EV_ABS:
                    self._raw[e.code] = e.value
                elif e.type == self._ec.EV_KEY:
                    self._keys[e.code] = 1 if e.value else 0
        except BlockingIOError:
            pass

    def _stick(self, code: int) -> float:
        lo, hi = self._abs.get(code, (-32768, 32767))
        center = (lo + hi) / 2.0
        half = (hi - lo) / 2.0 or 1.0
        v = max(-1.0, min(1.0, (self._raw.get(code, center) - center) / half))
        if abs(v) <= self.deadzone:
            return 0.0
        return (v - (self.deadzone if v > 0 else -self.deadzone)) / (1.0 - self.deadzone)

    def _trigger(self, code: int) -> float:
        lo, hi = self._abs.get(code, (0, 255))
        rng = (hi - lo) or 1
        return max(0.0, min(1.0, (self._raw.get(code, lo) - lo) / rng))

    def state(self) -> dict:
        ec = self._ec
        return {
            "lx": self._stick(ec.ABS_X), "ly": self._stick(ec.ABS_Y),
            "rx": self._stick(ec.ABS_RX), "ry": self._stick(ec.ABS_RY),
            "lt": self._trigger(ec.ABS_Z), "rt": self._trigger(ec.ABS_RZ),
            "lb": self._keys.get(ec.BTN_TL, 0), "rb": self._keys.get(ec.BTN_TR, 0),
        }

    def close(self) -> None:
        if self.dev is not None:
            try:
                self.dev.close()
            except Exception:  # noqa: BLE001
                pass
            self.dev = None


def _main() -> None:
    gp = Gamepad().open()
    print(f"opened {gp.name!r}. Move sticks / triggers / bumpers (Ctrl-C to stop).")
    try:
        while True:
            gp.read()
            s = gp.state()
            if any(abs(s[k]) > 0.03 for k in ("lx", "ly", "rx", "ry", "lt", "rt")) or s["lb"] or s["rb"]:
                print("lx=%+.2f ly=%+.2f | rx=%+.2f ry=%+.2f | lt=%.2f rt=%.2f | LB=%d RB=%d"
                      % (s["lx"], s["ly"], s["rx"], s["ry"], s["lt"], s["rt"], s["lb"], s["rb"]))
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        gp.close()


if __name__ == "__main__":
    _main()
