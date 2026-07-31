"""Read a 3Dconnexion SpaceMouse via the `hid` module (no system hidapi / no Homebrew).

macOS note: the 3Dconnexion driver (3DxWare) grabs the raw device and exposes only
"3Dconnexion Virtual ..." HID devices. **Quit 3DxWare** (menu-bar app, or Activity
Monitor -> quit "3DconnexionHelper"/"3DxWareMac") so the raw multi-axis device appears,
then this can read it.

Test / calibrate:  python -m ur5e_lerobot.teleop.spacemouse   (move the device)
"""
from __future__ import annotations

import struct
import time

import hid

VENDORS = (0x256F, 0x046D)
VIRTUAL_PIDS = {0xC670, 0xC671, 0xC672}  # driver's virtual mouse/keyboard/data devices
# 3Dconnexion's own vendor id (0x256F) is used ONLY by their devices, so any 0x256F device is
# a SpaceMouse. The legacy id 0x046D is *Logitech's* — SHARED with webcams, mice, keyboards —
# so for 0x046D we must allowlist the known 3Dconnexion SpaceMouse PIDs; otherwise a Logitech
# webcam/mouse gets grabbed as a "SpaceMouse" (exactly what happened on the Linux box).
SPACEMOUSE_046D_PIDS = {
    0xC603, 0xC605, 0xC606, 0xC621, 0xC623, 0xC625, 0xC626,
    0xC627, 0xC628, 0xC629, 0xC62B,  # SpaceNavigator/Explorer/Pilot/Traveler/Pro family
}


def _i16(lo: int, hi: int) -> int:
    return struct.unpack("<h", bytes((lo & 0xFF, hi & 0xFF)))[0]


def _is_spacemouse(d) -> bool:
    """True only for an actual 3Dconnexion SpaceMouse (never a Logitech webcam/mouse)."""
    vid, pid = d["vendor_id"], d["product_id"]
    if pid in VIRTUAL_PIDS:
        return False
    if vid == 0x256F:
        return True
    if vid == 0x046D:
        return pid in SPACEMOUSE_046D_PIDS
    return False


def driver_running() -> bool:
    """True if the 3Dconnexion driver's virtual devices are present (it grabs the raw one)."""
    return any(d["product_id"] in VIRTUAL_PIDS for d in hid.enumerate() if d["vendor_id"] in VENDORS)


def find_candidates() -> list:
    """SpaceMouse HID interfaces, best first (the 6-DOF multi-axis data interface)."""
    raws = [d for d in hid.enumerate() if _is_spacemouse(d)]

    def rank(d):
        up, u = d.get("usage_page"), d.get("usage")
        if up == 1 and u == 8:
            return 0  # multi-axis controller — the 6-DOF data (usage exposed: macOS / hidraw)
        # The Linux hidapi build here uses the libusb backend, which does NOT parse usage
        # (every interface reports 0/None). On 3Dconnexion receivers the 6-DOF data is on
        # interface 0, so prefer it when usage is unavailable.
        if not up and d.get("interface_number") == 0:
            return 1
        if up == 1 and u == 48:
            return 2
        if up and up >= 0xFF00:
            return 4  # vendor-defined
        return 3

    return sorted(raws, key=rank)


def find_device():
    cands = find_candidates()
    return cands[0] if cands else None


class SpaceMouse:
    """Latest-state reader. Axes normalized to ~[-1, 1]; buttons as a 0/1 list."""

    def __init__(self, scale: float = 350.0, deadband: float = 0.12):
        self.scale = scale
        self.deadband = deadband  # ignore |axis| < deadband (centering noise)
        self.dev = None
        self.info = None
        self.x = self.y = self.z = 0.0
        self.roll = self.pitch = self.yaw = 0.0
        self.buttons = [0, 0]
        self.last_report: list[int] = []

    def open(self) -> "SpaceMouse":
        cands = find_candidates()
        if not cands:
            raise RuntimeError("No 3Dconnexion device found. Is it plugged in / paired?")
        last = None
        for info in cands:
            dev = hid.device()
            try:
                dev.open_path(info["path"])
                dev.set_nonblocking(True)
                self.dev, self.info = dev, info
                return self
            except Exception as e:  # noqa: BLE001 — try the next interface
                last = e
                try:
                    dev.close()
                except Exception:
                    pass
        import sys

        if driver_running():
            hint = ("the 3Dconnexion driver (3DxWare) is still running and holding the device — "
                    "quit it (menu-bar icon / Activity Monitor: '3DconnexionHelper'/'3DxWareMac'), "
                    "then retry")
        elif sys.platform == "darwin":
            hint = ("on macOS, grant your terminal 'Input Monitoring' in System Settings > "
                    "Privacy & Security, then retry")
        else:
            hint = ("on Linux, raw device access is granted to the user logged in at the physical "
                    "screen — run this AT the machine, not over SSH; if it still fails, unplug/"
                    "replug the SpaceMouse and ensure spacenavd is stopped "
                    "(sudo systemctl mask --now spacenavd)")
        raise RuntimeError(f"could not open the SpaceMouse ({last}); {hint}.")

    def _norm(self, v: int) -> float:
        v = max(-1.0, min(1.0, v / self.scale))
        db = self.deadband
        if abs(v) <= db:
            return 0.0
        # rescale [db, 1] -> [0, 1] so motion ramps smoothly from the deadband edge
        return (v - db * (1.0 if v > 0 else -1.0)) / (1.0 - db)

    def _parse(self, d: list[int]) -> None:
        self.last_report = list(d)
        rid = d[0]
        if rid == 1:  # translation (and rotation too, on combined-report firmware)
            self.x, self.y, self.z = self._norm(_i16(d[1], d[2])), self._norm(_i16(d[3], d[4])), self._norm(_i16(d[5], d[6]))
            if len(d) >= 13:
                self.roll, self.pitch, self.yaw = self._norm(_i16(d[7], d[8])), self._norm(_i16(d[9], d[10])), self._norm(_i16(d[11], d[12]))
        elif rid == 2:  # rotation
            self.roll, self.pitch, self.yaw = self._norm(_i16(d[1], d[2])), self._norm(_i16(d[3], d[4])), self._norm(_i16(d[5], d[6]))
        elif rid == 3:  # buttons (bitmask)
            bits = d[1] if len(d) > 1 else 0
            self.buttons = [(bits >> i) & 1 for i in range(8)]

    def read(self) -> None:
        """Drain pending HID reports, updating the latest state."""
        if self.dev is None:
            return
        while True:
            data = self.dev.read(64)
            if not data:
                break
            self._parse(data)

    def state(self) -> dict:
        return {
            "x": self.x, "y": self.y, "z": self.z,
            "roll": self.roll, "pitch": self.pitch, "yaw": self.yaw,
            "buttons": list(self.buttons),
        }

    def close(self) -> None:
        if self.dev is not None:
            self.dev.close()
            self.dev = None


def _main() -> None:
    # 1) Show what's actually connected (helps tell driver-grab vs parsing issues).
    print("3Dconnexion HID devices seen:")
    any_raw = False
    for d in hid.enumerate():
        if d["vendor_id"] in VENDORS:
            virtual = d["product_id"] in VIRTUAL_PIDS
            any_raw = any_raw or not virtual
            print("  vid=%#06x pid=%#06x usage_page=%s usage=%s %-9s %r"
                  % (d["vendor_id"], d["product_id"], d.get("usage_page"), d.get("usage"),
                     "[VIRTUAL]" if virtual else "[RAW]", d.get("product_string")))
    if not any_raw:
        print("\n>> Only VIRTUAL devices found — the 3Dconnexion driver (3DxWare) is still holding")
        print(">> the SpaceMouse. Quit it (menu-bar icon / Activity Monitor: '3DconnexionHelper'")
        print(">> or '3DxWareMac'), then re-run this. (A re-plug after quitting can help.)")
        return

    sm = SpaceMouse()
    try:
        sm.open()
    except Exception as e:  # noqa: BLE001
        print("ERROR:", e)
        raise SystemExit(1)
    print(f"\nopened: {sm.info.get('product_string')!r}  vid={sm.info['vendor_id']:#06x} pid={sm.info['product_id']:#06x}")
    print("Move the SpaceMouse / press buttons (Ctrl-C to stop). Raw report shown for calibration.")
    try:
        while True:
            sm.read()
            s = sm.state()
            active = any(abs(s[k]) > 0.02 for k in ("x", "y", "z", "roll", "pitch", "yaw")) or any(s["buttons"])
            if active:
                print("x=%+.2f y=%+.2f z=%+.2f r=%+.2f p=%+.2f yaw=%+.2f btn=%s  raw=%s"
                      % (s["x"], s["y"], s["z"], s["roll"], s["pitch"], s["yaw"], s["buttons"][:2], sm.last_report))
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        sm.close()


if __name__ == "__main__":
    _main()
