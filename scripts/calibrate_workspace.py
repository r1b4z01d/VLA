"""Capture the workspace limits by hand-guiding the UR5e in FREEDRIVE, then print + save the values.

Point the arm's TCP at each limit and press ENTER to capture. Produces:
  1. z-floor            — table/floor height (min captured z)      -> rtde_arm z_floor
  2. teleop target box  — home +/- POS_DELTAS (per-axis reach)     -> manual_panel.POS_DELTAS
  3. robot no-go rect   — the mobile-base body footprint (x,y)     -> workspace.NO_GO_XY

    ~/rd_ws/VLA/run.sh scripts/calibrate_workspace.py --robot-ip 192.168.11.21

Freedrive is UR-controlled, so the software clamps are OFF while you guide — keep the e-stop in hand.
Freedrive is released on exit (including Ctrl-C). Each section is saved to --out as soon as it is done.

TWO WAYS THIS STOPPED THE UR while calibrating for a floor task (2026-10-24, from log/ur_events.log):
  * C150A0 "position close to joint limits (base)" -> protective stop. The base joint had been wound
    the long way round from home and sat at ~360°; pushing it further trips the limit. Every capture
    prints the base angle — past ±300°, unwind it the other way before going on.
  * C283 safety VIOLATION -> C306 FAULT, arm powered itself off. At ~0.90 m reach with the elbow
    straight, lifting the arm: through the elbow singularity a hand push turns into a joint-speed
    spike. Reach floor targets with the elbow BENT (bring the shoulder down instead of stretching
    out) and keep the printed reach under ~0.80 m.
A protective stop ends freedrive (the RTDE control script dies). This script notices, prints the
controller's code, and on ENTER clears it over the dashboard, reconnects and re-enters freedrive
with the points captured so far intact. After a FAULT the safety system is restarted the same way
(dashboard "restart safety"); if the controller refuses, do it on the pendant and press ENTER again.
"""
import argparse
import json
import math
import os
import threading
import time

import numpy as np

from ur5e_lerobot.robot.rtde_arm import RtdeArmInterface
from ur5e_lerobot.robot.ur_events import EventListener

REACH_WARN = 0.80   # m radial TCP distance from the base; the UR5e is straight at ~0.85 m + tool
BASE_WARN = 300.0   # deg; |base joint| beyond this is within 60° of the ±360° hard limit


class Session:
    """The arm + the controller's event stream + a watchdog that shouts when freedrive ends."""

    def __init__(self, arm: RtdeArmInterface, ip: str, events: EventListener):
        self.arm, self.ip, self.events = arm, ip, events
        self._seen = None
        self._stop = threading.Event()
        self._watch = threading.Thread(target=self._watchdog, daemon=True)

    def start(self) -> None:
        self._watch.start()

    def stop(self) -> None:
        self._stop.set()

    # -- state ---------------------------------------------------------------------------------
    def faulted(self) -> bool:
        try:
            return not self.arm.is_ready()
        except Exception:  # noqa: BLE001 — a dead interface counts as faulted
            return True

    def _why(self) -> str:
        e = self.events.last_safety_event()
        return e.text if e else "control script stopped (no safety event seen — pendant?)"

    def _watchdog(self) -> None:
        """Print the controller's reason the moment a stop happens, even while input() is waiting."""
        while not self._stop.is_set():
            e = self.events.last_safety_event()
            if e is not None and e is not self._seen:
                self._seen = e
                print(f"\n  ** UR STOPPED — {e.text}. Freedrive has ENDED; the arm is locked. "
                      f"Hands off, then press ENTER to recover. **", flush=True)
            time.sleep(0.2)

    def pose_line(self) -> str:
        p = [float(v) for v in self.arm.get_ee_pose()]
        q = self.arm.get_joint_positions()
        base = math.degrees(q[0])
        reach = math.hypot(p[0], p[1])
        line = (f"x={p[0]:+.3f}  y={p[1]:+.3f}  z={p[2]:+.3f} m   reach {reach:.2f} m   "
                f"base {base:+.0f}°")
        warn = []
        if abs(base) > BASE_WARN:
            warn.append(f"base joint {base:+.0f}° is near its ±360° limit — unwind it the other way")
        if reach > REACH_WARN:
            warn.append(f"reach {reach:.2f} m is near full stretch — bend the elbow, bring the shoulder down")
        if warn:
            line += "\n      ! " + "\n      ! ".join(warn)
        return line

    def capture(self) -> "list[float]":
        p = [float(v) for v in self.arm.get_ee_pose()]
        print("      captured  " + self.pose_line())
        return p

    # -- recovery ------------------------------------------------------------------------------
    def freedrive_on(self) -> bool:
        return bool(self.arm.start_freedrive())

    def recover(self) -> bool:
        """Freedrive ended (protective stop / violation / fault). Returns False if the operator aborts."""
        print(f"\n  UR stopped — {self._why()}")
        while True:
            s = input("  hands off the arm, then ENTER = clear + re-enter freedrive,  q = abort: ").strip().lower()
            if s == "q":
                return False
            try:
                from ur5e_lerobot.robot.ur_power import power_up
                print("  " + power_up(self.ip))          # powered off after a FAULT -> restart safety + brakes
                print("  " + self.arm.reconnect())       # clears a protective stop + reuploads the script
                if not self.faulted() and self.freedrive_on():
                    print("  FREEDRIVE back on — carry on.\n")
                    return True
                print("  still not ready — check the pendant, then ENTER to retry")
            except Exception as e:  # noqa: BLE001
                print(f"  recovery failed: {e} — fix it on the pendant, then ENTER to retry")


def capture_points(sess: Session, instruction: str) -> np.ndarray:
    """Hand-guide + ENTER to capture the current TCP pose; 'd' (done) to finish the section."""
    print(instruction)
    pts = []
    while True:
        s = input(f"    [{len(pts)} captured]  ENTER = capture point,  d = done: ").strip().lower()
        if s in ("d", "done", "q"):
            break
        if sess.faulted():
            if not sess.recover():
                raise KeyboardInterrupt
            continue  # the arm is where it stopped; guide it and press ENTER again to capture
        pts.append(sess.capture())
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 6))


def _wait_enter(sess: Session, prompt: str) -> None:
    while True:
        input(prompt)
        if not sess.faulted():
            return
        if not sess.recover():
            raise KeyboardInterrupt


def _save(calib: dict, out: str) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(calib, f, indent=2)
    os.replace(tmp, out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Freedrive workspace-limit calibrator for the UR5e.")
    ap.add_argument("--robot-ip", default="192.168.11.21")
    ap.add_argument("--out", default="outputs/workspace_calib.json")
    ap.add_argument("--no-ur-power", action="store_true",
                    help="skip powering the UR on + releasing brakes first (eval/teleop do this by default)")
    args = ap.parse_args()

    if not args.no_ur_power:
        # Same as eval_hw / the teleop panel: bring the arm up through the dashboard before RTDE, so
        # this script works from a powered-off arm instead of failing to connect.
        from ur5e_lerobot.robot.ur_power import power_up
        print(power_up(args.robot_ip))
    events = EventListener(args.robot_ip)   # passive: the controller's C-codes, for the watchdog
    events.start()
    arm = RtdeArmInterface(args.robot_ip)
    arm.connect()
    sess = Session(arm, args.robot_ip, events)
    sess.start()
    calib: dict = {}
    print("\nStarting FREEDRIVE — hand-guide the arm. **e-STOP IN HAND.**")
    print("  now at  " + sess.pose_line() + "\n")
    try:
        if not sess.freedrive_on():
            print("freedrive not available on this arm; aborting")
            return

        print("=== 1) FLOOR (z-floor / table height) ===")
        fp = capture_points(sess, "  Rest the TCP on the floor/table at a few spots (corners + middle); capture each.")
        if len(fp):
            calib["z_floor"] = round(float(fp[:, 2].min()), 4)
            print(f"  -> z_floor = {calib['z_floor']} m  (lowest of {len(fp)} points)\n")
            _save(calib, args.out)

        print("=== 2) TELEOP TARGET BOX (home +/- POS_DELTAS) ===")
        _wait_enter(sess, "  Move the arm to your HOME / start pose, then press ENTER…")
        home = [float(v) for v in arm.get_ee_pose()]
        print(f"  home TCP  x={home[0]:+.3f}  y={home[1]:+.3f}  z={home[2]:+.3f}")
        bp = capture_points(sess, "  Reach to the EXTREMES of where you want to teleop (edges/corners, up/down); capture each.")
        if len(bp):
            d = np.abs(bp[:, :3] - np.asarray(home[:3])).max(0)
            calib["home_tcp"] = [round(v, 4) for v in home]
            calib["pos_deltas"] = {"x": round(float(d[0]), 3), "y": round(float(d[1]), 3),
                                   "z": round(float(d[2]), 3)}
            print(f"  -> POS_DELTAS = {calib['pos_deltas']}  (max |reach - home| per axis)\n")
            _save(calib, args.out)

        print("=== 3) ROBOT BODY NO-GO RECTANGLE (NO_GO_XY) ===")
        rp = capture_points(sess, "  Touch the TCP to the CORNERS/edges of the robot-body keep-out footprint; capture each.")
        if len(rp):
            xmin, xmax = float(rp[:, 0].min()), float(rp[:, 0].max())
            ymin, ymax = float(rp[:, 1].min()), float(rp[:, 1].max())
            calib["no_go_xy"] = [round(xmin, 3), round(xmax, 3), round(ymin, 3), round(ymax, 3)]
            print(f"  -> NO_GO_XY = {tuple(calib['no_go_xy'])}  (xmin, xmax, ymin, ymax)\n")
            _save(calib, args.out)
    except KeyboardInterrupt:
        print("\naborted")
    finally:
        sess.stop()
        events.stop()
        try:
            arm.stop_freedrive()
        except Exception:  # noqa: BLE001
            pass
        try:
            arm.disconnect()
        except Exception:  # noqa: BLE001
            pass

    if calib:
        _save(calib, args.out)
        print(f"saved -> {args.out}\n")
    else:
        print("nothing captured — not saving\n")
    print("--- apply these ---")
    if "z_floor" in calib:
        print(f"  rtde_arm z_floor / eval_hw --z-floor : {calib['z_floor']}")
    if "pos_deltas" in calib:
        print(f"  ur5e_lerobot/teleop/manual_panel.py  POS_DELTAS = {calib['pos_deltas']}")
    if "no_go_xy" in calib:
        print(f"  ur5e_lerobot/robot/workspace.py      NO_GO_XY = {tuple(calib['no_go_xy'])}")


if __name__ == "__main__":
    main()
