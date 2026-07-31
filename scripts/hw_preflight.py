"""Staged hardware preflight for the UR5e + AmazingHand. Run ON THE ROBOT PC once the arm, hand,
and cameras are connected. Bring up one piece at a time — e-stop in hand.

    PYTHONPATH=. python scripts/hw_preflight.py --stage arm        # read TCP pose (NO motion)
    PYTHONPATH=. python scripts/hw_preflight.py --stage hand       # open/close cycle
    PYTHONPATH=. python scripts/hw_preflight.py --stage cameras    # grab + save a frame each
    PYTHONPATH=. python scripts/hw_preflight.py --stage all        # arm-read + hand + cameras (no motion)
    PYTHONPATH=. python scripts/hw_preflight.py --stage move --move # TINY arm nudge (explicit opt-in)

Defaults: UR 192.168.11.21, hand 192.168.11.117, scene cam 2, wrist cam 0.
Needs `pip install ur_rtde` on the robot PC for the arm stages.
"""
from __future__ import annotations

import argparse
import os
import time


def stage_arm(ip: str) -> None:
    from ur5e_lerobot.robot.rtde_arm import RtdeArmInterface
    arm = RtdeArmInterface(ip)
    arm.connect()
    print(f"[arm] connected {ip}")
    print(f"[arm] TCP pose : {[round(v, 4) for v in arm.get_ee_pose()]}")
    print(f"[arm] joints   : {[round(v, 4) for v in arm.get_joint_positions()]}")
    arm.disconnect()
    print("[arm] OK (read-only, no motion)")


def stage_move(ip: str, allow: bool) -> None:
    if not allow:
        print("[move] SKIPPED — pass --move to permit a small arm motion. Keep the e-stop in hand.")
        return
    from ur5e_lerobot.robot.rtde_arm import RtdeArmInterface
    arm = RtdeArmInterface(ip)
    arm.connect()
    start = arm.get_ee_pose()
    target = list(start)
    target[2] += 0.02  # +2 cm in z (up, away from the table)
    print(f"[move] start z={start[2]:.4f}; nudging +2cm up (slow servoL), then returning. e-STOP READY.")
    for _ in range(40):  # stream the target via the real teleop path (servoL + no-go + step clamp)
        arm.send_ee_pose(target)
        time.sleep(arm.dt)
    for _ in range(40):
        arm.send_ee_pose(start)
        time.sleep(arm.dt)
    print(f"[move] end z={arm.get_ee_pose()[2]:.4f} (expect ~={start[2]:.4f})")
    arm.disconnect()
    print("[move] OK")


def stage_hand(host: str) -> None:
    from ur5e_lerobot.hand import AmazingHandClient
    with AmazingHandClient(host) as hand:
        print(f"[hand] connected {host}")
        for label, curls in [("open", [0, 0, 0, 0]), ("half", [0.5] * 4),
                              ("close", [1, 1, 1, 1]), ("open", [0, 0, 0, 0])]:
            hand.send_curls(curls)
            print(f"[hand] -> {label}")
            time.sleep(1.0)
    print("[hand] OK")


def stage_cameras(scene_i, wrist_i) -> None:
    import cv2
    from ur5e_lerobot.sensors import UsbCamera
    from ur5e_lerobot.sensors.cameras import SCENE_CAM, WRIST_CAM
    scene_i = SCENE_CAM if scene_i is None else scene_i
    wrist_i = WRIST_CAM if wrist_i is None else wrist_i
    os.makedirs("outputs", exist_ok=True)
    for name, idx in [("scene", scene_i), ("wrist", wrist_i)]:
        cam = UsbCamera(idx)
        try:
            cam.connect()
            img = cam.read(320, 240)
            cv2.imwrite(f"outputs/preflight_{name}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"[cam] {name} (idx {idx}): {img.shape} -> outputs/preflight_{name}.png")
        except Exception as e:  # noqa: BLE001
            print(f"[cam] {name} (idx {idx}) FAILED: {e}")
        finally:
            cam.disconnect()
    print("[cameras] done")


def main() -> None:
    ap = argparse.ArgumentParser(description="Staged UR5e + AmazingHand hardware preflight.")
    ap.add_argument("--stage", choices=["arm", "move", "hand", "cameras", "all"], default="all")
    ap.add_argument("--robot-ip", default="192.168.11.21")
    ap.add_argument("--hand-host", default="192.168.11.117")
    ap.add_argument("--scene-cam", default=None, help="index or /dev path (default: stable by-id 4K cam)")
    ap.add_argument("--wrist-cam", default=None, help="index or /dev path (default: stable by-id ARC cam)")
    ap.add_argument("--move", action="store_true", help="permit the small arm motion in the move stage")
    args = ap.parse_args()

    s = args.stage
    runs = []
    if s in ("arm", "all"):
        runs.append(("arm", lambda: stage_arm(args.robot_ip)))
    if s in ("hand", "all"):
        runs.append(("hand", lambda: stage_hand(args.hand_host)))
    if s in ("cameras", "all"):
        runs.append(("cameras", lambda: stage_cameras(args.scene_cam, args.wrist_cam)))
    if s == "move":
        runs.append(("move", lambda: stage_move(args.robot_ip, args.move)))

    for name, fn in runs:  # one stage failing doesn't abort the rest
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
