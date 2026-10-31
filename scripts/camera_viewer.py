"""Quick look at the camera service from any box on the LAN — no robot, no ROS.

    ~/VLA/run.sh scripts/camera_viewer.py            # stats only (works headless / over ssh)
    ~/VLA/run.sh scripts/camera_viewer.py --save     # also write one PNG per camera to /tmp
    ~/VLA/run.sh scripts/camera_viewer.py --control 192.168.11.80 --status

Prints per-camera fps / latency / drops once a second, which is the first thing to check when a
consumer says "no frames": if this sees them, the service is fine and the problem is the consumer.
"""
from __future__ import annotations

import argparse
import json
import time

from ur5e_lerobot.sensors.camera_stream import (CAM_IDS, CONTROL_PORT, MCAST_GROUP, MCAST_PORT,
                                                 CameraSubscriber, ControlClient, local_ip_toward)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default=MCAST_GROUP)
    ap.add_argument("--port", type=int, default=MCAST_PORT)
    ap.add_argument("--seconds", type=float, default=10.0, help="how long to watch (0 = forever)")
    ap.add_argument("--save", action="store_true", help="write /tmp/cam_<name>.png from the latest frame")
    ap.add_argument("--host", default="127.0.0.1",
                    help="camera service host: picks the interface to join the multicast group on "
                         "(a multi-homed box otherwise joins on its default-route NIC and sees "
                         "nothing), and is where --status asks")
    ap.add_argument("--iface-ip", default=None, help="override the interface chosen from --host")
    ap.add_argument("--status", action="store_true", help="print the service's status JSON and exit")
    ap.add_argument("--squares", action="store_true", help="print each camera's current square crop and exit")
    ap.add_argument("--square", nargs=2, metavar=("CAM", "AT"),
                    help="move CAM's square crop to AT (0.0 top/left .. 1.0 bottom/right, or 'none') LIVE, "
                         "no service restart; then look at http://<host>:5602/CAM.raw.jpg. NOT persisted: "
                         "write the value into cameras.py once chosen")
    args = ap.parse_args()

    if args.status:
        print(json.dumps(ControlClient(args.host, CONTROL_PORT).call(cmd="status"), indent=2))
        return
    if args.squares:
        sq = ControlClient(args.host, CONTROL_PORT).call(cmd="get_square")["squares"]
        for n, at in sq.items():
            print(f"  {n:6s} square={at}   raw view: http://{args.host}:5602/{n}.raw.jpg")
        return
    if args.square:
        cam, at = args.square
        at_v = None if at.lower() in ("none", "off") else float(at)
        r = ControlClient(args.host, CONTROL_PORT).call(cmd="set_square", cam=cam, at=at_v)
        print(f"  {cam}: square -> {r['square']}   ({r['note']})   check http://{args.host}:5602/{cam}.raw.jpg")
        return

    iface = args.iface_ip or local_ip_toward(args.host)
    sub = CameraSubscriber(args.group, args.port, iface_ip=iface, decode=args.save)
    print(f"listening on {args.group}:{args.port} via {iface} …", flush=True)
    t0 = time.time()
    prev = {n: 0 for n in CAM_IDS}
    try:
        while not args.seconds or time.time() - t0 < args.seconds:
            time.sleep(1.0)
            parts = []
            for n in CAM_IDS:
                s = sub.stats[n]
                rec = sub.latest(n)
                fps = s["frames"] - prev[n]
                prev[n] = s["frames"]
                if rec is None:
                    parts.append(f"{n}: —")
                else:
                    parts.append(f"{n}: {fps:2d}fps {rec['w']}x{rec['h']} {len(rec['jpeg']) // 1024}KB "
                                 f"lat {s['latency_ms']:.0f}ms drop {s['dropped']}")
            print("  ".join(parts), flush=True)
    except KeyboardInterrupt:
        pass
    if args.save:
        import cv2
        for n in CAM_IDS:
            rec = sub.latest(n)
            if rec is not None and rec["rgb"] is not None:
                p = f"/tmp/cam_{n}.png"
                cv2.imwrite(p, cv2.cvtColor(rec["rgb"], cv2.COLOR_RGB2BGR))
                print(f"wrote {p}")
    sub.close()


if __name__ == "__main__":
    main()
