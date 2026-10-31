"""The camera service: owns the three USB cameras and streams them over UDP multicast, with a TCP
control port for exposure. Runs on the robot PC under supervisor as `vla_cameras`.

    ~/VLA/run.sh scripts/camera_server.py [--fps 15] [--quality 90] [--no-side]
                                          [--unicast HOST:PORT ...] [--iface-ip 192.168.11.80]

Consumers:
    ur5e_lerobot.sensors.camera_stream.CameraSubscriber         (any python with cv2 + numpy)
    make_engine("hardware", cameras="udp")                      (teleop panel / eval_hw)
    scripts/camera_ros_bridge.py                                (ROS 2 CompressedImage topics)
    scripts/camera_viewer.py                                    (quick look from any box)

Geometry: frames are published at the dataset box size (--width/--height, default 960x540) after
cameras.py's per-camera ROTATE and SQUARE crop — the scene/side cams come out 540x540 and the wrist
960x540, exactly what the dataset stores and the policy was trained on. Subscribers only resize.

Exposure: the service starts on the driver's auto, like a fresh UsbCamera would. The operator's
launch-time flow (auto-expose -> trim -> SET) now talks to this process over the control port, and
the locked value persists here until the next SET or a service restart — so an eval launched after a
recording session sees the exposure the data was recorded at, instead of re-metering from scratch.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

from ur5e_lerobot.sensors import UsbCamera
from ur5e_lerobot.sensors.camera_stream import (CONTROL_PORT, MCAST_GROUP, MCAST_PORT, MJPEG_PORT,
                                                 CameraPublisher, ControlServer, LockedCamera,
                                                 MjpegServer)
from ur5e_lerobot.sensors.cameras import (SCENE_AUTO_EXPOSURE, SCENE_CAM, SCENE_EXPOSURE, SCENE_GAIN,
                                          SCENE_ROTATE, SCENE_SQUARE, SIDE_AUTO_EXPOSURE, SIDE_CAM,
                                          SIDE_EXPOSURE, SIDE_GAIN, SIDE_ROTATE, SIDE_SQUARE,
                                          WRIST_AUTO_EXPOSURE, WRIST_CAM, WRIST_EXPOSURE, WRIST_GAIN,
                                          WRIST_ROTATE, WRIST_SQUARE, ExposureSession)


def _open(name: str, dev, rotate, gain, auto, expo, square, retries: int = 5) -> LockedCamera:
    """Open one camera, retrying briefly — at boot the USB bus can enumerate after supervisor starts us.

    A camera that still will not open is served UNCONNECTED rather than failing the whole service:
    the publisher's grab loop re-opens it by its by-path link every few seconds for as long as it
    runs, so the other cameras keep streaming meanwhile and the missing one comes back the moment it
    is plugged in. (Before 2026-10-27 one missing camera blocked the service for 60 s, then killed it,
    and supervisor restarted it into the same loop: the two live cameras were down too.)"""
    cam = UsbCamera(dev, rotate=rotate, gain=gain, square=square, auto_exposure=auto, exposure=expo)
    last = None
    for i in range(retries):
        try:
            cam.connect()
            print(f"[cam] {name}: {dev} open (rotate={rotate} square={square})", flush=True)
            return LockedCamera(cam)
        except Exception as e:  # noqa: BLE001
            last = e
            if i == 0 or i % 10 == 9:
                print(f"[cam] {name}: {e} — retrying ({i + 1}/{retries})", flush=True)
            time.sleep(2.0)
    print(f"[cam] {name}: could not open {dev} after {retries} tries ({last}) — serving WITHOUT it; the "
          f"publisher keeps re-opening it, plug it in and it comes back", flush=True)
    return LockedCamera(cam)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=960, help="box the frames are fit into (dataset size)")
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=float, default=15.0, help="max publish rate per camera (0 = every grab)")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality on the wire")
    ap.add_argument("--pace-us", type=float, default=60.0,
                    help="gap between a frame's UDP chunks; keeps 3-camera bursts under the kernel's "
                         "208 KB default socket buffer (0 = send as fast as possible)")
    ap.add_argument("--group", default=MCAST_GROUP)
    ap.add_argument("--port", type=int, default=MCAST_PORT)
    ap.add_argument("--control-port", type=int, default=CONTROL_PORT)
    ap.add_argument("--http-port", type=int, default=MJPEG_PORT,
                    help="HTTP MJPEG for VLC/browsers: http://<host>:PORT/scene.mjpg (0 = off)")
    ap.add_argument("--unicast", action="append", default=[], metavar="HOST:PORT",
                    help="also send to this host (for subscribers not on the multicast segment)")
    ap.add_argument("--iface-ip", default=None, help="egress interface for multicast (default: route)")
    ap.add_argument("--no-side", action="store_true", help="skip the side camera")
    ap.add_argument("--scene-cam", default=SCENE_CAM)
    ap.add_argument("--wrist-cam", default=WRIST_CAM)
    ap.add_argument("--side-cam", default=SIDE_CAM)
    ap.add_argument("--stats-every", type=float, default=30.0, help="seconds between stats lines (0 = off)")
    args = ap.parse_args()

    specs = [("scene", args.scene_cam, SCENE_ROTATE, SCENE_GAIN, SCENE_AUTO_EXPOSURE, SCENE_EXPOSURE, SCENE_SQUARE),
             ("wrist", args.wrist_cam, WRIST_ROTATE, WRIST_GAIN, WRIST_AUTO_EXPOSURE, WRIST_EXPOSURE, WRIST_SQUARE)]
    if not args.no_side:
        specs.append(("side", args.side_cam, SIDE_ROTATE, SIDE_GAIN, SIDE_AUTO_EXPOSURE, SIDE_EXPOSURE, SIDE_SQUARE))
    cams = {name: _open(name, *rest) for name, *rest in specs}

    unicast = []
    for u in args.unicast:
        host, _, port = u.partition(":")
        unicast.append((host, int(port or args.port)))

    pub = CameraPublisher(cams, args.width, args.height, fps=args.fps, quality=args.quality,
                          group=args.group, port=args.port, unicast=unicast, iface_ip=args.iface_ip,
                          pace_us=args.pace_us).start()
    exposure = ExposureSession(cams)       # drives the LockedCameras directly, server-side
    ctrl = ControlServer(pub, exposure, port=args.control_port).start()
    http = MjpegServer(pub, port=args.http_port).start() if args.http_port else None
    print(f"[server] publishing {list(cams)} -> {args.group}:{args.port}"
          f"{' + ' + str(unicast) if unicast else ''}  at <= {args.fps} fps, q{args.quality}; "
          f"control on :{args.control_port}"
          f"{f'; mjpeg http://0.0.0.0:{args.http_port}/<cam>.mjpg' if http else ''}", flush=True)

    stop = {"now": False}

    def _sig(*_):
        stop["now"] = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    prev = {n: 0 for n in cams}
    t_prev = time.time()
    try:
        while not stop["now"]:
            time.sleep(1.0)
            if args.stats_every and time.time() - t_prev >= args.stats_every:
                dt = time.time() - t_prev
                line = "  ".join(f"{n}: {(pub.stats[n]['frames'] - prev[n]) / dt:.1f}fps "
                                 f"grab {pub.stats[n]['grab_ms']:.0f}ms err {pub.stats[n]['errors']}"
                                 for n in cams)
                print(f"[stats] {line}  | exposure {exposure.summary()}"
                      f"{' LOCKED' if exposure.locked else ' auto'}", flush=True)
                prev = {n: pub.stats[n]["frames"] for n in cams}
                t_prev = time.time()
    finally:
        print("[server] stopping", flush=True)
        ctrl.stop()
        if http is not None:
            http.stop()
        pub.stop()
        for c in cams.values():
            try:
                c.cam.disconnect()
            except Exception:  # noqa: BLE001
                pass
    sys.exit(0)


if __name__ == "__main__":
    main()
