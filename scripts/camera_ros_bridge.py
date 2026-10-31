"""ROS 2 bridge: republish the camera service's UDP stream as sensor_msgs topics.

Runs under the SYSTEM python (rclpy lives there, not in the VLA venv); the only VLA import is the
stream module, which needs nothing beyond numpy + cv2:

    source /opt/ros/humble/setup.bash
    cd ~/VLA && PYTHONPATH=.:$PYTHONPATH python3 scripts/camera_ros_bridge.py [--raw] [--ns /vla/cameras]

(APPEND to PYTHONPATH — setup.bash puts the ROS site-packages there, and `PYTHONPATH=.` alone
replaces it, which fails with "No module named rclpy".)

Topics (per camera name scene / wrist / side):
    <ns>/<name>/image/compressed   sensor_msgs/CompressedImage  format=jpeg   (always)
    <ns>/<name>/image              sensor_msgs/Image            rgb8          (--raw)

CompressedImage is the default because the wire payload IS a JPEG: the bytes are forwarded as-is,
no decode, no re-encode, no cv_bridge — ~0 CPU per frame, and Foxglove / rqt / image_transport all
render it natively. `--raw` decodes to rgb8 for nodes that need pixels, at ~3 ms/frame.

QoS is SensorData (best-effort, depth 1): a late camera frame is worthless, so never queue them.
"""
from __future__ import annotations

import argparse
import array
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import CompressedImage, Image

from ur5e_lerobot.sensors.camera_stream import (CAM_IDS, MCAST_GROUP, MCAST_PORT,
                                                 CameraSubscriber, decode_jpeg)


def make_qos(kind: str) -> QoSProfile:
    """'reliable' (default) or 'best-effort', both KEEP_LAST depth 1.

    Best-effort is the textbook choice for sensor streams, but these samples are 30-120 KB and DDS
    fragments them into many UDP datagrams; with best-effort, ONE lost fragment loses the whole
    frame, while reliable + depth 1 retransmits fragments yet still replaces an unsent frame with
    the newest, so it cannot build a backlog. NB an earlier A/B here showed ~9 Hz for BOTH modes —
    that was the uint8[] setter cost in _worker, not QoS; re-measure before concluding anything
    about QoS from old numbers.
    """
    if kind == "best-effort":
        return qos_profile_sensor_data
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)


class CameraBridge(Node):
    def __init__(self, sub: CameraSubscriber, ns: str, raw: bool, qos: QoSProfile):
        super().__init__("vla_camera_bridge")
        self.sub, self.raw = sub, raw
        ns = ns.rstrip("/")
        self.pub_c = {n: self.create_publisher(CompressedImage, f"{ns}/{n}/image/compressed", qos)
                      for n in CAM_IDS}
        self.pub_r = ({n: self.create_publisher(Image, f"{ns}/{n}/image", qos)
                       for n in CAM_IDS} if raw else {})
        self.count = {n: 0 for n in CAM_IDS}
        self.t_log = time.time()
        self.create_timer(30.0, self.log)
        # Event-driven, one worker per camera: each blocks on the subscriber's condition variable and
        # publishes the moment a frame completes, instead of a timer polling for new frame ids —
        # no polling latency (up to a tick) and no work between frames. rclpy publish is
        # thread-safe. (The 10 fps / 150 ms seen with the first, timer-based version was NOT the
        # polling; it was the uint8[] setter cost below. Keep the event-driven shape anyway.)
        self._threads = [threading.Thread(target=self._worker, args=(n,), daemon=True, name=f"bridge-{n}")
                         for n in CAM_IDS]
        for t in self._threads:
            t.start()
        self.get_logger().info(f"bridging {MCAST_GROUP}:{MCAST_PORT} -> {ns}/<cam>/image/compressed"
                               f"{' + /image (rgb8)' if raw else ''}")

    def _worker(self, name: str) -> None:
        last = None
        while rclpy.ok():
            if not self.sub.wait(name, timeout=1.0, after=last):
                continue                      # no frame yet / stream paused — keep waiting
            rec = self.sub.latest(name)
            last = rec["frame_id"]
            self.count[name] += 1
            msg = CompressedImage()
            msg.header.stamp = _stamp(rec["ts"])   # capture time on the publisher, not bridge time
            msg.header.frame_id = f"vla_{name}_camera"
            msg.format = "jpeg"
            # array.array('B'), NEVER bytes. rclpy's generated setter for uint8[] takes an
            # array.array with typecode 'B' as-is, but anything else goes through a per-element
            # `assert isinstance(v, int) and 0 <= v < 256` under __debug__: measured 69.75 ms per
            # 120 KB frame against 0.00 ms for the array. At 45 frames/s that is ~3 s of GIL per
            # second — the bridge pinned the interpreter, starved its own UDP receive thread, and
            # delivered ~10 fps of a 15 fps stream with ~150 ms latency. Publishing is 0.13 ms.
            msg.data = array.array("B", rec["jpeg"])
            self.pub_c[name].publish(msg)
            if self.raw:
                rgb = rec["rgb"] if rec["rgb"] is not None else decode_jpeg(rec["jpeg"])
                im = Image()
                im.header = msg.header
                im.height, im.width = rgb.shape[:2]
                im.encoding = "rgb8"
                im.is_bigendian = 0
                im.step = im.width * 3
                im.data = array.array("B", rgb.tobytes())   # same setter trap, 1.5 MB per frame
                self.pub_r[name].publish(im)

    def log(self) -> None:
        dt = time.time() - self.t_log
        self.t_log = time.time()
        line = "  ".join(f"{n}: {self.count[n] / dt:.1f}fps lat {self.sub.stats[n]['latency_ms']:.0f}ms "
                         f"drop {self.sub.stats[n]['dropped']}" for n in CAM_IDS)
        self.count = {n: 0 for n in CAM_IDS}
        self.get_logger().info(line)


def _stamp(ts: float):
    from builtin_interfaces.msg import Time
    s = int(ts)
    return Time(sec=s, nanosec=int((ts - s) * 1e9))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", default="/vla/cameras")
    ap.add_argument("--raw", action="store_true", help="also publish decoded rgb8 Image topics")
    ap.add_argument("--qos", choices=["reliable", "best-effort"], default="reliable",
                    help="reliable (default) survives DDS fragmenting these 30-120 KB samples; "
                         "best-effort measured ~40%% frame loss on loopback. Subscribers must match.")
    ap.add_argument("--group", default=MCAST_GROUP)
    ap.add_argument("--port", type=int, default=MCAST_PORT)
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=sys.argv[:1] + ros_args)
    sub = CameraSubscriber(args.group, args.port, decode=args.raw)   # no decode unless someone needs pixels
    node = CameraBridge(sub, args.ns, args.raw, make_qos(args.qos))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
