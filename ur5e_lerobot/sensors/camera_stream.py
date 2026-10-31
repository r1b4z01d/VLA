"""Camera streaming: ONE process owns the USB cameras and publishes frames over UDP; anything else
(teleop panel, eval, the ROS bridge, a debug viewer, the GPU box) subscribes.

Why a separate process
----------------------
Today every consumer opens the three UsbCameras itself. That means: only one program can run at a
time (V4L2 devices are exclusive), each launch pays the ~7 s connect + auto-exposure warm-up, the
exposure the operator dialled in dies with the process, and nothing else — Foxglove, a recorder, a
second screen — can see the feeds. Owning the cameras in a long-lived supervisor service fixes all of
that: the exposure survives between runs, consumers attach and detach freely, and a camera fault is
one process to restart instead of the whole teleop session.

Transport
---------
Pure-stdlib UDP multicast. The robot PC has no internet and no zmq/msgpack, so the wire format is
just `struct` + JPEG from cv2 (which every consumer already has). A JPEG frame (29 KB for the 540²
scene cam at q90, 50 KB for the 960x540 wrist) is split into <=1400-byte chunks, each carrying a
26-byte header, so every datagram fits a 1500 MTU without IP fragmentation. Three cameras at 15 fps
is ~2 MB/s and ~1.5k datagrams/s — nothing on the gigabit link. A lost chunk drops THAT frame; the
next arrives within a frame period, which is the right trade for a live feed (TCP would stall on
retransmit instead). Multicast lets N subscribers join without the publisher knowing; unicast
targets can be added for hosts off the local segment.

JPEGs on the wire are REAL JPEGs (BGR-encoded), not the channel-swapped trick remote.py uses, so a
ROS/Foxglove/browser consumer decoding them sees correct colours. The cost is one cvtColor per frame
on each side (~1 ms).

Control
-------
A small TCP JSON-lines server on the publisher answers exposure commands (auto-expose, nudge, lock,
per-camera set/read) and status. Exposure state therefore lives IN the service, which is what makes
it survive across teleop/eval launches. `CameraProxy` presents each remote camera with the same
four methods `ExposureSession` drives on a UsbCamera, and `RemoteExposureSession` mirrors the
ExposureSession interface while delegating to the service, so the panel and eval keep their exact
launch-time flow (auto -> trim with two deck keys -> SET) with the work done server-side at grab
speed, all three cameras in parallel.

Frame geometry is the publisher's job: it applies cameras.py's ROTATE / SQUARE / exposure constants,
so every subscriber sees identical, dataset-geometry frames. Subscribers only resize.
"""
from __future__ import annotations

import json
import socket
import struct
import threading
import time
from typing import Callable

import cv2
import numpy as np

# ---------------------------------------------------------------------------------------------
# wire format
# ---------------------------------------------------------------------------------------------
MCAST_GROUP = "239.42.11.1"
MCAST_PORT = 5600
CONTROL_PORT = 5601
MJPEG_PORT = 5602      # HTTP multipart MJPEG, for VLC / browsers / Foxglove (see MjpegServer)
MAGIC = b"VCAM"
ENC_JPEG = 1
# <=1400 B payload + 26 B header + 28 B IP/UDP stays under a 1500 MTU; no IP fragmentation, which on
# Linux would otherwise make a single lost fragment cost the whole datagram.
CHUNK = 1400

# cam id on the wire <-> name. Fixed so the protocol has no handshake; add cameras at the end.
CAM_IDS = {"scene": 0, "wrist": 1, "side": 2}
CAM_NAMES = {v: k for k, v in CAM_IDS.items()}

# magic, cam_id, frame_id, timestamp, n_chunks, idx, w, h, enc
_HDR = struct.Struct("!4sBIdHHHHB")
HDR_LEN = _HDR.size  # 26


def pack_chunks(cam_id: int, frame_id: int, ts: float, w: int, h: int, payload: bytes) -> list:
    """Split one encoded frame into datagrams."""
    n = max(1, (len(payload) + CHUNK - 1) // CHUNK)
    return [_HDR.pack(MAGIC, cam_id, frame_id & 0xFFFFFFFF, ts, n, i, w, h, ENC_JPEG)
            + payload[i * CHUNK:(i + 1) * CHUNK] for i in range(n)]


def unpack_chunk(datagram: bytes):
    """-> (cam_id, frame_id, ts, n_chunks, idx, w, h, enc, payload) or None if not ours."""
    if len(datagram) < HDR_LEN:
        return None
    magic, cam_id, fid, ts, n, idx, w, h, enc = _HDR.unpack_from(datagram)
    if magic != MAGIC or idx >= n:
        return None
    return cam_id, fid, ts, n, idx, w, h, enc, datagram[HDR_LEN:]


def encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def decode_jpeg(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("jpeg decode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fit_resize(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize to fit inside (w, h), aspect preserved, NO padding — the publisher already applied the
    square crop / letterbox, so a subscriber must never change the geometry again."""
    ih, iw = rgb.shape[:2]
    if (iw, ih) == (w, h):
        return rgb
    s = min(w / iw, h / ih)
    return np.ascontiguousarray(cv2.resize(rgb, (max(1, round(iw * s)), max(1, round(ih * s)))))


# ---------------------------------------------------------------------------------------------
# publisher side
# ---------------------------------------------------------------------------------------------
class LockedCamera:
    """A UsbCamera behind a lock. cv2.VideoCapture is not thread-safe, and the grabber thread and the
    exposure routines (auto_expose reads frames to meter) both touch it."""

    def __init__(self, cam):
        self.cam = cam
        self.lock = threading.Lock()

    def read(self, w: int, h: int):
        with self.lock:
            return self.cam.read(w, h)

    def read_exposure(self):
        with self.lock:
            return self.cam.read_exposure()

    def set_exposure(self, value: float) -> float:
        with self.lock:
            return self.cam.set_exposure(value)

    def set_auto_exposure(self) -> None:
        with self.lock:
            self.cam.set_auto_exposure()

    @property
    def is_connected(self) -> bool:
        return self.cam.is_connected

    # --- square crop, tunable live (UsbCamera.read consults .square on every grab) -------------
    @property
    def square(self):
        return getattr(self.cam, "square", None)

    def set_square(self, at) -> None:
        """Move the square crop (fraction along the long axis; None = off). Takes effect on the next
        grab. The published frame stays the same SHAPE (a square is a square), so subscribers and the
        dataset declaration are unaffected — only the field of view moves."""
        with self.lock:
            self.cam.square = None if at is None else min(max(float(at), 0.0), 1.0)

    @property
    def last_full(self):
        """The upright, UNcropped frame from the last grab (UsbCamera.last_full), or None."""
        return getattr(self.cam, "last_full", None)


class CameraPublisher:
    """Grab each camera in its own thread, JPEG it, multicast it.

    One thread per camera matters: a grab is ~45 ms (USB MJPEG decode), so three in one loop would
    serialise to ~135 ms per round — slower than the 83 ms control period the policy runs at. In
    parallel, each camera streams at its own native rate and `fps` only rate-limits what goes on
    the wire.
    """

    def __init__(self, cams: "dict[str, LockedCamera]", width: int, height: int, fps: float = 15.0,
                 quality: int = 90, group: str = MCAST_GROUP, port: int = MCAST_PORT,
                 unicast: "list[tuple[str, int]] | None" = None, iface_ip: "str | None" = None,
                 ttl: int = 1, pace_us: float = 60.0):
        self.cams, self.w, self.h = cams, width, height
        self.period = 1.0 / fps if fps and fps > 0 else 0.0
        self.quality = quality
        # Gap between a frame's chunks. Without it a 120 KB frame is 86 datagrams in ~1 ms, and three
        # cameras can coincide — ~360 KB landing in a receiver whose socket buffer is the kernel
        # default of 208 KB (net.core.rmem_max) unless raised. 60 us/chunk stretches a frame over
        # ~5 ms so a burst cannot outrun a drain; cost ~5 ms latency. This is insurance, not a
        # measured fix: the drops seen on the robot PC turned out to be the ROS bridge pinning the
        # GIL (see camera_ros_bridge.py), and the system-wide UDP error counter there runs at ~38k/s
        # from other stacks regardless, so it cannot be used to judge this stream.
        self.pace_s = max(0.0, float(pace_us)) * 1e-6
        self.targets = [(group, port)] + list(unicast or [])
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        if iface_ip:   # pin the egress NIC (docker0 is also multicast-capable; default route wins otherwise)
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(iface_ip))
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list = []
        # latest frame per camera, for the control server (status thumbnails, metering)
        self.latest: "dict[str, tuple[np.ndarray, int, float]]" = {}
        # latest ENCODED frame per camera + a condition so MjpegServer can hand VLC/browsers the
        # exact bytes that went on the wire the moment they exist — no second encode, no polling
        self.latest_jpeg: "dict[str, tuple[bytes, int, float]]" = {}
        self._frame_cv = threading.Condition()
        self.stats = {n: {"frames": 0, "bytes": 0, "errors": 0, "last_ts": 0.0, "grab_ms": 0.0}
                      for n in cams}

    def wait_jpeg(self, name: str, after: "int | None", timeout: float):
        """Block until `name` has an encoded frame newer than frame id `after`. -> (jpeg, fid, ts) or None."""
        deadline = time.time() + timeout
        with self._frame_cv:
            while True:
                rec = self.latest_jpeg.get(name)
                if rec is not None and (after is None or rec[1] != after):
                    return rec
                left = deadline - time.time()
                if left <= 0:
                    return None
                self._frame_cv.wait(left)

    def start(self) -> "CameraPublisher":
        for name, cam in self.cams.items():
            t = threading.Thread(target=self._loop, args=(name, cam), daemon=True, name=f"grab-{name}")
            t.start()
            self._threads.append(t)
        return self

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)

    # Consecutive read failures before the camera is closed and re-opened. A USB camera that drops
    # off the bus (wrist: cable flexes with the arm; VIDIOC_REQBUFS errno 19) re-enumerates under a
    # new /dev/videoN; the old VideoCapture handle then fails forever. Re-opening by the stable
    # by-path link picks up the new node. ~2 s of failures at 0.1 s per retry.
    REOPEN_AFTER = 20
    REOPEN_EVERY_S = 3.0

    def _loop(self, name: str, cam: LockedCamera) -> None:
        cam_id = CAM_IDS[name]
        fid = 0
        next_send = 0.0
        st = self.stats[name]
        fails = 0
        last_reopen = 0.0
        reopen_fails = 0          # consecutive failed re-opens: logged on the 1st, then every 20th (~1/min)
        while not self._stop.is_set():
            try:
                t0 = time.time()
                rgb = cam.read(self.w, self.h)          # blocks until the camera has a new frame
                st["grab_ms"] = (time.time() - t0) * 1000
                if fails >= self.REOPEN_AFTER:
                    print(f"[pub] {time.strftime('%H:%M:%S')} {name}: frames back after {fails} failed reads",
                          flush=True)
                fails = 0
            except Exception as e:  # noqa: BLE001 — a dropped USB frame must not kill the stream
                st["errors"] += 1
                fails += 1
                if fails == 1 or fails % 500 == 0:
                    print(f"[pub] {time.strftime('%H:%M:%S')} {name}: read failed ({e}) x{fails}", flush=True)
                if fails >= self.REOPEN_AFTER and time.time() - last_reopen >= self.REOPEN_EVERY_S:
                    last_reopen = time.time()
                    try:
                        with cam.lock:
                            cam.cam.disconnect()
                            cam.cam.connect()       # re-resolves the by-path link; re-applies exposure/gain
                        print(f"[pub] {time.strftime('%H:%M:%S')} {name}: re-opened {cam.cam.device}"
                              + (f" after {reopen_fails} failed attempts" if reopen_fails else ""), flush=True)
                        reopen_fails = 0
                    except Exception as e2:  # noqa: BLE001 — still unplugged: try again in REOPEN_EVERY_S
                        reopen_fails += 1
                        if reopen_fails == 1 or reopen_fails % 20 == 0:
                            print(f"[pub] {time.strftime('%H:%M:%S')} {name}: re-open failed ({e2}) x{reopen_fails}",
                                  flush=True)
                time.sleep(0.1)
                continue
            now = time.time()
            self.latest[name] = (rgb, fid, now)
            if self.period and now < next_send:
                continue                                 # rate-limit the wire, not the grab
            # Advance the schedule by whole periods, NOT to now+period. With a ~52 ms grab against a
            # 66.7 ms (15 fps) period, now+period published every SECOND grab — 9.6 fps measured on the
            # rig for a 15 fps setting. Accumulating keeps the long-run rate at the target; the clamp
            # stops a burst after a stall (a USB hiccup would otherwise "catch up" several frames).
            next_send = max(next_send + self.period, now - self.period) if next_send else now + self.period
            try:
                payload = encode_jpeg(rgb, self.quality)
            except Exception as e:  # noqa: BLE001
                st["errors"] += 1
                print(f"[pub] {name}: encode failed ({e})", flush=True)
                continue
            with self._frame_cv:                         # hand the encoded bytes to MjpegServer
                self.latest_jpeg[name] = (payload, fid, now)
                self._frame_cv.notify_all()
            h, w = rgb.shape[:2]
            chunks = pack_chunks(cam_id, fid, now, w, h, payload)
            with self._send_lock:
                for tgt in self.targets:
                    for c in chunks:
                        try:
                            self.sock.sendto(c, tgt)
                        except OSError as e:
                            st["errors"] += 1
                            if st["errors"] % 50 == 1:
                                print(f"[pub] {name}: send to {tgt} failed ({e})", flush=True)
                            break
                        if self.pace_s:
                            time.sleep(self.pace_s)   # see pace_us: keep bursts under the rx buffer
            st["frames"] += 1
            st["bytes"] += len(payload)
            st["last_ts"] = now
            fid += 1


class ControlServer:
    """TCP JSON-lines: one request object per line, one reply per line. Threaded, many clients.

    Exposure commands run against the publisher's cameras through an `ExposureSession`, serialised
    by a lock so two clients can't interleave a ladder search. The search itself is the unchanged
    cameras.ExposureSession logic — one camera at a time, each ladder step settling on real grabs —
    so it costs ~12 s for three cameras on this rig. What the service adds is that the result
    OUTLIVES the client: lock once, and every later teleop/eval launch starts from it.
    """

    def __init__(self, pub: CameraPublisher, exposure, host: str = "0.0.0.0", port: int = CONTROL_PORT):
        self.pub, self.exposure = pub, exposure
        self.host, self.port = host, port
        self._lock = threading.Lock()        # exposure ops are not reentrant
        self._stop = threading.Event()
        self.t_start = time.time()

    def start(self) -> "ControlServer":
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(8)
        srv.settimeout(0.5)
        self._srv = srv
        threading.Thread(target=self._accept, daemon=True, name="ctrl-accept").start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        f = conn.makefile("rwb", buffering=0)
        try:
            for line in f:
                try:
                    req = json.loads(line)
                    rep = self.handle(req)
                except Exception as e:  # noqa: BLE001
                    rep = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                f.write((json.dumps(rep, default=float) + "\n").encode())
        except (OSError, ValueError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    # --- command handlers ---------------------------------------------------------------------
    def handle(self, req: dict) -> dict:
        cmd = req.get("cmd")
        ex = self.exposure
        if cmd == "status":
            return {"ok": True, "uptime_s": round(time.time() - self.t_start, 1),
                    "size": [self.pub.w, self.pub.h], "period_s": self.pub.period,
                    "targets": [list(t) for t in self.pub.targets],
                    "exposure": {"values": dict(ex.values), "locked": ex.locked},
                    "cams": {n: {"frames": s["frames"], "bytes": s["bytes"], "errors": s["errors"],
                                 "grab_ms": round(s["grab_ms"], 1),
                                 "age_s": round(time.time() - s["last_ts"], 2) if s["last_ts"] else None,
                                 "shape": list(self.pub.latest[n][0].shape[:2]) if n in self.pub.latest else None}
                             for n, s in self.pub.stats.items()}}
        if cmd == "get_exposure":
            return {"ok": True, "values": dict(ex.values), "locked": ex.locked}
        with self._lock:
            if cmd == "begin":
                ex.begin()
                return {"ok": True, "values": dict(ex.values), "locked": ex.locked}
            if cmd == "adopt_auto":
                return {"ok": True, "values": ex.adopt_auto(), "locked": ex.locked}
            if cmd == "auto_expose":
                vals = ex.auto_expose(self.pub.w, self.pub.h, settle=int(req.get("settle", 2)))
                return {"ok": True, "values": vals, "locked": ex.locked}
            if cmd == "nudge":
                return {"ok": True, "values": ex.nudge(float(req["factor"]), req.get("cam")), "locked": ex.locked}
            if cmd == "lock":
                return {"ok": True, "values": ex.lock(), "locked": ex.locked}
            if cmd == "set_values":
                # apply an explicit {cam: exposure} (e.g. the value a dataset was recorded at) and lock
                for n, v in dict(req["values"]).items():
                    if n in ex.cams:
                        ex.values[n] = float(v)
                return {"ok": True, "values": ex.lock(), "locked": ex.locked}
            if cmd == "get_square":
                return {"ok": True, "squares": {n: c.square for n, c in self.pub.cams.items()}}
            if cmd == "set_square":
                # live crop move: {cam, at} with at in [0,1] or null to disable. NOT persisted — write
                # the chosen value into cameras.py (SCENE_SQUARE / SIDE_SQUARE) or a restart reverts it.
                n = req.get("cam")
                if n not in self.pub.cams:
                    raise KeyError(f"unknown camera {n!r}")
                at = req.get("at")
                self.pub.cams[n].set_square(None if at is None else float(at))
                return {"ok": True, "cam": n, "square": self.pub.cams[n].square,
                        "note": "takes effect on the next frame; persist it in cameras.py"}
            cam = req.get("cam")
            if cmd in ("set_exposure", "read_exposure", "set_auto_exposure"):
                if cam not in self.pub.cams:
                    raise KeyError(f"unknown camera {cam!r}")
                c = self.pub.cams[cam]
                if cmd == "set_exposure":
                    v = c.set_exposure(float(req["value"]))
                    ex.values[cam] = v
                    return {"ok": True, "value": v}
                if cmd == "read_exposure":
                    return {"ok": True, "value": c.read_exposure()}
                c.set_auto_exposure()
                ex.values.pop(cam, None)
                return {"ok": True}
        raise ValueError(f"unknown cmd {cmd!r}")


class MjpegServer:
    """HTTP MJPEG (multipart/x-mixed-replace) for anything that can't speak the multicast format —
    VLC, a browser tab, Foxglove's image panel.

        http://<host>:5602/              index page with all three feeds
        http://<host>:5602/scene.mjpg    stream   (VLC: File > Open Network, or `vlc <url>`)
        http://<host>:5602/scene.jpg     one frame

    Forwards the publisher's encoded bytes unchanged — the frame VLC shows IS the frame that went
    to the policy. One thread per client (ThreadingHTTPServer); each client always gets the newest
    frame and skips any it missed, so a slow viewer lags only itself and never builds a backlog.
    Content-Length per part matters: VLC's mjpeg demux is unreliable without it.
    """

    def __init__(self, pub: CameraPublisher, host: str = "0.0.0.0", port: int = MJPEG_PORT):
        self.pub, self.host, self.port = pub, host, port
        self._httpd = None

    def start(self) -> "MjpegServer":
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        pub = self.pub
        names = list(pub.cams)

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_):      # one line per frame otherwise
                pass

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    return self._index()
                name, _, ext = path.lstrip("/").rpartition(".")
                raw = name.endswith(".raw")
                if raw:
                    name = name[:-4]
                if name not in names or ext not in ("mjpg", "mjpeg", "jpg", "jpeg"):
                    self.send_error(404, f"unknown feed; try {', '.join(n + '.mjpg' for n in names)} "
                                         f"or <cam>.raw.jpg for the uncropped frame")
                    return
                if raw:
                    return self._raw(name)
                if ext in ("jpg", "jpeg"):
                    return self._one(name)
                self._stream(name)

            def _raw(self, name):
                """The upright, UNcropped frame with the current square crop drawn on it — the only
                place the pixels OUTSIDE the square can be seen, hence the only way to choose
                SCENE_SQUARE / SIDE_SQUARE. Long side scaled to <= 720 px for the browser."""
                lc = pub.cams[name]
                full = lc.last_full
                if full is None:
                    self.send_error(503, "no frame yet")
                    return
                img = full.copy()
                h, w = img.shape[:2]
                at = lc.square
                if at is not None:
                    side = min(w, h)
                    if h >= w:
                        y0 = int(round((h - side) * at)); x0 = 0
                    else:
                        x0 = int(round((w - side) * at)); y0 = 0
                    # dim everything outside the square, draw its edge, label it
                    mask = np.zeros((h, w), dtype=bool); mask[y0:y0 + side, x0:x0 + side] = True
                    img[~mask] = (img[~mask] * 0.35).astype(img.dtype)
                    cv2.rectangle(img, (x0, y0), (x0 + side - 1, y0 + side - 1), (255, 220, 0), 4)
                    cv2.putText(img, f"{name} square={at:.2f}  rows {y0}-{y0 + side}" if h >= w
                                else f"{name} square={at:.2f}  cols {x0}-{x0 + side}",
                                (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 220, 0), 2)
                else:
                    cv2.putText(img, f"{name}: no square crop", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 220, 0), 2)
                if max(h, w) > 720:
                    s = 720 / max(h, w)
                    img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))))
                jpeg = encode_jpeg(img, 85)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(jpeg)

            def _index(self):
                body = ("<!doctype html><title>vla cameras</title><body style='background:#111;"
                        "color:#ccc;font-family:sans-serif;margin:8px'>"
                        + "".join(f"<figure style='display:inline-block;margin:4px'><img src='/{n}.mjpg' "
                                  f"style='max-height:45vh'><figcaption>{n} — <a href='/{n}.mjpg' "
                                  f"style='color:#8cf'>{n}.mjpg</a> · <a href='/{n}.jpg' style='color:#8cf'>"
                                  f"{n}.jpg</a> · <a href='/{n}.raw.jpg' style='color:#fc8'>{n}.raw.jpg</a>"
                                  f" <span style='color:#777'>(full frame, crop square drawn)</span>"
                                  f"</figcaption></figure>" for n in names)
                        + "<p style='color:#777;font-size:12px'>raw = the upright uncropped frame with the "
                          "current square crop drawn; move it live with scripts/camera_viewer.py --square CAM AT</p>"
                        + "</body>").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _one(self, name):
                rec = pub.wait_jpeg(name, None, 3.0)
                if rec is None:
                    self.send_error(503, "no frames yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(rec[0])))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(rec[0])

            def _stream(self, name):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                last = None
                try:
                    while True:
                        rec = pub.wait_jpeg(name, last, 5.0)
                        if rec is None:
                            continue                     # stream paused; keep the socket open
                        jpeg, last, _ = rec
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass                                 # viewer closed the window

        httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        httpd.daemon_threads = True
        self._httpd = httpd
        threading.Thread(target=httpd.serve_forever, daemon=True, name="mjpeg-http").start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()


# ---------------------------------------------------------------------------------------------
# subscriber side
# ---------------------------------------------------------------------------------------------
class CameraSubscriber:
    """Join the multicast group, reassemble frames, decode them on the receive thread.

    `read()` is a dict lookup + resize (~1 ms) — the ~45 ms per-camera grab that used to sit inside
    every control step is now the publisher's problem. `latest()` also exposes the raw JPEG bytes so
    a bridge (ROS CompressedImage) can forward frames without re-encoding.
    """

    def __init__(self, group: str = MCAST_GROUP, port: int = MCAST_PORT, iface_ip: str = "0.0.0.0",
                 stale_s: float = 1.0, decode: bool = True):
        self.group, self.port = group, port
        self.stale_s = stale_s
        self.decode = decode
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):          # several subscribers on ONE host share the port
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        sock.bind(("", port))
        mreq = struct.pack("4s4s", socket.inet_aton(group), socket.inet_aton(iface_ip))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(0.5)
        self.sock = sock
        self._partial: dict = {}      # (cam_id, frame_id) -> {"parts": {idx: bytes}, "n": n, "t": t, ...}
        self._latest: "dict[str, dict]" = {}
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self.stats = {n: {"frames": 0, "dropped": 0, "latency_ms": 0.0} for n in CAM_IDS}
        self._thread = threading.Thread(target=self._rx, daemon=True, name="cam-rx")
        self._thread.start()
        # Stop the receive thread BEFORE interpreter teardown. Left running, it can be inside
        # cv2.imdecode when the runtime tears down, which aborts with "terminate called without an
        # active exception" — an ugly crash on every teleop/eval exit, for nothing.
        import atexit
        atexit.register(self.close)

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.5)

    def _rx(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                self._gc()
                continue
            except OSError:
                break
            u = unpack_chunk(data)
            if u is None:
                continue
            cam_id, fid, ts, n, idx, w, h, enc, payload = u
            name = CAM_NAMES.get(cam_id)
            if name is None:
                continue
            key = (cam_id, fid)
            p = self._partial.get(key)
            if p is None:
                # a frame NEWER than this one already completed -> this is late, drop it. Only a SMALL
                # step back counts as late: when the publisher restarts its ids begin at 0 again, and
                # the old rule (any smaller id) then dropped every frame forever — the ROS bridge went
                # silent after each vla_cameras restart until it was restarted itself (2026-10-30).
                lat = self._latest.get(name)
                if lat is not None and fid <= lat["frame_id"] and lat["frame_id"] - fid < 30 and time.time() - lat["rx"] < 2.0:
                    continue
                p = self._partial[key] = {"parts": {}, "n": n, "t": time.time(), "ts": ts, "w": w, "h": h}
            p["parts"][idx] = payload
            if len(p["parts"]) < n:
                continue
            del self._partial[key]
            jpeg = b"".join(p["parts"][i] for i in range(n))
            rec = {"jpeg": jpeg, "frame_id": fid, "ts": ts, "w": w, "h": h, "rx": time.time(), "rgb": None}
            if self.decode:
                try:
                    rec["rgb"] = decode_jpeg(jpeg)
                except Exception:  # noqa: BLE001 — corrupt frame: skip it
                    self.stats[name]["dropped"] += 1
                    continue
            with self._cv:
                prev = self._latest.get(name)
                if prev is not None and fid > prev["frame_id"] + 1:
                    self.stats[name]["dropped"] += fid - prev["frame_id"] - 1
                self._latest[name] = rec
                self.stats[name]["frames"] += 1
                self.stats[name]["latency_ms"] = (rec["rx"] - ts) * 1000
                self._cv.notify_all()
            # a newer frame completed: anything older still partial is dead
            for k in [k for k in self._partial if k[0] == cam_id and k[1] < fid]:
                self.stats[name]["dropped"] += 1
                del self._partial[k]

    def _gc(self) -> None:
        now = time.time()
        for k in [k for k, p in self._partial.items() if now - p["t"] > self.stale_s]:
            del self._partial[k]

    # --- consumer API ---------------------------------------------------------------------------
    def cameras(self) -> "list[str]":
        with self._cv:
            return sorted(self._latest, key=lambda n: CAM_IDS[n])

    def wait(self, name: str, timeout: float = 3.0, after: "int | None" = None) -> bool:
        """Block until `name` has a frame (newer than frame id `after` if given). A frame id far BELOW
        `after` is a publisher that restarted (ids begin at 0 again) and counts as new too — the ROS
        bridge's workers otherwise waited forever after every vla_cameras restart (2026-10-30)."""
        deadline = time.time() + timeout
        with self._cv:
            while True:
                rec = self._latest.get(name)
                if rec is not None and (after is None or rec["frame_id"] > after or after - rec["frame_id"] >= 30):
                    return True
                left = deadline - time.time()
                if left <= 0:
                    return False
                self._cv.wait(left)

    def latest(self, name: str) -> "dict | None":
        with self._cv:
            return self._latest.get(name)

    def read(self, name: str, w: int, h: int, timeout: float = 3.0) -> np.ndarray:
        """Latest decoded frame fit into (w, h). Raises if the camera has never been seen or is stale —
        a silent black frame would be worse: the policy would act on it."""
        if not self.wait(name, timeout):
            raise RuntimeError(f"camera stream {name!r}: no frames on {self.group}:{self.port} "
                               f"(is the camera service running?)")
        rec = self.latest(name)
        if time.time() - rec["rx"] > self.stale_s:
            raise RuntimeError(f"camera stream {name!r}: last frame is {time.time() - rec['rx']:.1f}s old")
        return fit_resize(rec["rgb"], w, h)


class ControlClient:
    """JSON-lines client for ControlServer. One short-lived connection per call keeps it trivially
    reentrant and immune to a half-dead socket after the service restarts."""

    def __init__(self, host: str, port: int = CONTROL_PORT, timeout: float = 60.0):
        self.host, self.port, self.timeout = host, port, timeout

    def call(self, **req) -> dict:
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.sendall((json.dumps(req) + "\n").encode())
            f = s.makefile("rb")
            line = f.readline()
        if not line:
            raise RuntimeError("camera control: connection closed without a reply")
        rep = json.loads(line)
        if not rep.get("ok"):
            raise RuntimeError(f"camera control {req.get('cmd')}: {rep.get('error')}")
        return rep


class CameraProxy:
    """A remote camera with the four methods ExposureSession drives on a UsbCamera, plus `read`.

    `remote_session` is what `ExposureSession.for_cams` keys on to delegate the WHOLE flow to the
    service instead of running the ladder search over the network one frame at a time.
    """

    def __init__(self, name: str, sub: CameraSubscriber, ctrl: ControlClient, remote_session):
        self.name, self.sub, self.ctrl = name, sub, ctrl
        self.remote_session = remote_session
        self.is_connected = True

    def read(self, w: int, h: int) -> np.ndarray:
        """The NEXT frame after this call (not the cached one) — exposure metering needs a frame that
        was grabbed after the setting changed, which is exactly what UsbCamera.read gave it."""
        rec = self.sub.latest(self.name)
        after = rec["frame_id"] if rec else None
        self.sub.wait(self.name, timeout=3.0, after=after)
        return self.sub.read(self.name, w, h)

    def read_exposure(self):
        return self.ctrl.call(cmd="read_exposure", cam=self.name).get("value")

    def set_exposure(self, value: float) -> float:
        return float(self.ctrl.call(cmd="set_exposure", cam=self.name, value=float(value))["value"])

    def set_auto_exposure(self) -> None:
        self.ctrl.call(cmd="set_auto_exposure", cam=self.name)

    # --- square crop, live through the service (capture_meta.apply/squares_of use these) ---------
    @property
    def square(self):
        try:
            return self.ctrl.call(cmd="get_square")["squares"].get(self.name)
        except Exception:  # noqa: BLE001 — service gone / old build without get_square
            return None

    def set_square(self, at) -> None:
        self.ctrl.call(cmd="set_square", cam=self.name, at=at)

    def connect(self):
        return self

    def disconnect(self) -> None:
        pass


class RemoteExposureSession:
    """Same interface as cameras.ExposureSession; every operation runs inside the camera service.

    The service keeps the locked value between launches, so `values`/`locked` on construction
    reflect whatever the last operator left — which is the point: eval sees the exposure teleop
    recorded at, instead of re-metering from scratch each time.
    """

    STEP = 1.25

    def __init__(self, ctrl: ControlClient, names: "list[str]"):
        self.ctrl = ctrl
        self.cams = {n: None for n in names}
        self.values: dict = {}
        self.locked = False
        try:
            self._take(ctrl.call(cmd="get_exposure"))
        except Exception:  # noqa: BLE001 — service not up yet; begin()/auto_expose will say so
            pass

    def _take(self, rep: dict) -> dict:
        self.values = {k: float(v) for k, v in (rep.get("values") or {}).items()}
        self.locked = bool(rep.get("locked"))
        return dict(self.values)

    def begin(self) -> None:
        self._take(self.ctrl.call(cmd="begin"))

    def adopt_auto(self) -> dict:
        return self._take(self.ctrl.call(cmd="adopt_auto"))

    def auto_expose(self, w: int = 0, h: int = 0, settle: int = 2) -> dict:
        return self._take(self.ctrl.call(cmd="auto_expose", settle=settle))

    def nudge(self, factor: float, cam: "str | None" = None) -> dict:
        if cam is None:
            return self._take(self.ctrl.call(cmd="nudge", factor=float(factor)))
        # one camera: set_exposure exists on every service build, so this needs no service restart
        want = max(1.0, float(self.values.get(cam, 78.0)) * float(factor))
        rep = self.ctrl.call(cmd="set_exposure", cam=cam, value=want)
        self.values[cam] = float(rep.get("value", want))
        return dict(self.values)

    def lock(self) -> dict:
        return self._take(self.ctrl.call(cmd="lock"))

    def set_values(self, values: dict) -> dict:
        """Apply {cam: exposure} and lock — e.g. the exposure the dataset was recorded at."""
        return self._take(self.ctrl.call(cmd="set_values", values=values))

    @staticmethod
    def clip_pct(img) -> float:
        nz = img.max(axis=2) > 12
        px = img[nz] if nz.any() else img.reshape(-1, 3)
        return 100.0 * float((px.max(axis=-1) >= 250).mean())

    def summary(self) -> str:
        return "  ".join(f"{k}={v:.0f}" for k, v in sorted(self.values.items())) or "(none)"


def local_ip_toward(host: str) -> str:
    """The local address the kernel would source packets to `host` from — i.e. the interface that
    sits on the publisher's segment. A UDP connect() does no I/O; it only resolves the route.

    This is what a multicast JOIN has to be pinned to. Joining with 0.0.0.0 lets the kernel pick
    the default multicast interface, which on a multi-homed host is the default-route NIC: on the
    GPU box that is enp4s0 (192.168.1.x) while the camera stream arrives on the 192.168.11.x NIC —
    measured as a silent "no frames" with every port and route looking fine. Loopback falls back
    to 0.0.0.0: a same-host subscriber receives the publisher's loopback copy on its EGRESS
    interface, not on lo.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((host, 1))
        ip = s.getsockname()[0]
    except OSError:
        return "0.0.0.0"
    finally:
        s.close()
    return "0.0.0.0" if ip.startswith("127.") else ip


SERVICE_UNIT = "vla_cameras"   # the supervisor unit that owns the cameras (services/rd/vla_cameras.conf)


def _is_local_host(host: str) -> bool:
    """True if `host` is this machine, i.e. the camera service runs under the local supervisor."""
    if host in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
        return True
    try:
        ip = socket.gethostbyname(host)
    except OSError:
        return False
    if ip.startswith("127."):
        return True
    try:
        return local_ip_toward(ip) == ip
    except Exception:  # noqa: BLE001
        return False


def _supervisorctl(host: str, *args: str, timeout: float = 60.0) -> "tuple[int, str]":
    """`sudo -n supervisorctl <args>` on `host`: locally, or over ssh when the service lives on another
    box. Needs passwordless sudo for supervisorctl there (true for rd on the robot PC)."""
    import subprocess
    cmd = ["sudo", "-n", "supervisorctl", *args]
    if not _is_local_host(host):
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, " ".join(cmd)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)
    return r.returncode, (r.stdout + r.stderr).strip()


def service_status(host: str, unit: str = SERVICE_UNIT) -> str:
    """One line like 'vla_cameras RUNNING pid 8289, uptime 0:04:41', or 'unknown (...)'."""
    rc, out = _supervisorctl(host, "status", unit, timeout=15.0)
    return " ".join(out.split()) if out else f"unknown (rc={rc})"


def service_cameras(host: str, control_port: int = CONTROL_PORT) -> "list[str] | None":
    """Names the camera service on `host` publishes, or None if it cannot be asked right now."""
    try:
        st = ControlClient(host, control_port, timeout=5.0).call(cmd="status")
        return sorted((st.get("cams") or {}).keys())
    except Exception:  # noqa: BLE001
        return None


def restart_service(host: str, unit: str = SERVICE_UNIT, control_port: int = CONTROL_PORT,
                    wait_s: float = 45.0) -> str:
    """Restart the camera supervisor unit on `host` and wait until its control port answers again.

    The service re-opens every camera by its stable by-path link, so a camera that re-enumerated
    under a new /dev/videoN comes back. Raises RuntimeError with the reason when it cannot (no sudo,
    unit unknown, service not answering after the restart).
    """
    t0 = time.time()
    rc, out = _supervisorctl(host, "restart", unit)
    if rc != 0 or "ERROR" in out:
        raise RuntimeError(f"could not restart {unit} on {host}: {out or f'rc={rc}'} "
                           f"(needs passwordless sudo for supervisorctl on that host)")
    ctrl = ControlClient(host, control_port, timeout=5.0)
    deadline = t0 + wait_s
    while time.time() < deadline:
        try:
            ctrl.call(cmd="get_exposure")
            return (f"{unit} restarted on {host} ({out.splitlines()[-1] if out else 'ok'}); "
                    f"control port up after {time.time() - t0:.0f}s")
        except Exception:  # noqa: BLE001 — not up yet
            time.sleep(1.0)
    raise RuntimeError(f"{unit} was restarted on {host} but {host}:{control_port} did not answer within "
                       f"{wait_s:.0f}s — {service_status(host, unit)}; see log/{unit}.err.log there")


def open_stream(host: str, names: "list[str]", group: str = MCAST_GROUP, port: int = MCAST_PORT,
                control_port: int = CONTROL_PORT, timeout: float = 5.0, iface_ip: "str | None" = None,
                restart_on_fail: bool = True):
    """Subscriber + control + per-camera proxies for `make_engine(cameras="udp")`.

    Returns (subscriber, {name: grab_fn}) where each grab_fn(w, h) -> RGB and carries `.cam` (a
    CameraProxy), matching the contract `_grab` gives the local UsbCamera path. The multicast join
    is pinned to the interface that routes to `host` unless `iface_ip` overrides it.

    If the control port is down or a camera sends no frames and `restart_on_fail`, the camera
    supervisor unit on `host` is restarted once and the open retried. The service exits when a USB
    camera drops off the bus (VIDIOC_REQBUFS errno 19) and supervisor respawns it; a launch that
    lands in that window, or against a wedged service, used to die with "no frames".
    """
    try:
        return _open_stream(host, names, group, port, control_port, timeout, iface_ip)
    except RuntimeError as e:
        if not restart_on_fail:
            raise
        print(f"[cameras] {e}", flush=True)
        print(f"[cameras] restarting {SERVICE_UNIT} on {host} and retrying once …", flush=True)
        try:
            print(f"[cameras] {restart_service(host, control_port=control_port)}", flush=True)
        except Exception as re:  # noqa: BLE001
            raise RuntimeError(f"{e}; restart attempt failed: {re}") from e
        return _open_stream(host, names, group, port, control_port, max(timeout, 15.0), iface_ip)


def _open_stream(host: str, names: "list[str]", group: str, port: int, control_port: int,
                 timeout: float, iface_ip: "str | None"):
    probe = ControlClient(host, control_port, timeout=5.0)
    try:
        probe.call(cmd="get_exposure")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"camera service: control port {host}:{control_port} not answering ({e}) — "
                           f"is {SERVICE_UNIT} running on {host}?") from e
    sub = CameraSubscriber(group, port, iface_ip=iface_ip or local_ip_toward(host))
    ctrl = ControlClient(host, control_port)
    session = RemoteExposureSession(ctrl, names)
    missing = [n for n in names if not sub.wait(n, timeout)]
    if missing:
        sub.close()
        raise RuntimeError(f"camera service: no frames for {missing} on {group}:{port} within {timeout:.0f}s "
                           f"— is {SERVICE_UNIT} running on {host} with those cameras open?")
    grabs: "dict[str, Callable]" = {}
    for n in names:
        def read(w, h, _n=n):
            return sub.read(_n, w, h)
        read.cam = CameraProxy(n, sub, ctrl, session)
        grabs[n] = read
    return sub, grabs
