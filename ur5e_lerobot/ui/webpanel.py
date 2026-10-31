"""Browser UI for the robot-side apps (teleop panel, eval) — standard library only, works offline.

The control loop stays exactly where it is (Tk tick / eval loop on the robot PC). This module adds
a thin view + command channel over HTTP that any browser can open: the Mac, a tablet next to the
robot, or RD's own screen. Nothing here touches the robot: the loop pushes a state snapshot each
tick with set_state(), appends log lines with log(), and drains commands() — the same marshaling the
Stream Deck already uses for its key presses.

Endpoints (ThreadingHTTPServer, one thread per connection):
  GET  /                   the page (webui/static/robot/index.html; the manager's style.css is shared)
  GET  /static/...         static files from webui/static
  GET  /api/config         {mode, title, cams: {name: url}, keys}
  GET  /events             Server-Sent Events: `state` (latest snapshot, on change, <= 10 Hz) and `log`
  POST /api/cmd            {"name": "...", "args": {...}} -> queued for the control loop
  GET  /cam/<name>.mjpg    MJPEG from this process's own frames (sim engine / local cameras); with the
                           camera service running, config points the browser at its :5602 streams
  GET  /cam/<name>.jpg     one frame

Why SSE + POST and not WebSockets: no third-party package on the robot PC, no event loop to run next
to Tk, and a dropped connection just reconnects (EventSource does that by itself).
"""
from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

STATIC_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "webui", "static"))
# Bump when the page needs a newer command set from the app. The page is static and reloads at any
# time, but the app only picks up new code when it is relaunched; a mismatch shows a banner instead of
# silently logging "unknown command" (2026-10-26: grasp_hold against a panel started before it existed).
API_VERSION = 2
DEFAULT_PORT = 8090
_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "application/javascript",
          ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon", ".json": "application/json"}


def _encode_jpeg(rgb, quality: int = 80) -> "bytes | None":
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None
    except Exception:  # noqa: BLE001 — headless box without cv2: PIL
        try:
            import io
            from PIL import Image
            b = io.BytesIO()
            Image.fromarray(rgb).save(b, format="JPEG", quality=quality)
            return b.getvalue()
        except Exception:  # noqa: BLE001
            return None


class WebPanel:
    """One instance per app process. start() binds the port and returns the URL; a port already in
    use (another robot app running) raises OSError — the caller logs it and carries on without a
    browser UI, because the robot must never depend on this."""

    def __init__(self, mode: str, title: str, port: int = DEFAULT_PORT, host: str = "0.0.0.0",
                 cams: "dict[str, str | None] | None" = None, static_root: str = STATIC_ROOT, keys=None,
                 retry: bool = True, on_ready=None):
        self.mode, self.title, self.port, self.host = mode, title, port, host
        self.retry, self.on_ready = retry, on_ready     # keep trying a busy port; callback(url) when bound
        self._stopped = False
        self._retry_err = None
        self._handler_cls = None
        self.static_root = static_root
        # name -> URL template with "{host}" (camera service) or None (serve own frames)
        self.cams: "dict[str, str | None]" = dict(cams or {})
        self.keys = keys or {}
        self._state: dict = {}
        self._version = 0
        self._cv = threading.Condition()
        self._log: "deque[tuple[int, str]]" = deque(maxlen=400)
        self._log_seq = 0
        self._cmds: "queue.Queue[dict]" = queue.Queue()
        self._frames: dict = {}            # name -> (rgb, seq)
        self._frame_cv = threading.Condition()
        self._srv = None
        self._thread = None
        self.url = ""

    # ---- control-loop side -----------------------------------------------------------------------
    def set_state(self, d: dict) -> None:
        with self._cv:
            self._state = d
            self._version += 1
            self._cv.notify_all()

    def update(self, **kw) -> None:
        with self._cv:
            self._state = {**self._state, **kw}
            self._version += 1
            self._cv.notify_all()

    def log(self, line: str) -> None:
        with self._cv:
            self._log_seq += 1
            self._log.append((self._log_seq, f"{time.strftime('%H:%M:%S')}  {line}"))
            self._cv.notify_all()

    def set_frame(self, name: str, rgb) -> None:
        """Latest frame for /cam/<name>.mjpg (only used when the camera has no service URL)."""
        with self._frame_cv:
            seq = self._frames.get(name, (None, 0))[1] + 1
            self._frames[name] = (rgb, seq)
            self._frame_cv.notify_all()

    def commands(self) -> "list[dict]":
        out = []
        while True:
            try:
                out.append(self._cmds.get_nowait())
            except queue.Empty:
                return out

    # ---- server ---------------------------------------------------------------------------------
    def start(self) -> str:
        panel = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # noqa: N802 — silence per-request stderr noise
                pass

            def _send(self, code: int, body: bytes, ctype: str = "application/json", extra=None) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj, code: int = 200) -> None:
                self._send(code, json.dumps(obj).encode())

            def do_GET(self):  # noqa: N802
                path = urlparse(self.path).path
                if path in ("/", "/index.html"):
                    return self._static("robot/index.html")
                if path.startswith("/static/"):
                    return self._static(path[len("/static/"):])
                if path == "/api/config":
                    return self._json(panel._config(self.headers.get("Host", "")))
                if path == "/api/state":
                    with panel._cv:
                        return self._json(panel._state)
                if path == "/events":
                    return self._events()
                if path.startswith("/cam/"):
                    return self._cam(path[len("/cam/"):])
                self._send(404, b"not found", "text/plain")

            def do_POST(self):  # noqa: N802
                path = urlparse(self.path).path
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                if path == "/api/cmd":
                    try:
                        d = json.loads(raw or b"{}")
                        if not isinstance(d, dict) or not d.get("name"):
                            raise ValueError("expected {name, args}")
                    except Exception as e:  # noqa: BLE001
                        return self._json({"ok": False, "error": str(e)}, 400)
                    d.setdefault("args", {})
                    d["t"] = time.time()
                    panel._cmds.put(d)
                    return self._json({"ok": True})
                self._send(404, b"not found", "text/plain")

            def _static(self, rel: str) -> None:
                rel = os.path.normpath(rel).lstrip(os.sep)
                full = os.path.join(panel.static_root, rel)
                if not full.startswith(panel.static_root) or not os.path.isfile(full):
                    return self._send(404, b"not found", "text/plain")
                with open(full, "rb") as f:
                    body = f.read()
                self._send(200, body, _TYPES.get(os.path.splitext(full)[1], "application/octet-stream"))

            def _events(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                seen_version = -1
                log_cursor = 0
                with panel._cv:  # replay the recent log so a fresh tab shows context
                    backlog = list(panel._log)
                try:
                    for seq, line in backlog[-60:]:
                        self.wfile.write(f"event: log\ndata: {json.dumps(line)}\n\n".encode())
                        log_cursor = seq
                    self.wfile.flush()
                    last_keep = time.time()
                    while not panel._stopped:
                        with panel._cv:
                            panel._cv.wait(0.5)
                            version, state = panel._version, panel._state
                            new_logs = [(s, l) for s, l in panel._log if s > log_cursor]
                        if version != seen_version:
                            self.wfile.write(f"event: state\ndata: {json.dumps(state)}\n\n".encode())
                            seen_version = version
                        for seq, line in new_logs:
                            self.wfile.write(f"event: log\ndata: {json.dumps(line)}\n\n".encode())
                            log_cursor = seq
                        if time.time() - last_keep > 15:
                            self.wfile.write(b": keepalive\n\n")
                            last_keep = time.time()
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

            def _cam(self, rest: str) -> None:
                name, _, ext = rest.partition(".")
                if name not in panel.cams:
                    return self._send(404, b"unknown camera", "text/plain")
                if ext == "jpg":
                    with panel._frame_cv:
                        rec = panel._frames.get(name)
                    jpg = _encode_jpeg(rec[0]) if rec else None
                    if jpg is None:
                        return self._send(503, b"no frame yet", "text/plain")
                    return self._send(200, jpg, "image/jpeg")
                boundary = "frame"
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last_seq = -1
                try:
                    while not panel._stopped:
                        with panel._frame_cv:
                            panel._frame_cv.wait(1.0)
                            rec = panel._frames.get(name)
                        if rec is None or rec[1] == last_seq:
                            continue
                        last_seq = rec[1]
                        jpg = _encode_jpeg(rec[0])
                        if jpg is None:
                            continue
                        self.wfile.write(f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                                         f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                        self.wfile.flush()
                        time.sleep(0.08)   # ~10 fps is plenty for a preview and keeps the encode cheap
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        self._handler_cls = Handler
        try:
            self._bind()
        except OSError as e:
            if not self.retry:
                raise
            # Port busy (a previous robot app still holds it, or its writer children do while it exits):
            # keep trying in the background and announce the URL when it binds. The robot never waits.
            self._retry_err = e
            threading.Thread(target=self._retry_loop, daemon=True, name="webpanel-retry").start()
            return ""
        return self.url

    def _bind(self) -> None:
        self._srv = ThreadingHTTPServer((self.host, self.port), self._handler_cls)
        self._srv.daemon_threads = True
        # server_close() would otherwise JOIN every handler thread — and a browser's /events (SSE)
        # and /cam/*.mjpg connections never end while the tab is open. That hung the panel's clean
        # shutdown until supervisor SIGKILLed it (2026-10-26 02:16). The loops below watch _stopped.
        self._srv.block_on_close = False
        # Forked children (LeRobot's image-writer processes) inherit the listening socket and keep the
        # port alive after the app exits: close it in every child right after the fork.
        try:
            os.register_at_fork(after_in_child=self._close_listener_in_child)
        except (AttributeError, OSError):
            pass
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True, name="webpanel")
        self._thread.start()
        self.url = f"http://{_advertised_host()}:{self.port}/"

    def _close_listener_in_child(self) -> None:
        try:
            if self._srv is not None:
                self._srv.socket.close()
        except Exception:  # noqa: BLE001
            pass

    def _retry_loop(self) -> None:
        while self._srv is None and not self._stopped:
            time.sleep(3.0)
            try:
                self._bind()
            except OSError:
                continue
            if self.on_ready:
                try:
                    self.on_ready(self.url)
                except Exception:  # noqa: BLE001
                    pass

    def stop(self) -> None:
        self._stopped = True
        with self._cv:            # wake the SSE loops so they see _stopped and exit
            self._cv.notify_all()
        with self._frame_cv:
            self._frame_cv.notify_all()
        if self._srv is not None:
            try:
                self._srv.shutdown()       # returns once serve_forever's loop exits (<= 0.5 s)
                self._srv.server_close()   # no join: block_on_close is False
            except Exception:  # noqa: BLE001
                pass
            self._srv = None

    def _config(self, request_host: str) -> dict:
        host = request_host.split(":")[0] or "127.0.0.1"
        cams = {}
        for name, tmpl in self.cams.items():
            cams[name] = tmpl.replace("{host}", host) if tmpl else f"/cam/{name}.mjpg"
        return {"mode": self.mode, "title": self.title, "cams": cams, "keys": self.keys, "api": API_VERSION}


def _advertised_host() -> str:
    """The LAN address to print at launch (what another device would type)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.11.1", 1))
        ip = s.getsockname()[0]
        s.close()
        if not ip.startswith("127."):
            return ip
    except OSError:
        pass
    return socket.gethostname()


def service_cam_urls(names, camera_host: str, port: int = 5602) -> "dict[str, str]":
    """URL templates for cameras served by the camera service. A loopback camera_host (the usual
    --camera-host 127.0.0.1 on the robot PC) becomes "{host}", i.e. whatever host the browser used
    to reach this panel — a browser on the Mac cannot use 127.0.0.1."""
    h = "{host}" if camera_host in ("127.0.0.1", "localhost", "0.0.0.0") else camera_host
    return {n: f"http://{h}:{port}/{n}.mjpg" for n in names}
