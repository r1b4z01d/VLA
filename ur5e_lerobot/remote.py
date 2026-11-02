"""Remote-inference bridge — run a policy (e.g. SmolVLA) in a SERVER process and stream actions to the
eval CLIENT over TCP. Since 2026-10-10 both live on the robot's Jetson Orin: scripts/infer_server.py runs
there as the supervisor unit vla_infer_server and the client (eval_hw.py, cameras + RTDE + hand) connects
over loopback (127.0.0.1:8777). The GPU box can still be the server, over the LAN, as a fallback.

Why a separate process even on one box: the server owns the GPU, the checkpoint(s) and the prefetch/RTC
chunk scheduler, so a model switch or a server restart never touches the control loop, and the same
client works unchanged against a remote GPU. Only observations + actions cross the socket.

Wire format: 4-byte big-endian length prefix + a pickled dict of PLAIN Python types only (lists / str /
bytes) so it is portable across differing numpy/torch versions. Images are JPEG-compressed. **Trusted
LAN only** — pickle is not safe against a hostile peer.

Topology: on the robot, `eval_hw.py --remote 127.0.0.1:8777`. Against the GPU box instead (it is on the
robot Wi-Fi, 192.168.11.119): `eval_hw.py --remote 192.168.11.119:8777` (the server binds 0.0.0.0; open
the port in the GPU box's firewall). If the two ever sit on different subnets bridged by the workstation,
relay the port through it —
    ssh -N -L 0.0.0.0:8777:localhost:8777 GPU      # needs `GatewayPorts yes` in the workstation's sshd_config
and point the client at <WORKSTATION_IP>:8777.
"""
from __future__ import annotations

import pickle
import socket
import struct


def send_msg(sock, obj) -> None:
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_all(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed mid-message")
        buf += chunk
    return bytes(buf)


def recv_msg(sock):
    (n,) = struct.unpack(">I", _recv_all(sock, 4))
    return pickle.loads(_recv_all(sock, n))


# Downscale before the wire. The policy fits every image into `resize_imgs_with_padding` (512x512 for
# SmolVLA) preserving aspect, so anything larger is bytes we pay for and the policy immediately throws
# away. Shrinking to the SAME fit-inside-512 box the policy would produce keeps geometry identical
# (960x540 -> 512x288, then the policy pads to 512x512 exactly as before) while roughly halving the
# payload — measured 222 KB -> ~100 KB per 3-camera observation.
# NB this must stay a FIT (aspect-preserving, shrink-only). Resizing to a square 512x512 would squish
# rotated/portrait views and silently break train/deploy geometry.
WIRE_MAX_WH = 512
JPEG_QUALITY = 92  # unchanged from before the downscale — lower it separately if you want more savings


def _fit_for_wire(a, max_wh: int = WIRE_MAX_WH):
    """Shrink `a` to fit inside (max_wh, max_wh) preserving aspect. Never upscales."""
    import cv2

    h, w = a.shape[:2]
    scale = min(max_wh / w, max_wh / h)
    if scale >= 1.0:
        return a
    return cv2.resize(a, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def encode_obs(state, images: dict, task: str, max_wh: int = WIRE_MAX_WH,
               quality: int = JPEG_QUALITY) -> dict:
    """Pack an observation into wire types: state->list, images->{name: JPEG bytes}. `images` maps a
    camera name (scene/wrist/side/…) to an RGB HWC array — whatever set the trained policy expects.
    JPEG round-trips the array channel-for-channel regardless of RGB/BGR interpretation (no swap).

    Images are shrunk to fit inside (max_wh, max_wh) first — see WIRE_MAX_WH. Pass max_wh=0 to disable
    (send at capture size) if a policy ever wants more resolution than it resizes to."""
    import cv2

    def jpg(a) -> bytes:
        if max_wh:
            a = _fit_for_wire(a, max_wh)
        ok, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes()

    return {"state": [float(v) for v in state], "task": task,
            "images": {name: jpg(rgb) for name, rgb in images.items()}}


def decode_obs(msg: dict):
    import cv2
    import numpy as np

    def unjpg(b):
        return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)

    return (np.asarray(msg["state"], dtype=np.float32),
            {name: unjpg(b) for name, b in msg["images"].items()}, msg["task"])


class RemotePolicyClient:
    """Thin obs->action RPC. Stands in for a local policy in the eval loop; the server holds the policy
    (and its action-chunk queue / reactivity), so this just ships observations and gets one action back."""

    def __init__(self, host: str, port: int, timeout: float = 15.0):
        self.host, self.port, self.timeout = host, port, timeout
        self.sock = None

    def connect(self) -> "RemotePolicyClient":
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return self

    def infer(self, state, images: dict, task: str):
        import numpy as np

        send_msg(self.sock, encode_obs(state, images, task))
        resp = recv_msg(self.sock)
        if "error" in resp:
            raise RuntimeError(f"remote policy error: {resp['error']}")
        return np.asarray(resp["action"], dtype=np.float32)

    def reset(self) -> None:
        """Reset the server-side policy (clears its action-chunk queue) — call on PLAY/RESET."""
        send_msg(self.sock, {"cmd": "reset"})
        recv_msg(self.sock)  # ack

    def info(self) -> dict:
        """What the server is serving: resolved checkpoint path, reactivity flags. {} on an old server
        that does not know the message (it would treat it as an observation and return an error)."""
        send_msg(self.sock, {"cmd": "info"})
        rep = recv_msg(self.sock)
        return rep if isinstance(rep, dict) and rep.get("ok") else {}

    def select(self, model: str) -> dict:
        """Make a multi-model server serve `model` (run name, checkpoint path or realpath): instant
        when it is resident. Returns the server's reply ({'ok': True, ...info} or {'ok': False,
        'error': ...}); an old server answers with an 'error' (it took the message for an observation)."""
        send_msg(self.sock, {"cmd": "select", "model": model})
        rep = recv_msg(self.sock)
        return rep if isinstance(rep, dict) else {"ok": False, "error": "bad reply"}

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:  # noqa: BLE001
                pass
            self.sock = None


def serve(host: str, port: int, infer_fn, reset_fn=None, info_fn=None, cmd_fn=None) -> None:
    """Blocking single-client server. `infer_fn(state, images: dict, task) -> action` (array or list),
    where `images` maps camera name -> RGB array. Handles {'cmd':'reset'} and {'cmd':'info'} control
    messages and survives client reconnects. `info_fn() -> dict` describes what is being served
    (resolved checkpoint, reactivity flags) so the client can RECORD it: an eval that only knows
    "127.0.0.1:8777" cannot later be grouped by model, which is how 136 runs ended up unattributable."""
    import numpy as np

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[server] listening on {host}:{port}", flush=True)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[server] client connected: {addr}", flush=True)
        try:
            while True:
                msg = recv_msg(conn)
                if isinstance(msg, dict) and msg.get("cmd") == "reset":
                    if reset_fn is not None:
                        reset_fn()
                    send_msg(conn, {"ok": True})
                    continue
                if isinstance(msg, dict) and msg.get("cmd") == "info":
                    try:
                        send_msg(conn, {"ok": True, **(info_fn() if info_fn is not None else {})})
                    except Exception as e:  # noqa: BLE001
                        send_msg(conn, {"ok": False, "error": repr(e)})
                    continue
                if isinstance(msg, dict) and "cmd" in msg:     # other control messages (e.g. select)
                    try:
                        rep = cmd_fn(msg) if cmd_fn is not None else None
                        send_msg(conn, rep if isinstance(rep, dict) else
                                 {"ok": False, "error": f"unknown cmd {msg.get('cmd')!r}"})
                    except Exception as e:  # noqa: BLE001
                        send_msg(conn, {"ok": False, "error": repr(e)})
                    continue
                state, images, task = decode_obs(msg)
                try:
                    action = infer_fn(state, images, task)
                    send_msg(conn, {"action": [float(v) for v in np.asarray(action).ravel()]})
                except Exception as e:  # noqa: BLE001 — report inference errors to the client, keep serving
                    send_msg(conn, {"error": repr(e)})
        except (ConnectionError, EOFError, OSError) as e:
            print(f"[server] client disconnected ({e}); awaiting reconnect", flush=True)
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
