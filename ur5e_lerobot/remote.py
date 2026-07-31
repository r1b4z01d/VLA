"""Remote-inference bridge — run a heavy policy (e.g. SmolVLA) on a GPU host and stream actions to the
robot PC over TCP. The robot PC (cameras + RTDE + hand) is the CLIENT; the GPU box is the SERVER.

Why: SmolVLA is ~2.6 s/inference on the robot PC's CPU (measured) — unusable for real-time control —
but ~tens of ms on the 4090. So inference lives on the GPU; only observations + actions cross the wire.

Wire format: 4-byte big-endian length prefix + a pickled dict of PLAIN Python types only (lists / str /
bytes) so it is portable across differing numpy/torch versions. Images are JPEG-compressed. **Trusted
LAN only** — pickle is not safe against a hostile peer.

Topology (simplest): put the GPU box and the robot PC on the SAME subnet — the client then connects
straight to the server, `eval_hw.py --remote <GPU_IP>:8777` (the server binds 0.0.0.0; just open the
port in the GPU's firewall). Keep the GPU box dual-homed if it also needs internet for training.

Fallback, if they must stay on different subnets bridged by the Mac: relay the port through the Mac —
    ssh -N -L 0.0.0.0:8777:localhost:8777 gpu      # needs `GatewayPorts yes` in the Mac's sshd_config
and point the client at <MAC_IP>:8777.
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


def encode_obs(state, scene_rgb, wrist_rgb, task: str) -> dict:
    """Pack an observation into wire types: state->list, images->JPEG bytes (encode/decode round-trips
    the array channel-for-channel regardless of RGB/BGR interpretation, so no color swap)."""
    import cv2

    def jpg(a) -> bytes:
        ok, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes()

    return {"state": [float(v) for v in state], "task": task,
            "scene": jpg(scene_rgb), "wrist": jpg(wrist_rgb)}


def decode_obs(msg: dict):
    import cv2
    import numpy as np

    def unjpg(b):
        return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)

    return (np.asarray(msg["state"], dtype=np.float32), unjpg(msg["scene"]), unjpg(msg["wrist"]), msg["task"])


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

    def infer(self, state, scene_rgb, wrist_rgb, task: str):
        import numpy as np

        send_msg(self.sock, encode_obs(state, scene_rgb, wrist_rgb, task))
        resp = recv_msg(self.sock)
        if "error" in resp:
            raise RuntimeError(f"remote policy error: {resp['error']}")
        return np.asarray(resp["action"], dtype=np.float32)

    def reset(self) -> None:
        """Reset the server-side policy (clears its action-chunk queue) — call on PLAY/RESET."""
        send_msg(self.sock, {"cmd": "reset"})
        recv_msg(self.sock)  # ack

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:  # noqa: BLE001
                pass
            self.sock = None


def serve(host: str, port: int, infer_fn, reset_fn=None) -> None:
    """Blocking single-client server. `infer_fn(state, scene_rgb, wrist_rgb, task) -> action` (array or
    list). Handles a {'cmd':'reset'} control message and survives client reconnects."""
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
                state, scene, wrist, task = decode_obs(msg)
                try:
                    action = infer_fn(state, scene, wrist, task)
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
