"""Episode player — rerun web viewer fed by a STATIC .rrd over plain HTTP.

Why not gRPC: lerobot's `--mode distant` serves the data over a gRPC proxy that speaks HTTP/2
*cleartext* (h2c). Browsers cannot open h2c connections, so the embedded viewer's connection to that
port is reset ("The connection was reset"). That gRPC path is really meant for the *native* desktop
viewer (`rerun rerun+http://…`).

Instead we:
  1. render each episode once to a static `<ds>__ep<N>.rrd` (lerobot_dataset_viz --save 1), cached;
  2. keep ONE persistent `rr.serve_web_viewer` process that serves only the WASM viewer app (no gRPC);
  3. point the iframe at  http://<host>:<web_port>/?url=<http url of the .rrd on :8080>.
The browser then fetches the .rrd over ordinary HTTP GET (Range-friendly) — no gRPC, no h2c. The
served viewer sets no COOP/COEP, so the cross-origin .rrd fetch needs only Access-Control-Allow-Origin.
"""
from __future__ import annotations

import glob
import os
import shutil
import signal
import subprocess
import sys
import time

VIZ_MODULE = "lerobot.scripts.lerobot_dataset_viz"
# tiny blocking server that serves ONLY the viewer app (no recording, no gRPC).
# serve_web_viewer returns immediately (serves on a background thread), so the main thread must block;
# use a bounded sleep loop — time.sleep(huge) raises OSError [Errno 22] on macOS.
_APP_SERVER = (
    "import sys, time, rerun as rr\n"
    "rr.serve_web_viewer(web_port=int(sys.argv[1]), open_browser=False)\n"
    "while True:\n"
    "    time.sleep(3600)\n"
)


class ViewerManager:
    def __init__(self, repo_root: str, python_exe: str | None = None, web_port: int = 9090,
                 cache_dir: str | None = None):
        self.repo_root = repo_root
        self.python_exe = python_exe or sys.executable
        self.web_port = web_port
        self.cache_dir = cache_dir or os.path.join(repo_root, "outputs", "webui", "rrd")
        self._app: subprocess.Popen | None = None

    # ---- the persistent viewer-app server (no gRPC) ----
    def _app_alive(self) -> bool:
        return self._app is not None and self._app.poll() is None

    def _free_port(self) -> None:
        """Kill anything holding the app port (e.g. an orphan from a hard restart)."""
        try:
            out = subprocess.run(["lsof", "-ti", f"tcp:{self.web_port}"],
                                 capture_output=True, text=True, timeout=5)
            for pid in out.stdout.split():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
        except Exception:  # noqa: BLE001 — lsof missing / nothing to reclaim
            pass

    def ensure_app(self) -> None:
        if self._app_alive():
            return
        self._free_port()
        os.makedirs(self.cache_dir, exist_ok=True)
        log = open(os.path.join(os.path.dirname(self.cache_dir), "viewer_app.log"), "w")
        self._app = subprocess.Popen([self.python_exe, "-c", _APP_SERVER, str(self.web_port)],
                                     cwd=self.repo_root, stdout=log, stderr=subprocess.STDOUT)
        time.sleep(1.5)  # let it bind before the iframe loads

    def stop(self) -> None:
        if self._app is not None:
            try:
                self._app.terminate()
                try:
                    self._app.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    self._app.kill()
            except Exception:  # noqa: BLE001
                pass
        self._app = None

    # ---- per-episode .rrd (cached) ----
    def render_rrd(self, dataset_root: str, repo_id: str, episode: int) -> str:
        """Return the path to this episode's .rrd, rendering it if missing/stale. Cache is keyed by
        dataset name + episode and invalidated when the dataset's info.json is newer (i.e. after an edit)."""
        ds_name = os.path.basename(dataset_root.rstrip("/"))
        os.makedirs(self.cache_dir, exist_ok=True)
        cache = os.path.join(self.cache_dir, f"{ds_name}__ep{int(episode)}.rrd")
        info = os.path.join(dataset_root, "meta", "info.json")
        if os.path.isfile(cache) and os.path.isfile(info) \
                and os.path.getmtime(cache) >= os.path.getmtime(info):
            return cache
        tmp = os.path.join(self.cache_dir, f".{ds_name}__ep{int(episode)}.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        try:
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": self.repo_root}
            # --num-workers 0: run the DataLoader in-process so it needs no /dev/shm. Docker's default
            # /dev/shm is 64MB and torch workers overflow it ("unable to allocate shared memory"); worker
            # parallelism gave no render speedup here anyway, so 0 is both safe and free.
            cmd = [self.python_exe, "-m", VIZ_MODULE, "--repo-id", repo_id, "--root", dataset_root,
                   "--episode-index", str(int(episode)), "--save", "1", "--output-dir", tmp,
                   "--num-workers", "0"]
            r = subprocess.run(cmd, cwd=self.repo_root, env=env, capture_output=True, text=True,
                               timeout=600)
            hits = glob.glob(os.path.join(tmp, "*.rrd"))
            if not hits:
                raise RuntimeError(f"viz wrote no .rrd (exit {r.returncode}): {r.stderr[-400:]}")
            os.replace(hits[0], cache)  # atomic into place
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return cache

    def viewer_url(self, host: str, rrd_url: str) -> str:
        return f"http://{host}:{self.web_port}/?url={rrd_url}"

    def status(self, episode: int | None = None) -> dict:
        return {"running": self._app_alive(), "web_port": self.web_port, "episode": episode}
