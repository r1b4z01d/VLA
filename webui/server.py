"""FastAPI backend + static frontend for the dataset-management web UI.

    HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python -m webui.server --host 0.0.0.0 --port 8080
    # then browse from the Mac at  http://192.168.11.119:8081

Read endpoints (GET) scan parquet metadata only (instant). Mutating endpoints (POST) delegate to
webui.ops; destructive ones require {"confirm": true}. The folder monitor is client-side polling of
GET /api/datasets, which re-scans the folder every call and returns a signature the UI diffs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

from . import datasets as ds_mod
from . import annotations as anno
from . import ops
from . import models as models_mod
from . import evals as evals_mod
from .viewer import ViewerManager

app = FastAPI(title="RobotDisco Dataset Manager")
# The rerun viewer app (served on :web_port) fetches the .rrd from us cross-origin, with a Range header
# that triggers a CORS preflight. Allow all — this is a trusted-LAN internal tool — so the preflight
# OPTIONS is answered (not 405) and the actual fetch is permitted.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# populated in main(); defaults let `uvicorn webui.server:app` work from the repo root too
DATASETS_DIR = os.path.abspath(os.environ.get("WEBUI_DATASETS_DIR", "outputs/datasets"))
REPO_ROOT = os.path.abspath(os.environ.get("WEBUI_REPO_ROOT", "."))
VIEWER = ViewerManager(REPO_ROOT)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def _guard(name: str) -> str:
    """Resolve a dataset name to its root, rejecting path traversal."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(400, f"bad dataset name: {name!r}")
    root = os.path.join(DATASETS_DIR, name)
    if not ds_mod.is_dataset(root):
        raise HTTPException(404, f"no dataset {name!r}")
    return root


def _run(fn, *a, **k):
    try:
        return fn(*a, **k)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — surface op errors as 400 with the message
        raise HTTPException(400, f"{type(e).__name__}: {e}")


def _require_confirm(payload: dict) -> None:
    if not payload.get("confirm"):
        raise HTTPException(400, "this operation is destructive; resend with confirm=true")


# --------------------------------------------------------------------------- read
@app.get("/api/datasets")
def api_datasets():
    items = ds_mod.list_datasets(DATASETS_DIR)
    sig = hashlib.sha1(
        json.dumps([(d.get("name"), d.get("episodes"), d.get("frames"), d.get("size_mb")) for d in items],
                   sort_keys=True).encode()
    ).hexdigest()[:12]
    return {"datasets": items, "signature": sig, "scanned_at": time.time(),
            "datasets_dir": DATASETS_DIR}


@app.get("/api/datasets/{name}/episodes")
def api_episodes(name: str):
    _guard(name)
    return {"name": name, "episodes": _run(ds_mod.list_episodes, DATASETS_DIR, name)}


# --------------------------------------------------------------------------- in-place edits
@app.post("/api/datasets/{name}/rename_task")
def api_rename_task(name: str, payload: dict = Body(...)):
    _guard(name)
    et = payload.get("episode_tasks")
    return _run(ops.rename_task, DATASETS_DIR, name, new_task=payload.get("new_task"), episode_tasks=et)


@app.post("/api/datasets/{name}/annotate")
def api_annotate(name: str, payload: dict = Body(...)):
    root = _guard(name)
    if "episode" not in payload:
        raise HTTPException(400, "annotate needs an episode index")
    rec = _run(anno.set_episode, root, int(payload["episode"]),
               {k: payload[k] for k in anno.FIELDS if k in payload})
    return {"ok": True, "episode": int(payload["episode"]), "annotation": rec}


# --------------------------------------------------------------------------- rebuild edits (+ .bak)
@app.post("/api/datasets/{name}/delete_episodes")
def api_delete_episodes(name: str, payload: dict = Body(...)):
    _guard(name)
    _require_confirm(payload)
    eps = payload.get("episodes") or []
    return _run(ops.delete_episodes, DATASETS_DIR, name, [int(e) for e in eps])


@app.post("/api/datasets/{name}/trim")
def api_trim(name: str, payload: dict = Body(...)):
    _guard(name)
    _require_confirm(payload)
    for f in ("episode", "cut_start_s", "cut_end_s"):
        if f not in payload:
            raise HTTPException(400, f"trim needs {f}")
    return _run(ops.trim_episode, DATASETS_DIR, name, int(payload["episode"]),
                float(payload["cut_start_s"]), float(payload["cut_end_s"]))


@app.post("/api/datasets/{name}/move")
def api_move(name: str, payload: dict = Body(...)):
    _guard(name)
    dst = payload.get("dst")
    if not dst:
        raise HTTPException(400, "move needs dst")
    # dst may be a NEW dataset (created from the moved episode); ops validates the name + existing case
    return _run(ops.move_episode, DATASETS_DIR, name, int(payload["episode"]), dst)


# --------------------------------------------------------------------------- new-dataset ops
@app.post("/api/datasets/merge")
def api_merge(payload: dict = Body(...)):
    names = payload.get("names") or []
    out = payload.get("out_name")
    if not out:
        raise HTTPException(400, "merge needs out_name")
    for n in names:
        _guard(n)
    return _run(ops.merge, DATASETS_DIR, names, out)


@app.post("/api/datasets/{name}/to_video")
def api_to_video(name: str, payload: dict = Body(default={})):
    _guard(name)
    return _run(ops.to_video, DATASETS_DIR, name, payload.get("out_name"))


# --------------------------------------------------------------------------- lifecycle
@app.post("/api/datasets/{name}/delete")
def api_delete(name: str, payload: dict = Body(...)):
    _guard(name)
    _require_confirm(payload)
    VIEWER.stop()
    return _run(ops.delete_dataset, DATASETS_DIR, name)


@app.post("/api/datasets/{name}/restore")
def api_restore(name: str):
    return _run(ops.restore_backup, DATASETS_DIR, name)


@app.post("/api/datasets/{name}/train")
def api_train(name: str, payload: dict = Body(default={})):
    """Kick off a training run on this dataset (on THIS host's GPU); appears in the Models tab."""
    _guard(name)
    return _run(models_mod.start_training, REPO_ROOT, DATASETS_DIR, name,
                payload.get("policy", "smolvla"), payload.get("steps", 20000),
                payload.get("batch_size", 64), payload.get("save_freq", 2500),
                dry_run=bool(payload.get("dry_run", False)))


@app.post("/api/datasets/{name}/rename")
def api_rename_dataset(name: str, payload: dict = Body(...)):
    _guard(name)
    new = payload.get("new_name")
    if not new:
        raise HTTPException(400, "rename needs new_name")
    VIEWER.stop()  # any cached viewer points at the old name
    return _run(ops.rename_dataset, DATASETS_DIR, name, new)


@app.post("/api/datasets/add")
def api_add(payload: dict = Body(...)):
    return _run(ops.add_dataset, DATASETS_DIR, source=payload.get("source"),
                remote=payload.get("remote"), as_name=payload.get("as_name"))


# --------------------------------------------------------------------------- player
def _repo_id_of(root: str, name: str) -> str:
    return json.load(open(os.path.join(root, "meta", "info.json"))).get("repo_id") or f"local/{name}"


def _browser_host_port(request: Request) -> tuple[str, int]:
    """The host:port the BROWSER can reach us at. request.url reflects what the client used — except a
    bind-all host (0.0.0.0 / ::) is NOT routable from a browser (Docker Desktop often links 0.0.0.0:8080),
    so fall back to localhost. Both the .rrd URL and the viewer-app URL are built from this."""
    host = request.url.hostname or "localhost"
    if host in ("0.0.0.0", "::", ""):
        host = "localhost"
    return host, request.url.port or 8080


@app.get("/api/datasets/{name}/episodes/{ep}/viewer")
def api_viewer(name: str, ep: int, request: Request):
    """Ensure the rerun viewer app is up + the episode's .rrd is rendered, then hand back a URL that
    loads that .rrd into the app over plain HTTP (no gRPC — browsers can't do the h2c gRPC path)."""
    root = _guard(name)
    VIEWER.ensure_app()
    _run(VIEWER.render_rrd, root, _repo_id_of(root, name), int(ep))  # pre-render -> /rrd is a cache hit
    host, ui_port = _browser_host_port(request)
    # the browser fetches the .rrd from us (:ui_port), then the viewer app on :web_port loads it via ?url=.
    # The path MUST end in `.rrd` — rerun's data-source parser classifies http(s) recordings by that
    # extension and otherwise rejects the URL ("Failed to parse URL") and shows its welcome screen.
    rrd_url = f"http://{host}:{ui_port}/api/datasets/{name}/episodes/{ep}/rerun.rrd"
    return {"episode": int(ep), "web_port": VIEWER.web_port, "url": VIEWER.viewer_url(host, rrd_url)}


@app.get("/api/datasets/{name}/episodes/{ep}/rerun.rrd")
def api_rrd(name: str, ep: int):
    """Serve the episode's static .rrd (path ends in `.rrd` so rerun recognizes it). The viewer app
    (served cross-origin on :web_port) fetches this; it sets no COEP, so a plain
    Access-Control-Allow-Origin suffices. FileResponse handles Range."""
    root = _guard(name)
    path = _run(VIEWER.render_rrd, root, _repo_id_of(root, name), int(ep))
    return FileResponse(path, media_type="application/octet-stream",
                        headers={"Cache-Control": "no-cache"})  # CORS handled by the middleware


@app.post("/api/viewer/stop")
def api_viewer_stop():
    VIEWER.stop()
    return {"ok": True}


# --------------------------------------------------------------------------- models
def _guard_plain(name: str) -> str:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(400, f"bad name: {name!r}")
    return name


@app.get("/api/models")
def api_models():
    return {"models": _run(models_mod.list_models, REPO_ROOT)}


@app.post("/api/models/{name}/rename")
def api_model_rename(name: str, payload: dict = Body(...)):
    _guard_plain(name)
    if not payload.get("new_name"):
        raise HTTPException(400, "rename needs new_name")
    return _run(models_mod.rename_model, REPO_ROOT, name, payload["new_name"])


@app.post("/api/models/{name}/delete")
def api_model_delete(name: str, payload: dict = Body(...)):
    _guard_plain(name)
    _require_confirm(payload)
    return _run(models_mod.delete_model, REPO_ROOT, name)


@app.get("/api/models/{name}/download")
def api_model_download(name: str, step: str | None = None):
    """Zip the checkpoint's pretrained_model (the deployable artifact) and stream it."""
    import shutil as _sh
    from starlette.background import BackgroundTask

    _guard_plain(name)
    src = _run(models_mod.checkpoint_dir, REPO_ROOT, name, step)
    cache = os.path.join(REPO_ROOT, "outputs", "webui")
    os.makedirs(cache, exist_ok=True)
    base = os.path.join(cache, f"dl_{name}_{int(time.time())}")
    _sh.make_archive(base, "zip", src)
    zp = base + ".zip"
    return FileResponse(zp, filename=f"{name}.zip", media_type="application/zip",
                        background=BackgroundTask(lambda: os.path.isfile(zp) and os.remove(zp)))


# --------------------------------------------------------------------------- evals
@app.get("/api/evals")
def api_evals():
    return {"evals": _run(evals_mod.list_evals, REPO_ROOT)}


@app.post("/api/evals")
def api_eval_create(payload: dict = Body(...)):
    if not payload.get("name"):
        raise HTTPException(400, "eval needs a name")
    return _run(evals_mod.create_eval, REPO_ROOT, payload["name"], payload)


@app.post("/api/evals/bulk_annotate")
def api_evals_bulk_annotate(payload: dict = Body(...)):
    """Set the same fields on many evals at once — e.g. assign a model/checkpoint to every run in an
    'unassigned' group after the fact. {names: [...], fields: {...}}"""
    names = payload.get("names") or []
    fields = payload.get("fields") or {}
    if not names or not isinstance(fields, dict):
        raise HTTPException(400, "bulk_annotate needs names[] and fields{}")
    return _run(evals_mod.bulk_annotate, REPO_ROOT, [str(n) for n in names], fields)


@app.get("/api/evals/groups")
def api_evals_groups():
    """Per-model roll-up: run count, rated count, combined (mean) rating, success rate."""
    return {"groups": _run(evals_mod.group_summary, REPO_ROOT)}


@app.post("/api/evals/{name}/annotate")
def api_eval_annotate(name: str, payload: dict = Body(...)):
    _guard_plain(name)
    return _run(evals_mod.annotate_eval, REPO_ROOT, name, payload)


@app.post("/api/evals/{name}/rename")
def api_eval_rename(name: str, payload: dict = Body(...)):
    _guard_plain(name)
    if not payload.get("new_name"):
        raise HTTPException(400, "rename needs new_name")
    return _run(evals_mod.rename_eval, REPO_ROOT, name, payload["new_name"])


@app.post("/api/evals/{name}/delete")
def api_eval_delete(name: str, payload: dict = Body(...)):
    _guard_plain(name)
    _require_confirm(payload)
    return _run(evals_mod.delete_eval, REPO_ROOT, name)


@app.get("/api/evals/{name}/download")
def api_eval_download(name: str):
    import shutil as _sh
    from starlette.background import BackgroundTask

    _guard_plain(name)
    src = _run(evals_mod.eval_dir, REPO_ROOT, name)
    cache = os.path.join(REPO_ROOT, "outputs", "webui")
    os.makedirs(cache, exist_ok=True)
    base = os.path.join(cache, f"eval_{name}_{int(time.time())}")
    _sh.make_archive(base, "zip", src)
    zp = base + ".zip"
    return FileResponse(zp, filename=f"{name}.zip", media_type="application/zip",
                        background=BackgroundTask(lambda: os.path.isfile(zp) and os.remove(zp)))


@app.get("/api/evals/{name}/video")
def api_eval_video(name: str):
    import glob as _g

    _guard_plain(name)
    run = _run(evals_mod.eval_dir, REPO_ROOT, name)
    vids = sorted(v for v in _g.glob(os.path.join(run, "*.mp4")) if not v.endswith(".h264.mp4"))
    if not vids:
        raise HTTPException(404, "no video for this eval")
    # The robot PC writes rollouts with cv2's "mp4v" fourcc = MPEG-4 Part 2 (ffprobe: codec mpeg4,
    # Simple Profile). No browser decodes that in <video>, which is why the ▶ button showed a black
    # player. The headless OpenCV wheel on the robot PC has no H.264 encoder, so the fix lives here:
    # transcode to H.264 once on first request, cache it beside the original, serve the cache. Run
    # niced and on two threads so a request during a training run cannot starve its dataloader.
    src = vids[0]
    h264 = src[:-4] + ".h264.mp4"
    if not os.path.isfile(h264) or os.path.getmtime(h264) < os.path.getmtime(src):
        import shutil as _sh
        import subprocess as _sp
        if _sh.which("ffmpeg"):
            tmp = h264 + ".tmp.mp4"
            r = _sp.run(["nice", "-n", "19", "ffmpeg", "-y", "-v", "error", "-threads", "2", "-i", src,
                         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                         "-movflags", "+faststart", "-an", tmp], capture_output=True, text=True, timeout=300)
            if r.returncode == 0 and os.path.isfile(tmp):
                os.replace(tmp, h264)
            else:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                h264 = src          # serve the original rather than fail (download still works)
        else:
            h264 = src
    return FileResponse(h264, media_type="video/mp4",
                        headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-cache"})


@app.on_event("shutdown")
def _shutdown():
    VIEWER.stop()


@app.get("/healthz")
def healthz():
    return JSONResponse({"ok": True, "datasets_dir": DATASETS_DIR})


class _NoCacheStatic(StaticFiles):
    """Serve the UI assets with `no-cache` so a redeploy is picked up on the next page load.

    Not `no-store`: StaticFiles still sends an ETag, so the browser revalidates and normally gets a
    304 with an empty body — nothing on a LAN. Without this, the default (no Cache-Control at all)
    lets the browser heuristically cache app.js while revalidating index.html, so the row markup and
    the header markup come from different deploys. That does not fail loudly; it silently shifts every
    cell one column left and looks exactly like the new columns "not rendering".
    """

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


def _mount_static():
    if os.path.isdir(STATIC_DIR):
        app.mount("/", _NoCacheStatic(directory=STATIC_DIR, html=True), name="static")


ROBOT_URL = os.environ.get("VLA_ROBOT_URL", "http://192.168.11.80:8090/")
ROUTE_URL = os.environ.get("VLA_ROUTE_URL", "http://192.168.11.80:8092/")   # rd_ws/Scripts/route_editor.py on the robot
# the map's source: the robot's route editor (live) or route_editor.py --sim (rd_ws/Scripts/dev_stack.sh,
# normally on the box running this manager). The Map tab's live / sim switch picks it, for the Map and
# 3D tabs only — the Teleop and Routes tabs always talk to the robot.
SIM_ROUTE_URL = os.environ.get("VLA_SIM_ROUTE_URL", "http://127.0.0.1:8094/")   # 8094: never the live editor's 8092
MAP_SOURCE = os.environ.get("VLA_MAP_SOURCE", "live")      # "live" | "sim"
SIM_EDITOR = os.environ.get("VLA_SIM_EDITOR", "")          # rd_ws/Scripts/route_editor.py; "" = look in the parent checkout
_sim_proc = None                                           # the simulator this manager started (switch -> sim)


def _sim_editor_path() -> "str | None":
    # this repo is the rd_ws/VLA submodule, so rd_ws/Scripts is one level up from REPO_ROOT
    cands = ([SIM_EDITOR] if SIM_EDITOR else []) + [os.path.join(REPO_ROOT, "..", "Scripts", "route_editor.py"),
                                                     os.path.expanduser("~/rd_ws/Scripts/route_editor.py")]
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    return None


def _url_reachable(url: str, probe: str = "api/map", timeout: float = 1.0) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(_slash(url) + probe, timeout=timeout):
            return True
    except Exception:  # noqa: BLE001
        return False


def sim_start() -> dict:
    """Switching the map to sim: start route_editor.py --sim here (the box running this manager) unless
    something already answers at the sim URL. Same interpreter as the manager (numpy / Pillow / PyYAML)."""
    global _sim_proc
    from urllib.parse import urlparse
    import subprocess
    import sys
    import time as _t
    if _url_reachable(SIM_ROUTE_URL):
        return {"reachable": True, "started": False}
    path = _sim_editor_path()
    if not path:
        return {"reachable": False, "started": False, "error": "route_editor.py not found: set --sim-editor / VLA_SIM_EDITOR"}
    if _sim_proc is not None and _sim_proc.poll() is None:
        return {"reachable": False, "started": False, "error": f"simulator already starting (pid {_sim_proc.pid})"}
    port = urlparse(SIM_ROUTE_URL).port or 8094
    os.makedirs(os.path.join(REPO_ROOT, "log"), exist_ok=True)
    log_path = os.path.join(REPO_ROOT, "log", "sim_editor.log")
    log = open(log_path, "ab")
    _sim_proc = subprocess.Popen([sys.executable, path, "--sim", "--port", str(port)], cwd=os.path.dirname(path),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(24):                                    # up to ~12 s for the map + first sim tick
        _t.sleep(0.5)
        if _url_reachable(SIM_ROUTE_URL):
            return {"reachable": True, "started": True, "pid": _sim_proc.pid, "log": log_path}
        if _sim_proc.poll() is not None:
            break
    return {"reachable": False, "started": True, "pid": _sim_proc.pid, "log": log_path,
            "error": f"simulator did not come up (see {log_path})"}


def sim_stop() -> bool:
    """Stop the simulator THIS manager started (never one started by hand)."""
    global _sim_proc
    if _sim_proc is None or _sim_proc.poll() is not None:
        _sim_proc = None
        return False
    _sim_proc.terminate()
    try:
        _sim_proc.wait(5)
    except Exception:  # noqa: BLE001
        _sim_proc.kill()
    _sim_proc = None
    return True


import atexit as _atexit

_atexit.register(sim_stop)


def _slash(url: str) -> str:
    return url if url.endswith("/") else url + "/"


def _map_route_url() -> str:
    return _slash(SIM_ROUTE_URL if MAP_SOURCE == "sim" else ROUTE_URL)


def _browser_url(url: str, request: Request) -> str:
    """A loopback host in a configured URL means "this box" to the server, but the BROWSER loads these
    pages into iframes, where 127.0.0.1 is the viewer's own machine. Replace it with the host the
    browser used to reach us (works for the robot-side manager and for a Mac-side one alike)."""
    from urllib.parse import urlparse, urlunparse

    u = urlparse(url)
    if u.hostname in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
        host = request.url.hostname or "localhost"
        netloc = f"{host}:{u.port}" if u.port else host
        return urlunparse(u._replace(netloc=netloc))
    return url


@app.get("/api/robot")
def api_robot(request: Request):
    """The robot-side browser page (teleop panel / eval serve it on the robot PC, port 8090) and
    whether it answers right now. The RobotDisco Teleop tab embeds it and shows this status."""
    import urllib.request
    url = _slash(ROBOT_URL)
    cfg = None
    try:
        with urllib.request.urlopen(url + "api/config", timeout=1.5) as r:
            cfg = json.loads(r.read())
    except Exception:  # noqa: BLE001 — nothing running on the robot, or it is off
        cfg = None
    return {"url": _browser_url(url, request), "reachable": cfg is not None, "config": cfg}


@app.get("/api/route_editor")
def api_route_editor(request: Request, scope: str = "routes"):
    """The route editor the robot PC serves (rd_ws/Scripts/route_editor.py, supervisor unit
    robot_disco_route_editor, port 8092): the nav2 map, the robot's live pose and the route files for
    path.py. The Routes tab embeds it and shows whether it answers. `scope=map` (the Map and 3D tabs)
    follows the Map tab's live / sim switch; the Routes tab always gets the robot's."""
    import urllib.request
    url = _map_route_url() if scope == "map" else _slash(ROUTE_URL)
    info = None
    try:
        with urllib.request.urlopen(url + "api/map", timeout=1.5) as r:
            info = {"map": json.loads(r.read()).get("name")}
        with urllib.request.urlopen(url + "api/pose", timeout=1.5) as r:
            info.update(json.loads(r.read()))
    except Exception:  # noqa: BLE001 — the unit is not running, or the robot is off
        info = None
    return {"url": _browser_url(url, request), "reachable": info is not None, "info": info,
            "source": (MAP_SOURCE if scope == "map" else "live")}


# ---------------------------------------------------------------------------- Services tab
# supervisor's XML-RPC on the robot PC (the same API its own :9999 page uses). Native page instead of an
# iframe: fits the manager, works from the Mac, and adds the ordered base-stack restart.
SUPERVISOR_URL = os.environ.get("VLA_SUPERVISOR_URL", "http://192.168.11.80:9999/RPC2")
# the map page's node last: it, too, keeps dead publishers after a stack restart (no robot on the page)
BASE_STACK = [("robot_disco", 25.0), ("robot_disco_localization", 8.0), ("robot_disco_navigation", 5.0),
              ("robot_disco_collision_monitor", 3.0), ("robot_disco_route_editor", 0.0)]
_svc_job = {"running": False, "log": [], "t": 0.0}


def _supervisor():
    import http.client
    import xmlrpc.client

    class _Transport(xmlrpc.client.Transport):          # the stock transport has no timeout
        def make_connection(self, host):
            return http.client.HTTPConnection(host, timeout=4)
    return xmlrpc.client.ServerProxy(SUPERVISOR_URL, transport=_Transport(), allow_none=True)


@app.get("/api/services")
def api_services():
    """Every supervisor unit with its state, like `supervisorctl status`, plus the ordered-restart job."""
    try:
        infos = _supervisor().supervisor.getAllProcessInfo()
    except Exception as e:  # noqa: BLE001 — supervisor unreachable (robot off)
        return {"reachable": False, "error": f"{type(e).__name__}: {e}", "url": SUPERVISOR_URL, "job": _svc_job}
    now = time.time()
    out = []
    for i in infos:
        running = i["statename"] == "RUNNING"
        out.append({"name": i["name"], "group": i["group"], "state": i["statename"], "pid": i["pid"] or None,
                    "uptime_s": int(now - i["start"]) if running and i["start"] else None,
                    "description": i["description"], "stdout": i.get("stdout_logfile"), "stderr": i.get("stderr_logfile")})
    return {"reachable": True, "url": SUPERVISOR_URL, "services": out, "job": _svc_job}


@app.post("/api/services/restart_base")
def api_services_restart_base():
    """Restart the base stack IN ORDER with settle times: base launch (controller manager, lidars),
    then localization, navigation, collision monitor. A unit restarted before the base settles ends up
    bound to publishers that no longer exist (seen 2026-10-30: AMCL and the collision monitor with a
    frozen odom buffer, robot stopped). Runs in the background; progress in /api/services' `job`."""
    import threading
    if _svc_job["running"]:
        raise HTTPException(409, "already running")

    def run():
        _svc_job.update(running=True, log=[], t=time.time())
        sv = _supervisor().supervisor
        try:
            for name, settle in BASE_STACK:
                _svc_job["log"].append(f"restarting {name}")
                try:
                    sv.stopProcess(name, True)
                except Exception:  # noqa: BLE001
                    pass
                sv.startProcess(name, True)
                if settle:
                    _svc_job["log"].append(f"settling {settle:.0f} s")
                    time.sleep(settle)
            _svc_job["log"].append("done — set the initial pose again if the robot was localized")
        except Exception as e:  # noqa: BLE001
            _svc_job["log"].append(f"failed: {e}")
        finally:
            _svc_job["running"] = False
    threading.Thread(target=run, daemon=True).start()
    return {"ok": True}


class ServiceAction(BaseModel):
    action: str


@app.post("/api/services/{name}")
def api_service_action(name: str, body: ServiceAction):
    """start | stop | restart one unit (restart = stop, wait, start — what supervisorctl does)."""
    sv = _supervisor().supervisor
    try:
        if body.action == "start":
            sv.startProcess(name, True)
        elif body.action == "stop":
            sv.stopProcess(name, True)
        elif body.action == "restart":
            try:
                sv.stopProcess(name, True)
            except Exception:  # noqa: BLE001 — already stopped
                pass
            sv.startProcess(name, True)
        else:
            raise HTTPException(400, "action must be start|stop|restart")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"supervisor: {e}")
    return {"ok": True, "name": name, "action": body.action}


@app.get("/api/services/{name}/log")
def api_service_log(name: str, stream: str = "stdout", lines: int = 80):
    """The tail of a unit's stdout / stderr log, the way the supervisor page shows it."""
    sv = _supervisor().supervisor
    try:
        fn = sv.tailProcessStderrLog if stream == "stderr" else sv.tailProcessStdoutLog
        data, offset, overflow = fn(name, 0, 60000)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"supervisor: {e}")
    import re as _re
    text = _re.sub(r"\x1b\[[0-9;]*m", "", data)          # the ROS colour codes
    tail = text.splitlines()[-max(1, min(lines, 2000)):]
    return {"name": name, "stream": stream, "lines": tail}


@app.get("/api/map_source")
def api_map_source_get(request: Request):
    """The Map tab's switch: which route editor the Map and 3D tabs embed (live robot or simulation)."""
    return {"mode": MAP_SOURCE, "urls": {"live": _browser_url(_slash(ROUTE_URL), request), "sim": _browser_url(_slash(SIM_ROUTE_URL), request)},
            "sim": {"running": _sim_proc is not None and _sim_proc.poll() is None, "editor": _sim_editor_path()}}


class MapSourceBody(BaseModel):
    mode: str


@app.post("/api/map_source")
def api_map_source_set(body: MapSourceBody, request: Request):
    global MAP_SOURCE
    if body.mode not in ("live", "sim"):
        raise HTTPException(400, "mode must be live or sim")
    MAP_SOURCE = body.mode
    out = api_map_source_get(request)
    if body.mode == "sim":
        out["start"] = sim_start()                     # starts the simulator here if nothing answers
        out["sim"]["running"] = _sim_proc is not None and _sim_proc.poll() is None
    else:
        out["stopped"] = sim_stop()
        out["sim"]["running"] = False
    return out
_mount_static()


def main() -> None:
    global DATASETS_DIR, REPO_ROOT, VIEWER, ROBOT_URL, ROUTE_URL, SIM_ROUTE_URL, MAP_SOURCE, SIM_EDITOR, SUPERVISOR_URL
    ap = argparse.ArgumentParser(description="VLA dataset-management web UI")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--datasets-dir", default="outputs/datasets")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--web-port", type=int, default=9090, help="rerun web-viewer app port")
    ap.add_argument("--robot-url", default=ROBOT_URL,
                    help="where the robot PC serves its browser page (teleop panel / eval, --ui web)")
    ap.add_argument("--route-url", default=ROUTE_URL,
                    help="the route editor on the robot PC (VLA_ROUTE_URL), embedded in the Routes tab")
    ap.add_argument("--sim-route-url", default=SIM_ROUTE_URL, help="route_editor.py --sim of the simulation stack (VLA_SIM_ROUTE_URL)")
    ap.add_argument("--map-source", choices=["live", "sim"], default=MAP_SOURCE, help="what the Map tab embeds at start (its switch changes it)")
    ap.add_argument("--supervisor-url", default=SUPERVISOR_URL, help="supervisor's XML-RPC on the robot PC (VLA_SUPERVISOR_URL), for the Services tab")
    ap.add_argument("--sim-editor", default=SIM_EDITOR, help="rd_ws/Scripts/route_editor.py to launch with --sim when the switch goes to sim (VLA_SIM_EDITOR; default: next to this repo)")
    args = ap.parse_args()
    ROBOT_URL = args.robot_url
    ROUTE_URL = args.route_url
    SIM_ROUTE_URL = args.sim_route_url
    MAP_SOURCE = args.map_source
    SIM_EDITOR = args.sim_editor
    SUPERVISOR_URL = args.supervisor_url

    DATASETS_DIR = os.path.abspath(args.datasets_dir)
    REPO_ROOT = os.path.abspath(args.repo_root)
    VIEWER = ViewerManager(REPO_ROOT, web_port=args.web_port)
    print(f"datasets_dir = {DATASETS_DIR}\nrepo_root    = {REPO_ROOT}\nserving      http://{args.host}:{args.port}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
