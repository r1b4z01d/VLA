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
import re
import threading
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
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
FOLKS_URL = os.environ.get("VLA_FOLKS_URL", "http://192.168.11.80:8096/")   # robot_face's "Folks" page (known people) on the robot
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
    whether it answers right now. The VLA Teleop tab embeds it and shows this status."""
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


@app.get("/api/folks")
def api_folks(request: Request):
    """The face node's Folks page on the robot PC (robot_face, port 8096): the people the robot has
    learned, with their pictures and editable names. The Folks tab embeds it and shows whether it answers."""
    import urllib.request
    url = _slash(FOLKS_URL)
    info = None
    try:
        with urllib.request.urlopen(url + "api/people", timeout=1.5) as r:
            info = {"count": len(json.loads(r.read()))}
        with urllib.request.urlopen(url + "api/live", timeout=1.5) as r:
            info["live"] = json.loads(r.read())
    except Exception:  # noqa: BLE001 — the face node is not running, or the robot is off
        info = None
    return {"url": _browser_url(url, request), "reachable": info is not None, "info": info}


def _folks_json(method: str, path: str, body: "dict | None" = None, timeout: float = 4.0):
    """JSON from the face node's Folks server on the robot; 502 when it is not up."""
    import urllib.error
    import urllib.request
    url = _slash(FOLKS_URL) + path.lstrip("/")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error")
        except Exception:  # noqa: BLE001
            msg = None
        raise HTTPException(e.code, msg or f"face node: {e.code}")
    except Exception as e:  # noqa: BLE001 — the face node is not running, or the robot is off
        raise HTTPException(502, f"face node not reachable at {url}: {e}")


class VolumeBody(BaseModel):
    volume: "float | None" = None    # 0-100 %
    muted: "bool | None" = None


@app.get("/api/volume")
def api_volume():
    """The robot speaker's volume. The face node owns the speaker sink (robot_face audio_sink.py) and
    serves this next to its Folks page; the header slider in every tab relays through here."""
    return _folks_json("GET", "api/volume")


@app.post("/api/volume")
def api_volume_set(body: VolumeBody):
    return _folks_json("POST", "api/volume", {k: v for k, v in body.dict().items() if v is not None})


# ---------------------------------------------------------------------------- Landmarks tab
# Named places the robot can navigate to. The data lives with the robot's route editor
# (rd_ws/Scripts/route_editor.py: landmarks/landmarks.yaml + the photos, /api/landmarks*), which also
# has the robot's live map pose, the camera frames and the /goal_pose publisher — this manager only
# relays, following the Map tab's live / sim switch like the Map and 3D tabs do.
def _editor_call(method: str, path: str, body: "dict | None" = None, timeout: float = 8.0) -> tuple:
    """(status, content-type, bytes) from the route editor the Map tab follows; 502 when it is not up."""
    import urllib.error
    import urllib.request
    url = _map_route_url() + path.lstrip("/")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()
    except Exception as e:  # noqa: BLE001 — the editor is not running, or the robot is off
        raise HTTPException(502, f"route editor not reachable at {url}: {e}")


def _editor_json(method: str, path: str, body: "dict | None" = None, timeout: float = 8.0):
    status, _ctype, raw = _editor_call(method, path, body, timeout)
    try:
        obj = json.loads(raw)
    except Exception:  # noqa: BLE001
        raise HTTPException(502, f"route editor: {status} {raw[:120]!r}")
    if status >= 400:
        raise HTTPException(status, (obj.get("error") if isinstance(obj, dict) else None) or f"route editor: {status}")
    return obj


class LandmarkBody(BaseModel):
    name: "str | None" = None
    x: "float | None" = None
    y: "float | None" = None
    yaw: "float | None" = None
    pose: "bool | None" = None       # recapture: move the landmark to where the robot is now
    images: "bool | None" = None     # recapture: take the photos again (default)


def _lm_fields(body: LandmarkBody, *keys) -> dict:
    return {k: getattr(body, k) for k in keys if getattr(body, k) is not None}


@app.get("/api/landmarks")
def api_landmarks(request: Request):
    """The landmarks, the map the robot runs, its live pose and which cameras have a frame right now."""
    d = _editor_json("GET", "api/landmarks", timeout=4.0)
    d["source"] = MAP_SOURCE
    d["editor_url"] = _browser_url(_map_route_url(), request)
    return d


@app.post("/api/landmarks")
def api_landmark_add(body: LandmarkBody):
    """Add a landmark where the robot stands (or at a typed x / y / yaw); the photos are taken now."""
    return _editor_json("POST", "api/landmarks", _lm_fields(body, "name", "x", "y", "yaw"), timeout=20.0)


@app.post("/api/landmarks/{lid}")
def api_landmark_edit(lid: str, body: LandmarkBody):
    return _editor_json("POST", f"api/landmarks/{lid}", _lm_fields(body, "name", "x", "y", "yaw"))


@app.post("/api/landmarks/{lid}/goto")
def api_landmark_goto(lid: str):
    """Queue the landmark's pose as a nav2 /goal_pose (the editor's publisher)."""
    return _editor_json("POST", f"api/landmarks/{lid}/goto", {})


@app.post("/api/landmarks/{lid}/recapture")
def api_landmark_recapture(lid: str, body: LandmarkBody):
    return _editor_json("POST", f"api/landmarks/{lid}/recapture", _lm_fields(body, "pose", "images"), timeout=20.0)


@app.delete("/api/landmarks/{lid}")
def api_landmark_delete(lid: str):
    return _editor_json("DELETE", f"api/landmarks/{lid}")


# A landmark's TASKS tie a policy run (an eval) to the place: "pick up tigger" at the Right Bookshelf = serve
# tigger_1_merged_smolvla_1, task "Grab and hold tigger", vla_action.py options. The voice assistant on the robot
# reads the same landmarks.yaml ("Robot Disco, pick up tigger" drives there first); ▶ Run hands it the job.
class TaskBody(BaseModel):
    name: "str | None" = None          # what you say
    model: "str | None" = None         # training run (or run@step) to serve
    task: "str | None" = None          # the instruction the policy was trained with
    seconds: "float | None" = None
    still: "float | None" = None
    min_seconds: "float | None" = None
    done_when: "str | None" = None     # none | closed | open
    still_mode: "str | None" = None    # all | fingers
    keep_grasp: "bool | None" = None
    home: "bool | None" = None
    extra: "str | None" = None


_TASK_KEYS = ("name", "model", "task", "seconds", "still", "min_seconds", "done_when", "still_mode", "keep_grasp", "home", "extra")


def _task_fields(body: TaskBody) -> dict:
    return {k: getattr(body, k) for k in _TASK_KEYS if getattr(body, k) is not None}


@app.post("/api/landmarks/{lid}/tasks")
def api_landmark_task_add(lid: str, body: TaskBody):
    return _editor_json("POST", f"api/landmarks/{lid}/tasks", _task_fields(body))


@app.post("/api/landmarks/{lid}/tasks/{tid}")
def api_landmark_task_edit(lid: str, tid: str, body: TaskBody):
    return _editor_json("POST", f"api/landmarks/{lid}/tasks/{tid}", _task_fields(body))


@app.post("/api/landmarks/{lid}/tasks/{tid}/run")
def api_landmark_task_run(lid: str, tid: str):
    """Hand the job to the voice assistant (robot_voice): it drives to the landmark, then runs the policy."""
    return _editor_json("POST", f"api/landmarks/{lid}/tasks/{tid}/run", {})


@app.delete("/api/landmarks/{lid}/tasks/{tid}")
def api_landmark_task_delete(lid: str, tid: str):
    return _editor_json("DELETE", f"api/landmarks/{lid}/tasks/{tid}")


@app.get("/api/landmarks/img/{name}")
def api_landmark_image(name: str):
    status, _ctype, raw = _editor_call("GET", f"landmarks/img/{name}", timeout=6.0)
    if status != 200:
        raise HTTPException(404, "no such image")
    return Response(content=raw, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@app.get("/api/landmarks/map")
def api_landmark_map():
    """The nav map's geometry (resolution, origin, size, name), for the cards' map thumbnails."""
    return _editor_json("GET", "api/map", timeout=4.0)


@app.get("/api/landmarks/map.png")
def api_landmark_map_png():
    status, _ctype, raw = _editor_call("GET", "map.png", timeout=6.0)
    if status != 200:
        raise HTTPException(404, "no map image")
    return Response(content=raw, media_type="image/png", headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------------------- Services tab
# supervisor's XML-RPC on the robot PC (the same API its own :9999 page uses). Native page instead of an
# iframe: fits the manager, works from the Mac, and adds the ordered base-stack restart.
SUPERVISOR_URL = os.environ.get("VLA_SUPERVISOR_URL", "http://192.168.11.80:9999/RPC2")
# then the AprilTag relocalizer (needs AMCL's map frame + the camera) and the map page's node last: it,
# too, keeps dead publishers after a stack restart (no robot on the page)
BASE_STACK = [("robot_disco", 25.0), ("robot_disco_localization", 8.0), ("robot_disco_navigation", 5.0),
              ("robot_disco_collision_monitor", 3.0), ("robot_disco_apriltag_relocalization", 2.0),
              ("robot_disco_route_editor", 0.0)]
_svc_job = {"running": False, "log": [], "t": 0.0}


# supervisor copies a log tail into its XML reply as-is, and XML 1.0 forbids the control characters that
# logs do carry (the ESC of ROS colour codes — the realsense node's warnings, the collision monitor —, the
# NUL a power cut leaves in a file), so the parser refused the whole reply: "not well-formed (invalid
# token): line 424, column 27". The transport swaps each such byte for a character reference into the
# private-use area before the parser sees it and swaps it back afterwards, so the proxy returns exactly
# what supervisor sent.
_XML_FORBIDDEN = re.compile(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_XML_RESTORE = {0xE000 + n: chr(n) for n in range(0x20)}


def _xml_restore(v):
    if isinstance(v, str):
        return v.translate(_XML_RESTORE)
    if isinstance(v, list):
        return [_xml_restore(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_xml_restore(x) for x in v)
    if isinstance(v, dict):
        return {k: _xml_restore(x) for k, x in v.items()}
    return v


def _supervisor():
    import http.client
    import xmlrpc.client

    class _Transport(xmlrpc.client.Transport):          # the stock transport has no timeout
        def make_connection(self, host):
            return http.client.HTTPConnection(host, timeout=4)

        def getparser(self):
            parser, unmarshaller = super().getparser()
            feed, close = parser.feed, unmarshaller.close
            parser.feed = lambda data: feed(_XML_FORBIDDEN.sub(lambda m: b"&#x%04X;" % (0xE000 + m.group()[0]), data))
            unmarshaller.close = lambda: _xml_restore(close())
            return parser, unmarshaller
    return xmlrpc.client.ServerProxy(SUPERVISOR_URL, transport=_Transport(), allow_none=True)


def _stop_wait(sv, name: str, timeout: float = 25.0) -> str:
    """Stop a unit and wait until supervisor says it is down. stopProcess(wait=True) blocks for up to the
    unit's stopwaitsecs (10 s) — longer than the RPC transport's 4 s timeout — so the call raised, the
    error was swallowed and the following start raced the stop ('ABNORMAL_TERMINATION', 2026-11-03)."""
    try:
        sv.stopProcess(name, False)
    except Exception:  # noqa: BLE001 — already stopped
        pass
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            st = sv.getProcessInfo(name)["statename"]
        except Exception:  # noqa: BLE001
            st = "?"
        if st in ("STOPPED", "EXITED", "FATAL", "UNKNOWN"):
            return st
        time.sleep(0.5)
    return "STOPPING"


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
                _stop_wait(sv, name)
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
            _stop_wait(sv, name)
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
    import xmlrpc.client
    sv = _supervisor().supervisor
    fn = sv.tailProcessStderrLog if stream == "stderr" else sv.tailProcessStdoutLog
    for length in (60000, 59999, 59998, 59997):
        try:
            data, offset, overflow = fn(name, 0, length)
            break
        except xmlrpc.client.ProtocolError as e:         # HTTP 500: supervisor decodes the tail as strict UTF-8,
            err = e                                      # and this window starts inside a multi-byte character
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"supervisor: {e}")
    else:
        raise HTTPException(502, f"supervisor: {err}")
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", data)         # ANSI escapes (the ROS colour codes)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)    # NUL from a power cut, stray controls
    tail = text.splitlines()[-max(1, min(lines, 2000)):]
    return {"name": name, "stream": stream, "lines": tail}


# ---------------------------------------------------------------------------- Live view activity log
# One merged, filtered feed of what the robot's units are saying — the Live view's collapsible log pane.
# Built from the supervisor log tails (the same source as the Services tab), a short window per unit so a
# poll stays cheap; ROS lines carry an epoch stamp that orders the merge, other lines inherit the previous
# line's stamp in their file.
ACTIVITY_UNITS = [("robot_disco_navigation", "nav"), ("robot_disco_collision_monitor", "collision"),
                  ("robot_disco_localization", "loc"), ("robot_disco_route_editor", "route"),
                  ("robot_voice", "voice"), ("robot_face", "face"), ("vla_eval", "eval"),
                  ("vla_infer_server", "infer"), ("robot_disco", "base"),
                  ("robot_disco_apriltag_relocalization", "tag")]
_ROS_LINE = re.compile(r"^\[(INFO|WARN|WARNING|ERROR|FATAL|DEBUG)\]\s*\[(\d{9,}(?:\.\d+)?)\]\s*\[([^\]]+)\]:\s*(.*)$")
_ROS_LINE2 = re.compile(r"^\[([\w.\-]+)-\d+\]\s*\[(INFO|WARN|WARNING|ERROR|FATAL|DEBUG)\]\s*\[(\d{9,}(?:\.\d+)?)\]\s*\[([^\]]+)\]:\s*(.*)$")
_PY_LEVEL = re.compile(r"\b(ERROR|CRITICAL|Traceback|WARNING|WARN)\b")
# chatter that says nothing about what the robot is doing
_ACT_NOISE = re.compile(r"(Publishing:|Received|get_state|lifecycle_manager.*(Waiting for service|Creating bond)|"
                        r"Transform data too old|Failed to bring up all nodes|GET /|POST /|HTTP/1\.[01]\" 200|"
                        r"Message Filter dropping|UDP|keepalive|heartbeat)", re.I)
_ACT_WAIT = re.compile(r"(wait|stop|abort|fail|timeout|time out|stuck|reject|cancel|no path|collision|blocked|"
                       r"recover|retry|lost|not reachable|unavailable)", re.I)


def _unit_tail(sv, name: str, stream: str, nbytes: int = 16000) -> list:
    fn = sv.tailProcessStderrLog if stream == "stderr" else sv.tailProcessStdoutLog
    for length in (nbytes, nbytes - 1, nbytes - 2, nbytes - 3):     # see api_service_log: window may split a UTF-8 char
        try:
            data = fn(name, 0, length)[0]
            break
        except Exception:  # noqa: BLE001
            data = None
    if not data:
        return []
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", data)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.splitlines()[1:]                                    # first line is probably cut


def _activity_lines(unit: str, short: str, lines: list) -> list:
    out, last_t = [], 0.0
    for ln in lines:
        ln = ln.rstrip()
        if not ln.strip():
            continue
        m = _ROS_LINE.match(ln)
        m2 = None if m else _ROS_LINE2.match(ln)
        if m:
            lvl, t, node, msg = m.group(1), float(m.group(2)), m.group(3), m.group(4)
        elif m2:
            lvl, t, node, msg = m2.group(2), float(m2.group(3)), m2.group(4), m2.group(5)
        else:
            lvl, t, node, msg = ("ERROR" if _PY_LEVEL.search(ln) and "WARN" not in ln.upper() else
                                 "WARN" if "WARN" in ln.upper() else "INFO"), last_t, "", ln
        if lvl == "DEBUG" or _ACT_NOISE.search(msg):
            continue
        last_t = t or last_t
        out.append({"t": t or last_t, "unit": short, "node": node.split(".")[-1], "level": "WARN" if lvl == "WARNING" else lvl,
                    "msg": msg[:400], "wait": bool(_ACT_WAIT.search(msg))})
    return out


@app.get("/api/activity")
def api_activity(lines: int = 200, units: str = ""):
    """Recent log lines from the robot's units, merged by time and with the noise removed, plus each
    unit's supervisor state — what the Live view's log pane shows ("what is RD doing / waiting for")."""
    from concurrent.futures import ThreadPoolExecutor
    want = {u for u in units.split(",") if u}
    try:
        infos = {i["name"]: i for i in _supervisor().supervisor.getAllProcessInfo()}
    except Exception as e:  # noqa: BLE001
        return {"reachable": False, "error": f"{type(e).__name__}: {e}", "lines": [], "units": []}
    now = time.time()
    chosen = [(n, s) for n, s in ACTIVITY_UNITS if n in infos and (not want or s in want)]

    def pull(item):
        n, s = item
        if infos[n]["statename"] not in ("RUNNING", "STARTING", "BACKOFF", "FATAL", "EXITED"):
            return []
        sv = _supervisor().supervisor                                # one proxy per thread
        rows = []
        for stream in ("stdout", "stderr"):
            try:
                rows += _activity_lines(n, s, _unit_tail(sv, n, stream))
            except Exception:  # noqa: BLE001
                pass
        return rows
    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = [r for part in ex.map(pull, chosen) for r in part]
    rows.sort(key=lambda r: r["t"])
    merged = []                                                      # collapse consecutive repeats into x N
    for r in rows:
        p = merged[-1] if merged else None
        if p and p["unit"] == r["unit"] and p["msg"] == r["msg"] and p["level"] == r["level"]:
            p["count"] = p.get("count", 1) + 1
            p["t"] = r["t"]
        else:
            merged.append(dict(r))
    return {"reachable": True, "now": now, "lines": merged[-max(10, min(lines, 800)):],
            "units": [{"unit": s, "name": n, "state": infos[n]["statename"],
                       "uptime_s": int(now - infos[n]["start"]) if infos[n]["statename"] == "RUNNING" and infos[n]["start"] > 1e9 else None}
                      for n, s in ACTIVITY_UNITS if n in infos]}


# ---------------------------------------------------------------------------- Maps tab
# The robot's nav maps live with the route editor (rd_ws/src/robot_disco/maps, /api/maps there); the
# manager relays, and starts / stops the units involved: SLAM (robot_disco_slam) while a map is made or
# re-mapped, localization + navigation when another map is activated. In sim (the Live tab's switch)
# only the editor is involved.
MAPS_UNITS = {"slam": "robot_disco_slam", "localization": "robot_disco_localization", "navigation": "robot_disco_navigation"}
_maps_job = {"running": False, "log": [], "t": 0.0}


def _unit_states() -> dict:
    if MAP_SOURCE == "sim":
        return {}
    try:
        return {p["name"]: p["statename"] for p in _supervisor().supervisor.getAllProcessInfo() if p["name"] in MAPS_UNITS.values()}
    except Exception as e:  # noqa: BLE001
        return {"error": f"supervisor: {e}"}


def _unit(action: str, name: str, log: list) -> None:
    sv = _supervisor().supervisor
    try:
        if action in ("stop", "restart"):
            _stop_wait(sv, name)
        if action in ("start", "restart"):
            sv.startProcess(name, True)
        log.append(f"{action} {name}")
    except Exception as e:  # noqa: BLE001
        log.append(f"{action} {name} failed: {e}")
        raise


@app.get("/api/maps")
def api_maps():
    """The robot's maps (+ which one is active), the SLAM / localization / navigation unit states, the
    activation job and whether SLAM is mapping right now."""
    d = _editor_json("GET", "api/maps", timeout=6.0)
    d["units"] = _unit_states()
    d["job"] = _maps_job
    d["source"] = MAP_SOURCE
    return d


@app.get("/api/maps/{name}.png")
def api_map_png(name: str):
    status, ctype, raw = _editor_call("GET", f"api/maps/{name}.png", timeout=10.0)
    if status >= 400:
        raise HTTPException(status, raw[:120].decode(errors="replace"))
    return Response(content=raw, media_type="image/png", headers={"Cache-Control": "max-age=30"})


class SlamBody(BaseModel):
    mode: "str | None" = None       # new | continue
    name: "str | None" = None
    seed_pose: bool = True          # continue: resume at the robot's current pose, not the posegraph's dock


@app.post("/api/maps/slam/start")
def api_maps_slam_start(body: SlamBody):
    """Start mapping: a fresh map (mode new) or continue an existing map's posegraph. The editor records
    the request for the SLAM launch; localization (AMCL) must stop (both own map -> odom), then the
    SLAM unit starts. Drive the robot around; the Live tab shows the map grow."""
    # The seed is resolved editor-side BEFORE localization is stopped below — once AMCL is down there
    # is no map->base_footprint left to read the robot's position from.
    d = _editor_json("POST", "api/maps/slam/start",
                     {"mode": body.mode or "new", "name": body.name or "", "seed_pose": body.seed_pose},
                     timeout=6.0)
    log: list = []
    if MAP_SOURCE != "sim":
        states = _unit_states()
        if states.get(MAPS_UNITS["localization"]) == "RUNNING":
            _unit("stop", MAPS_UNITS["localization"], log)
        _unit("restart", MAPS_UNITS["slam"], log)
    d["units"] = log
    return d


@app.post("/api/maps/slam/save")
def api_maps_slam_save(body: SlamBody):
    """Save what SLAM has so far as <name> (map image + yaml + posegraph), through slam_toolbox's services."""
    return _editor_json("POST", "api/maps/slam/save", {"name": body.name or ""}, timeout=75.0)


@app.post("/api/maps/slam/stop")
def api_maps_slam_stop():
    """Stop the SLAM unit and go back to the file map (activate the saved map to run on it)."""
    log: list = []
    if MAP_SOURCE != "sim":
        try:
            _unit("stop", MAPS_UNITS["slam"], log)
        except Exception:  # noqa: BLE001
            pass
    d = _editor_json("POST", "api/maps/slam/stop", {}, timeout=10.0)
    d["units"] = log
    return d


class ActivateBody(BaseModel):
    restart: bool = True


@app.post("/api/maps/{name}/activate")
def api_map_activate(name: str, body: ActivateBody):
    """Make <name> the robot's map: the editor serves it and tags new routes / landmarks with it; the
    localization and navigation units are restarted (in that order, with a settle) so AMCL and the
    costmaps load it — in the background, progress in /api/maps' job."""
    import threading
    d = _editor_json("POST", f"api/maps/{name}/activate", {}, timeout=10.0)
    if body.restart and MAP_SOURCE != "sim":
        if _maps_job["running"]:
            raise HTTPException(409, "a map switch is already running")

        def run():
            _maps_job.update(running=True, log=[f"activating {name}"], t=time.time())
            try:
                states = _unit_states()
                for key, settle in (("localization", 6.0), ("navigation", 0.0)):
                    unit = MAPS_UNITS[key]
                    if states.get(unit) == "RUNNING":
                        _unit("restart", unit, _maps_job["log"])
                        if settle:
                            _maps_job["log"].append(f"settling {settle:.0f} s")
                            time.sleep(settle)
                    else:
                        _maps_job["log"].append(f"{unit} not running — left alone")
                _maps_job["log"].append("done — set the initial pose on the new map")
            except Exception as e:  # noqa: BLE001
                _maps_job["log"].append(f"failed: {e}")
            finally:
                _maps_job["running"] = False
        threading.Thread(target=run, daemon=True).start()
        d["job"] = "started"
    return d


class DrawBody(BaseModel):
    png: str                        # data URL or base64 PNG, greyscale, the map's size
    save_as: "str | None" = None


@app.post("/api/maps/{name}/draw")
def api_map_draw(name: str, body: DrawBody):
    """Write the drawing editor's result as <name> (or as a new map, save_as)."""
    return _editor_json("POST", f"api/maps/{name}/draw", {"png": body.png, "save_as": body.save_as}, timeout=40.0)


@app.delete("/api/maps/{name}")
def api_map_delete(name: str):
    return _editor_json("DELETE", f"api/maps/{name}", timeout=10.0)


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

# ---------------------------------------------------------------------------- Hand tab
# The AmazingHand is an ESP32 on WiFi that takes ASCII lines on UDP :8766 (ur5e_lerobot/hand/
# amazing_hand_client.py has the protocol): "J:a0,..,a7,speed" sets the 8 servo offsets (deg), one
# character 0-9/a/b runs a firmware gesture, "Q" answers "K:<uptime ms>". The tab is a native page
# (presets, per-finger sliders, the gestures, a raw line) and this server sends the datagrams, like the
# teleop panel and the eval do: fire-and-forget, so a hole in the lossy link only costs the lines inside
# it. Nothing holds a session with the hand — a teleop or eval stream to it just wins the next line.
HAND_HOST = os.environ.get("VLA_HAND_HOST", "192.168.11.117")
HAND_UDP_PORT = int(os.environ.get("VLA_HAND_UDP_PORT", "8766"))
HAND_GESTURES = {"0": "open", "1": "close", "2": "open, progressive", "3": "spread", "4": "clench",
                 "5": "index pointing", "6": "no-no-no", "7": "perfect (OK)", "8": "victory", "9": "scissors",
                 "a": "pinch", "b": "middle finger"}   # parseHandData() in AmazingHand-main/.../HAND_CTRL.h
HAND_SPEED_MIN, HAND_SPEED_MAX = 50, 2000               # the firmware constrains J: speeds to this range
_hand_lock = threading.Lock()
_hand = {"curls": [0.0] * 4, "abduct": [0.0] * 4, "speed": 200, "sent": 0, "errors": 0, "last": None, "alive_t": None}


def _hand_mapping():
    """The curl/abduct -> servo-offset tables live with the hand client; import lazily and find the repo
    root ourselves when the server was started without PYTHONPATH=. (the robot's unit sets it)."""
    try:
        from ur5e_lerobot.hand import amazing_hand_client as m
    except ImportError:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ur5e_lerobot.hand import amazing_hand_client as m
    return m


def _hand_send(line: str, host: "str | None" = None, reply_wait: float = 0.0, count: bool = True) -> dict:
    """One datagram to the hand; with reply_wait, also wait that long for an answer (only "Q" gets one)."""
    import socket
    host = (host or HAND_HOST).strip()
    out = {"host": host, "line": line, "reply": None, "error": None}
    with _hand_lock:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(reply_wait if reply_wait > 0 else 0.2)
                sock.connect((host, HAND_UDP_PORT))
                sock.send((line.rstrip("\n") + "\n").encode("ascii"))
                if count:
                    _hand["sent"] += 1
                if reply_wait > 0:
                    try:
                        out["reply"] = sock.recv(256).decode("ascii", "ignore").strip()
                    except socket.timeout:
                        pass
        except (OSError, UnicodeEncodeError) as e:
            if count:
                _hand["errors"] += 1
            out["error"] = f"{type(e).__name__}: {e}"
        if count:
            _hand["last"] = {"line": line, "t": time.time(), "error": out["error"]}
    return out


def _hand_probe(host: "str | None" = None, tries: int = 4, wait: float = 0.35) -> dict:
    """Liveness: "Q" -> "K:<uptime ms>". Several tries, because the link drops 20-50% of packets (a probe
    needs both directions to survive); a miss is reported together with when the hand last answered."""
    last = None
    for _ in range(tries):
        r = _hand_send("Q", host, reply_wait=wait, count=False)
        if r["reply"] and r["reply"].startswith("K:"):
            try:
                up = int(r["reply"][2:].strip())
            except ValueError:
                up = None
            with _hand_lock:
                _hand["alive_t"] = time.time()
            return {"alive": True, "uptime_ms": up, "error": None}
        last = r["error"] or last
    return {"alive": False, "uptime_ms": None, "error": last or "no answer to the liveness probe (hand off, rebooting, or the WiFi link is dropping everything)"}


@app.get("/api/hand")
def api_hand(host: "str | None" = None):
    """The hand's liveness (UDP probe), the last command this server sent it, and the gesture table."""
    h = (host or HAND_HOST).strip()
    p = _hand_probe(h)
    with _hand_lock:
        st = {k: _hand[k] for k in ("curls", "abduct", "speed", "sent", "errors", "last", "alive_t")}
    seen = None if st.pop("alive_t") is None else round(time.time() - _hand["alive_t"], 1)
    return {"host": h, "udp_port": HAND_UDP_PORT, "alive": p["alive"], "uptime_ms": p["uptime_ms"], "error": p["error"], "last_alive_s": seen,
            "console_url": f"http://{h}/webserial", "gestures": HAND_GESTURES,
            "speed_range": [HAND_SPEED_MIN, HAND_SPEED_MAX], "state": st}


class HandCurls(BaseModel):
    # the 4 curl channels in schema order (named index, middle, ring, thumb), 0 open .. 1 closed. On this hand
    # they drive middle, pinky, index, thumb (measured 2026-10-26, ur5e_lerobot/teleop/manual_panel.py) — the
    # page labels its sliders by physical finger; the channel order stays, recorded datasets replay through it.
    curls: list[float]
    abduct: "list[float] | None" = None  # per channel, -1 .. 1 (side-to-side spread), 0 = none
    speed: "int | None" = None
    host: "str | None" = None


@app.post("/api/hand/curls")
def api_hand_curls(body: HandCurls):
    """Per-finger curl (+ optional abduction) -> one J: line, through the hand client's mapping tables."""
    m = _hand_mapping()
    if len(body.curls) != 4:
        raise HTTPException(400, "curls must have 4 values (the channels index, middle, ring, thumb)")
    curls = [min(1.0, max(0.0, float(c))) for c in body.curls]
    abduct = [0.0] * 4
    if body.abduct is not None:
        if len(body.abduct) != 4:
            raise HTTPException(400, "abduct must have 4 values")
        abduct = [min(1.0, max(-1.0, float(a))) for a in body.abduct]
    speed = int(body.speed if body.speed is not None else _hand["speed"])
    speed = min(HAND_SPEED_MAX, max(HAND_SPEED_MIN, speed))
    offsets = m.flex_abduct_to_offsets(curls, abduct) if any(abs(a) > 1e-6 for a in abduct) else m.curls_to_offsets(curls)
    line = m.format_joint_command(offsets, speed).strip()
    r = _hand_send(line, body.host)
    with _hand_lock:
        _hand["curls"], _hand["abduct"], _hand["speed"] = curls, abduct, speed
    if r["error"]:
        raise HTTPException(502, f"send to {r['host']}:{HAND_UDP_PORT}/udp failed: {r['error']}")
    return {"line": line, "offsets": [round(o, 2) for o in offsets], "curls": curls, "abduct": abduct, "speed": speed}


class HandLine(BaseModel):
    line: str
    host: "str | None" = None


@app.post("/api/hand/gesture")
def api_hand_gesture(body: HandLine):
    """One of the firmware's gestures (a single character; it moves at the hand's own speed)."""
    c = body.line.strip()
    if c not in HAND_GESTURES:
        raise HTTPException(400, f"gesture must be one of: {' '.join(HAND_GESTURES)}")
    r = _hand_send(c, body.host)
    if r["error"]:
        raise HTTPException(502, f"send failed: {r['error']}")
    return {"line": c, "gesture": HAND_GESTURES[c]}


@app.post("/api/hand/raw")
def api_hand_raw(body: HandLine):
    """Any one line, as typed (J:…, a gesture character, Q). The reply, if the firmware sends one."""
    line = body.line.strip()
    if not line or len(line) > 150 or not line.isascii() or not line.isprintable():
        raise HTTPException(400, "one printable ASCII line of at most 150 characters")
    r = _hand_send(line, body.host, reply_wait=0.6 if line == "Q" else 0.15)
    if r["error"]:
        raise HTTPException(502, f"send failed: {r['error']}")
    return r


_mount_static()


def main() -> None:
    global DATASETS_DIR, REPO_ROOT, VIEWER, ROBOT_URL, ROUTE_URL, SIM_ROUTE_URL, MAP_SOURCE, SIM_EDITOR, SUPERVISOR_URL, HAND_HOST
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
    ap.add_argument("--hand-host", default=HAND_HOST, help="the AmazingHand's IP (VLA_HAND_HOST), for the Hand tab; UDP port 8766")
    args = ap.parse_args()
    ROBOT_URL = args.robot_url
    ROUTE_URL = args.route_url
    SIM_ROUTE_URL = args.sim_route_url
    MAP_SOURCE = args.map_source
    SIM_EDITOR = args.sim_editor
    SUPERVISOR_URL = args.supervisor_url
    HAND_HOST = args.hand_host

    DATASETS_DIR = os.path.abspath(args.datasets_dir)
    REPO_ROOT = os.path.abspath(args.repo_root)
    VIEWER = ViewerManager(REPO_ROOT, web_port=args.web_port)
    print(f"datasets_dir = {DATASETS_DIR}\nrepo_root    = {REPO_ROOT}\nserving      http://{args.host}:{args.port}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
