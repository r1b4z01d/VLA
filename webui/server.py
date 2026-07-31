"""FastAPI backend + static frontend for the dataset-management web UI.

    HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python -m webui.server --host 0.0.0.0 --port 8080
    # then browse from the Mac at  http://192.168.11.130:8080

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
    vids = sorted(_g.glob(os.path.join(run, "*.mp4")))
    if not vids:
        raise HTTPException(404, "no video for this eval")
    return FileResponse(vids[0], media_type="video/mp4",
                        headers={"Access-Control-Allow-Origin": "*"})


@app.on_event("shutdown")
def _shutdown():
    VIEWER.stop()


@app.get("/healthz")
def healthz():
    return JSONResponse({"ok": True, "datasets_dir": DATASETS_DIR})


def _mount_static():
    if os.path.isdir(STATIC_DIR):
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


_mount_static()


def main() -> None:
    global DATASETS_DIR, REPO_ROOT, VIEWER
    ap = argparse.ArgumentParser(description="VLA dataset-management web UI")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--datasets-dir", default="outputs/datasets")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--web-port", type=int, default=9090, help="rerun web-viewer app port")
    args = ap.parse_args()

    DATASETS_DIR = os.path.abspath(args.datasets_dir)
    REPO_ROOT = os.path.abspath(args.repo_root)
    VIEWER = ViewerManager(REPO_ROOT, web_port=args.web_port)
    print(f"datasets_dir = {DATASETS_DIR}\nrepo_root    = {REPO_ROOT}\nserving      http://{args.host}:{args.port}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
