# RobotDisco VLA Manager (`webui/`)

A browser console for the whole loop between recording and deployment, with three tabs — **Datasets**
(browse/merge/delete/import, per-episode trim/delete/rate/annotate/rename-task/move, and an episode
player that renders each episode to a static rerun **`.rrd`** shown in rerun's WASM viewer over plain
HTTP), **Models** (every training run + live progress for the one training now), and **Evals** (each
eval run with rating/notes + metrics). The ops delegate to `lerobot.datasets.dataset_tools` and
`lerobot.scripts.lerobot_dataset_viz` (no bespoke dataset surgery except sub-episode trim, which has
no library equivalent). It's self-contained (only needs lerobot + FastAPI), so it runs the same on the
**GPU box**, your **Mac**, or in **Docker**.

![Datasets tab — dataset explorer, episode manager, and the embedded rerun player](https://raw.githubusercontent.com/r1b4z01d/VLA/main/docs/Episodes.png)

## Run it

**Native (GPU box or Mac)** — one-time `uv pip install -r webui/requirements.txt` into the project venv, then:
```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python -m webui.server \
  --host 0.0.0.0 --port 8080 --datasets-dir outputs/datasets
# GPU box: browse from the Mac at http://192.168.11.130:8080  ·  Mac: http://localhost:8080
```

**Docker** (multi-arch amd64/arm64; `outputs/datasets` is mounted, not baked in, so edits persist on the host):
```bash
docker compose -f webui/docker-compose.yml up --build       # then http://localhost:8080
# portable image for the x86 GPU/robot infra AND the Mac:
docker buildx build --platform linux/amd64,linux/arm64 -f webui/Dockerfile -t <registry>/vla-webui:0.4.4 --push .
```
The image bundles lerobot + ffmpeg (CPU-only torch). On a **Linux** host uncomment `user: "${UID}:${GID}"`
in the compose service so volume files aren't root-owned; the *import-from-robot-PC* button needs your
SSH key mounted (`~/.ssh:/root/.ssh:ro`).

## Tabs

- **Explorer** — cards per dataset (episodes/frames/fps/size/task/media); check ≥2 → **Merge**;
  **+ Add** imports a dataset from the robot PC (`rsync` from `rd@.80`) or copies a local path;
  the header polls the folder every 4 s and flags newly-recorded/-changed datasets.
- **Episode manager** — per row: **▶** (play), **trim** (drop a time window), **move** (to another
  dataset), **del**, click-to-set **rating** stars, inline **notes** + **operator**. Rename-task
  applies to the whole dataset or just the selected episodes.
- **Player** — **▶** renders the episode to a rerun `.rrd` (first play of an episode ~30s, then cached)
  and loads it in rerun's WASM viewer over ordinary HTTP — **no gRPC**, so it works in any browser.
  Needs ports **8080** (UI + the `.rrd`) and **9090** (viewer app) reachable; the Docker setup publishes both.
- **Models tab** — lists every run under `outputs/train/` (policy, dataset, steps, checkpoints, size)
  and shows the actively-training one **live** (step/loss/ETA, auto-refreshing); finished models get
  download (zipped checkpoint) / rename / delete.
- **Evals tab** — each eval run as a card with **★ rating**, notes, operator + metrics (model,
  checkpoint, task, episodes, success rate). Bridge evals are **auto-captured GPU-side** into
  `outputs/evals/` by `scripts/infer_server.py`; you can also add manual entries. Rename / delete /
  download / play-video per eval.

![Models tab — every training run, with live progress for the one training now](https://raw.githubusercontent.com/r1b4z01d/VLA/main/docs/Models.png)

## Safety
Rename-task + annotations are in-place; delete/trim/merge/move **rebuild** the dataset and keep the
pre-edit copy as `<name>.bak` (one-click **restore .bak**). Destructive calls require a confirm. `move`
pre-checks fps/feature compatibility *before* touching either side, so a rejected move can't half-apply.
Ratings/notes/operator live in a sidecar (`meta/annotations.json`), outside the LeRobot schema, and are
remapped through the new episode indices on every rebuild.
