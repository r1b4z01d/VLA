# Dataset-management web UI — self-contained image (lerobot + rerun + FastAPI).
# Build context is the repo root (so it can COPY webui/); see webui/docker-compose.yml.
#
#   docker compose -f webui/docker-compose.yml up --build          # local, native arch
#   docker buildx build --platform linux/amd64,linux/arm64 \       # portable multi-arch artifact
#       -f webui/Dockerfile -t <registry>/vla-webui:0.4.4 --push .
#
# syntax=docker/dockerfile:1
FROM python:3.10-slim

# ffmpeg (Debian bookworm ships libsvtav1) for `to_video`; lsof for the viewer port-reclaim;
# rsync + ssh client for the optional "import from robot PC" path; build-essential to compile the
# few C-extension sdists lerobot pulls with no aarch64 wheel (e.g. evdev, via pynput).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg lsof rsync openssh-client build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch FIRST so the subsequent lerobot install does not drag in the multi-GB
# CUDA build (the web UI does pure CPU dataset I/O — no inference/training). torchcodec is skipped
# on linux-aarch64 by lerobot's own dependency marker, so this same install works on arm64.
# --extra-index-url keeps PyPI available for torch's own deps (typing-extensions, sympy, …) which
# the pytorch CPU index does not host; the CPU wheels still win (their +cpu local version sorts ahead).
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "torch>=2.2.1,<2.11" "torchvision>=0.21,<0.26"

# lerobot pinned to the 0.4.4 training-env format (dataset_tools + the rerun dataset viz), plus the
# two web deps. Everything else webui needs (pyarrow/pandas/numpy/pillow/rerun-sdk) comes with lerobot.
RUN pip install --no-cache-dir "lerobot==0.4.4" fastapi "uvicorn[standard]"

COPY webui/ /app/webui/
# viewer.log lives under outputs/webui; make it world-writable so the container works under `--user`
RUN mkdir -p /app/outputs/webui && chmod -R 777 /app/outputs

ENV HF_HUB_OFFLINE=1 \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1
WORKDIR /app
# 8080 = web UI + per-episode .rrd files · 9090 = rerun WASM viewer app (loads, then fetches the .rrd from :8080)
EXPOSE 8080 9090

CMD ["python", "-m", "webui.server", "--host", "0.0.0.0", "--port", "8080", \
     "--datasets-dir", "/data", "--repo-root", "/app"]
