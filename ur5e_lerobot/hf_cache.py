"""Keep the Hugging Face `datasets` cache from eating the disk.

LeRobot loads a dataset's parquet through `datasets.load_dataset("parquet", ...)`, which materialises
an Arrow copy under ~/.cache/huggingface/datasets/parquet/default-<fingerprint>. The fingerprint
changes whenever the parquet files change, i.e. on EVERY saved episode, so recording leaves one full
copy of the dataset behind per save: on 2026-10-24 the robot PC had 98 of them (16 GB for a 3.8 GB
datasets folder) and 671 MB free. Training runs on the GPU box do the same once per run.

prune() deletes all but the newest `keep` entries. The newest is the one the running process may still
have memory-mapped; on Linux an unlinked mapped file stays readable until closed, so even that would
be safe, but keeping a couple costs little and avoids re-conversion on resume.
"""
from __future__ import annotations

import os
import shutil
import time


def cache_dir() -> str:
    base = os.environ.get("HF_DATASETS_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "datasets")
    return os.path.join(base, "parquet")


def prune(keep: int = 2) -> str:
    """Delete all but the `keep` newest parquet cache entries (+ their lock files). Returns a log line."""
    d = cache_dir()
    if not os.path.isdir(d):
        return "hf datasets cache: nothing to prune"
    entries = []
    for name in os.listdir(d):
        p = os.path.join(d, name)
        if os.path.isdir(p):
            try:
                entries.append((os.path.getmtime(p), p))
            except OSError:
                pass
    entries.sort(reverse=True)
    freed = 0
    removed = 0
    for _, p in entries[max(keep, 0):]:
        try:
            size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
            shutil.rmtree(p, ignore_errors=True)
            freed += size
            removed += 1
        except OSError:
            pass
    # stale .lock files next to the parquet dir (one per entry ever created)
    parent = os.path.dirname(d)
    cutoff = time.time() - 3600
    for name in os.listdir(parent):
        p = os.path.join(parent, name)
        if name.endswith(".lock") and os.path.isfile(p):
            try:
                if os.path.getmtime(p) < cutoff:
                    os.remove(p)
            except OSError:
                pass
    return f"hf datasets cache: removed {removed} old entries ({freed / 1e9:.1f} GB), kept {min(keep, len(entries))}"
