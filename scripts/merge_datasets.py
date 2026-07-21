"""Merge multiple LeRobot recording sessions into one dataset (for training).

Each teleop session is its own LeRobotDataset (the panel writes a fresh root per run). This
combines several into a single dataset via LeRobot's built-in aggregate_datasets. All sessions
must share the same features (same cameras/resolution, state/action, fps) and the same image-vs-
video format — they will if recorded with the same panel config.

    .venv/bin/python scripts/merge_datasets.py outputs/datasets/manual_box outputs/datasets/manual_box_1 \
        --out outputs/datasets/merged --repo-id local/ur5e_amazinghand_merged
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # local datasets only — never reach for the HF Hub

from lerobot.datasets.aggregate import aggregate_datasets


def _validate(roots: list[Path]) -> None:
    """Fail clearly (before LeRobot's confusing Hub-fallback 401) if a session is incomplete or
    a different format than the others."""
    dtypes = set()
    for r in roots:
        info = r / "meta" / "info.json"
        eps = r / "meta" / "episodes"
        if not info.exists():
            raise SystemExit(f"{r}: not a LeRobot dataset (no meta/info.json)")
        if not (eps.is_dir() and any(eps.rglob("*.parquet"))):
            raise SystemExit(
                f"{r}: incomplete dataset — no meta/episodes/*.parquet (the recording didn't "
                f"finalize). Re-record it, or drop it from the merge.")
        dtypes.add(json.loads(info.read_text())["features"]["observation.images.scene"]["dtype"])
    if len(dtypes) > 1:
        raise SystemExit(f"mixed image/video formats {dtypes} — all sessions must match to merge")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge LeRobot recording sessions into one dataset.")
    ap.add_argument("inputs", nargs="+", help="input dataset roots to merge (2+)")
    ap.add_argument("--out", required=True, help="output (merged) dataset root")
    ap.add_argument("--repo-id", default="local/merged", help="repo_id for the merged dataset")
    args = ap.parse_args()

    roots = [Path(p) for p in args.inputs]
    _validate(roots)
    repo_ids = [f"local/{p.name}" for p in roots]  # per-input ids (must be unique; dir names are)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"output root {out} already exists — remove it or pick another --out")

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=args.repo_id,
        roots=roots,
        aggr_root=out,
    )
    print(f"merged {len(roots)} sessions -> {out}")


if __name__ == "__main__":
    main()
