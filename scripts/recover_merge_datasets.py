"""Recover unfinalized LeRobot sessions and merge them into one dataset by re-writing every frame.

Some teleop sessions wrote their frame data (data/chunk-*/file-*.parquet, with images embedded as
bytes) but never finalized meta/episodes, so LeRobot can't load them. aggregate_datasets can't help
(it needs loadable inputs). Instead we read the raw data parquets directly, decode each frame, and
re-add it through LeRobot's writer — which regenerates ALL metadata correctly and merges in one go.

    PYTHONPATH=. .venv/bin/python scripts/recover_merge_datasets.py \
        outputs/datasets/manual_1 outputs/datasets/manual_9 \
        --out outputs/datasets/merged --repo-id local/ur5e_amazinghand_merged
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REAL_FEATURES = ("observation.state", "observation.images.scene", "action", "observation.images.wrist")


def read_source(root: str) -> tuple[pd.DataFrame, dict]:
    """Concatenate a source's data parquets (skipping corrupt files) and map task_index->string."""
    frames = []
    for f in sorted(glob.glob(os.path.join(root, "data", "chunk-*", "file-*.parquet"))):
        try:
            frames.append(pq.read_table(f).to_pandas())
        except Exception as e:  # corrupt/unfinalized file -> skip, keep the rest
            print(f"  ! skipping unreadable {f}: {str(e)[:60]}")
    if not frames:
        raise SystemExit(f"{root}: no readable data parquets")
    df = pd.concat(frames, ignore_index=True).sort_values(["episode_index", "frame_index"])
    tasks = pq.read_table(os.path.join(root, "meta", "tasks.parquet")).to_pandas()
    idx2task = {v: k for k, v in tasks["task_index"].items()}  # {task_index: task_string}
    return df, idx2task


def decode(img) -> np.ndarray:
    """LeRobot/HF image cell {bytes, path} -> HWC uint8 RGB array."""
    return np.asarray(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="source dataset roots (in the order to append)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default="local/merged")
    args = ap.parse_args()

    if os.path.exists(args.out):
        raise SystemExit(f"output {args.out} already exists — remove it or pick another --out")

    info = json.load(open(os.path.join(args.inputs[0], "meta", "info.json")))
    # shape must be a tuple: validate_frame compares array.shape (tuple) == feature["shape"]
    # strictly, and a JSON list [16] never equals the tuple (16,).
    features = {k: {**info["features"][k], "shape": tuple(info["features"][k]["shape"])}
                for k in REAL_FEATURES}
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=info["fps"], features=features,
                               root=args.out, robot_type=info.get("robot_type"), use_videos=False)

    total_ep = 0
    for root in args.inputs:
        df, idx2task = read_source(root)
        eps = sorted(df["episode_index"].unique())
        print(f"{root}: {len(df)} frames, {len(eps)} episodes {eps}")
        for ep in eps:
            edf = df[df["episode_index"] == ep]
            for _, row in edf.iterrows():
                ds.add_frame({
                    "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                    "action": np.asarray(row["action"], dtype=np.float32),
                    "observation.images.scene": decode(row["observation.images.scene"]),
                    "observation.images.wrist": decode(row["observation.images.wrist"]),
                    "task": idx2task[int(row["task_index"])],
                })
            ds.save_episode()
            total_ep += 1
            print(f"  saved merged episode {total_ep - 1} (from {os.path.basename(root)} ep {ep}, {len(edf)} frames)")

    ds.finalize()  # write parquet footers (data + episode metadata) so the merged dataset is valid

    # Best-effort in-process verify: reloading right after the final metadata-buffer flush can race
    # and spuriously fail even though every episode is written. Never let that crash the merge.
    try:
        re = LeRobotDataset(repo_id=args.repo_id, root=args.out)
        print(f"\nMERGED -> {args.out}\n  episodes={re.num_episodes}  frames={re.num_frames}  tasks={re.meta.total_tasks}")
    except Exception as e:  # noqa: BLE001
        print(f"\nMERGED -> {args.out}\n  wrote {total_ep} episodes ({type(e).__name__} on the in-process "
              f"reload — files are written; reload with LeRobotDataset in a fresh process to confirm)")


if __name__ == "__main__":
    main()
