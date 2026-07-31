"""Re-encode an image LeRobotDataset into a VIDEO dataset (far faster training-time loading — the
4090 was data-starved on the image version). Reads the raw data parquets and re-emits every frame,
in episode order, into a new dataset created with use_videos=True (camera features -> dtype 'video').

    PYTHONPATH=. python scripts/to_video.py --root outputs/datasets/hw_pickplace \
        --out outputs/datasets/hw_pickplace_video --repo-id local/hw_pickplace
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


def read_source(root: str):
    frames = []
    for f in sorted(glob.glob(os.path.join(root, "data", "chunk-*", "file-*.parquet"))):
        frames.append(pq.read_table(f).to_pandas())
    if not frames:
        raise SystemExit(f"{root}: no data parquets")
    df = pd.concat(frames, ignore_index=True).sort_values(["episode_index", "frame_index"])
    tasks = pq.read_table(os.path.join(root, "meta", "tasks.parquet")).to_pandas()
    idx2task = {v: k for k, v in tasks["task_index"].items()}
    return df, idx2task


def decode(img) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default="local/video")
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists — remove it or pick another --out")

    info = json.load(open(os.path.join(args.root, "meta", "info.json")))
    features = {}
    for k in REAL_FEATURES:
        f = {**info["features"][k], "shape": tuple(info["features"][k]["shape"])}
        if "image" in k:
            f["dtype"] = "video"  # <- the whole point: store cameras as video, not per-frame images
        features[k] = f
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=info["fps"], features=features,
                               root=args.out, robot_type=info.get("robot_type"), use_videos=True)

    df, idx2task = read_source(args.root)
    eps = sorted(df["episode_index"].unique())
    print(f"{args.root}: {len(df)} frames, {len(eps)} episodes -> video dataset")
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
        print(f"  encoded ep {int(ep)} ({len(edf)} frames)")

    ds.finalize()
    try:
        re = LeRobotDataset(repo_id=args.repo_id, root=args.out)
        vids = [k for k, v in re.meta.features.items() if v.get("dtype") == "video"]
        print(f"\nVIDEO -> {args.out}\n  episodes={re.num_episodes}  frames={re.num_frames}  video features={vids}")
    except Exception as e:  # noqa: BLE001
        print(f"\nVIDEO -> {args.out}: wrote episodes ({type(e).__name__} on reload; files are written)")


if __name__ == "__main__":
    main()
