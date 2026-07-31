"""Split ONE episode of a LeRobotDataset into two by deleting a middle time-window (e.g. two demos
accidentally recorded as one episode). Reads the raw data parquets and re-emits every frame into a
NEW dataset: the target episode becomes two episodes — frames before --cut-start, then frames after
--cut-end — with the window in between dropped; all other episodes are copied unchanged.

    PYTHONPATH=. python scripts/split_episode.py --root outputs/datasets/hw_pickplace \
        --out outputs/datasets/hw_pickplace_split --episode 1 --cut-start 22.15 --cut-end 61.6
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
    ap.add_argument("--repo-id", default="local/split")
    ap.add_argument("--episode", type=int, required=True, help="episode index to split")
    ap.add_argument("--cut-start", type=float, required=True, help="seconds: start of the deleted window")
    ap.add_argument("--cut-end", type=float, required=True, help="seconds: end of the deleted window")
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists — remove it or pick another --out")

    info = json.load(open(os.path.join(args.root, "meta", "info.json")))
    fps = info["fps"]
    features = {k: {**info["features"][k], "shape": tuple(info["features"][k]["shape"])} for k in REAL_FEATURES}
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=fps, features=features, root=args.out,
                               robot_type=info.get("robot_type"), use_videos=False)

    df, idx2task = read_source(args.root)
    f_lo, f_hi = args.cut_start * fps, args.cut_end * fps  # frame_index bounds of the deleted window

    def emit(edf, tag: str) -> None:
        for _, row in edf.iterrows():
            ds.add_frame({
                "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                "action": np.asarray(row["action"], dtype=np.float32),
                "observation.images.scene": decode(row["observation.images.scene"]),
                "observation.images.wrist": decode(row["observation.images.wrist"]),
                "task": idx2task[int(row["task_index"])],
            })
        ds.save_episode()
        print(f"  {tag}: {len(edf)} frames")

    out_ep = 0
    for ep in sorted(df["episode_index"].unique()):
        edf = df[df["episode_index"] == ep]
        if ep == args.episode:
            a = edf[edf["frame_index"] < f_lo]
            b = edf[edf["frame_index"] > f_hi]
            print(f"splitting ep {ep} ({len(edf)}f) -> A={len(a)}f + drop {len(edf) - len(a) - len(b)}f + B={len(b)}f")
            emit(a, f"out ep {out_ep} (split A of {ep})"); out_ep += 1
            emit(b, f"out ep {out_ep} (split B of {ep})"); out_ep += 1
        else:
            emit(edf, f"out ep {out_ep} (copy of {ep})"); out_ep += 1

    ds.finalize()
    try:
        re = LeRobotDataset(repo_id=args.repo_id, root=args.out)
        print(f"\nSPLIT -> {args.out}\n  episodes={re.num_episodes}  frames={re.num_frames}")
    except Exception as e:  # noqa: BLE001
        print(f"\nSPLIT -> {args.out}: wrote {out_ep} episodes ({type(e).__name__} on reload; files are written)")


if __name__ == "__main__":
    main()
