"""Review a recorded LeRobotDataset: a montage (1 frame/episode) and a scene|wrist video.

    .venv/bin/python -m ur5e_lerobot.sim.playback --root outputs/datasets/manual_5 \
        --montage outputs/playback/montage.png --video outputs/playback/demos.mp4
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _hwc_uint8(t) -> np.ndarray:
    a = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
    if a.ndim == 3 and a.shape[0] in (1, 3):  # CHW -> HWC
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8) if float(a.max()) <= 1.0 + 1e-3 else a.astype(np.uint8)
    return np.ascontiguousarray(a)


def _episode_bounds(ds):
    edi = getattr(ds, "episode_data_index", None)
    if edi is not None and "from" in edi:
        return list(zip(edi["from"].tolist(), edi["to"].tolist()))
    # fallback: scan episode_index
    bounds, cur, start = [], None, 0
    for i in range(len(ds)):
        e = int(ds[i]["episode_index"])
        if cur is None:
            cur = e
        elif e != cur:
            bounds.append((start, i)); start, cur = i, e
    bounds.append((start, len(ds)))
    return bounds


def load_dataset(root: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = json.load(open(os.path.join(root, "meta", "info.json"))).get("repo_id", "local/dataset")
    return LeRobotDataset(repo_id, root=root)


def make_montage(ds, bounds, out_png: str, cols: int = 4) -> None:
    import cv2

    tiles = []
    for ep, (s, e) in enumerate(bounds):
        img = _hwc_uint8(ds[(s + e) // 2]["observation.images.scene"])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(img, f"ep {ep}  ({e - s}f)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(img)
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    cv2.imwrite(out_png, grid)
    print(f"montage -> {out_png}")


def make_video(ds, bounds, out_mp4: str, fps: int, step: int = 1) -> None:
    import cv2
    import imageio.v2 as imageio

    has_wrist = "observation.images.wrist" in ds.features
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
    # H.264 (opens in QuickTime/anything); stream frames so we don't buffer them all.
    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8, macro_block_size=16)
    n = 0
    for ep, (s, e) in enumerate(bounds):
        for i in range(s, e, step):
            it = ds[i]
            scene = _hwc_uint8(it["observation.images.scene"])  # RGB
            if has_wrist:
                wrist = _hwc_uint8(it["observation.images.wrist"])
                if wrist.shape[:2] != scene.shape[:2]:
                    wrist = cv2.resize(wrist, (scene.shape[1], scene.shape[0]))
                frame = np.hstack([scene, wrist])
            else:
                frame = scene
            cv2.putText(frame, f"ep {ep}  frame {i - s}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            writer.append_data(frame)
            n += 1
    writer.close()
    print(f"video -> {out_mp4}  ({n} frames @ {fps}fps h264, {'scene|wrist' if has_wrist else 'scene'})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Play back a recorded LeRobotDataset.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--montage", default="")
    ap.add_argument("--video", default="")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--step", type=int, default=1, help="frame subsample for the video")
    args = ap.parse_args()

    ds = load_dataset(args.root)
    bounds = _episode_bounds(ds)
    lens = [e - s for s, e in bounds]
    print(f"{len(bounds)} episodes, {len(ds)} frames; per-episode lengths: {lens}")
    if args.montage:
        make_montage(ds, bounds, args.montage)
    if args.video:
        make_video(ds, bounds, args.video, args.fps, args.step)


if __name__ == "__main__":
    main()
