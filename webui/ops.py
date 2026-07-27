"""Dataset/episode mutating operations for the web UI.

Delegates to lerobot.datasets.dataset_tools wherever a primitive exists (delete / merge / rename /
to-video); the ONE op with no library equivalent — sub-episode trim (drop a time window) — reuses
the raw read->re-emit pattern from scripts/split_episode.py.

Two write disciplines:
  * IN-PLACE  — rename_task (modify_tasks) and annotations: touch only small metadata, no rebuild.
  * REBUILD   — delete / trim / move: a new dataset is built to a temp dir, then swapped into the
                original name keeping the pre-edit copy as "<name>.bak" (one level, recoverable).
  * NEW       — merge / to_video: emit a brand-new dataset; sources are never touched.

Every rebuild renumbers episode_index, so the annotations sidecar is remapped through the
old->new index mapping (annotations.remap) before the swap.
"""
from __future__ import annotations

import glob
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

# auto-managed columns LeRobot regenerates on add_frame/save_episode — excluded from re-emit
_AUTO = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


# --------------------------------------------------------------------------- helpers
def _read_info(root: str) -> dict:
    with open(os.path.join(root, "meta", "info.json")) as f:
        return json.load(f)


def _repo_id(root: str) -> str:
    return _read_info(root).get("repo_id") or f"local/{os.path.basename(root)}"


def _set_repo_id(root: str, repo_id: str) -> None:
    """Pin repo_id in info.json (a dataset newly split off inherits the split key as its repo_id)."""
    p = os.path.join(root, "meta", "info.json")
    try:
        info = json.load(open(p))
        info["repo_id"] = repo_id
        with open(p, "w") as f:
            json.dump(info, f, indent=4)
    except Exception:  # noqa: BLE001
        pass


def _load(root: str):
    """Load a local dataset (read-mode) for the dataset_tools functions."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(_repo_id(root), root=root)


def _rmtree(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


_WORK_SUFFIXES = (".rebuild.tmp", ".staged", ".peel.tmp", ".swap")


def _clean_working(datasets_dir: str, *names: str) -> None:
    """Remove any leftover working dirs (from a prior interrupted op) for these dataset names."""
    for n in names:
        for suf in _WORK_SUFFIXES:
            _rmtree(os.path.join(datasets_dir, f".{n}{suf}"))


def _kept_mapping(total: int, deleted: set[int]) -> dict[int, int]:
    """{old_ep: new_ep} after deleting `deleted` and renumbering the survivors contiguously."""
    kept = [e for e in range(total) if e not in deleted]
    return {old: new for new, old in enumerate(kept)}


def _stage_out(new_ds, tmp_parent: str, datasets_dir: str, name_hint: str) -> str:
    """Pull the just-built dataset out of its temp parent to a stable sibling path; return it.

    dataset_tools may write to tmp_parent directly or nest a repo_id under it, so we trust
    new_ds.root and stage that. Cleans tmp_parent afterward.
    """
    from . import annotations as _a  # noqa: F401 (kept for symmetry / lazy import policy)

    new_root = str(new_ds.root)
    staged = os.path.join(datasets_dir, f".{name_hint}.staged")
    _rmtree(staged)
    shutil.move(new_root, staged)
    _rmtree(tmp_parent)
    return staged


def _swap_in_place(datasets_dir: str, name: str, staged: str) -> str:
    """Replace datasets_dir/<name> with `staged`, keeping the old copy as <name>.bak."""
    target = os.path.join(datasets_dir, name)
    bak = target + ".bak"
    _rmtree(bak)
    if os.path.exists(target):
        os.rename(target, bak)
    os.rename(staged, target)
    return target


# --------------------------------------------------------------------------- IN-PLACE
def rename_task(datasets_dir: str, name: str, new_task: str | None = None,
                episode_tasks: dict[int, str] | None = None) -> dict[str, Any]:
    """Rename the task for the whole dataset (`new_task`) or per-episode (`episode_tasks`).
    In-place: modify_tasks rewrites only tasks.parquet, the task_index column, the episodes
    `tasks` column, and info.total_tasks."""
    from lerobot.datasets import dataset_tools as dt

    root = os.path.join(datasets_dir, name)
    ds = _load(root)
    if episode_tasks:
        dt.modify_tasks(ds, episode_tasks={int(k): v for k, v in episode_tasks.items()})
    elif new_task is not None:
        dt.modify_tasks(ds, new_task=new_task)
    else:
        raise ValueError("rename_task needs new_task or episode_tasks")
    return {"ok": True, "name": name}


# --------------------------------------------------------------------------- REBUILD (replace + .bak)
def delete_episodes(datasets_dir: str, name: str, episodes: list[int]) -> dict[str, Any]:
    from lerobot.datasets import dataset_tools as dt

    from . import annotations as anno

    root = os.path.join(datasets_dir, name)
    total = _read_info(root)["total_episodes"]
    dele = {int(e) for e in episodes}
    if not dele:
        raise ValueError("no episodes given")
    if dele >= set(range(total)):
        raise ValueError("refusing to delete every episode")

    _clean_working(datasets_dir, name)
    tmp = os.path.join(datasets_dir, f".{name}.rebuild.tmp")
    new_ds = dt.delete_episodes(_load(root), sorted(dele), output_dir=tmp, repo_id=_repo_id(root))
    staged = _stage_out(new_ds, tmp, datasets_dir, name)
    anno.remap(root, staged, _kept_mapping(total, dele))
    _swap_in_place(datasets_dir, name, staged)
    return {"ok": True, "name": name, "deleted": sorted(dele),
            "episodes": _read_info(root)["total_episodes"]}


def trim_episode(datasets_dir: str, name: str, episode: int,
                 cut_start_s: float, cut_end_s: float) -> dict[str, Any]:
    """Drop the frames in [cut_start_s, cut_end_s] of one episode, stitching the remainder into a
    SINGLE episode (episode count and order unchanged, so annotations map 1:1). Image datasets only.

    Reuses the scripts/split_episode.py raw pattern (read parquet -> filter frames -> re-emit),
    generalized to any feature set derived from info.json.
    """
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from PIL import Image

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = os.path.join(datasets_dir, name)
    info = _read_info(root)
    fps = info["fps"]
    feats_all = info["features"]
    real = [k for k in feats_all if k not in _AUTO]
    img_keys = {k for k in real if feats_all[k].get("dtype") in ("image", "video")}
    if any(feats_all[k].get("dtype") == "video" for k in img_keys):
        raise ValueError("trim supports image datasets only (this one stores video); trim before to_video")

    # read raw frames
    dfs = [pq.read_table(f).to_pandas()
           for f in sorted(glob.glob(os.path.join(root, "data", "chunk-*", "file-*.parquet")))]
    df = pd.concat(dfs, ignore_index=True).sort_values(["episode_index", "frame_index"])
    tasks = pq.read_table(os.path.join(root, "meta", "tasks.parquet")).to_pandas()
    idx2task = {v: k for k, v in tasks["task_index"].items()}

    ep_df = df[df["episode_index"] == episode]
    if ep_df.empty:
        raise ValueError(f"episode {episode} not found")
    f_lo, f_hi = cut_start_s * fps, cut_end_s * fps
    keep = ep_df[(ep_df["frame_index"] < f_lo) | (ep_df["frame_index"] > f_hi)]
    if keep.empty:
        raise ValueError("trim would delete every frame of the episode")
    dropped = len(ep_df) - len(keep)
    if dropped <= 0:
        raise ValueError("the trim window removes no frames")

    features = {k: {**feats_all[k], "shape": tuple(feats_all[k]["shape"])} for k in real}
    _clean_working(datasets_dir, name)
    tmp = os.path.join(datasets_dir, f".{name}.rebuild.tmp")
    ds = LeRobotDataset.create(repo_id=_repo_id(root), fps=fps, features=features, root=tmp,
                               robot_type=info.get("robot_type"), use_videos=False)

    def decode(cell) -> "np.ndarray":
        return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))

    def emit(rows) -> None:
        for _, row in rows.iterrows():
            frame = {"task": idx2task[int(row["task_index"])]}
            for k in real:
                frame[k] = decode(row[k]) if k in img_keys else np.asarray(row[k], dtype=np.float32)
            ds.add_frame(frame)
        ds.save_episode()

    for ep in sorted(df["episode_index"].unique()):
        emit(keep if ep == episode else df[df["episode_index"] == ep])
    ds.finalize()

    staged = os.path.join(datasets_dir, f".{name}.staged")
    _rmtree(staged)
    shutil.move(str(ds.root), staged)
    _rmtree(tmp)
    # episode count/order unchanged -> identity annotation mapping (content of `episode` changed only)
    from . import annotations as anno
    anno.remap(root, staged, {e: e for e in range(info["total_episodes"])})
    _swap_in_place(datasets_dir, name, staged)
    return {"ok": True, "name": name, "episode": episode, "dropped_frames": int(dropped),
            "kept_frames": int(len(keep))}


def move_episode(datasets_dir: str, src: str, episode: int, dst: str) -> dict[str, Any]:
    """Move one episode from src to dst.

    If dst EXISTS, the episode is appended (fps/features must match); dst is rebuilt (+.bak).
    If dst does NOT exist, it is CREATED from the moved episode — LeRobot datasets can't be empty, so
    a "new dataset" is born from its first episode (later moves append to it). src is always rebuilt
    (delete) keeping a .bak. Everything rejectable (incompatibility, emptying src, bad new name) is
    checked BEFORE any dataset is touched, so a rejected move can never half-apply. Annotation travels.
    """
    from lerobot.datasets import dataset_tools as dt

    from . import annotations as anno

    src_root = os.path.join(datasets_dir, src)
    dst_root = os.path.join(datasets_dir, dst)
    if src == dst:
        raise ValueError("src and dst are the same dataset")
    si = _read_info(src_root)
    src_total = si["total_episodes"]
    ep = int(episode)
    if ep not in range(src_total):
        raise ValueError(f"episode {ep} out of range for {src}")
    if src_total <= 1:
        raise ValueError(f"{src} has only {src_total} episode(s); moving it would empty the dataset")

    dst_exists = os.path.isdir(dst_root)
    if dst_exists:
        di = _read_info(dst_root)
        dst_total = di["total_episodes"]
        if si.get("fps") != di.get("fps"):
            raise ValueError(f"fps mismatch: {src}={si.get('fps')} vs {dst}={di.get('fps')} — cannot merge")
        if set(si.get("features", {})) != set(di.get("features", {})):
            raise ValueError(f"feature set differs between {src} and {dst} — cannot merge")
    elif not dst or "/" in dst or "\\" in dst or dst.startswith("."):
        raise ValueError(f"bad new dataset name: {dst!r}")

    moved_note = anno.load(src_root).get(str(ep))
    _clean_working(datasets_dir, src, dst)
    try:
        # 1) peel the episode off src into a standalone 1-episode dataset
        tmp_peel = os.path.join(datasets_dir, f".{src}.peel.tmp")
        peeled = dt.split_dataset(_load(src_root), {"moved": [ep]}, output_dir=tmp_peel)["moved"]

        if dst_exists:  # 2a) append -> merge([dst, peeled]) rebuilds dst (+.bak)
            tmp_dst = os.path.join(datasets_dir, f".{dst}.rebuild.tmp")
            new_dst = dt.merge_datasets([_load(dst_root), peeled], output_repo_id=_repo_id(dst_root),
                                        output_dir=tmp_dst)
            staged_dst = _stage_out(new_dst, tmp_dst, datasets_dir, dst)
            anno.remap(dst_root, staged_dst, {e: e for e in range(dst_total)})  # dst keeps its indices
            if moved_note:
                d = anno.load(staged_dst); d[str(dst_total)] = moved_note; anno.save(staged_dst, d)
            _swap_in_place(datasets_dir, dst, staged_dst)
        else:  # 2b) CREATE dst from the peeled 1-episode dataset (no .bak — nothing existed)
            staged_dst = os.path.join(datasets_dir, f".{dst}.staged")
            _rmtree(staged_dst)
            shutil.move(str(peeled.root), staged_dst)
            _set_repo_id(staged_dst, f"local/{dst}")  # was the split key, not the folder name
            anno.save(staged_dst, {"0": moved_note} if moved_note else {})
            os.rename(staged_dst, dst_root)

        _rmtree(tmp_peel)

        # 3) delete the episode from src (rebuild + .bak)
        tmp_src = os.path.join(datasets_dir, f".{src}.rebuild.tmp")
        new_src = dt.delete_episodes(_load(src_root), [ep], output_dir=tmp_src, repo_id=_repo_id(src_root))
        staged_src = _stage_out(new_src, tmp_src, datasets_dir, src)
        anno.remap(src_root, staged_src, _kept_mapping(src_total, {ep}))
        _swap_in_place(datasets_dir, src, staged_src)
    finally:  # scrub build scratch; never touch a live target here
        for p in (f".{src}.peel.tmp", f".{src}.rebuild.tmp", f".{dst}.rebuild.tmp", f".{dst}.staged"):
            _rmtree(os.path.join(datasets_dir, p))

    return {"ok": True, "src": src, "dst": dst, "episode": ep, "created": not dst_exists,
            "src_episodes": _read_info(src_root)["total_episodes"],
            "dst_episodes": _read_info(dst_root)["total_episodes"]}


# --------------------------------------------------------------------------- NEW dataset
def merge(datasets_dir: str, names: list[str], out_name: str) -> dict[str, Any]:
    from lerobot.datasets import dataset_tools as dt

    from . import annotations as anno

    if len(names) < 2:
        raise ValueError("merge needs >= 2 datasets")
    target = os.path.join(datasets_dir, out_name)
    if os.path.exists(target):
        raise ValueError(f"{out_name} already exists")
    roots = [os.path.join(datasets_dir, n) for n in names]
    _clean_working(datasets_dir, out_name)
    tmp = os.path.join(datasets_dir, f".{out_name}.rebuild.tmp")
    new_ds = dt.merge_datasets([_load(r) for r in roots], output_repo_id=f"local/{out_name}",
                               output_dir=tmp)
    new_root = str(new_ds.root)
    _rmtree(target)
    shutil.move(new_root, target)
    _rmtree(tmp)
    # carry annotations across with cumulative offsets (episodes concatenated in `names` order)
    merged: dict[str, Any] = {}
    offset = 0
    for r in roots:
        n = _read_info(r)["total_episodes"]
        for k, v in anno.load(r).items():
            try:
                merged[str(int(k) + offset)] = v
            except ValueError:
                pass
        offset += n
    if merged:
        anno.save(target, merged)
    return {"ok": True, "name": out_name, "episodes": _read_info(target)["total_episodes"],
            "sources": names}


def to_video(datasets_dir: str, name: str, out_name: str | None = None) -> dict[str, Any]:
    from lerobot.datasets import dataset_tools as dt

    from . import annotations as anno

    out_name = out_name or f"{name}_video"
    root = os.path.join(datasets_dir, name)
    target = os.path.join(datasets_dir, out_name)
    if os.path.exists(target):
        raise ValueError(f"{out_name} already exists")
    _clean_working(datasets_dir, out_name)
    tmp = os.path.join(datasets_dir, f".{out_name}.rebuild.tmp")
    # NB: this one takes a pathlib.Path (it does output_dir / "temp_images"), unlike the others
    new_ds = dt.convert_image_to_video_dataset(_load(root), output_dir=Path(tmp),
                                               repo_id=f"local/{out_name}")
    new_root = str(new_ds.root)
    _rmtree(target)
    shutil.move(new_root, target)
    _rmtree(tmp)
    # episode indices unchanged
    anno.remap(root, target, {e: e for e in range(_read_info(root)["total_episodes"])})
    return {"ok": True, "name": out_name, "source": name}


# --------------------------------------------------------------------------- lifecycle
def delete_dataset(datasets_dir: str, name: str) -> dict[str, Any]:
    root = os.path.join(datasets_dir, name)
    if not os.path.isdir(root):
        raise ValueError(f"{name} not found")
    _rmtree(root)
    _rmtree(root + ".bak")
    return {"ok": True, "deleted": name}


def rename_dataset(datasets_dir: str, name: str, new_name: str) -> dict[str, Any]:
    """Rename a dataset folder (cheap mv, not a rebuild). Also updates a pinned repo_id in info.json
    and carries the .bak alongside. The repo_id otherwise derives from the folder name (see _repo_id)."""
    new_name = (new_name or "").strip()
    if not new_name or "/" in new_name or "\\" in new_name or new_name.startswith("."):
        raise ValueError(f"bad new name: {new_name!r}")
    src = os.path.join(datasets_dir, name)
    dst = os.path.join(datasets_dir, new_name)
    if not os.path.isdir(src):
        raise ValueError(f"{name} not found")
    if new_name == name:
        return {"ok": True, "old": name, "new": new_name}
    if os.path.exists(dst):
        raise ValueError(f"{new_name} already exists")
    os.rename(src, dst)
    info_path = os.path.join(dst, "meta", "info.json")
    try:  # keep a pinned repo_id consistent (most datasets have none -> derived from the folder)
        info = json.load(open(info_path))
        if info.get("repo_id"):
            info["repo_id"] = f"local/{new_name}"
            with open(info_path, "w") as f:
                json.dump(info, f, indent=4)
    except Exception:  # noqa: BLE001 — don't fail the rename over info.json cosmetics
        pass
    if os.path.isdir(src + ".bak"):
        _rmtree(dst + ".bak")
        os.rename(src + ".bak", dst + ".bak")
    return {"ok": True, "old": name, "new": new_name}


def restore_backup(datasets_dir: str, name: str) -> dict[str, Any]:
    """Undo the last rebuild edit: swap <name> and <name>.bak back."""
    target = os.path.join(datasets_dir, name)
    bak = target + ".bak"
    if not os.path.isdir(bak):
        raise ValueError(f"no backup for {name}")
    swap = os.path.join(datasets_dir, f".{name}.swap")
    _rmtree(swap)
    if os.path.exists(target):
        os.rename(target, swap)
    os.rename(bak, target)
    if os.path.exists(swap):
        os.rename(swap, bak)  # the previously-current version becomes the new .bak
    return {"ok": True, "restored": name}


def add_dataset(datasets_dir: str, *, source: str | None = None,
                remote: str | None = None, as_name: str | None = None) -> dict[str, Any]:
    """Register a new dataset. Either copy a local dataset dir (`source`) or rsync one from the
    robot PC (`remote` = dataset name under rd@.80:~/VLA/outputs/datasets)."""
    import subprocess

    if remote:
        name = as_name or remote
        dest = os.path.join(datasets_dir, name)
        if os.path.exists(dest):
            raise ValueError(f"{name} already exists")
        src = f"rd@192.168.11.80:~/VLA/outputs/datasets/{remote}/"
        r = subprocess.run(["rsync", "-a", "--info=stats1", src, dest + "/"],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            _rmtree(dest)
            raise RuntimeError(f"rsync failed: {r.stderr.strip()[:300]}")
        return {"ok": True, "name": name, "imported_from": src}

    if source:
        if not os.path.isdir(os.path.join(source, "meta")):
            raise ValueError(f"{source} is not a LeRobot dataset (no meta/)")
        name = as_name or os.path.basename(os.path.normpath(source))
        dest = os.path.join(datasets_dir, name)
        if os.path.exists(dest):
            raise ValueError(f"{name} already exists")
        shutil.copytree(source, dest)
        return {"ok": True, "name": name, "copied_from": source}

    raise ValueError("add_dataset needs `source` (local path) or `remote` (name on .80)")
