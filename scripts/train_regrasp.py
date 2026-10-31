"""Train with the RE-GRASP segments oversampled, so recovery isn't drowned out by first-try success.

Why this exists. Behaviour cloning only learns what it sees, and a dataset of clean first-try picks
contains no example of "I missed — now what", so the policy has no recovery behaviour and small errors
compound. Deliberately recording re-grasps fixes that, but they are rare: if 5% of frames are recovery
the model treats them as noise. This oversamples them.

It also attacks the causal-confusion problem measured on this robot. `scripts/debug_eval.py` and the
t=0 aim test showed the policy's apparent object-tracking was almost entirely proprioceptive — swapping
in another episode's images moved the prediction ~2mm, while the arm's own pose explained nearly all of
it. State works as a shortcut because each pose occurs in ONE trajectory context, so pose->action is
near single-valued. A re-grasp breaks that: the arm revisits a similar pose but the correct next action
now depends on where the object actually is, so the same state maps to different actions and vision
becomes necessary. That makes recovery data a more principled fix than `train_nostate.py`, which deletes
the shortcut's input rather than adding the data that disambiguates it.

    .venv/bin/python scripts/train_regrasp.py --regrasp-weight 5 \
      --policy.path=lerobot/smolvla_base --dataset.repo_id=local/<ds> --dataset.root=outputs/datasets/<ds> \
      --batch_size=64 --steps=20000 --policy.device=cuda --output_dir=outputs/train/<run>

`--regrasp-weight N` repeats each re-grasp frame N times per epoch (1 = off, just report the balance).
Ranges come from meta/annotations.json, written by the teleop panel's deck key 0 while recording.

HOW IT HOOKS IN: SmolVLA has no `drop_n_last_frames`, so lerobot_train builds its DataLoader with
`sampler=None, shuffle=True` — there is no sampler to subclass. So we wrap torch's DataLoader and
substitute an index sampler, then hand off to the stock `train()` (same approach as train_nostate.py).

ALTERNATIVE NOT TAKEN: conditioning instead of weighting — give recovery segments their own task
string so the language-conditioned policy can be TOLD to re-grasp. That needs a trigger at deploy time
(the policy cannot detect its own failure), so it only pays off with a human-in-the-loop key during
eval. Weighting needs no deploy-side change, so start here.
"""
from __future__ import annotations

import json
import os
import sys

import torch.utils.data as _tud


def _pop_arg(name: str, default):
    """Pull our own flag out of argv before draccus parses the rest (it rejects unknown args)."""
    if name in sys.argv:
        i = sys.argv.index(name)
        val = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return type(default)(val)
    return default


REGRASP_WEIGHT = _pop_arg("--regrasp-weight", 5)


def _regrasp_global_indices(dataset) -> "tuple[list[int], int]":
    """Global dataset indices covered by a re-grasp range. Returns (indices, total_frames).

    annotations.json holds EPISODE-LOCAL frame ranges; `meta.episodes['dataset_from_index']` gives each
    episode's offset into the flat dataset, so global = offset + local. Ranges are clipped to the
    episode (a trim can shift frames inside an episode without the sidecar knowing — see
    webui/annotations.FIELDS).
    """
    root = dataset.root if isinstance(dataset.root, str) else str(dataset.root)
    path = os.path.join(root, "meta", "annotations.json")
    if not os.path.isfile(path):
        return [], len(dataset)
    try:
        with open(path) as f:
            ann = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"[regrasp] could not read {path}: {e}", flush=True)
        return [], len(dataset)

    eps = dataset.meta.episodes
    starts, ends = list(eps["dataset_from_index"]), list(eps["dataset_to_index"])
    out: list[int] = []
    clipped = 0
    for ep_str, rec in (ann or {}).items():
        ranges = (rec or {}).get("regrasp_ranges") or []
        try:
            ep = int(ep_str)
        except (TypeError, ValueError):
            continue
        if ep >= len(starts):
            continue
        base, cap = starts[ep], ends[ep]
        for rng in ranges:
            if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
                continue
            lo, hi = int(rng[0]), int(rng[1])
            g0, g1 = base + max(0, lo), min(base + max(0, hi), cap)
            if g1 <= g0:
                continue
            if base + hi > cap:
                clipped += 1
            out.extend(range(g0, g1))
    if clipped:
        print(f"[regrasp] {clipped} range(s) extended past their episode and were clipped "
              f"(was the episode trimmed after marking?)", flush=True)
    return sorted(set(out)), len(dataset)


class _OversampleSampler:
    """Every frame once, plus re-grasp frames repeated (weight-1) more times. Reshuffled per epoch.

    A repeated-index list rather than WeightedRandomSampler: the oversample factor is then exact and
    every non-recovery frame is still seen once per epoch, instead of sampling with replacement and
    changing what an "epoch" means.
    """

    def __init__(self, n: int, hot: "list[int]", weight: int):
        import torch

        self._torch = torch
        self.indices = list(range(n)) + list(hot) * max(0, int(weight) - 1)

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        order = self._torch.randperm(len(self.indices)).tolist()
        return iter([self.indices[i] for i in order])


_orig_dataloader = _tud.DataLoader


class _RegraspDataLoader(_orig_dataloader):
    """DataLoader that swaps in the oversampling sampler, then removes the patch.

    Must be a SUBCLASS, not a wrapper function: accelerate does
    `isinstance(obj, torch.utils.data.DataLoader)` when preparing, and a function is not a type.
    Once the training loader is built we restore the original class, so anything constructed later
    (eval loaders, accelerate internals) is untouched and still passes that isinstance check.
    """

    def __init__(self, dataset, *a, **kw):
        # the training loader is the one built with shuffle=True and no sampler
        if kw.get("sampler") is None and kw.get("shuffle") and hasattr(dataset, "meta"):
            hot, total = _regrasp_global_indices(dataset)
            pct = 100.0 * len(hot) / max(total, 1)
            if not hot:
                print(f"[regrasp] no re-grasp ranges in this dataset's annotations.json — "
                      f"sampling unchanged ({total} frames)", flush=True)
            elif REGRASP_WEIGHT <= 1:
                print(f"[regrasp] {len(hot)}/{total} frames ({pct:.1f}%) are re-grasp; "
                      f"weight {REGRASP_WEIGHT} -> reporting only, sampling unchanged", flush=True)
            else:
                kw["sampler"] = _OversampleSampler(total, hot, REGRASP_WEIGHT)
                kw["shuffle"] = False
                after = 100.0 * (len(hot) * REGRASP_WEIGHT) / max(len(kw["sampler"]), 1)
                print(f"[regrasp] {len(hot)}/{total} frames ({pct:.1f}%) are re-grasp; "
                      f"weight x{REGRASP_WEIGHT} -> {after:.1f}% of samples, "
                      f"epoch {total} -> {len(kw['sampler'])} samples", flush=True)
            _tud.DataLoader = _orig_dataloader   # one-shot: leave everything after this alone
        super().__init__(dataset, *a, **kw)


_tud.DataLoader = _RegraspDataLoader


def _carry_capture_geometry() -> None:
    """Copy the dataset's meta/capture.json (the camera crop it was recorded with) into
    outputs/train/<run>/capture.json, so the checkpoint carries its own geometry: the inference
    server reports it with {'cmd':'info'} and the eval applies it to the cameras. Without this a
    floor-task model served after a shelf-task model would be evaluated through the wrong crop."""
    root = out = None
    for a in sys.argv:
        if a.startswith("--dataset.root="):
            root = a.split("=", 1)[1]
        elif a.startswith("--output_dir="):
            out = a.split("=", 1)[1]
    if not root or not out:
        return
    src = os.path.join(root, "meta", "capture.json")
    if not os.path.isfile(src):
        print(f"[capture] {src} not found — dataset predates capture.json; checkpoint will carry no "
              f"geometry (eval keeps the cameras as they are)", flush=True)
        return
    # Do NOT create the output dir here: lerobot refuses to train into a directory that already
    # exists ("Output directory ... already exists and resume is False" — 2026-10-26, the manager's
    # Train button). lerobot only makes the dir at its first checkpoint save, so copy from a poller
    # once it appears, and again at exit for a run that ends right after its last checkpoint
    # (a 2-step test run finished before the 2 s poll saw the dir).
    import atexit
    import shutil
    import threading
    import time

    done = threading.Event()

    def carry(reason: str) -> bool:
        if done.is_set() or not os.path.isdir(out):
            return done.is_set()
        try:
            shutil.copy2(src, os.path.join(out, "capture.json"))
            print(f"[capture] geometry carried into {out}/capture.json ({reason})", flush=True)
        except OSError as e:
            print(f"[capture] could not copy capture.json: {e}", flush=True)
        done.set()
        return True

    def poll() -> None:
        while not done.is_set():
            if carry("first checkpoint"):
                return
            time.sleep(2.0)

    threading.Thread(target=poll, daemon=True, name="capture-carry").start()
    atexit.register(lambda: carry("at exit") or print(
        f"[capture] {out} never appeared — capture.json not carried", flush=True))


_carry_capture_geometry()

# Every training run leaves a full Arrow copy of the dataset in ~/.cache/huggingface/datasets (the
# parquet loader re-materialises it per fingerprint): 203 copies / 17 GB on the GPU box on 2026-10-24.
try:
    from ur5e_lerobot import hf_cache as _hf_cache
    print("[cache] " + _hf_cache.prune(keep=3), flush=True)
except Exception as _e:  # noqa: BLE001
    print(f"[cache] prune skipped: {_e}", flush=True)

from lerobot.scripts.lerobot_train import train  # noqa: E402  (patch must land before train imports)

if __name__ == "__main__":
    train()
