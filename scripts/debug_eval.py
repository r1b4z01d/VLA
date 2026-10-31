"""Offline eval diagnostic — run a trained policy on its OWN dataset (no robot) to split
'the model didn't learn' from 'the eval hardware pipeline is wrong'.

Two checks:
  1) predicted vs recorded action on training frames (n_action_steps=1: re-plan every frame).
       small ee error  => the model REPRODUCES the demos; the hardware eval failing then points at a
                          train/deploy OBSERVATION mismatch (image orientation/brightness/camera-swap/
                          resolution/state/task) — compare the eval images to the dataset next.
       large ee error  => the model / normalization / data is the problem.
  2) vision dependence — how much the predicted action changes when the images are blanked.
       near zero        => the policy IGNORES vision (memorized trajectories; it can't servo to the
                          object no matter where it is).

    .venv/bin/python scripts/debug_eval.py \
      --ckpt outputs/train/pickup_smolvla/checkpoints/last/pretrained_model \
      --dataset-root outputs/datasets/pickup --repo-id local/pickup --episodes 3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

import eval_hw  # reuse the exact policy-load + reactivity helpers the real eval uses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", default="local/x")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    img_keys = [k for k in ds.meta.features if k.startswith("observation.images.")]
    print(f"dataset: {ds.num_episodes} eps, {ds.num_frames} frames, images={img_keys}")
    print(f"sample task: {ds[0]['task']!r}")

    policy, ptype = eval_hw._load_policy(args.ckpt, args.device)
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": args.device}})
    eval_hw._set_reactivity(policy, ptype, 1, None)  # re-plan every frame for a fair per-frame compare
    print(f"loaded {ptype}\n")

    try:  # per-frame episode ids WITHOUT decoding images (column read)
        epi = np.asarray(ds.hf_dataset["episode_index"])
    except Exception:  # noqa: BLE001 — fallback: scan (slower, decodes)
        epi = np.asarray([int(ds[i]["episode_index"]) for i in range(len(ds))])
    ep_idxs = {e: np.where(epi == e)[0] for e in range(min(args.episodes, ds.num_episodes))}

    def predict(item, images, task=None):
        obs = {"observation.state": item["observation.state"][None].to(args.device),
               "task": [task if task is not None else item["task"]]}
        for k in img_keys:
            obs[k] = images[k][None].to(args.device)
        with torch.no_grad():
            return post(policy.select_action(pre(obs)))[0].cpu().numpy()

    print("=== 1) predicted vs recorded action  |pred - recorded| ===")
    ee_all = []
    for e in range(min(args.episodes, ds.num_episodes)):
        policy.reset()
        errs = []
        for i in ep_idxs[e]:
            item = ds[int(i)]
            errs.append(np.abs(predict(item, {k: item[k] for k in img_keys}) - item["action"].numpy()))
        errs = np.array(errs)
        ee = errs[:, :3].mean()
        ee_all.append(ee)
        print(f"  ep {e}: ee-pos={ee * 1000:5.0f} mm | rotvec={errs[:, 3:6].mean():.3f} | "
              f"curls={errs[:, 6:].mean():.3f} | {len(errs)} frames")
    print(f"  MEAN ee-pos error = {np.mean(ee_all) * 1000:.0f} mm   "
          f"(small = the model reproduces the demos)\n")

    print("=== 2) vision dependence (action change when images are blanked) ===")
    mid = ep_idxs[0][len(ep_idxs[0]) // 2]
    item = ds[int(mid)]
    policy.reset()
    a_real = predict(item, {k: item[k] for k in img_keys})
    policy.reset()
    a_blank = predict(item, {k: torch.zeros_like(item[k]) for k in img_keys})
    d = np.abs(a_real - a_blank)
    print(f"  |action(real) - action(blank)|: ee-pos={d[:3].mean() * 1000:.0f} mm, "
          f"rotvec={d[3:6].mean():.3f}, curls={d[6:].mean():.3f}")
    print("  (near-zero ee => the policy ignores vision -> memorized, won't servo to the object)")

    if len(ep_idxs) >= 2:
        print("\n=== 3) does the action follow the OBJECT image + TASK string? ===")
        ea, eb = ep_idxs[0], ep_idxs[1]
        ia = ds[int(ea[len(ea) // 3])]
        ib = ds[int(eb[len(eb) // 3])]  # different episode => object in a different place
        imgs_a = {k: ia[k] for k in img_keys}
        imgs_b = {k: ib[k] for k in img_keys}
        policy.reset(); base = predict(ia, imgs_a)
        policy.reset(); swap = predict(ia, imgs_b)  # SAME state, different-episode images
        policy.reset(); wrong = predict(ia, imgs_a, task="grab the object")
        di, dt = np.abs(base - swap), np.abs(base - wrong)
        print(f"  swap to a different-episode image: ee-pos change = {di[:3].mean() * 1000:.0f} mm")
        print(f"  wrong task string:                 ee-pos change = {dt[:3].mean() * 1000:.0f} mm")
        print("  (ref: typical per-step motion ~3 mm, workspace spread ~160 mm; a vision-driven "
              "policy should swing a LOT when the object image changes)")


if __name__ == "__main__":
    main()
