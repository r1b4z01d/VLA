"""Retrain a policy with the proprioceptive `observation.state` DROPPED from its inputs — to test /
break the 'copycat' shortcut (the policy predicting actions from the arm's own pose instead of the
cameras). See scripts/debug_eval.py for the diagnosis this addresses.

It monkeypatches lerobot's feature extraction so the policy (and its normalization) are built
images-only, then hands off to the normal lerobot `train()` — so every usual flag still works:

    .venv/bin/python scripts/train_nostate.py \
      --policy.type=smolvla --dataset.repo_id=local/pickup --dataset.root=outputs/datasets/pickup \
      --batch_size=64 --steps=20000 --save_freq=5000 --policy.device=cuda --policy.push_to_hub=false \
      --num_workers=8 --eval_freq=0 --wandb.enable=false --output_dir=outputs/train/pickup_nostate

The dataset is unchanged; the batch still contains observation.state, the policy just never consumes
it. Eval works unchanged too (the policy simply ignores the state the bridge sends).
"""
import lerobot.policies.factory as _factory

_DROP = "observation.state"
_orig_features = _factory.dataset_to_policy_features


def _drop_state(features):
    feats = _orig_features(features)
    if _DROP in feats:
        feats.pop(_DROP)
        imgs = [k for k in feats if k.startswith("observation.images")]
        print(f"[train_nostate] dropped {_DROP}: policy inputs are images-only -> {imgs}", flush=True)
    return feats


_factory.dataset_to_policy_features = _drop_state

from lerobot.scripts.lerobot_train import train  # noqa: E402  (patch must land before train imports)

if __name__ == "__main__":
    train()
