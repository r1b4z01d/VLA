# Training · Eval · Playback · Merge

On the GPU box (`HF_HUB_OFFLINE=1` for local datasets; `PYTHONPATH=.` for the scripts).

```bash
# Train ACT (no gym env -> eval_freq=0; no hub push):
HF_HUB_OFFLINE=1 .venv/bin/python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/<id> --dataset.root=outputs/datasets/<dir> \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --steps=50000 --batch_size=16 --num_workers=24 --eval_freq=0 \
  --output_dir=outputs/train/<run> --wandb.enable=false

# Fine-tune SmolVLA from the pretrained base instead (needs internet + transformers; more sample-efficient
# than ACT-from-scratch). Full recipe + deploy/version notes: docs/smolvla.md
lerobot-train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/<id> --dataset.root=outputs/datasets/<dir> \
  --batch_size=64 --steps=20000 --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/train/<run>_smolvla --wandb.enable=false
# eval_hw.py auto-detects act vs smolvla from the checkpoint config — same deploy command.

# Roll out a policy in sim, score pick-place (+ --video):
PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/eval_policy.py \
  --ckpt outputs/train/<run>/checkpoints/last/pretrained_model --episodes 10 --steps 400 --device cuda [--randomize] [--video]

# Replay recorded episodes (montage + video):
PYTHONPATH=. .venv/bin/python -m ur5e_lerobot.sim.playback --root outputs/datasets/<dir> --montage out/montage.png --video out/demos.mp4

# Merge/recover recording sessions into one dataset (handles unfinalized/partly-corrupt sources):
PYTHONPATH=. .venv/bin/python scripts/recover_merge_datasets.py \
  outputs/datasets/a outputs/datasets/b --out outputs/datasets/merged --repo-id local/merged
```

> **Camera / dataset note.** `eval_policy.py` runs in sim at `320×240`. Hardware datasets are now
> recorded at **960×540** with three cameras (scene · wrist · side); when a 3-camera checkpoint is
> trained, the eval bridge (`remote.py` / `infer_server.py` / `eval_hw.py`) needs the third image wired
> through and `W,H` set to `960×540` to match. See [SmolVLA notes](https://github.com/r1b4z01d/VLA/blob/main/docs/smolvla.md).

**Result so far:** an ACT run in sim at a single fixed position scored **100% in-distribution, ~10%
randomized** — a memorization signature. The fix is **more position-varied demos**, not a bigger model.
