# Training · Eval · Playback · Merge

On the GPU box (`HF_HUB_OFFLINE=1` for local datasets; `PYTHONPATH=.` for the scripts).

```bash
# Train ACT (no gym env -> eval_freq=0; no hub push):
HF_HUB_OFFLINE=1 .venv/bin/python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/<id> --dataset.root=outputs/datasets/<dir> \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --steps=50000 --batch_size=16 --num_workers=24 --eval_freq=0 \
  --output_dir=outputs/train/<run> --wandb.enable=false

# Fine-tune SmolVLA from the pretrained base (the normal path; the manager's Train button runs exactly
# this). train_regrasp.py wraps lerobot_train: re-grasp segments from meta/annotations.json are
# oversampled, the dataset's meta/capture.json (crop) is carried into the run, and the HF datasets cache
# is pruned first. --policy.path (NOT --policy.type=smolvla, which trains from scratch) + --rename_map
# (scene/wrist/side -> camera1/2/3). num_workers 4: 8 filled host RAM.
PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/train_regrasp.py --regrasp-weight 3 \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/<id> --dataset.root=outputs/datasets/<dir> \
  --rename_map='{"observation.images.scene":"observation.images.camera1","observation.images.wrist":"observation.images.camera2","observation.images.side":"observation.images.camera3"}' \
  --batch_size=64 --steps=20000 --num_workers=4 --save_freq=2500 --policy.device=cuda \
  --policy.push_to_hub=false --output_dir=outputs/train/<run>_smolvla --wandb.enable=false
# then ship the checkpoint to the robot and point outputs/serve_ckpt at it (Machines & Network).
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

> **Camera / dataset note.** `eval_policy.py` runs in sim at `320×240`. Hardware datasets are recorded
> with three cameras (scene · wrist · side) at the 960×540 box, the sideways ones square-cropped; the
> crop is recorded in `meta/capture.json` and must match between the training data and the eval — the
> bridge carries it from the checkpoint to the cameras automatically. Datasets with different crops
> cannot be merged. See [SmolVLA notes](https://github.com/r1b4z01d/VLA/blob/main/docs/smolvla.md).

**Result so far:** an ACT run in sim at a single fixed position scored **100% in-distribution, ~10%
randomized** — a memorization signature. The fix is **more position-varied demos**, not a bigger model.
