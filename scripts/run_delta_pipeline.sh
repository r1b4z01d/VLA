#!/usr/bin/env bash
# Delta-action experiment pipeline — runs AFTER the pickup_nostate run finishes, so jobs don't
# contend for the GPU. Steps: wait for no-state watcher -> convert pickup to delta actions ->
# train SmolVLA on it (state INCLUDED, matching leesweqq/ur5_vla_pick_cube_project) -> debug_eval.
# Launch detached:  setsid bash scripts/run_delta_pipeline.sh </dev/null >/dev/null 2>&1 &
set -u
cd ~/VLA || exit 1
LOG=outputs/train/delta_pipeline.log
exec >>"$LOG" 2>&1
echo "===== delta pipeline started $(date) ====="

# 1. Wait for the no-state run + its debug_eval to finish (frees the GPU).
echo "waiting for pickup_nostate watcher (WATCHER_DONE) ..."
for _ in $(seq 1 600); do   # up to 600*60s = 10h
  grep -q WATCHER_DONE outputs/train/pickup_nostate_debug.txt 2>/dev/null && break
  sleep 60
done
if ! grep -q WATCHER_DONE outputs/train/pickup_nostate_debug.txt 2>/dev/null; then
  echo "TIMEOUT waiting for pickup_nostate; aborting"; exit 1
fi
while pgrep -f "output_dir=outputs/train/pickup_nostate" >/dev/null; do sleep 30; done
sleep 20
echo "no-state done, GPU free -> starting delta job $(date)"

# 2. Convert pickup -> pickup_delta (absolute EE pose action -> delta vs current pose).
echo "===== converting pickup -> pickup_delta $(date) ====="
HF_HUB_OFFLINE=1 .venv/bin/python scripts/make_delta_dataset.py \
  --src outputs/datasets/pickup --dst outputs/datasets/pickup_delta || { echo "CONVERT FAILED"; exit 1; }

# 3. Train SmolVLA on the delta dataset (proprioceptive state kept; delta actions break the shortcut).
echo "===== training pickup_delta $(date) ====="
rm -rf outputs/train/pickup_delta outputs/train/pickup_delta.log
HF_HUB_OFFLINE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python \
  -m lerobot.scripts.lerobot_train \
  --policy.type=smolvla --dataset.repo_id=local/pickup_delta \
  --dataset.root="$(pwd)/outputs/datasets/pickup_delta" \
  --batch_size=64 --steps=20000 --save_freq=5000 --policy.device=cuda --policy.push_to_hub=false \
  --num_workers=8 --eval_freq=0 --wandb.enable=false --output_dir=outputs/train/pickup_delta \
  >outputs/train/pickup_delta.log 2>&1 || { echo "TRAIN FAILED"; exit 1; }

# 4. debug_eval on the delta model (predicted vs recorded delta + vision/object/task sensitivity).
echo "===== debug_eval pickup_delta $(date) ====="
CK=outputs/train/pickup_delta/checkpoints/020000/pretrained_model
[ -d "$CK" ] || CK=outputs/train/pickup_delta/checkpoints/last/pretrained_model
echo "debug_eval pickup_delta ($CK) - $(date)" >outputs/train/pickup_delta_debug.txt
HF_HUB_OFFLINE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python \
  scripts/debug_eval.py --ckpt "$CK" \
  --dataset-root outputs/datasets/pickup_delta --repo-id local/pickup_delta --episodes 2 \
  >>outputs/train/pickup_delta_debug.txt 2>&1
echo "PIPELINE_DONE $(date)" >>outputs/train/pickup_delta_debug.txt
echo "===== delta pipeline done $(date) ====="
