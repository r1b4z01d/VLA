# SmolVLA: training and serving

**Why SmolVLA.** ACT is trained from scratch and needs ~50-100+ demos per task; on 23 real demos it
diverged (0 % success, runaway targets). SmolVLA is a pretrained foundation VLA (~450M, SmolVLM-2
backbone + flow-matching action expert) that **fine-tunes** from `lerobot/smolvla_base`, is far more
sample-efficient and is language-conditioned (it uses the `task` string). Same dataset, same 10-D
action / 16-D state, same LeRobot pipeline. Merged on 2026-10; **lerobot 0.4.4** serves it on both the
GPU box and the robot.

## Training (GPU box: CUDA + internet)
The manager's **Train** button on a dataset card is the normal path; it runs
`scripts/train_regrasp.py`, which wraps `lerobot_train` and oversamples the re-grasp segments the
teleop panel flagged (`meta/annotations.json` `regrasp_ranges`). The equivalent command:
```bash
.venv/bin/python scripts/train_regrasp.py --regrasp-weight 2 \
  --policy.path=lerobot/smolvla_base \
  --rename_map='{"observation.images.scene":"observation.images.camera1","observation.images.wrist":"observation.images.camera2","observation.images.side":"observation.images.camera3"}' \
  --dataset.repo_id=local/tigger_1_merged --dataset.root=outputs/datasets/tigger_1_merged \
  --policy.device=cuda --policy.push_to_hub=false --num_workers=4 \
  --steps=20000 --batch_size=64 --save_freq=2500 --eval_freq=0 --wandb.enable=false \
  --output_dir=outputs/train/tigger_1_merged_smolvla
```
- `--policy.path=lerobot/smolvla_base` loads the pretrained VLA; `--policy.type=smolvla` would train
  **from scratch** and is never what you want (`tigger_1_merged_smolvla`, 2026-10-16, was such a run and
  must not be served).
- `--rename_map` maps the dataset's cameras to `camera1/2/3`, what the base was trained on; the manager
  builds it from the dataset's actual image keys.
- `--regrasp-weight`: the manager picks 3 when the dataset's recovery share is under 10 %, 2 under 20 %,
  else 1 (report only). A fixed 3 on a 23 % set would make a policy that mostly practises retrying.
- `--num_workers=4`: 8 took the box down by OOM on 2026-09-24. The manager also wraps the run in
  `systemd-run --user` with `MemoryMax=40G`.
- The run directory gets the dataset's `capture.json` (the crop the policy trained on) and the HF
  datasets cache is pruned afterwards (`ur5e_lerobot/hf_cache.py`, keep 3).
- ~20k steps is plenty for a single task; the first proper fine-tune (`tigger_1_merged_smolvla_1`,
  2026-10-16, 73 episodes) reached loss 0.004 / cos-vs-base 0.992.

## Serving: the inference server runs ON the robot (Jetson Orin)
**History:** the Intel robot PC had no GPU and SmolVLA took ~2.6 s per inference on its CPU, so the
policy was split into a server behind a framed-pickle TCP bridge and served from the GPU box. Since
2026-10-10 the robot PC is a **Jetson AGX Orin** and the same server runs **on the robot**
(`services/rd/vla_infer_server.conf`, manual start, loopback :8777); the GPU box only trains. Bridge:
`ur5e_lerobot/remote.py` (protocol + client) · `scripts/infer_server.py` (server) · `eval_hw.py --remote
HOST:PORT` (client — no policy, checkpoint or torch needed on the client side).
```bash
# 1) ship the checkpoint from the GPU box (pretrained_model only, ~865 MB) — directly, or via the Mac
rsync -a ~/VLA/outputs/train/<run>/checkpoints/<step>/ rd@192.168.11.80:rd_ws/VLA/outputs/train/<run>/checkpoints/<step>/
# 2) robot: point the serve symlink at it and restart the unit (or: ~/rd_ws/VLA/run.sh -m ur5e_lerobot.services serve <run>)
ln -sfn ~/rd_ws/VLA/outputs/train/<run>/checkpoints/<step>/pretrained_model ~/rd_ws/VLA/outputs/serve_ckpt
sudo supervisorctl restart vla_infer_server
# 3) robot: eval client against loopback (e-stop in hand), or the vla_eval unit with config/eval.env
~/rd_ws/VLA/run.sh scripts/eval_hw.py --remote 127.0.0.1:8777 --cameras udp --task "pick up tigger" --fps 12 --video
```
The server answers `{"cmd":"info"}` with the resolved checkpoint, its reactivity flags and the capture
geometry from the run's `capture.json`; the eval writes that into its registry record, so the manager
groups runs per model.

**Reactivity lives in the unit file, server side.** In remote mode `--n-action-steps`, `--prefetch`,
`--prefetch-at` and `--rtc` are server flags; the client only streams observations and plays the
chunks. On the Orin a chunk costs ~0.9-1.2 s, so the unit runs `--n-action-steps 24 --prefetch
--prefetch-at 21 --rtc` (measured 2026-10-11; the policy re-observes every 2 s, RTC blends each seam)
and keeps the route's models **resident** with `--also <run>` so a stage switch is a `select` (<1 s)
instead of a restart (~16 s). The base VLM is loaded from `~/.cache/huggingface/hub` with
`HF_HUB_OFFLINE=1` (the robot has no internet).

**Fallback — serve from the GPU box:** `sudo supervisorctl start vla_infer_server` there and point the
client at `--remote 192.168.11.119:8777`. Its unit (`services/vla_infer_server.conf`) currently runs
`--n-action-steps 4` without prefetch/RTC — a 2026-10-12 experiment, not a tuned setting — and the GPU
box reaches the robot segment over Wi-Fi only, so real-time actions over that link stutter. Not the
normal path.
