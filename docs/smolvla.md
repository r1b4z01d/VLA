# Migrating ACT → SmolVLA (branch `smolvla`)

**Why.** ACT is trained from scratch and needs ~50–100+ demos per task; on 23 real demos it diverged
(0% success, runaway targets — see the eval-log diagnosis). SmolVLA is a **pretrained foundation VLA**
(~450M, SmolVLM-2 backbone + flow-matching action expert) that **fine-tunes** from `lerobot/smolvla_base`,
so it's far more sample-efficient and is language-conditioned (it actually uses the `task` string). Same
dataset, same 10-D action / 16-D state, same LeRobot pipeline — swap `--policy.type`.

## Code changes on this branch
- `scripts/eval_hw.py` now **auto-detects the policy class** from the checkpoint's `config.json` `type`
  (`act` or `smolvla`) via `_load_policy()` — no other eval change needed. The Stream Deck controls,
  safety clamps, and per-step logging are policy-agnostic.
- Reactivity flags: `--n-action-steps` applies to **both** (SmolVLA also defaults to open-loop:
  `chunk_size = n_action_steps = 50`). `--temporal-ensemble` is **ACT-only** and is ignored for SmolVLA.

## Training recipe (GPU box — has CUDA + internet)
SmolVLA fine-tunes from the pretrained base, so the box must be able to fetch `lerobot/smolvla_base`
(and the SmolVLM-2 weights) once. Same video dataset as ACT.
```bash
# deps (once): transformers, accelerate, num2words, safetensors — for the SmolVLM backbone
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/hw_pickplace \
  --dataset.root=outputs/datasets/hw_pickplace_video \
  --batch_size=64 --steps=20000 \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/train/hw_pickplace_smolvla --wandb.enable=false
```
- `--policy.path=lerobot/smolvla_base` loads the pretrained VLA and fine-tunes (vs `--policy.type=act`
  from scratch). Drop `batch_size` if the 4090 OOMs; SmolVLA is designed to fine-tune on modest VRAM.
- SmolVLA fine-tunes fast (~20k steps is plenty for a single task); with more demos, scale steps.

## Deploy / eval — remote inference, server ON THE ROBOT (Jetson Orin)
**History:** the Intel robot PC had no GPU and SmolVLA was **~2.6 s/inference on its CPU** (vs ~1 ms for
ACT), so the policy was split into a server process behind a framed-pickle TCP bridge and served from the
GPU box. Since 2026-10-10 the robot PC is a **Jetson AGX Orin** and the same server runs **on the robot**
(`services/rd/vla_infer_server.conf`, ~0.8 s per chunk, covered by `--prefetch --rtc`); the GPU box only
trains. Bridge: `ur5e_lerobot/remote.py` (protocol + client) · `scripts/infer_server.py` (server) ·
`eval_hw.py --remote HOST:PORT` (client — no policy/checkpoint/deps needed locally in this mode).
```bash
# 1) GPU box: train (web manager Train button / scripts/train_regrasp.py), then ship the checkpoint
rsync -a ~/VLA/outputs/train/<run>/checkpoints/<step>/ rd@192.168.11.80:VLA/outputs/train/<run>/checkpoints/<step>/

# 2) robot (RD): point the serve symlink at it and (re)start the unit — reactivity is set in the unit
ln -sfn ~/VLA/outputs/train/<run>/checkpoints/<step>/pretrained_model ~/VLA/outputs/serve_ckpt
sudo supervisorctl restart vla_infer_server

# 3) robot (RD): eval client against loopback (e-stop in hand)
~/VLA/run.sh scripts/eval_hw.py --remote 127.0.0.1:8777 --cameras udp --task "pick up tigger" --fps 12 --video
```
The server answers `{"cmd":"info"}` with the resolved checkpoint, its reactivity flags and the capture
geometry carried from the dataset (`capture.json`); the eval writes that into its registry record, so the
web manager groups runs per model. The action-chunk design absorbs the chunk compute: on the Orin
`--n-action-steps 24 --prefetch --prefetch-at 21 --rtc` (measured in the unit file), vs
`--n-action-steps 8 --prefetch --rtc` on a 4090.

_Fallback — serve from the GPU box:_ `sudo supervisorctl start vla_infer_server` there
(`services/vla_infer_server.conf`, dual-homed on the robot subnet) and point the client at
`--remote <GPU_IP>:8777`. Not the normal path any more: the GPU box reaches the robot segment over Wi-Fi
only (its wired NIC is dead) and real-time actions over that link stutter.

## OPEN DECISION — lerobot version alignment (verify when the machines are back)
SmolVLA requires the `smolvla` policy + `transformers`. **The pinned `.venv` on both the GPU box and `.80`
is lerobot 0.4.4 — confirm it ships SmolVLA.** If not:
- Train in a **lerobot 0.5.x venv** (the `.gitignore` hints a `.venv-v05` already exists), and
- the **eval env on `.80` must match** (0.5.x + `transformers`) so the checkpoint loads — ACT loads on
  0.4.4, SmolVLA may force `.80` onto 0.5.x too. Resolve this before training so train/deploy agree.

## Verify-when-online checklist
- [ ] `import lerobot.policies.smolvla.modeling_smolvla` works in the GPU-box training venv
- [ ] `.80` eval venv has smolvla + `transformers` (already has `transformers==4.57.6`)
- [ ] GPU box can fetch `lerobot/smolvla_base` (internet)
- [ ] `hw_pickplace_video` dataset present on the GPU box
- [ ] Collect more demos first (SmolVLA is sample-efficient, but 23 is still thin)
- [ ] Train → move checkpoint to `.80` → `eval_hw.py --ckpt … --n-action-steps <tuned>` → measure `infer_ms`
