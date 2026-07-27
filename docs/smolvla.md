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

## Deploy / eval — remote inference (robot PC `.80` has no GPU)
**Measured:** SmolVLA is **~2.6 s/inference on `.80`'s CPU** (vs ~1 ms for ACT) — unusable for real-time.
So inference runs on the **GPU box** and only observations/actions cross the wire. Bridge:
`ur5e_lerobot/remote.py` (framed-pickle TCP protocol + client) · `scripts/infer_server.py` (GPU server) ·
`eval_hw.py --remote HOST:PORT` (robot-PC client — no policy/checkpoint/deps needed locally in this mode).

**Put the GPU box on the robot's subnet** (keep it dual-homed so it still has internet for training). Then
`.80` connects straight to the GPU — no Mac relay, no tunnel:
```bash
# 1) GPU box: serve the policy (reactivity is set HERE, server-side; open port 8777 in the firewall)
.venv/bin/python scripts/infer_server.py \
  --ckpt outputs/train/hw_pickplace_smolvla/checkpoints/last/pretrained_model \
  --device cuda --port 8777 --n-action-steps 8

# 2) robot PC .80: run the eval client straight at the GPU (e-stop in hand)
~/VLA/run.sh scripts/eval_hw.py --remote <GPU_IP>:8777 --task "pick up the Home Depot bucket" --fps 12 --video
```
On the GPU SmolVLA is ~tens of ms, so `--n-action-steps` can be small (reactive); the action-chunk design
also absorbs network latency. Validated offline with a mock server (protocol/client round-trip + reset);
the live path needs both machines on the network to confirm.

_Fallback — if they must stay on different subnets:_ relay through the Mac with
`ssh -N -L 0.0.0.0:8777:localhost:8777 gpu` (needs `GatewayPorts yes`) and point the client at `<MAC_IP>:8777`.
_Permanent fix:_ a GPU in the robot PC — then drop `--remote` and run local.

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
