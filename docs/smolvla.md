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

## Deploy / eval (robot PC `.80`, CPU-only) — the hard constraint
`eval_hw.py` loads the SmolVLA checkpoint the same way (`--ckpt .../pretrained_model`). **But `.80` has no
GPU**, and SmolVLA (~450M) on CPU is ~**seconds per inference**, not the ~1 ms ACT enjoyed. So:
- Watch `infer_ms` in `outputs/eval_hw_log.csv`. It will not hold 12 fps on CPU.
- Short-term: raise `--n-action-steps` so one slow model call feeds many executed steps (open-loop-ish
  between calls) — trades reactivity for feasibility.
- Proper fix (TODO): **remote inference** — run SmolVLA on the GPU box, stream actions to `.80`
  (latest-wins UDP; SAFE-STALL on stale packets). The action-chunk design tolerates a low query rate,
  so cross-subnet latency may be acceptable. Or put a GPU in the robot PC.

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
