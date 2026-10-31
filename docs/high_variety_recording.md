# High-Variety Re-Record Checklist (fix the copycat failure)

## Why this matters (the one principle)

Three offline experiments on the `pickup` dataset (absolute+state, no-state, delta+state) all produced
a policy that **ignores the cameras** — `scripts/debug_eval.py` showed the predicted action barely
changed (1–4 mm) when the object image was swapped. The model, the 3-camera bridge, the resolution, and
the observation pipeline were all verified correct. The failure is the **copycat / causal-confusion**
trap: whenever the demos allow a *non-visual* solution (the arm's trajectory is predictable from its own
pose because the object lives in a limited set of places), the policy takes that shortcut and never
learns to look.

No action/observation reformulation fixes this on low-variety data. The one thing that does:

> **The correct action must depend ONLY on where the object is.** So randomize the object position
> *widely* across episodes, and hold everything the policy could cheat on *constant*.

The reference project that works (leesweqq/ur5_vla_pick_cube_project) succeeds because its PyBullet
scripted expert **randomizes the cube spawn every episode** — high object variety makes proprioception
useless for predicting the motion, so vision becomes mandatory.

---

## Before the session (setup)

- [ ] **Cameras fixed and untouched.** Scene (0:7.3), wrist (0:5), side (0:8.4) — do NOT bump or
      re-mount them during collection *or* before eval. Geometry consistency train↔deploy matters more
      than anything optical.
- [ ] **Lighting consistent** and adequate (gain is on auto now). Check all three feeds in the panel
      look well-exposed and stable before starting.
- [ ] **SpaceMouse** selected as input (NOT freedrive) and calibrated (roll/yaw already fixed).
- [ ] **Set Home** to a good, safe transport start pose (arm tucked, workspace clear). This becomes the
      consistent start + the auto-return-on-save target.
- [ ] **Pick a real task string** and reuse it verbatim at eval — e.g. `--task "pick up the red cube"`.
      (The old data still says `"put the block on the green pad"` — don't repeat that mistake.)
- [ ] Launch on the robot PC into a **fresh dataset**:
      ```bash
      ~/VLA/run.sh -m ur5e_lerobot.teleop.manual_panel --engine hardware --side-cam \
        --use-videos --resume --root outputs/datasets/pickup_v2 \
        --task "pick up the red cube" --operator "Bryan Ribas"
      ```
      (`--resume` appends into one dataset so all episodes stay mergeable; each Save auto-returns home,
      auto-uploads to the GPU, and lights the deck for a 1–5★ rating.)

## The object variety (the core of the whole thing)

- [ ] **Move the object to a genuinely different spot every single episode** — spanning the *entire*
      reachable workspace: near/far, left/right, all four corners, center, and the edges. If you plotted
      every object position, it should **fill the area, not cluster.**
- [ ] **Randomize the object's orientation** too (rotate it), not just its position.
- [ ] Don't let your hand fall into a habit of similar placements. Use an aid: mentally tile the table
      into ~a dozen zones and cycle/shuffle through them, or roll a die for zone + a coin for
      orientation. Deliberate scatter beats "eyeball random" (which clusters).
- [ ] If the object drifts to only a comfortable sub-region, you'll reproduce the exact failure — the
      spread of object positions **is** the fix.

## Per-episode loop

1. **Reset the object** to a new, deliberately-different position + orientation.
2. **Arm at home** (it returns there automatically after the previous Save; deck **Reset** if needed).
3. **● Start rec.**
4. **Teleop the pick** with the SpaceMouse: approach → grasp → lift. Let the motion genuinely follow
      where the object is (that's the vision signal you're teaching).
5. **■ Save** — the arm auto-returns to home (recorded), the episode auto-uploads to the GPU.
6. **Rate 1–5★** on the deck bottom row (skip/discard fumbles rather than saving bad demos).
7. Repeat with a new object position.

## Keep these CONSTANT (so the policy learns the task, not noise)

- Camera poses, lighting, the task string, and the home/start pose. A consistent home start is *good*
  here: the approach *from* home must aim at the object, which is exactly the vision-dependent behavior
  we want. Vary the **object**, not the rig.

## How many

- Aim for **~100+ clean episodes** (the reference used 100). The old set had 55 with *narrow* variety —
  wide variety matters more than raw count, but more of both is better. Discard bad demos; don't pad.

---

## After collection — GATE before touching the robot

1. **Train** SmolVLA on the new dataset (Train button, or CLI). Start **standard** (state + absolute
   actions) — if the variety is real, that alone should force vision use:
   ```
   --policy.type=smolvla --batch_size=64 --steps=20000 --save_freq=5000 ...
   ```
2. **Run the offline gate** — do NOT drive the robot until this passes:
   ```bash
   .venv/bin/python scripts/debug_eval.py \
     --ckpt outputs/train/<run>/checkpoints/last/pretrained_model \
     --dataset-root outputs/datasets/pickup_v2 --repo-id local/pickup_v2
   ```
   **Pass criterion:** in check (3), *swap object image* should swing the action by **tens of mm**
   (comparable to the per-step motion / workspace scale) — not the ~1–4 mm we saw. Also check (2) blank
   should swing a lot. That's proof the policy is servoing to the object.
3. **If it passes** → ship the checkpoint to the robot, repoint `outputs/serve_ckpt`, `sudo supervisorctl
   restart vla_infer_server` there, and run `scripts/eval_hw.py --remote 127.0.0.1:8777 --cameras udp` on
   the robot (e-stop in hand) — the server runs on the Jetson, not on the GPU box.
4. **If it's still stuck low** → the variety wasn't wide enough; collect more (wider) before retraining.
   Only then reach for the stronger formulations we already built: **delta actions**
   (`scripts/make_delta_dataset.py`) and/or **no proprioception** (`scripts/train_nostate.py`) — these
   help *on top of* good variety, but they cannot substitute for it.

## Tooling already built and proven

- `scripts/debug_eval.py` — the 2-minute offline vision-use gate (run it every time).
- `scripts/make_delta_dataset.py` — absolute→delta action conversion (deploy: `target = pose + delta`).
- `scripts/train_nostate.py` — retrain with `observation.state` dropped.
- 3-camera eval bridge (`remote.py` / `infer_server.py` / `eval_hw.py`) at 960×540 — verified correct.
