# VLA on UR5e with LeRobot

Experimenting with Vision-Language-Action policies on a Universal Robots UR5e (with an AmazingHand
end-effector), using [LeRobot](https://github.com/huggingface/lerobot). Simulation-first;
**real-hardware teleop + recording now works** (first datasets captured on the robot). RTDE drives
the arm for teleop; ROS 2 is reserved for autonomous deployment.

See **[ROADMAP.md](ROADMAP.md)** for the full plan and **[docs/decisions.md](docs/decisions.md)**
for open architectural choices.

## Machines & network
- **NVIDIA Linux workstation** (`gpu` = `192.168.1.64`) — RTX 4090; GPU training + eval. The workhorse.
- **MacBook Pro (M2 Pro)** — development + sim teleop. No CUDA.
- **Robot PC** (Intel, `robotdisco` = `192.168.11.80`) — hardware teleop/recording: arm + hand + cameras + SpaceMouse/gamepad.
- **Robot subnet `192.168.11.x`** — UR5e `.21` (RTDE `:30004`, dashboard `:29999`), AmazingHand **ESP32 `.117`** (TCP `:8765`).
  The subnet is **isolated (no internet)**; install on the robot PC by tunnelling `pip` over SSH through a machine that has internet.

Envs: **lerobot 0.4.4** in the project `.venv` on **both the GPU box and the robot PC** (`.80`,
Python 3.10) — so trained checkpoints load across them without a version mismatch; the **Mac** dev
env is lerobot 0.5.x / Python 3.12 (sim + development only, not in the train→deploy path).

## Robot PC (`.80`): `run.sh`
Every robot-PC command goes through **`~/VLA/run.sh`**, which `cd`s to `~/VLA`, sets `PYTHONPATH=.`,
`PYTHONNOUSERSITE=1` (isolates from a conflicting `~/.local` ML stack), and `HF_HUB_OFFLINE=1` (the
subnet has no internet — stops lerobot/HF from hanging on DNS), and uses the project venv:
```bash
~/VLA/run.sh -m ur5e_lerobot.teleop.manual_panel --engine hardware
```

## Hardware teleop + recording (validated ✅)
Runs **on the robot PC** (SpaceMouse/gamepad + the GUI are local; raw-HID + display need the local seat).
Arm over **RTDE** (`robot/rtde_arm.py`: `servoL` streaming with a workspace **no-go** clamp (TCP **and**
elbow/wrist links via FK), per-command `max_step` (translation) + `max_rot_step` (wrist orientation) clamps,
an optional table **z-floor**, and an **IK-reachability guard** so an over-reach holds instead of faulting the
controller). Hand over TCP (`hand/amazing_hand_client.py` → `.117`). Cameras by **stable
`/dev/v4l/by-id` paths** (scene = 4K wide, wrist = ARC eye-in-hand); the scene feed is flipped 180°
(mounted upside down). The flip lives in the shared hardware engine, so recording and eval stay consistent.

> **Safety note.** These clamps are **software guards in the servo command path only** — not a substitute
> for UR **safety planes** (set those in PolyScope for a safety-rated, controller-enforced limit). In
> particular they are **bypassed in freedrive** (teach mode is UR-controlled), and the elbow/wrist-link
> check is a crude nominal-FK approximation. Keep the e-stop in hand.

**Bring-up (e-stop in hand):**
1. UR pendant → **Remote Control**; if powered off, power on + release brakes (dashboard) and clear the **position-verify** dialog.
2. The hand is powered from the **UR tool port** — enable **Tool Output Voltage** on the pendant so the ESP32 (`.117`) boots.
3. Launch and record (append into one dataset with `--resume`):
   ```bash
   ~/VLA/run.sh -m ur5e_lerobot.teleop.manual_panel --engine hardware \
     --resume --root outputs/datasets/hw_pickplace --task "pick up the block and place it on the pad"
   ```

**Panel** (`teleop/manual_panel.py`), single-column layout (stacked cameras → controls → bottom button bar):
- **Input dropdown**, live-switchable: `spacemouse` · `gamepad` (Xbox, evdev) · `freedrive` (hand-guide the arm; grasp via keys/SpaceMouse buttons) · `sliders`.
- **Calibrate once** so axes match your viewpoint — `sm_calibrate` / `gp_calibrate` terminal tools, or the in-GUI **Calibrate GP** button (writes `outputs/{sm,gp}_calib.json`, auto-loaded). **The gamepad won't drive the arm until calibrated.**
- **⟳ Reconnect UR** — recover from a fault / stopped control script (clears a protective stop, reuploads the script) without restarting.
- **Grasp**: SpaceMouse buttons or keyboard `c`/`o`; gamepad bumpers LB/RB.
- **● Start rec → demo → ■ Save ep**. Closing the window finalizes cleanly (parquet footers). Records both cameras + the 10-D action; **vary the block position** across episodes.

First hardware capture verified: **2 episodes / 900 frames**, both cameras, grasp exercised, `state ≠ action` (real servo lag) — genuine, learnable demos.

**Replay / review a capture** — writes a montage (1 frame/episode) + a scene|wrist MP4. Headless
(reads the recorded frames; no robot needed), so it runs over SSH:
```bash
~/VLA/run.sh -m ur5e_lerobot.sim.playback --root outputs/datasets/<dir> \
  --montage outputs/playback/<name>.png --video outputs/playback/<name>.mp4 --fps 12
```
`--fps 12` plays hardware recordings at real time (raise to fast-forward); `--step N` subsamples;
`--montage`/`--video` are independent. Files land in `outputs/playback/` — open on the robot PC's
desktop or copy off with `scp`.

## Recording sim demos (teleop)
Drive the UR5e + AmazingHand in sim (MuJoCo) and record LeRobot datasets. **Prefer the workstation** — it
records a *video* dataset where you train (image datasets load far slower). Run on the box's own desktop;
`MUJOCO_GL=egl` renders headless on the 4090, `--use-videos` needs system FFmpeg (present on the box).
```bash
cd ~/VLA && .venv/bin/python -m ur5e_lerobot.teleop.spacemouse         # confirm the SpaceMouse reads
cd ~/VLA && MUJOCO_GL=egl .venv/bin/python -m ur5e_lerobot.teleop.manual_panel \
  --input spacemouse --use-videos --root outputs/datasets/teleop
```
Drop `--use-videos` for images; `--input sliders` for keyboard/mouse; `--resume` appends. On the Mac,
images only, and quit the 3Dconnexion 3DxWare driver first so the raw SpaceMouse is readable.

**Grasping & scene.** The AmazingHand can't form a stable *physics* grasp in sim (4-finger geometry won't
oppose), so a **grasp aid welds the block to the hand on close and releases on open** — pick-place "just
works" for demo collection (`build_combined_model(grasp_aid=False)` to disable). Block = 4×4×9 cm on the
floor → green pad. Panel sim controls (hidden on `--engine hardware`): **Randomize Box/Goal** + a **ROYGBIV
box-color** dropdown. Collect *position variety*; a fixed position overfits (see eval below).

## Train · eval · playback · merge
On the GPU box (`HF_HUB_OFFLINE=1` for local datasets; `PYTHONPATH=.` for the scripts).
```bash
# Train ACT (no gym env -> eval_freq=0; no hub push):
HF_HUB_OFFLINE=1 .venv/bin/python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/<id> --dataset.root=outputs/datasets/<dir> \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --steps=50000 --batch_size=16 --num_workers=24 --eval_freq=0 \
  --output_dir=outputs/train/<run> --wandb.enable=false

# Roll out a policy in sim, score pick-place (+ --video):
PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/eval_policy.py \
  --ckpt outputs/train/<run>/checkpoints/last/pretrained_model --episodes 10 --steps 400 --device cuda [--randomize] [--video]

# Replay recorded episodes (montage + video):
PYTHONPATH=. .venv/bin/python -m ur5e_lerobot.sim.playback --root outputs/datasets/<dir> --montage out/montage.png --video out/demos.mp4

# Merge/recover recording sessions into one dataset (handles unfinalized/partly-corrupt sources):
PYTHONPATH=. .venv/bin/python scripts/recover_merge_datasets.py \
  outputs/datasets/a outputs/datasets/b --out outputs/datasets/merged --repo-id local/merged
```
Current ACT result (sim, single fixed position): **100% in-distribution, ~10% randomized** — a
memorization signature; the fix is more position-varied demos, not a bigger model.

## Deploy a trained policy on real hardware (`eval_hw.py`) — e-stop in hand
Runs a trained checkpoint on the **real UR5e + AmazingHand** from live camera + proprioceptive
observations (`scripts/eval_hw.py`, on the robot PC). **Motion-producing** — but the same `rtde_arm`
clamps stay active (workspace no-go, per-step `max_step` + `max_rot_step`, IK-reachability guard), so the arm *ramps*
toward the policy's targets and can't jump; **Ctrl-C** aborts (stops the servo + control script).

**One-time offline setup** (the robot subnet has no internet): the ACT ResNet18 backbone tries to
fetch ImageNet weights the first time a checkpoint loads. Seed the torch cache once — copy
`resnet18-f37072fd.pth` into `~/.cache/torch/hub/checkpoints/` from a machine that has it (the GPU box
or Mac) — and `run.sh` already exports `HF_HUB_OFFLINE=1`. The trained weights overwrite the backbone
on load, so the ImageNet init is only needed to build the module.

**Interactive — Stream Deck driven, launches PAUSED.** No panel-first dance: freedrive is built in, and
`eval_hw` powers the UR **tool voltage** itself if the first connect fails (the hand is tool-powered).
Three keys (a **keyboard fallback** maps the same letters if no deck is found):
- **PLAY / PAUSE** (`p`) — toggle the policy. Paused holds the arm; play (re)starts and resets the
  policy so it re-observes from the current pose. Key is **green** paused, **red** while the arm is live.
- **FREEDRIVE** (`f`) — gravity-comp teach mode: hand-guide the arm to a **training** start pose (policy
  suspended). Pressing PLAY exits freedrive and starts.
- **RESET** (`r`) — reset the policy's internal state and pause (arm holds). Abort a run and start clean.
- **EXIT** (`x`) — stop + disconnect cleanly (Ctrl-C also aborts).

**Run (e-stop in hand):** UR in **Remote Control** + brakes released; place the target as in the demos.
```bash
~/VLA/run.sh scripts/eval_hw.py \
  --ckpt outputs/train/hw_pickplace_act/checkpoints/last/pretrained_model \
  --task "pick up the Home Depot bucket" \
  --temporal-ensemble 0.01 --fps 12 --video
```
**Reactivity (important):** an ACT checkpoint runs `n_action_steps == chunk_size` (here 100) by default
— **fully open-loop**: one observation, then 100 blind actions (~8 s at 12 fps). For a contact-rich pick
this alone can mean 0%. `--temporal-ensemble 0.01` re-observes **every step** (ALOHA-style blending);
`--n-action-steps 8` re-observes every 8. Both are inference-time only — **no retraining**.
Typical loop: launch (paused) → **FREEDRIVE** to a training-like start → **FREEDRIVE off** → **PLAY** →
watch → **PAUSE** / reposition / **PLAY** → **EXIT**. `--fps` **must match the training data** (12 here);
`--device cpu` by default (no CUDA on the robot PC — ACT is small, CPU keeps up at 12 fps); `--video`
writes `outputs/eval_hw.mp4` of the played segments; `--steps N` auto-pauses after N played steps (0 =
unlimited); `--task` is cosmetic for ACT (no language encoder). There is **no automatic success metric**
on hardware — watch it and judge, or review the MP4. First rollouts from ~20 demos read as "reaches the
right region" more than clean grasps; the fix is more position-varied demos, not a bigger model.

## 8-DOF hand (per-finger flex + abduct) — in progress
Adding palm-back **buttons (PCF8574 on the ESP32 I²C bus)** for per-finger **flex *and* abduct**, recorded
as a **14-D** action. Groundwork done: `schema.ActionV2` (14-D) + `STATE_V2` (20-D), and
`amazing_hand_client` `send_flex_abduct` / `read_state` (the `abduct=0` path is bit-identical to today's
curls). Firmware + `:8765` protocol (`F:` command, `S:` report) spec:
**[docs/amazinghand_esp32_buttons.md](docs/amazinghand_esp32_buttons.md)**. Panel "hand-buttons" recording
source lands when the board is wired + flashed. The live pipeline stays **10-D**; v2 is opt-in.

## Sync
Robot PC has no rsync — use **tar-over-ssh**:
```bash
tar czf - --exclude=__pycache__ --exclude='._*' ur5e_lerobot scripts | ssh rd@192.168.11.80 'tar xzf - -C ~/VLA'
```
Workstation (has rsync):
```bash
rsync -az --exclude=__pycache__ --exclude='._*' ur5e_lerobot/ scripts/ assets/ gpu:~/VLA/
```

## Layout
- `scripts/` — setup, eval, merge, sim-grasp diagnostics; `hw_preflight.py` (staged hardware bring-up: arm/hand/cameras/move).
- `ur5e_lerobot/` — integration package.
  - `schema.py` — action/obs contract: **v1 10-D / 16-D state** (live) + **v2 14-D / 20-D** (8-DOF hand, opt-in) ✅
  - `hand/amazing_hand_client.py` — TCP client for the AmazingHand ESP32 (`.117:8765`): curls + flex/abduct + `read_state` ✅
  - `robot/` — LeRobot adapter `URAmazingHand`; `rtde_arm.py` (RTDE teleop: servoL + no-go/max_step/max_rot_step/z-floor/IK guard + freedrive + reconnect); `workspace.py` (no-go zone incl. elbow/wrist-link FK check, body at +y); `arm_interface.py` `Ros2ArmInterface` (stub) ✅
  - `sensors/cameras.py` — USB cameras by stable `by-id` path (+ 180° flip option) ✅
  - `sim/` — MuJoCo UR5e + AmazingHand, mink IK, grasp-aid weld, record/playback (`--engine mujoco|kinematic`) ✅
  - `teleop/manual_panel.py` — Tkinter panel: input dropdown (spacemouse/gamepad/freedrive/sliders), Reconnect-UR, calibrator loaders, `--engine hardware`, clean finalize ✅
  - `teleop/spacemouse.py` · `gamepad.py` (evdev) · `sm_calibrate.py` · `gp_calibrate.py` — input readers + one-shot axis calibrators ✅
- `docs/` — decisions, data schema, camera setup, **`amazinghand_esp32_buttons.md`** (8-DOF firmware/protocol spec).
- `assets/amazing_hand/` — vendored MuJoCo hand model. UR5e model from `robot_descriptions`.
- `ros2_ws/` — UR5e ROS 2 bringup (deployment path; build on the Linux box).