# Machines & Network

- **NVIDIA Linux workstation** (`gpu` = `192.168.11.130`) — RTX 4090; GPU training + eval. The workhorse.
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
