# Machines & Network

- **NVIDIA Linux workstation** (`GPU` ssh alias = `4090TI`, Ubuntu 24.04, `192.168.11.119` on the robot Wi-Fi / `192.168.1.135` wired; replaced `bryan-ubuntu` 2026-10-26) — RTX 4090; **training + the web manager** (and the master eval registry). It no longer serves inference to the robot; its `vla_infer_server` unit is kept as a fallback / experiment host only.
- **MacBook Pro (M2 Pro)** — development + sim teleop. No CUDA.
- **Robot PC** (`robotdisco` = `192.168.11.80`; since 2026-10-10 an **NVIDIA Jetson AGX Orin** 64 GB, replacing the Intel PC) — hardware teleop/recording (arm + hand + cameras + SpaceMouse/gamepad) **and the inference server**: the `vla_infer_server` supervisor unit (`services/rd/`) serves `outputs/serve_ckpt` on `127.0.0.1:8777` and `eval_hw.py --remote 127.0.0.1:8777` runs on the same box, so nothing real-time crosses the network. Its 64 GB eMMC is the system disk; since 2026-10-26 `~/VLA` and `~/rd_ws` are **symlinks to `/mnt/data/VLA` and `/mnt/data/rd_ws` on a 512 GB NVMe** (ext4, label `data`, fstab `nofail`; `/etc/systemd/system/supervisor.service.d/nvme-data.conf` orders supervisor after the mount), so every `/home/rd/...` path in these pages, the supervisor confs and the rsync deploys is unchanged.
- **Robot subnet `192.168.11.x`** — UR5e `.21` (RTDE `:30004`, dashboard `:29999`), AmazingHand **ESP32 `.117`** (UDP `:8766` by default, TCP `:8765` fallback).
  The subnet is **isolated (no internet)**; install on the robot PC by tunnelling `pip` over SSH through a machine that has internet.

Envs: **lerobot 0.4.4** in the project `.venv` on **both the GPU box and the robot PC** (`.80`,
Python 3.10) — so trained checkpoints load across them without a version mismatch; the **Mac** dev
env is lerobot 0.5.x / Python 3.12 (sim + development only, not in the train→deploy path).

## Where inference runs (since the Jetson)
The policy is always a separate server process behind the framed-pickle bridge (`ur5e_lerobot/remote.py`,
`scripts/infer_server.py`), and **that server runs on the robot**. The GPU box only trains:
```bash
# GPU box: train (web manager Train button or scripts/train_regrasp.py), then ship the checkpoint
rsync -a ~/VLA/outputs/train/<run>/checkpoints/<step>/ rd@192.168.11.80:VLA/outputs/train/<run>/checkpoints/<step>/
# robot: repoint the serve symlink and (re)start the unit — reactivity flags live in services/rd/vla_infer_server.conf
ln -sfn ~/VLA/outputs/train/<run>/checkpoints/<step>/pretrained_model ~/VLA/outputs/serve_ckpt
sudo supervisorctl restart vla_infer_server
# robot: eval against loopback
~/VLA/run.sh scripts/eval_hw.py --remote 127.0.0.1:8777 --cameras udp --fps 12 --video
```
Supervisor units on RD (`services/rd/`, installed by `scripts/install_services.sh rd`): `vla_cameras` and
`vla_camera_ros_bridge` (autostart), and the manual-start `vla_infer_server`, `vla_teleop` (recording panel,
args in `~/VLA/config/teleop.env`) and `vla_eval` (policy eval, `config/eval.env`), started and stopped from
the supervisor web UI at `http://192.168.11.80:9999/` (the route editor is on `:8092`, embedded in the manager's Routes tab); the web manager also runs on the robot at `:8081` (`services/rd/vla_webui.conf`) or `sudo supervisorctl start|stop <unit>` on RD.
A SmolVLA chunk costs ~0.8 s on the Orin (vs ~0.12 s on the 4090), covered by `--n-action-steps 24
--prefetch --prefetch-at 21 --rtc` in the unit. Serving from the GPU box (`--remote <GPU_IP>:8777`) still
works as a fallback, but the GPU box reaches the robot subnet over Wi-Fi only (its wired NIC is dead) and
real-time actions over that link stutter.

## Robot PC (`.80`): `run.sh`
Every robot-PC command goes through **`~/VLA/run.sh`**, which `cd`s to `~/VLA`, sets `PYTHONPATH=.`,
`PYTHONNOUSERSITE=1` (isolates from a conflicting `~/.local` ML stack), and `HF_HUB_OFFLINE=1` (the
subnet has no internet — stops lerobot/HF from hanging on DNS), and uses the project venv:
```bash
~/VLA/run.sh -m ur5e_lerobot.teleop.manual_panel --engine hardware
```
