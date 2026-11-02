# Machines & Network

- **NVIDIA Linux workstation** (`GPU` ssh alias = `4090TI`, Ubuntu 24.04, `192.168.11.119` on the robot Wi-Fi / `192.168.1.135` wired; replaced `bryan-ubuntu` 2026-10-26) — RTX 4090; **training + the web manager** (and the master eval registry). It no longer serves inference to the robot; its `vla_infer_server` unit is kept as a fallback / experiment host only.
- **MacBook Pro (M2 Pro)** — development + sim teleop. No CUDA.
- **Robot PC** (`robotdisco` = `192.168.11.80`; since 2026-10-10 an **NVIDIA Jetson AGX Orin** 64 GB, replacing the Intel PC) — hardware teleop/recording (arm + hand + cameras + SpaceMouse/gamepad) **and the inference server**: the `vla_infer_server` supervisor unit (`services/rd/`) serves `outputs/serve_ckpt` on `127.0.0.1:8777` and `eval_hw.py --remote 127.0.0.1:8777` runs on the same box, so nothing real-time crosses the network. Its 64 GB eMMC is the system disk; since 2026-10-26 `~/rd_ws` is a **symlink to `/mnt/data/rd_ws` on a 512 GB NVMe** (ext4, label `data`, fstab `nofail`; `/etc/systemd/system/supervisor.service.d/nvme-data.conf` orders supervisor after the mount), so every `/home/rd/...` path in these pages, the supervisor confs and the rsync deploys is unchanged by the drive. Since 2026-11-02 VLA lives inside it as **`~/rd_ws/VLA`** (this repo is rd_ws's submodule); there is no `~/VLA`.
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
rsync -a ~/VLA/outputs/train/<run>/checkpoints/<step>/ rd@192.168.11.80:rd_ws/VLA/outputs/train/<run>/checkpoints/<step>/
# robot: repoint the serve symlink and (re)start the unit — reactivity flags live in services/rd/vla_infer_server.conf
ln -sfn ~/rd_ws/VLA/outputs/train/<run>/checkpoints/<step>/pretrained_model ~/rd_ws/VLA/outputs/serve_ckpt
sudo supervisorctl restart vla_infer_server
# robot: eval against loopback
~/rd_ws/VLA/run.sh scripts/eval_hw.py --remote 127.0.0.1:8777 --cameras udp --fps 12 --video
```
Supervisor units on RD (`services/rd/`, installed by `scripts/install_services.sh rd`):

| unit | autostart | what |
|---|---|---|
| `vla_cameras` | yes | the camera service (`scripts/camera_server.py`): owns the three USB cameras, multicasts frames |
| `vla_camera_ros_bridge` | yes | the same frames as ROS 2 `CompressedImage` topics (`/vla/cameras/<cam>/image/compressed`) |
| `vla_webui` | yes | the web manager on the robot, `:8081`, pointed at the robot's own teleop page and route editor |
| `vla_infer_server` | manual | the inference server on `127.0.0.1:8777` (start it before driving) |
| `vla_teleop` | manual | the recording panel, arguments in `~/rd_ws/VLA/config/teleop.env` (`scripts/service_teleop.sh`) |
| `vla_eval` | manual | the policy eval, arguments in `config/eval.env` (`scripts/service_eval.sh`) |

Start and stop them from the manager's **Services** tab ([Web Manager](Web-Manager#services)), the supervisor
web UI at `http://192.168.11.80:9999/`, or `sudo supervisorctl start|stop <unit>` on RD. The route editor
(`rd_ws/Scripts/route_editor.py`, `:8092`) and the Folks page (`:8096`) are rd_ws units, embedded by the
manager's Map / Routes / Landmarks and Folks tabs.
A SmolVLA chunk costs ~0.8 s on the Orin (vs ~0.12 s on the 4090), covered by `--n-action-steps 24
--prefetch --prefetch-at 21 --rtc` in the unit. Serving from the GPU box (`--remote <GPU_IP>:8777`) still
works as a fallback, but the GPU box reaches the robot subnet over Wi-Fi only (its wired NIC is dead) and
real-time actions over that link stutter.

## Robot PC (`.80`): `run.sh`
Every robot-PC command goes through **`~/rd_ws/VLA/run.sh`**, which `cd`s to `~/rd_ws/VLA`, sets `PYTHONPATH=.`,
`PYTHONNOUSERSITE=1` (isolates from a conflicting `~/.local` ML stack), and `HF_HUB_OFFLINE=1` (the
subnet has no internet — stops lerobot/HF from hanging on DNS), and uses the project venv. The file is
**not in this repo**: `rd_ws/provision/provision.sh` (phase `vla`) writes it next to the venv it builds.
```bash
~/rd_ws/VLA/run.sh -m ur5e_lerobot.teleop.manual_panel --engine hardware --cameras udp
```
`--cameras udp` is not a default (the panel and `eval_hw.py` default to `local`, which would open the
cameras the always-on `vla_cameras` unit already owns); only the `vla_teleop` / `vla_eval` wrappers pass
it for you, which is why the units are the normal way to run both.
