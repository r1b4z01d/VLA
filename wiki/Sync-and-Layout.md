# Sync & Repo Layout

## Sync
The Mac repo is the source of truth; `~/VLA` on the GPU box and on RD are **rsync targets, not git
checkouts** (they silently reverted once when an older tree was pushed over them — md5-compare before an
eval if in doubt). Workstation / Mac (have rsync):
```bash
rsync -az --exclude=__pycache__ --exclude='._*' ur5e_lerobot/ scripts/ assets/ gpu:~/VLA/
```
If a machine has no rsync, use **tar-over-ssh**:
```bash
tar czf - --exclude=__pycache__ --exclude='._*' ur5e_lerobot scripts | ssh rd@192.168.11.80 'tar xzf - -C ~/VLA'
```
Saved episodes also **auto-upload** from the robot PC to the GPU (see
[Hardware Teleop & Recording](Hardware-Teleop-and-Recording#auto-upload--stream-deck-star-rating)).

## Layout
- `scripts/` — `eval_hw.py` (deploy ACT|SmolVLA, local or `--remote`, run registry, browser page); `infer_server.py` (remote-inference server for SmolVLA — runs **on the robot's Jetson** as the `vla_infer_server` unit; the GPU box is only a fallback); `camera_server.py` / `camera_viewer.py` / `camera_ros_bridge.py` (the camera service, its live crop/exposure tool, ROS CompressedImage bridge); `calibrate_workspace.py` (freedrive workspace limits → `outputs/workspace_calib.json`); `ur_event_log.py` (UR controller event log); `train_regrasp.py` (fine-tune with re-grasp oversampling, used by the manager's Train button); `hw_preflight.py`; merge / sim-grasp diagnostics.
- `services/` — supervisor units: `vla_infer_server.conf` (GPU box, fallback), `rd/` (`vla_cameras`, `vla_camera_ros_bridge`, `vla_infer_server` on the robot); `scripts/install_services.sh [rd]` installs them.
- `ur5e_lerobot/` — integration package.
  - `remote.py` — remote-inference bridge (framed-pickle TCP): `RemotePolicyClient` + `serve()`, so SmolVLA runs in a separate server process (on the robot's Jetson since 2026-10-10, over loopback; the GPU box over the LAN is the fallback) and streams actions to the eval client.
  - `schema.py` — action/obs contract: **v1 10-D / 16-D state** (live) + **v2 14-D / 20-D** (8-DOF hand, opt-in).
  - `hand/amazing_hand_client.py` — UDP (`.117:8766`, default) / TCP (`:8765`) client for the AmazingHand ESP32: curls + flex/abduct + `read_state`.
  - `robot/` — LeRobot adapter `URAmazingHand`; `rtde_arm.py` (50 Hz servoJ streaming thread with one controller IK per hand-over, no-go/max_step/max_rot_step/z-floor clamps, freedrive, reconnect); `workspace.py` (no-go zone + the calibration file); `ur_power.py` (dashboard power up/down, protective-stop clear, safety restart); `ur_events.py` (passive primary-interface listener decoding C-codes); `ur_kin.py`.
  - `sensors/cameras.py` — three USB cameras by stable USB-port `by-path`, per-camera rotation, gain and **square crop** (`SCENE_SQUARE` / `SIDE_SQUARE`), exposure session; `camera_stream.py` — the camera service (multicast publisher, control port, MJPEG, subscriber + proxies, restart-and-retry); `capture_meta.py` — `meta/capture.json` (crop geometry + recording dates carried dataset → checkpoint → eval).
  - `ui/webpanel.py` — the browser page server embedded in the panel and the eval (stdlib: static, SSE state/log, POST commands, MJPEG for own frames).
  - `hf_cache.py` — prunes the HF `datasets` Arrow cache that every save / training run leaves behind.
  - `sim/` — MuJoCo UR5e + AmazingHand, mink IK, grasp-aid weld, record/playback (`--engine mujoco|kinematic`).
  - `teleop/manual_panel.py` — the recording app: control loop (Tk-scheduled, 12 Hz), browser page by default (`--ui web`, Tk window behind `--ui tk`), `--input spacemouse|gamepad|freedrive|sliders`, async episode save + home + rating + background upload, dataset switch, Stream Deck, `--engine hardware|mujoco|kinematic`.
  - `teleop/spacemouse.py` · `gamepad.py` (evdev) · `sm_calibrate.py` · `gp_calibrate.py` — input readers + one-shot axis calibrators.
- `webui/` — the manager (FastAPI + static vanilla-JS) incl. `static/robot/` (the robot page's HTML/CSS/JS, served by the robot apps): `datasets.py` (fast metadata listing), `ops.py` (delete/trim/merge/move/to-video wrapping `dataset_tools`), `annotations.py` (rating/notes/operator sidecar), `models.py` / `evals.py` (training runs + eval records), `viewer.py` (rerun web-viewer subprocess), `server.py` (REST + static).
- `docs/` — decisions, data schema, camera setup, **`amazinghand_esp32_buttons.md`** (8-DOF firmware/protocol spec), `smolvla.md` (fine-tune recipe).
- `assets/amazing_hand/` — vendored MuJoCo hand model. UR5e model from `robot_descriptions`.
- `ros2_ws/` — UR5e ROS 2 bringup (deployment path; build on the Linux box).
