# Sync & Repo Layout

## Sync
Workstation / Mac (have rsync):
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
- `scripts/` — setup, eval, merge, sim-grasp diagnostics; `hw_preflight.py` (staged hardware bring-up); `eval_hw.py` (deploy ACT|SmolVLA, local or `--remote`); `infer_server.py` (GPU-side remote-inference server for SmolVLA).
- `ur5e_lerobot/` — integration package.
  - `remote.py` — remote-inference bridge (framed-pickle TCP): `RemotePolicyClient` + `serve()`, so SmolVLA runs on the GPU box and streams actions to the robot PC.
  - `schema.py` — action/obs contract: **v1 10-D / 16-D state** (live) + **v2 14-D / 20-D** (8-DOF hand, opt-in).
  - `hand/amazing_hand_client.py` — TCP client for the AmazingHand ESP32 (`.117:8765`): curls + flex/abduct + `read_state`.
  - `robot/` — LeRobot adapter `URAmazingHand`; `rtde_arm.py` (RTDE teleop: servoL + no-go/max_step/max_rot_step/z-floor/IK guard + freedrive + reconnect); `workspace.py` (no-go zone incl. elbow/wrist-link FK check, body at +y); `arm_interface.py` `Ros2ArmInterface` (stub).
  - `sensors/cameras.py` — three USB cameras by stable USB-port `by-path` (same wide-FOV model), per-camera rotation + gain; MJPG 720p capture → 960×540 dataset size.
  - `sim/` — MuJoCo UR5e + AmazingHand, mink IK, grasp-aid weld, record/playback (`--engine mujoco|kinematic`).
  - `teleop/manual_panel.py` — Tkinter panel: input dropdown (spacemouse/gamepad/freedrive/sliders), Reconnect-UR, calibrator loaders, `--engine hardware`, auto-upload + star rating + auto-home-on-save, clean finalize.
  - `teleop/spacemouse.py` · `gamepad.py` (evdev) · `sm_calibrate.py` · `gp_calibrate.py` — input readers + one-shot axis calibrators.
- `webui/` — dataset-management web UI (FastAPI + static vanilla-JS): `datasets.py` (fast metadata listing), `ops.py` (delete/trim/merge/move/to-video wrapping `dataset_tools`), `annotations.py` (rating/notes/operator sidecar), `models.py` / `evals.py` (training runs + eval records), `viewer.py` (rerun web-viewer subprocess), `server.py` (REST + static).
- `docs/` — decisions, data schema, camera setup, **`amazinghand_esp32_buttons.md`** (8-DOF firmware/protocol spec), `smolvla.md` (fine-tune recipe).
- `assets/amazing_hand/` — vendored MuJoCo hand model. UR5e model from `robot_descriptions`.
- `ros2_ws/` — UR5e ROS 2 bringup (deployment path; build on the Linux box).
