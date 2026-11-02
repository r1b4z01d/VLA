# Sync & Repo Layout

## Sync
VLA is the **`rd_ws/VLA` git submodule** (since 2026-11-02): the Mac's `rd_ws/VLA` checkout is the source of
truth and rd_ws pins its commit. Neither machine pulls it with git:
- **RD** (no internet): the tree lives at `~/rd_ws/VLA` (really `/mnt/data/rd_ws/VLA`, through the `~/rd_ws`
  link) and is pushed from rd_ws with `provision/deploy_to_rd.sh --vla` (rsync, never `--delete`; the
  robot's `.venv`, `outputs/`, `log/` and `config/*.env` are left alone; it reinstalls `services/rd/*.conf`
  afterwards). Every robot-side path is `~/rd_ws/VLA/...`; there is no `~/VLA` any more
  (`provision_jetson.sh vlamove` did the move, `vlacheck` audits for leftovers).
- **GPU box**: `~/VLA` stays a plain rsync target (its units say `/home/bryan/VLA`), pushed from `rd_ws/VLA`:
```bash
rsync -a --exclude='/outputs/' --exclude='.venv*' --exclude='.git/' --exclude='__pycache__/' --exclude='/log/' \
      --exclude='/ros2_ws/build/' --exclude='/ros2_ws/install/' --exclude='/ros2_ws/log/' ./ GPU:~/VLA/
```
Both copies silently reverted once when an older tree was pushed over them — md5-compare before an eval if in
doubt. The GPU box pulls the robot's data with `rd_ws/provision/gpu/provision_gpu.sh data` (`RD:rd_ws/VLA/outputs/`).

Saved episodes also **auto-upload** from the robot PC to the GPU (see
[Hardware Teleop & Recording](Hardware-Teleop-and-Recording#save-home-rating-upload)).

## Layout
- `scripts/` —
  - deploy: `eval_hw.py` (run ACT|SmolVLA on the robot, local or `--remote`, interactive or `--auto`, stages, run registry, browser page); `infer_server.py` (the inference server — runs **on the robot's Jetson** as the `vla_infer_server` unit; the GPU box is only a fallback); `vla_action.py` (one policy run as a command: preflight, switch the server, `eval_hw --auto`; what the route runner and the voice assistant call); `pick_place_loop.py` (grab → place → repeat via stages); `service_teleop.sh` / `service_eval.sh` (the `vla_teleop` / `vla_eval` unit wrappers: free port 8090, preflight, `run.sh`); `install_services.sh [rd]`; `hw_preflight.py`.
  - cameras: `camera_server.py` / `camera_viewer.py` / `camera_ros_bridge.py` (the camera service, its stats/crop tool, the ROS CompressedImage bridge); `tune_exposure.py` (sweep exposure → clip %); `calibrate_workspace.py` (freedrive workspace limits → `outputs/workspace_calib.json`); `capture_backfill.py` (write `meta/capture.json` for older datasets); `ur_event_log.py` (UR controller event log).
  - training: `train_regrasp.py` (fine-tune with re-grasp oversampling, used by the manager's Train button); `train_nostate.py`, `make_delta_dataset.py`, `run_delta_pipeline.sh` (experiments); `debug_eval.py` (replay a dataset through a checkpoint, the gate before a hardware eval); `eval_policy.py` (sim rollouts).
  - datasets: `merge_datasets.py`, `recover_merge_datasets.py`, `split_episode.py`, `to_video.py`, `plot_workspace_calib.py`.
  - sim / hand: `build_disco_sim.py` (the mobile-base MuJoCo scene from `rd_ws/src/robot_disco/description`), `characterize_hand.py`, `grasp_search.py`, `grip_aid_test.py`; `setup_*.sh` (dev environments).
- `services/` — supervisor units. GPU box: `vla_webui.conf` (autostart) and `vla_infer_server.conf` (manual, fallback). Robot, `rd/`: `vla_cameras`, `vla_camera_ros_bridge`, `vla_webui` (autostart), `vla_infer_server`, `vla_teleop`, `vla_eval` (manual) plus `teleop.env.example` / `eval.env.example`, copied to `config/*.env` on first install; `scripts/install_services.sh [rd]` installs them.
- `ur5e_lerobot/` — integration package.
  - `remote.py` — remote-inference bridge (framed-pickle TCP): `RemotePolicyClient` + `serve()`, so SmolVLA runs in a separate server process (on the robot's Jetson since 2026-10-10, over loopback; the GPU box over the LAN is the fallback) and streams actions to the eval client.
  - `schema.py` — action/obs contract: **v1 10-D / 16-D state** (live) + **v2 14-D / 20-D** (8-DOF hand, opt-in).
  - `hand/amazing_hand_client.py` — UDP (`.117:8766`, default) / TCP (`:8765`) client for the AmazingHand ESP32: curls + flex/abduct + `read_state`.
  - `services.py` — the preflight every eval/panel launch runs (stops the other unit, makes sure the server answers) and the `status` / `serve` / `info` / `teleop-preflight` CLI.
  - `robot/` — LeRobot adapter `URAmazingHand` (`ur_amazinghand_robot.py`, `arm_interface.py`, `config.py`); `rtde_arm.py` (50 Hz servoJ streaming thread with one controller IK per hand-over, no-go/max_step/max_rot_step/z-floor clamps, freedrive, reconnect); `workspace.py` (no-go zone + the calibration file); `ur_power.py` (dashboard power up/down, protective-stop clear, safety restart); `ur_events.py` (passive primary-interface listener decoding C-codes); `ur_kin.py`.
  - `sensors/cameras.py` — three USB cameras by stable USB-port `by-path`, per-camera rotation, gain and **square crop** (`SCENE_SQUARE` / `SIDE_SQUARE`), exposure session; `camera_stream.py` — the camera service (multicast publisher, control port, MJPEG, subscriber + proxies, restart-and-retry); `capture_meta.py` — `meta/capture.json` (crop geometry + recording dates carried dataset → checkpoint → eval).
  - `ui/webpanel.py` — the browser page server embedded in the panel and the eval (stdlib: static, SSE state/log, POST commands, MJPEG for own frames).
  - `hf_cache.py` — prunes the HF `datasets` Arrow cache that every save / training run leaves behind.
  - `sim/` — MuJoCo UR5e + AmazingHand, mink IK, grasp-aid weld, record/playback (`--engine mujoco|kinematic`).
  - `teleop/manual_panel.py` — the recording app: control loop (Tk-scheduled, 12 Hz), browser page by default (`--ui web`, Tk window behind `--ui tk`), `--input spacemouse|gamepad|freedrive|sliders`, async episode save + home + rating + background upload, dataset switch, Stream Deck, `--engine hardware|mujoco|kinematic`.
  - `teleop/spacemouse.py` · `gamepad.py` (evdev) · `streamdeck.py` (the deck wrapper) · `sm_calibrate.py` · `gp_calibrate.py` — input readers + one-shot axis calibrators.
  - `sim/deploy.py` — run a checkpoint in the MuJoCo scene.
- `webui/` — the manager (FastAPI + static vanilla-JS) incl. `static/robot/` (the robot page's HTML/CSS/JS, served by the robot apps): `datasets.py` (fast metadata listing), `ops.py` (delete/trim/merge/move/to-video wrapping `dataset_tools`), `annotations.py` (rating/notes/operator sidecar), `models.py` / `evals.py` (training runs + eval records), `viewer.py` (rerun web-viewer subprocess), `server.py` (REST + static; it also relays the route editor and its landmarks API, the robot's supervisor (Services tab), the hand's UDP protocol (Hand tab) and the Folks page).
- `docs/` — decisions, data schema, camera setup, **`amazinghand_esp32_buttons.md`** (8-DOF firmware/protocol spec), `smolvla.md` (fine-tune recipe).
- `assets/amazing_hand/` — vendored MuJoCo hand model. UR5e model from `robot_descriptions`.
- `ros2_ws/` — UR5e ROS 2 bringup (deployment path; build on the Linux box).
