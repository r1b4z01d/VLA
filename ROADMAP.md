# UR5e + AmazingHand VLA — Roadmap

State on 2026-11-02. The robot ("Robot Disco") is a mobile base with a UR5e, a Pollen AmazingHand and
three webcams, run from a Jetson AGX Orin; policies are LeRobot SmolVLA fine-tunes trained on a 4090
box. This repo is the `rd_ws/VLA` submodule of the ROS 2 workspace. The original 2026-07 plan is at the
end for reference.

## Built
- **Hardware recording** — the teleop panel (`ur5e_lerobot/teleop/manual_panel.py`): SpaceMouse /
  gamepad / freedrive input, RTDE servoJ streaming at 50 Hz with workspace clamps, UDP hand, a browser
  page on :8090 (the manager's Teleop tab), Stream Deck, exposure lock, re-grasp marks, 1-5★ rating,
  auto-upload to the GPU box, run as the `vla_teleop` unit from `config/teleop.env`.
- **The camera service** — one process owns the three USB webcams (by USB port), multicasts JPEG,
  serves MJPEG and a control port; the square crop travels with the dataset (`meta/capture.json`) into
  the checkpoint, the server and the eval.
- **Datasets** — V1 schema (10-D action / 16-D state / three 540 px images at 12 fps, video storage),
  annotations sidecar, merge / trim / move / rename tools in the manager.
- **Training** — SmolVLA fine-tune from `lerobot/smolvla_base` with re-grasp oversampling
  (`scripts/train_regrasp.py`, adaptive weight), launched from the manager's Train button under a
  memory cap; ACT for comparison; offline vision-use gate (`debug_eval.py`); delta / no-state variants.
- **Inference on the robot** — `infer_server.py` as the `vla_infer_server` unit on the Orin (loopback
  :8777) with prefetch / RTC / `--n-action-steps 24` and resident models (`--also`), the framed-pickle
  bridge (`remote.py`), `{"cmd":"info"}` reporting checkpoint + crop; `ur5e_lerobot.services`
  (`status` / `serve` / `info` / preflights).
- **Eval** — `eval_hw.py`: interactive (Stream Deck / page / keyboard) or `--auto` with stillness and
  grasp-level end conditions, stages, rollable runs with ratings, the run registry
  (`outputs/evals/<run>/eval.json`), rollout video, upload to the GPU box; run as the `vla_eval` unit
  from `config/eval.env`.
- **Unattended arm tasks** — `vla_action.py` (preflight, serve, eval `--auto`, home) and
  `pick_place_loop.py`; called by the rd_ws route runner's `vla:` post-actions, by landmark tasks from
  the manager, and by the robot's voice assistant ("Robot Disco, pick up tigger").
- **The web manager** — ten tabs: Map (3D + 2D, live or simulated), Routes, Landmarks (+ tasks),
  Folks, Services (supervisor control, ordered base restart), VLA Teleop, Hand, Datasets,
  Models, Evals. Runs on the GPU box, the robot and a laptop (`rd_ws/Scripts/dev_stack.sh`).
- **Simulation** — MuJoCo UR5e + AmazingHand with mink IK and a grasp-aid weld, used by the laptop
  stack and for sim demos; the route editor's `--sim` fabricates the whole mobile robot.
- **Machines and provisioning** — Jetson (JetPack 6, NVMe data drive), GPU box, both rebuilt from
  `rd_ws/provision/`; supervisor units in `services/` and `services/rd/`.

## In progress
- Higher-variety datasets for the current tasks (`docs/high_variety_recording.md`) and multi-stage
  routes (grab at the bookshelf → place in the banana bed) run end to end from a landmark task.
- Screenshots of the robot-only pages (Folks, Services, live Routes, eval page) in the docs.

## Next
- The **8-DOF hand** (`docs/amazinghand_esp32_buttons.md`): button firmware with `F:` / `S:` lines, the
  panel's hand-buttons input source, V2 datasets (14-D / 20-D).
- Base velocities in the action space, if a task needs arm and base together.
- Force/torque in the observation for contact-rich tasks.
- Merge in the manager carrying `capture.json`; trim shifting `regrasp_ranges`.

## Original plan (2026-07, for the record)
Start in simulation (Gazebo for control parity, Isaac Lab-Arena for vision), bridge LeRobot to the UR5e
through ROS 2 / MoveIt Servo, teleop with a SpaceMouse and a Manus glove, hand over a TCP service,
RealSense as the scene camera, ACT first then SmolVLA, inference on the NVIDIA workstation. What
survived: LeRobot, the SpaceMouse, MuJoCo for the hand and arm sim, the `J:` line protocol (now UDP),
ACT → SmolVLA. What changed: RTDE instead of ROS for the arm, three webcams instead of the RealSense, a
camera service, inference on the Jetson, no glove, no Gazebo / Isaac. Phase 0's PushT smoke recipe on
the Mac still works:
```
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/lerobot-train \
  --dataset.repo_id=lerobot/pusht --dataset.video_backend=pyav \
  --policy.type=diffusion --policy.device=mps --env.type=pusht \
  --steps=100 --batch_size=8 --eval.n_episodes=2 --wandb.enable=false \
  --output_dir=outputs/train/pusht_smoke
```

## Sources
- [LeRobot](https://github.com/huggingface/lerobot) · [SmolVLA](https://huggingface.co/lerobot/smolvla_base)
- [AmazingHand](https://github.com/pollen-robotics/AmazingHand) (our fork: [r1b4z01d/AmazingHand](https://github.com/r1b4z01d/AmazingHand))
- [ur_rtde](https://sdurobotics.gitlab.io/ur_rtde/)
