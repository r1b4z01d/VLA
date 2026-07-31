# UR5e + VLA + LeRobot — Project Roadmap

Goal: experiment with Vision-Language-Action (VLA) policies on a **UR5e**, using
**LeRobot** as the learning framework, starting in **simulation** before touching
real hardware.

## Hardware / compute on hand
- **UR5e** (6-DOF industrial arm, controlled over RTDE).
- **AmazingHand** end-effector — Pollen Robotics open-source hand. 8-DOF (4 fingers
  × 2), driven by 8× **Feetech SCS0009** serial-bus servos. ~400 g, 3D-printed.
  Repo: https://github.com/pollen-robotics/AmazingHand
- **Intel RealSense** camera(s) + **USB webcam(s)**.
- **Teleop:** 3Dconnexion **SpaceMouse** (6-DOF → arm) + **Manus glove** (finger
  tracking → AmazingHand). See teleop mapping below.
- **NVIDIA Linux workstation** — the workhorse (ROS 2, sim, GPU training/inference).
- **MacBook Pro M2 Pro / 16 GB** — development + light LeRobot exploration only
  (no CUDA; not for training VLAs).

## The central reality
LeRobot (v0.5.0) does **not** support the UR5e out of the box. Its supported robots
are SO-100, LeKiwi, Koch, HopeJR, OMX, EarthRover, Reachy2, Unitree G1. Our core
engineering task is a **custom LeRobot `Robot` adapter** that bridges to the UR5e
(via ROS 2) so LeRobot's record → train → eval pipeline works unchanged.

LeRobot policies available to us: ACT, Diffusion Policy, **π0 / π0-FAST**, **SmolVLA**
(the VLAs), plus Real-Time Chunking for responsive inference.

## Two simulators (the key fork)
| | Gazebo (`ur_simulation_gz`) | Isaac Lab-Arena |
|---|---|---|
| Control parity w/ real UR5e | ✅ same `ros2_control` interfaces | ⚠️ needs UR5e embodiment added |
| Vision realism (for VLAs) | ⚠️ basic | ✅ RTX photoreal |
| LeRobot integration | via our bridge | ✅ native (EnvHub) |
| Best for | motion / control / bridge validation | VLA training & eval at scale |

Plan: **Gazebo for the control + bridge foundation; Isaac for the VLA/vision phase.**

---

## Phases

### Phase 0 — Learn LeRobot (no UR, no ROS) — *runs on the Mac* ✅ pipeline verified
Install LeRobot, run the train → eval loop on a built-in gym env (PushT). Goal:
internalize the dataset format, config system, and CLIs. → `scripts/setup_mac_dev.sh`.
Working smoke recipe (Apple Silicon):
```
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/lerobot-train \
  --dataset.repo_id=lerobot/pusht --dataset.video_backend=pyav \
  --policy.type=diffusion --policy.device=mps --env.type=pusht \
  --steps=100 --batch_size=8 --eval.n_episodes=2 --wandb.enable=false \
  --output_dir=outputs/train/pusht_smoke
```
Gotcha: the default `torchcodec` video backend needs system FFmpeg (absent here) —
use `--dataset.video_backend=pyav` (bundles its own FFmpeg). Real training → Linux box.

### Phase 1 — UR5e digital twin in ROS 2 — *Linux box*
Install ROS 2, `Universal_Robots_ROS2_Driver` + `ur_simulation_gz`. Drive a simulated
UR5e via `ros2_control` (+ optionally MoveIt). Optionally run **URSim** for controller
realism. → `scripts/setup_linux_stack.sh`

### Phase 2 — LeRobot ↔ ROS 2 bridge (core integration)
Write a custom LeRobot `Robot` (an `rclpy` node) that:
- subscribes to `/joint_states` + camera topics,
- publishes joint commands to the UR controllers,
- (later) commands the AmazingHand over its Feetech bus.
Validate by teleoperating the sim UR5e and recording a LeRobot dataset.
**Teleop in:** SpaceMouse via the `lerobot-teleoperator-spacemouse` package (emits EE
targets) and/or **MoveIt Servo** (twist → real-time IK). Study existing bridge
[`lerobot-ros`](https://github.com/ycheng517/lerobot-ros) before building from scratch.
**Gotcha:** `rclpy` lives in the ROS 2 (apt) Python env; LeRobot installs via pip.
They must share one interpreter — use a venv with `--system-site-packages` or install
LeRobot into the ROS Python. See `docs/decisions.md`.

### Phase 3 — VLA in sim (Isaac Lab-Arena)
Add the UR5e embodiment to Isaac Lab-Arena (import UR URDF), generate
LeRobot-compatible datasets, fine-tune/eval **SmolVLA** or **π0** with RTX rendering.
Glove-based dexterous data collection here can use **NVIDIA Isaac Teleop** (Manus is
its official data glove) — the retargeting pipeline is shared sim↔real.

### Phase 4 — AmazingHand integration
Don't build from scratch — a working stack exists in
[r1b4z01d/AmazingHand](https://github.com/r1b4z01d/AmazingHand/tree/main/Demo) (our fork; vendored
in rd_ws as the `AmazingHand-main` submodule)
(Dora dataflow): `MediaPipe → MuJoCo+Mink IK retarget → AHControl (Rust, rustypot/serial)`.
Reuse from it:
- the **MuJoCo model** of the hand (AHSimulation) for our sim,
- the **retargeting** (camera→joint targets); Manus glove is the alternative input via
  Manus Core (ref [`Bidex_Manus_Teleop`](https://github.com/leap-hand/Bidex_Manus_Teleop)).
- **Logical interface = 8 joint targets** (4 fingers × 2). Expose it as a **TCP service**
  (e.g. `J:a1..a8,speed\n`) so it stays out of the rclpy+lerobot Python env (see D5).
Native transport is serial (`/dev/ttyACM0`); confirm the user's TCP endpoint (D4).
- ✅ **MuJoCo hand sim runs on the Mac** — `ur5e_lerobot/sim/amazing_hand_mujoco.py`
  drives the vendored model by 4 curls (open↔closed verified). Reusable for Phase 3.
- ✅ **Sim-backed recording on the Mac (no hardware)** — `SimArm` + `SimHand` plug into
  the `URAmazingHand` adapter; `ur5e_lerobot/sim/record_sim.py` writes a real
  LeRobotDataset (state[16] + scene image + action[10]), verified to reload. The
  scripted action source swaps for the **SpaceMouse teleop** once it's available.
- ✅ **Full MuJoCo UR5e + AmazingHand sim on the Mac** — `sim/mujoco_arm.py` (UR5e +
  mink IK, EE-pose→joints to 0.2 mm), `sim/combined_model.py` + `sim/mujoco_cell.py`
  (hand attached to the flange; `CellArm`/`CellHand` back the adapter). NOTE: only the
  *Gazebo/ROS 2* sim is Linux-bound — MuJoCo runs the UR5e fine on macOS.
- ✅ **Manual control panel** — `ur5e_lerobot/teleop/manual_panel.py` (Tkinter): sliders
  for EE pose + grasp, live arm+hand preview, Record/Save. `--engine mujoco` (real arm)
  or `kinematic` (fast). Collect sim demos NOW, no SpaceMouse.
- ✅ **`record_sim --engine mujoco`** records real arm+hand demos (state ≠ action;
  physical flange vs command) — verified to reload.
- ✅ **Physics grasp works with the AmazingHand** — combined scene has a graspable block;
  the hand picks it up and holds it aloft (~0.23 m, contact physics). Key: hand-mesh
  collision must be enabled in the spec BEFORE compile (MuJoCo builds the convex hull only
  then). Recipe: top-down over block (x≈0, y≈0.45), descend flange z≈0.12, close, lift.

### Phase 5 — Real UR5e
Swap the ROS 2 driver from sim → real robot (same interfaces). Collect real demos,
fine-tune on the workstation, deploy with async inference.

## Open decisions
Tracked in `docs/decisions.md` (ROS 2 distro, sim track, teleop method, hand control).

## Sources
- [LeRobot v0.5.0 release](https://huggingface.co/blog/lerobot-release-v050) ·
  [repo](https://github.com/huggingface/lerobot) · [π0 docs](https://huggingface.co/docs/lerobot/en/pi0)
- [AmazingHand](https://github.com/pollen-robotics/AmazingHand) ·
  [announcement](https://huggingface.co/blog/pollen-robotics/amazing-hand)
- [UR ROS 2 GZ Simulation](https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation) ·
  [UR ROS 2 Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver)
- [Isaac Lab-Arena + LeRobot](https://huggingface.co/docs/lerobot/envhub_isaaclab_arena) ·
  [NVIDIA blog](https://developer.nvidia.com/blog/simplify-generalist-robot-policy-evaluation-in-simulation-with-nvidia-isaac-lab-arena/)
