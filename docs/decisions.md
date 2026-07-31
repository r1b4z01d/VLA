# Open decisions

Lightweight ADR log — resolve these as the project progresses.

## D1 — ROS 2 distribution  ✅ RESOLVED
**Jazzy / Ubuntu 24.04**, using new **Gazebo Harmonic** (Gz Sim, not Classic).
- UR sim available as apt binary: `ros-jazzy-ur-simulation-gz` (no source build needed).
- Driver metapackage: `ros-jazzy-ur`. Control bridge: `ros-jazzy-gz-ros2-control`.
- Jazzy gotcha: launch arg renamed `prefix` → `tf_prefix`.
See `scripts/setup_linux_stack.sh`.

## D2 — Sim track for VLAs
Gazebo (control parity) vs Isaac Lab-Arena (vision/VLA). Plan: Gazebo for Phase 1–2
foundation, Isaac for Phase 3 VLA work. **Status:** tentative — note that Isaac Teleop
+ Manus (official glove) for dexterous data collection tilts toward adopting Isaac
earlier than originally planned.

## D3 — Teleoperation / demo collection method  ✅ RESOLVED
**Arm:** 3Dconnexion **SpaceMouse** → 6-DOF EE control.
  - LeRobot path: `lerobot-teleoperator-spacemouse` (PyPI) — emits EE-target actions.
  - ROS 2 path: `spacenav` node → **MoveIt Servo** (twist → real-time IK servoing).
**Hand:** **Manus glove** → Manus Core ROS 2 package (built-in human→robot retargeting)
  → AmazingHand joint targets. In Isaac, use **Isaac Teleop** (Manus = official glove).
**Open sub-question:** start sim with scripted trajectories, or wire teleop into sim
from day one? (Leaning: get SpaceMouse→sim-arm working early; add glove in Phase 4.)

## D4 — AmazingHand control path
**Reference stack** ([r1b4z01d/AmazingHand](https://github.com/r1b4z01d/AmazingHand) `Demo/`, our fork — vendored in rd_ws as the `AmazingHand-main` submodule), a Dora dataflow:
`MediaPipe tracking → MuJoCo+Mink IK retarget (AHSimulation) → AHControl (Rust) → servos`.
- **Logical interface:** 8 servo joint targets (4 fingers × 2: index/middle/ring/thumb),
  per `FINGER_TO_SERVO`. A higher-level per-finger "curl" 0..1 maps to two servo angles
  (offsets −90..90°). Native control is **serial `/dev/ttyACM0` via `rustypot`** (Feetech).
- **Retargeting is already solved** two ways → both emit hand joint targets:
  (a) camera + MuJoCo/Mink IK (this repo), (b) Manus glove + Manus Core
  (ref `leap-hand/Bidex_Manus_Teleop`).
- **There is a MuJoCo model of the AmazingHand** (AHSimulation) — reuse for our sim phase.
- **Transport — CONFIRMED:** raw TCP to **192.168.1.194:8765**, ASCII line
  `J:a0..a7,<speed>\n` (8 per-servo deg offsets + int speed), TCP_NODELAY, ~20 Hz.
  Client implemented: `ur5e_lerobot/hand/amazing_hand_client.py` (open/close verified
  against the demo's offset tables). Keeps the hand out of the rclpy+lerobot env (D5).
**Still open:** (a) speed units/range (default 200 is a placeholder); (b) action-space
representation — expose 8 per-servo offsets or 4 per-finger curls to the policy?;
(c) what hosts the TCP server (ESP32 firmware vs a host bridging to rustypot/serial).
**Status:** transport resolved; representation OPEN — Phase 4.

## D5 — LeRobot/ROS Python environment
`rclpy` (apt) and `lerobot` (pip) must share one interpreter. Decide: venv with
`--system-site-packages`, or pip-install lerobot into the ROS Python, or a bridge
process boundary (e.g. ZMQ) to keep envs separate. **Status:** OPEN — Phase 2.

## D6 — Action / observation space  ✅ RESOLVED (schema in `data_schema.md`)
- **Arm:** end-effector pose, 6-DoF Cartesian — position (x,y,z, m) + rotation vector
  (rx,ry,rz, rad) in the UR base frame. Executed via MoveIt Servo / Cartesian control.
- **Hand:** 4 per-finger curls in [0,1] (index, middle, ring, thumb) → expanded to 8
  servo offsets by `AmazingHandClient.send_curls`.
- **Action = 10-D**: `[ee_x,ee_y,ee_z, ee_rx,ee_ry,ee_rz, curl_index,curl_middle,curl_ring,curl_thumb]`.
- **Observation:** images (scene/wrist/side per `camera_setup.md`) + `state` (16-D:
  ee_pose[6] + arm_joints[6] + hand_curls[4]).
- Code contract: `ur5e_lerobot/schema.py`.
**Sub-decisions (defaults chosen, easy to revisit):** absolute targets (not deltas);
rotation stored as axis-angle, convert to 6D at the model input; force/torque not logged
yet. **Status:** RESOLVED for scaffolding; revisit before large-scale recording.

## D7 — Camera placement
Full plan in `camera_setup.md`. Default: C1 scene (RealSense RGB-D) + C2 wrist
(webcam/D405) + C3 side (webcam), optional C4 top-down. Wrist cam defined in the URDF
for sim↔real parity. **Status:** plan drafted — needs exact camera inventory + models
and workspace/mount geometry to finalize the tier and C1/C2 assignment.

## D8 — Scope for v1 (first VLA): keep it simple
The robot is a mobile manipulator (diff-drive base + UR5e + AmazingHand, all ROS 2), but
**v1 treats the base as FIXED** — it is out of scope. Rationale: first VLA; minimize DOF
and moving parts to reach a working policy fast.
- v1 robot = UR5e (6-DoF EE pose) + AmazingHand (4 curls) + 1–2 cameras + one short task.
- Adding the base later is **additive**, not a rewrite: a `BaseInterface` → `/cmd_vel`
  (ros2 `diff_drive_controller`), action 10→12-D (`base.x.vel`, `base.theta.vel`), and
  base odometry into the observation. Precedent: LeKiwi (omni) / Mobile ALOHA (diff-drive).
- Policy path: start with **ACT** (simplest imitation learning), then swap to **SmolVLA**
  (the language-conditioned VLA) via `--policy.type` — a one-line change, same pipeline.
**Status:** v1 scope set — fixed base.
