# Decisions

The ADR log, rewritten on 2026-11-02 to say what was actually decided. The original 2026-07 plan is
summarised in `../ROADMAP.md`.

## D1 — Arm control: RTDE, not ROS / MoveIt  ✅
`robot/rtde_arm.py` talks to the UR5e over `ur_rtde` directly: a **servoJ streaming thread at 50 Hz**
(20 ms servo time) with one controller IK solution per hand-over, joint-space interpolation, the no-go
clamp (TCP and elbow/wrist links via FK), per-step `max_step` / `max_rot_step`, the z-floor from the
workspace calibration, freedrive and reconnect. `ur_power.py` drives the dashboard (power up, clear
protective stop, restart safety, park on exit). ROS keeps the mobile base, nav2 and the face; the ROS
UR driver (`robot_disco_ur5e_driver`) is stopped by the preflight whenever the panel or the eval needs
the arm. Why: MoveIt Servo over a shared rclpy + lerobot interpreter (the old D5) was never needed, and
RTDE gave the stutter-free stream the 83 ms servoL windows could not.

## D2 — Teleop input: SpaceMouse / gamepad / freedrive, own readers  ✅
`teleop/spacemouse.py` (default), `teleop/gamepad.py` (Xbox, evdev) and freedrive (hand-guide the arm,
grasp from keys); sliders in the sim. No Manus glove, no `lerobot-teleoperator-spacemouse`. The hand is
pinch or full grasp from a button (`--grasp`), per-finger from the Stream Deck.

## D3 — Hand transport: UDP line protocol to the ESP32  ✅
The AmazingHand's ESP32 (Waveshare servo driver) hosts the server itself: `J:a0..a7,speed\n` as **one
UDP datagram per setpoint to 192.168.11.117:8766**, a `Q` probe answered by `K:<uptime ms>`, firmware
gestures `0-9 a b`, speed 50-2000. TCP :8765 remains as the legacy transport (`--hand-transport tcp`).
Why UDP (2026-10-24): the hand Wi-Fi loses 12-25 % of packets and TCP replayed stale setpoints
seconds late; a lost datagram is simply superseded by the next. Powered from the UR tool port (12 V).

## D4 — Cameras: three webcams by USB port, one camera service  ✅
Three identical USB webcams (scene, side, wrist) addressed by `by-path`, owned by one process
(`camera_server.py`, multicast + control + MJPEG) that every consumer subscribes to. The RealSense is
ROS-only. Details and the Jetson port map: `camera_setup.md`.

## D5 — The square crop travels with the data  ✅
Scene and side are square-cropped portrait frames; the crop is part of the observation geometry, so it
is written to `meta/capture.json` at record time, copied into the training run, reported by the
inference server and applied by the eval. Merged datasets need `capture_backfill.py`.

## D6 — Action / observation space  ✅
V1: 10-D absolute EE pose + 4 curls, 16-D state, three images at 540×540 / 540×960, 12 fps
(`data_schema.md`). V2 (14-D / 20-D, flex + abduct) exists in `schema.py` for the 8-DOF hand and is
opt-in. Delta actions and no-state variants were tried offline (`make_delta_dataset.py`,
`train_nostate.py`) and did not fix a low-variety dataset; object variety did (`high_variety_recording.md`).

## D7 — Policy: SmolVLA fine-tune, not ACT from scratch  ✅
ACT from scratch diverged on 23 demos. SmolVLA fine-tunes from `lerobot/smolvla_base` (lerobot 0.4.4 on
both boxes), is language-conditioned and sample-efficient. The manager's Train button runs
`scripts/train_regrasp.py` (re-grasp segments oversampled with an adaptive weight); ACT stays available
for comparison. `smolvla.md` has the recipe.

## D8 — Inference on the robot, behind a server  ✅
The policy runs as a separate process behind a framed-pickle TCP bridge (`remote.py`,
`infer_server.py`) and the eval is the client. Since 2026-10-10 the server runs **on the robot's Jetson
Orin** (`vla_infer_server`, loopback :8777, `--n-action-steps 24 --prefetch --prefetch-at 21 --rtc`,
resident models via `--also`); the GPU box only trains and hosts the master manager. Why: the Orin
makes the robot self-contained and the GPU box only reaches the robot segment over Wi-Fi.

## D9 — Scope: fixed base first, then landmark tasks  ✅
The first policies treat the base as fixed. The base now drives *between* arm tasks: `path.py` routes
with `vla:` post-actions, landmark tasks from the manager and the voice assistant ("pick up tigger" =
navigate to the landmark, then `vla_action.py`). The base is still not part of the action space.

## D10 — Machines  ✅
Mac = source of truth and development (`rd_ws/VLA`, the submodule since 2026-11-02); GPU box
(4090TI, Ubuntu 24.04) = training + master manager + the voice assistant's LLM; robot PC = Jetson AGX
Orin (JetPack 6, Ubuntu 22.04, ROS 2 Humble) running teleop, eval, the camera service, the inference
server and the ROS stack. See `../wiki/Machines-and-Network.md`.

## Open
- Hand-buttons input source for the 8-DOF hand (firmware `F:`/`S:` lines are specified, the panel source
  is not written).
- Base velocities in the action space (a 12-D action), if a task ever needs arm + base together.
- Force/torque in the observation.
