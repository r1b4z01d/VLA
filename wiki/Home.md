# VLA on UR5e with LeRobot

Vision-Language-Action policies on a Universal Robots UR5e (with an AmazingHand end-effector) on the
Robot Disco mobile base, using [LeRobot](https://github.com/huggingface/lerobot). The full loop runs on
hardware: teleop recording on the robot PC (a Jetson AGX Orin) through a browser page, SmolVLA
fine-tuning on the GPU box from the web manager, and evaluation with the inference server running on the
robot itself. RTDE drives the arm; ROS 2 (the parent repo `rd_ws`, of which this repo is the `VLA/`
submodule since 2026-11-02) drives the mobile base, and a route waypoint or a spoken landmark task runs a
policy where the robot stops.

See **[ROADMAP.md](https://github.com/r1b4z01d/VLA/blob/main/ROADMAP.md)** for what is built, in
progress and next, and **[docs/decisions.md](https://github.com/r1b4z01d/VLA/blob/main/docs/decisions.md)**
for the architectural choices as taken.

## Contents
- **[Machines & Network](Machines-and-Network)** — the three machines, the isolated robot subnet, where inference runs, the robot's supervisor units and `run.sh`.
- **[Hardware Teleop & Recording](Hardware-Teleop-and-Recording)** — the recording app and its browser page, the camera service and crop, workspace calibration, Stream Deck, save/home/rating/upload, UR faults.
- **[Recording Sim Demos](Recording-Sim-Demos)** — MuJoCo teleop capture and the sim grasp aid.
- **[Training & Eval](Training-and-Eval)** — fine-tune SmolVLA (what the Train button runs), train ACT, sim eval, playback, merge/recover.
- **[Web Manager (RobotDisco)](Web-Manager)** — the browser console, ten tabs: Map (3D/2D, live or simulated) · Routes · Landmarks (+ tasks) · Folks · Services · RobotDisco Teleop · Hand · Datasets · Models (Train button) · Evals grouped by model.
- **[Deploy on Hardware](Deploy-on-Hardware)** — run a trained policy on the real UR5e (`eval_hw.py`) with the robot-side inference server; deck / page / keyboard driven, unattended `--auto` runs and stages, `vla_action.py`, the route runner and landmark tasks, the run registry.
- **[8-DOF Hand](8-DOF-Hand)** — per-finger flex + abduct (in progress).
- **[Sync & Layout](Sync-and-Layout)** — how each machine gets the code (submodule, `deploy_to_rd.sh --vla`, rsync to the GPU box) and the repo map.
