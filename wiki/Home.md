# VLA on UR5e with LeRobot

Experimenting with Vision-Language-Action policies on a Universal Robots UR5e (with an AmazingHand
end-effector), using [LeRobot](https://github.com/huggingface/lerobot). The full loop runs on hardware:
teleop recording on the robot PC (a Jetson Orin) through a browser page, SmolVLA fine-tuning on the
GPU box from the web manager, and evaluation with the inference server running on the robot itself.
RTDE drives the arm; ROS 2 is reserved for the mobile base and autonomous deployment.

See **[ROADMAP.md](https://github.com/r1b4z01d/VLA/blob/main/ROADMAP.md)** for the full plan and
**[docs/decisions.md](https://github.com/r1b4z01d/VLA/blob/main/docs/decisions.md)** for open
architectural choices.

## Contents
- **[Machines & Network](Machines-and-Network)** — the three machines, the isolated robot subnet, and `run.sh` on the robot PC.
- **[Hardware Teleop & Recording](Hardware-Teleop-and-Recording)** — the recording app and its browser page, the camera service and crop, workspace calibration, Stream Deck, save/home/rating/upload, UR faults.
- **[Recording Sim Demos](Recording-Sim-Demos)** — MuJoCo teleop capture and the sim grasp aid.
- **[Training & Eval](Training-and-Eval)** — train ACT / fine-tune SmolVLA, sim eval, playback, merge/recover.
- **[Web Manager (RobotDisco)](Web-Manager)** — the browser console: live robot page · datasets · models (Train button) · evals grouped by model.
- **[Deploy on Hardware](Deploy-on-Hardware)** — run a trained policy on the real UR5e (`eval_hw.py`) with the robot-side inference server; deck / page / keyboard driven, run registry.
- **[8-DOF Hand](8-DOF-Hand)** — per-finger flex + abduct (in progress).
- **[Sync & Layout](Sync-and-Layout)** — moving code between machines and the repo map.
