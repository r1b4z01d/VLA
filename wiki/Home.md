# VLA on UR5e with LeRobot

Experimenting with Vision-Language-Action policies on a Universal Robots UR5e (with an AmazingHand
end-effector), using [LeRobot](https://github.com/huggingface/lerobot). Simulation-first;
**real-hardware teleop + recording works** (datasets captured on the robot). RTDE drives the arm for
teleop; ROS 2 is reserved for autonomous deployment.

See **[ROADMAP.md](https://github.com/r1b4z01d/VLA/blob/main/ROADMAP.md)** for the full plan and
**[docs/decisions.md](https://github.com/r1b4z01d/VLA/blob/main/docs/decisions.md)** for open
architectural choices.

## Contents
- **[Machines & Network](Machines-and-Network)** — the three machines, the isolated robot subnet, and `run.sh` on the robot PC.
- **[Hardware Teleop & Recording](Hardware-Teleop-and-Recording)** — the RTDE teleop panel, cameras, Stream Deck, home pose, auto-upload + star rating + auto-home-on-save.
- **[Recording Sim Demos](Recording-Sim-Demos)** — MuJoCo teleop capture and the sim grasp aid.
- **[Training & Eval](Training-and-Eval)** — train ACT / fine-tune SmolVLA, sim eval, playback, merge/recover.
- **[Web Manager (RobotDisco)](Web-Manager)** — the browser console for datasets · models · evals (native or Docker).
- **[Deploy on Hardware](Deploy-on-Hardware)** — run a trained policy on the real UR5e (`eval_hw.py`), Stream-Deck driven.
- **[8-DOF Hand](8-DOF-Hand)** — per-finger flex + abduct (in progress).
- **[Sync & Layout](Sync-and-Layout)** — moving code between machines and the repo map.
