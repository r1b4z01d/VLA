# ur5e_amazinghand_bringup

UR5e + AmazingHand bringup for **ROS 2 Jazzy / Gazebo Harmonic** (decision D1).
Scaffolded on the Mac; **build and run on the NVIDIA Linux workstation.**

## Contents
- `launch/ur5e_sim.launch.py` — brings up the UR5e in Gazebo via `ur_simulation_gz`.
- `urdf/wrist_camera.xacro` — eye-in-hand camera frame; the **sim↔real parity contract**
  (D7). The placeholder transform must be replaced with the hand-eye calibration result.
- `config/servo.yaml` — MoveIt Servo starting config for SpaceMouse teleop (D3).

## Build & run (on the workstation)
```bash
# prerequisites installed by scripts/setup_linux_stack.sh:
#   ros-jazzy-ur  ros-jazzy-ur-simulation-gz  ros-jazzy-gz-ros2-control  ros-jazzy-moveit-servo
cd ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select ur5e_amazinghand_bringup
source install/setup.bash

ros2 launch ur5e_amazinghand_bringup ur5e_sim.launch.py ur_type:=ur5e
```

## Next (Phase 1 → 2)
1. Confirm the bare UR5e moves in Gazebo (joint controllers respond).
2. Load MoveIt Servo (`config/servo.yaml`) and drive the TCP with a twist command;
   then wire the SpaceMouse → `delta_twist_cmds`.
3. Add `wrist_camera.xacro` to the spawned description and bridge the Gz image topic
   via `ros_gz_bridge`.
4. Build the LeRobot ↔ ROS 2 arm bridge that implements
   `ur5e_lerobot.robot.Ros2ArmInterface` against this sim (then the real robot).
