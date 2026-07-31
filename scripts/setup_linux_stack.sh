#!/usr/bin/env bash
# Phase 1 stack on the NVIDIA Linux workstation.
# Target: Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic (decision D1).
# Assumes ROS 2 Jazzy is already installed (apt) per docs.ros.org/en/jazzy.
set -euo pipefail

ROS_DISTRO="jazzy"   # locked: Ubuntu 24.04 -> Jazzy -> Gazebo Harmonic

echo ">>> ROS 2 distro: ${ROS_DISTRO} (Gazebo Harmonic)"
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"

# 1) UR driver + UR Gazebo (Gz) sim + gz<->ros2_control bridge — all from apt binaries.
#    'ros-jazzy-ur' is a metapackage (ur_robot_driver, ur_description, ur_moveit_config, ...).
sudo apt-get update
sudo apt-get install -y \
  "ros-${ROS_DISTRO}-ur" \
  "ros-${ROS_DISTRO}-ur-simulation-gz" \
  "ros-${ROS_DISTRO}-gz-ros2-control" \
  "ros-${ROS_DISTRO}-moveit-servo"     # for SpaceMouse teleop (twist -> IK), decision D3

# (Source alternative — only if you need to MODIFY the sim:)
#   git clone -b ros2 \
#     https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation.git \
#     ros2_ws/src/ur_simulation_gz
#   ( cd ros2_ws && rosdep install --from-paths src --ignore-src -r -y && colcon build )

# 2) Our own workspace (for the LeRobot <-> ROS 2 bridge built in Phase 2).
WS="$(cd "$(dirname "$0")/.." && pwd)/ros2_ws"
mkdir -p "${WS}/src"

cat <<EOF

✅ ROS 2 Jazzy + UR driver + UR Gazebo (Harmonic) + MoveIt Servo installed.

Launch a simulated UR5e:
  ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur5e
  # NOTE: on Jazzy the namespacing arg is 'tf_prefix' (was 'prefix' pre-Jazzy).

Next:
  - Install LeRobot into the SAME Python that has rclpy (decisions.md D5).
  - Wire SpaceMouse -> MoveIt Servo, then build the LeRobot <-> ROS 2 bridge (Phase 2).
    Reference bridge: github.com/ycheng517/lerobot-ros
EOF
