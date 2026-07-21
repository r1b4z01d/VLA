"""Bring up the UR5e in Gazebo (Harmonic) via the apt-installed ur_simulation_gz.

This wraps the official sim launch so our project owns one entrypoint. Build on the
workstation (ROS 2 Jazzy), then:

    ros2 launch ur5e_amazinghand_bringup ur5e_sim.launch.py

TODO (next, Phase 1/2):
  * load MoveIt Servo with config/servo.yaml (SpaceMouse twist -> IK),
  * add the wrist camera (urdf/wrist_camera.xacro) to the spawned description,
  * bridge Gz camera topics to ROS via ros_gz_bridge.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    ur_type = LaunchConfiguration("ur_type")
    launch_rviz = LaunchConfiguration("launch_rviz")

    declared_args = [
        DeclareLaunchArgument("ur_type", default_value="ur5e", description="UR model"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
    ]

    ur_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [FindPackageShare("ur_simulation_gz"), "launch", "ur_sim_control.launch.py"]
                )
            ]
        ),
        launch_arguments={
            "ur_type": ur_type,
            "launch_rviz": launch_rviz,
        }.items(),
    )

    return LaunchDescription(declared_args + [ur_sim])
