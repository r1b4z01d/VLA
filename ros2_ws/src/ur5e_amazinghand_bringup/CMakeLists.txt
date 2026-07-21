cmake_minimum_required(VERSION 3.8)
project(ur5e_amazinghand_bringup)

find_package(ament_cmake REQUIRED)

# Install resource directories (launch files, configs, robot description).
install(
  DIRECTORY launch config urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
