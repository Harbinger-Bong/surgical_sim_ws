cmake_minimum_required(VERSION 3.8)
project(kuka_vacuum_gripper)

find_package(ament_cmake REQUIRED)

# This is the magic line that was missing!
install(DIRECTORY urdf DESTINATION share/${PROJECT_NAME})

ament_package()
