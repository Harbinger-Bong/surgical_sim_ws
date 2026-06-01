#!/usr/bin/env python3
"""
surgical_rviz.launch.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Launches the full surgical demo in RViz with fake (mock) hardware.
No Gazebo, no real robot required.

Start order
  t=0   kuka moveit_planning_fake_hardware.launch.py  (RViz + move_group)
  t=10  surgical_control_server   (builds scene, exposes /execute_task)
  t=12  vision_logic_coordinator  (subscribes /voice_command)
  t=14  voice_terminal_mock       (interactive stdin publisher)

Usage
  ros2 launch kuka_surgical_demo surgical_rviz.launch.py

Optional args
  launch_voice:=false   — skip the terminal node (use your own publisher)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    launch_voice = LaunchConfiguration('launch_voice')

    # ── 1. MoveIt2 with fake hardware (RViz + move_group) ─────────
    #    This launch file already starts:
    #      • robot_state_publisher
    #      • ros2_control_node (mock hardware)
    #      • joint_state_broadcaster
    #      • joint_trajectory_controller
    #      • move_group
    #      • rviz2
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('kuka_kr_moveit_config'),
                'launch',
                'moveit_planning_fake_hardware.launch.py',
            ])
        ),
        launch_arguments={
            'robot_model':  'kr6_r900_sixx_with_gripper',
            'robot_family': 'agilus',
        }.items(),
    )

    # ── 2. Surgical control server ─────────────────────────────────
    control_server = TimerAction(
        period=10.0,
        actions=[Node(
            package='kuka_surgical_demo',
            executable='surgical_control_server',
            output='screen',
        )],
    )

    # ── 3. Vision / logic coordinator ──────────────────────────────
    vision_logic = TimerAction(
        period=12.0,
        actions=[Node(
            package='kuka_surgical_demo',
            executable='vision_logic_mock',
            output='screen',
        )],
    )

    # ── 4. Voice terminal (interactive stdin) ──────────────────────
    voice_terminal = TimerAction(
        period=14.0,
        actions=[Node(
            package='kuka_surgical_demo',
            executable='voice_terminal_mock',
            output='screen',
            emulate_tty=True,
            condition=IfCondition(launch_voice),
        )],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'launch_voice',
            default_value='true',
            description='Launch the interactive voice terminal node'),

        moveit_launch,
        control_server,
        vision_logic,
        voice_terminal,
    ])
