#!/usr/bin/env python3
"""
surgical_gazebo.launch.py  — v5 (Final Polish)
Root cause fix: kuka_kr_moveit_config/config/pilz_industrial_motion_planner_planning.yaml
uses YAML block scalar (>-) for request_adapters, which loads as a single string.
MoveIt2 Jazzy requires string_array. Fix: load a patched copy of that file
stored in kuka_surgical_demo/config/ with correct list syntax.
All other yamls loaded from kuka_kr_moveit_config as before.
Includes emulate_tty=True for the interactive terminal node.
"""

import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def load_yaml(package_name, file_path):
    pkg_path = get_package_share_directory(package_name)
    abs_path = os.path.join(pkg_path, file_path)
    with open(abs_path, "r") as f:
        return yaml.safe_load(f)


def spawner(controller_name, active=True):
    args = [controller_name, "-c", "/controller_manager"]
    if not active:
        args.append("--inactive")
    return Node(
        package="controller_manager",
        executable="spawner",
        arguments=args,
        output="screen",
    )


def launch_setup(context, *args, **kwargs):
    use_gui      = LaunchConfiguration("use_gui")
    gz_world     = LaunchConfiguration("gz_world")
    launch_voice = LaunchConfiguration("launch_voice")

    kuka_agilus_share = get_package_share_directory("kuka_agilus_support")
    kuka_gazebo_share = get_package_share_directory("kuka_gazebo")
    kuka_moveit_share = get_package_share_directory("kuka_kr_moveit_config")

    # ── URDF ──────────────────────────────────────────────────────
    robot_description_content = ParameterValue(
        Command([
            FindExecutable(name="xacro"), " ",
            os.path.join(
                kuka_agilus_share, "urdf",
                "kr6_r900_sixx_with_gripper.urdf.xacro"
            ),
            " mode:=gazebo",
            " prefix:=",
            " driver_version:=rsi_only",
            " verify_robot_model:=false",
        ]),
        value_type=str,
    )
    robot_description = {"robot_description": robot_description_content}

    # ── SRDF ──────────────────────────────────────────────────────
    srdf_path = os.path.join(kuka_moveit_share, "urdf",
                             "kr6_r900_sixx_with_gripper.srdf")
    with open(srdf_path, "r") as f:
        robot_description_semantic = {"robot_description_semantic": f.read()}

    # ── Kinematics ────────────────────────────────────────────────
    kinematics_yaml = load_yaml("kuka_kr_moveit_config", "config/kinematics.yaml")
    robot_description_kinematics = {"robot_description_kinematics": kinematics_yaml}

    # ── OMPL planning ─────────────────────────────────────────────
    ompl_planning_yaml = load_yaml("kuka_kr_moveit_config", "config/ompl_planning.yaml")

    # ── Pilz planning — load PATCHED version from surgical demo ───
    pilz_planning_yaml = load_yaml(
        "kuka_surgical_demo", "config/pilz_industrial_motion_planner_planning.yaml")

    # ── Pilz cartesian limits ─────────────────────────────────────
    pilz_cartesian_limits_yaml = load_yaml(
        "kuka_kr_moveit_config", "config/pilz_cartesian_limits.yaml")

    # ── MoveIt controllers ────────────────────────────────────────
    moveit_controllers_yaml = load_yaml(
        "kuka_kr_moveit_config", "config/moveit_controllers.yaml")

    # ── Planning pipelines ────────────────────────────────────────
    planning_pipelines = {
        "planning_pipelines": ["pilz_industrial_motion_planner", "ompl"],
        "default_planning_pipeline": "pilz_industrial_motion_planner",
        "pilz_industrial_motion_planner": pilz_planning_yaml,
        "ompl": ompl_planning_yaml,
    }

    # ── 1. robot_state_publisher ───────────────────────────────────
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    # ── 2. Gazebo Harmonic ─────────────────────────────────────────
    world_path = gz_world.perform(context)
    if not os.path.isabs(world_path):
        world_path = os.path.join(kuka_gazebo_share, "world", world_path)

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
            )
        ),
        launch_arguments={
            "gz_args": world_path + " -r -v1",
            "on_exit_shutdown": "true",
        }.items(),
        condition=IfCondition(use_gui),
    )

    # ── 3. Spawn robot ─────────────────────────────────────────────
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic", "robot_description",
            "-name",  "kr6_r900_sixx_with_gripper",
            "-allow_renaming",
            "-x", "0", "-y", "0", "-z", "0",
        ],
        output="screen",
    )

    # ── 4. Clock bridge only ───────────────────────────────────────
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    # ── 5 & 6. Controller spawners ────────────────────────────────
    jsb_spawner = TimerAction(
        period=5.0,
        actions=[spawner("joint_state_broadcaster")],
    )
    jtc_spawner = TimerAction(
        period=7.0,
        actions=[spawner("joint_trajectory_controller")],
    )

    # ── 7. MoveIt2 move_group ──────────────────────────────────────
    move_group_node = TimerAction(
        period=10.0,
        actions=[Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[
                robot_description,
                robot_description_semantic,
                robot_description_kinematics,
                planning_pipelines,
                pilz_cartesian_limits_yaml,
                moveit_controllers_yaml,
                {"use_sim_time": True},
                {"publish_robot_description_semantic": True},
            ],
        )],
    )

    # ── 8. surgical_control_server ─────────────────────────────────
    surgical_server = TimerAction(
        period=18.0,
        actions=[Node(
            package="kuka_surgical_demo",
            executable="surgical_control_server",
            output="screen",
            parameters=[{"use_sim_time": True}],
        )],
    )

    # ── 9. vision_logic_mock ───────────────────────────────────────
    vision_logic = TimerAction(
        period=19.0,
        actions=[Node(
            package="kuka_surgical_demo",
            executable="vision_logic_mock",
            output="screen",
            parameters=[{"use_sim_time": True}],
        )],
    )

    # ── 10. voice_terminal_mock ────────────────────────────────────
    voice_terminal = TimerAction(
        period=20.0,
        actions=[Node(
            package="kuka_surgical_demo",
            executable="voice_terminal_mock",
            output="screen",
            emulate_tty=True, # <--- Added to prevent EOF phantom inputs
            condition=IfCondition(launch_voice),
        )],
    )

    return [
        rsp_node,
        gz_sim,
        spawn_robot,
        clock_bridge,
        jsb_spawner,
        jtc_spawner,
        move_group_node,
        surgical_server,
        vision_logic,
        voice_terminal,
    ]


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument("use_gui", default_value="true"),
        DeclareLaunchArgument("gz_world", default_value="surgical_or.sdf"),
        DeclareLaunchArgument("launch_voice", default_value="true"),
    ]
    return LaunchDescription(declared_args + [OpaqueFunction(function=launch_setup)])
