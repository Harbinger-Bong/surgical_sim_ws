# Surgical Simulation Workspace

ROS2 Jazzy surgical manipulation workspace built around a KUKA KR6 R900 sixx robot integrated with a custom vacuum gripper, MoveIt2 planning, and modular voice-command orchestration.

Current setup is focused on RViz-based simulation and workflow validation. Physical robot integration and real hardware execution are planned next.

---

# System Overview

```text
Voice Command Interface
        ↓
Speech Processing and Command Interpretation
        ↓
Vision and Task Understanding Layer
        ↓
Surgical Task Coordination System
        ↓
Motion Planning and Safety Validation
        ↓
Simulated KUKA KR6 Robot with Vacuum Gripper Execution
```

---

# Workspace Structure

```text
surgical_sim_ws/
├── src/
│   ├── kuka_robot_descriptions
│   ├── kuka_surgical_demo
│   ├── kuka_vacuum_gripper
│   └── surgical_msgs
```

---

# Packages

| Package                   | Purpose                                                 |
| ------------------------- | ------------------------------------------------------- |
| `surgical_msgs`           | Custom ROS2 service interfaces                          |
| `kuka_vacuum_gripper`     | Vacuum gripper URDF/Xacro package                       |
| `kuka_surgical_demo`      | Surgical orchestration, planning, and voice pipeline    |
| `kuka_robot_descriptions` | Modified KR6 robot description and MoveIt configuration |

---

# Features

* MoveIt2 motion planning
* Pilz PTP/LIN trajectories
* Custom vacuum gripper integration
* Single and multi-instrument workflows
* Voice-command driven task execution
* Offline speech recognition using Vosk
* RViz fake hardware simulation

---

# Build

Always source the workspace before running nodes:

```bash
source install/setup.bash
```

Build workspace:

```bash
cd ~/surgical_sim_ws
colcon build
```

---

# Launch Workflow

## Base Visualization Launch

Terminal 1:

```bash
ros2 launch kuka_kr_moveit_config moveit_planning_fake_hardware.launch.py \
robot_model:=kr6_r900_sixx_with_gripper \
robot_family:=agilus
```

This launch is the base visualization and MoveIt2 setup required for all workflows.

---

# Single Pick-and-Place

Terminal 2:

```bash
ros2 run kuka_surgical_demo surgical_pick_place
```

---

# Multi-Instrument Pick-and-Place

Terminal 2:

```bash
ros2 run kuka_surgical_demo multi_instrument_pick_place
```

---

# Terminal Input Control + Mock Vision

Terminal 1:

```bash
ros2 launch kuka_kr_moveit_config moveit_planning_fake_hardware.launch.py \
robot_model:=kr6_r900_sixx_with_gripper \
robot_family:=agilus
```

Terminal 2:

```bash
ros2 run kuka_surgical_demo surgical_control_server
```

Terminal 3:

```bash
ros2 run kuka_surgical_demo voice_terminal_mock
```

Terminal 4:

```bash
ros2 run kuka_surgical_demo vision_logic_mock
```

---

# Voice Control

Use the same launch sequence as the terminal input control workflow, but replace the terminal input node with:

```bash
ros2 run kuka_surgical_demo voice_ai_node
```

---

# Acknowledgements

This project is built on top of the Kroshu KUKA robot description repository:

https://github.com/kroshu/kuka_robot_descriptions

# Notes

* Large AI models and binary artifacts are excluded from Git tracking.
* Mesh assets are retained for URDF and MoveIt visualization support.
