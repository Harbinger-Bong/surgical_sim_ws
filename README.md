# Surgical Simulation Workspace

ROS2 Jazzy surgical manipulation workspace built around a KUKA KR6 R900 sixx robot. The system integrates MoveIt2 planning, custom vacuum gripper manipulation, modular voice-command orchestration, computer vision, and direct hardware execution via the KUKA Ethernet KRL XML (EKI) interface.

## Acknowledgements & Credits

* This project is built on top of the Kroshu KUKA robot description repository[cite: 1].
* KUKA EKI integration utilizes the `kuka_eki` package: [https://gitingest.com/tingelst/kuka_eki/blob/ros2/kuka_ek/](https://gitingest.com/tingelst/kuka_eki/blob/ros2/kuka_ek/)

---

## System Overview

```text
Voice/Vision/Keyboard Interfaces
        ↓
Command Interpretation & Target Generation
        ↓
MoveIt2 Motion Planning (Simulation/Validation)
        ↓
kuka_eki_bridge (Trajectory Interception)
        ↓
KUKA KRC4 Controller (Physical Hardware execution via EKI TCP/IP)

```

---

## Workspace Structure

```text
harbinger-bong-surgical_sim_ws/
├── keyboard_control.py         # Direct EKI keyboard teleoperation
├── keyboard_nudge.py           # Single-joint EKI micro-nudging
├── palm_approach_node.py       # MediaPipe palm detection to EKI PTP
├── src/
│   ├── kuka_eki_bridge         # MoveIt-to-EKI trajectory dispatchers
│   ├── kuka_robot_descriptions # KUKA URDFs and MoveIt configs
│   ├── kuka_surgical_demo      # Surgical logic and Vosk voice pipelines
│   ├── kuka_vacuum_gripper     # Vacuum gripper Xacro definitions
│   └── surgical_msgs           # Custom ROS2 interfaces

```

---

## Packages

| Package | Purpose |
| --- | --- |
| `kuka_eki_bridge` | EKI bridges translating MoveIt2 trajectories and vision logic to KRC4 hardware. |
| `kuka_robot_descriptions` | Modified KR-series robot descriptions, meshes, and MoveIt2 configurations. |
| `kuka_surgical_demo` | Surgical orchestration, mock pipelines, and voice control AI. |
| `kuka_vacuum_gripper` | Vacuum gripper URDF/Xacro models. |
| `surgical_msgs` | Custom ROS2 service interfaces (`TaskPickPlace.srv`). |

---

## Features

* **MoveIt2 Motion Planning**: Integration with Pilz PTP/LIN trajectories and safety validation.
* **Hardware EKI Bridge**: Real-time trajectory extraction from `/display_planned_path` directly to the KUKA state/motion servers.
* **Vision-Driven Autonomy**:
* OpenCV multi-color detection for automated suction triggering (`vision_gripper_bridge.py`).
* MediaPipe hand landmark tracking for dynamic hover positioning (`palm_approach_node.py`).


* **Voice Control**: Offline Vosk-powered speech recognition mapping spoken digits to 3D grid coordinates.
* **Teleoperation**: Safe, absolute/relative keyboard control scripts mapped directly to EKI motion clients.

---

## Build Instructions

Always source the workspace before running nodes:

```bash
source install/setup.bash

```

Build the workspace:

```bash
cd ~/surgical_sim_ws
colcon build

```

---

## Launch Workflows

### 1. Base RViz Simulation & MoveIt2

Required as the base layer for any MoveIt-based planning workflows.

**Terminal 1:**

```bash
ros2 launch kuka_kr_moveit_config moveit_planning_fake_hardware.launch.py \
robot_model:=kr6_r900_sixx_with_gripper \
robot_family:=agilus

```

### 2. Hardware Execution (EKI Bridges)

Ensure the KUKA robot is at `192.168.1.147` and the EKI server (`ros_eki.src`) is running on the SmartPad.

**Standard MoveIt Bridge:**

```bash
ros2 run kuka_eki_bridge bridge_node

```

**Keyboard-Toggled Gripper Bridge:**

```bash
ros2 run kuka_eki_bridge gripper_bridge

```

**Vision-Automated Gripper Bridge (OpenCV):**

```bash
ros2 run kuka_eki_bridge vision_gripper_bridge

```

**Voice-Controlled Grid Bridge (Vosk):**

```bash
ros2 run kuka_eki_bridge voice_bridge_node

```

### 3. Standalone Hardware Scripts (No MoveIt Required)

These scripts communicate directly with the KUKA controller via the `kuka_eki` Python library.

**Direct Keyboard Teleoperation:**

```bash
python3 keyboard_control.py

```

**Single-Joint Keyboard Nudging:**

```bash
python3 keyboard_nudge.py

```

**MediaPipe Palm Detection & Approach:**

```bash
# Append --dry-run to test detection without robot motion
python3 palm_approach_node.py

```
