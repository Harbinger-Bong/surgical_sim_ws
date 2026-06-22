import sys
import os
import math
import threading
import time
import cv2
import numpy as np

# Force Python to look in the kuka_eki directory regardless of bash environment
sys.path.append(os.path.expanduser('~/surgical_sim_ws/src/kuka_eki'))

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import DisplayTrajectory
from kuka_eki.eki import EkiMotionClient
from kuka_eki.krl import Axis

def build_gripper_packet(state: int) -> bytes:
    """Type=0 → no motion case in KRL switch, but Gripper field is still read and $OUT[1] set."""
    return (
        b'<RobotCommand>'
        b'<Type>0</Type>'
        b'<Axis A1="0" A2="0" A3="0" A4="0" A5="0" A6="0"/>'
        b'<Cart X="0" Y="0" Z="0" A="0" B="0" C="0"/>'
        b'<Velocity>0.1</Velocity>'
        b'<Gripper>' + str(state).encode() + b'</Gripper>'
        b'</RobotCommand>'
    )

class VisionGripperBridge(Node):
    def __init__(self):
        super().__init__('vision_gripper_bridge')
        self.kuka_ip = "192.168.1.147"
        self.get_logger().info(f"Connecting to KUKA at {self.kuka_ip}...")

        self.motion_client = EkiMotionClient(self.kuka_ip)
        self.motion_client.connect()
        self.get_logger().info("--- EKI BRIDGE CONNECTED ---")

        self.subscription = self.create_subscription(
            DisplayTrajectory,
            '/display_planned_path',
            self.display_trajectory_callback,
            10
        )
        self.get_logger().info("Listening on /display_planned_path ...")

        # Gripper state
        self._gripper_state = 0
        self._gripper_lock = threading.Lock()

        # Start Camera thread
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()
        self.get_logger().info("Multi-Color Detector active — Show RED, BLUE, GREEN, or YELLOW to pick for 47 seconds.")

    # ── Gripper ───────────────────────────────────────────────────────────────

    def _send_gripper(self, state: int):
        """Send gripper packet through the existing EKI motion socket."""
        try:
            self.motion_client._tcp_client.sendall(build_gripper_packet(state))
            label = "ON  (pick)" if state else "OFF (place)"
            self.get_logger().info(f"Gripper {label}")
        except Exception as e:
            self.get_logger().error(f"Gripper send failed: {e}")

    def _set_gripper(self, state: int):
        with self._gripper_lock:
            # ONLY send packet if the state is actually changing
            if self._gripper_state != state:
                self._gripper_state = state
                self._send_gripper(self._gripper_state)

    def _camera_loop(self):
        cap = cv2.VideoCapture(2)
        if not cap.isOpened():
            self.get_logger().warn("Failed to open camera 2. Trying index 0...")
            cap = cv2.VideoCapture(0)

        # Define HSV color ranges
        # Note: Red wraps around the hue scale in OpenCV, so it gets two ranges
        color_ranges = {
            "Yellow": [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
            "Blue":   [(np.array([100, 150, 50]), np.array([140, 255, 255]))],
            "Green":  [(np.array([35, 100, 100]), np.array([85, 255, 255]))],
            "Red":    [(np.array([0, 120, 70]), np.array([10, 255, 255])),
                       (np.array([170, 120, 70]), np.array([180, 255, 255]))]
        }
        
        # Debounce and threshold variables
        min_color_area = 5000  # Minimum number of colored pixels to consider it a solid detection
        color_frames = 0
        
        # Non-blocking timer variables
        is_timing = False
        suction_end_time = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            current_time = time.time()

            # 1. If we are currently holding suction for the 47 seconds
            if is_timing:
                if current_time >= suction_end_time:
                    self.get_logger().info("⏱️ 47 seconds complete. Vacuum OFF. Ready for next detection.")
                    self._set_gripper(0)
                    is_timing = False
                    color_frames = 0  # Reset debounce
                
                # Keep consuming frames to clear the buffer, but skip detection until time is up
                continue

            # 2. Convert to HSV and detect colors
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            detected_color = None
            
            for color_name, ranges in color_ranges.items():
                mask = None
                for lower, upper in ranges:
                    current_mask = cv2.inRange(hsv, lower, upper)
                    if mask is None:
                        mask = current_mask
                    else:
                        mask = cv2.bitwise_or(mask, current_mask)
                
                # Count how many pixels fall within the color range
                area = cv2.countNonZero(mask)
                if area > min_color_area:
                    detected_color = color_name
                    break  # Stop checking other colors if we found a match

            # 3. Trigger Logic
            if detected_color:
                color_frames += 1
                if color_frames >= 5:  # Require 5 consecutive frames to avoid spurious noise
                    self.get_logger().info(f"🎨 {detected_color} object detected! Vacuum ON for 47 seconds.")
                    self._set_gripper(1)
                    
                    # Start the non-blocking timer
                    is_timing = True
                    suction_end_time = current_time + 47.0
            else:
                color_frames = 0

    # ── Motion ────────────────────────────────────────────────────────────────

    def display_trajectory_callback(self, msg: DisplayTrajectory):
        if not msg.trajectory:
            self.get_logger().warn("Empty trajectory received, skipping.")
            return

        joint_traj = msg.trajectory[0].joint_trajectory

        if not joint_traj.points:
            self.get_logger().warn("No points in trajectory.")
            return

        final_point = joint_traj.points[-1]
        joint_names = joint_traj.joint_names

        target_angles = [0.0] * 6
        for i, name in enumerate(joint_names):
            if "joint_1" in name:   target_angles[0] = math.degrees(final_point.positions[i])
            elif "joint_2" in name: target_angles[1] = math.degrees(final_point.positions[i])
            elif "joint_3" in name: target_angles[2] = math.degrees(final_point.positions[i])
            elif "joint_4" in name: target_angles[3] = math.degrees(final_point.positions[i])
            elif "joint_5" in name: target_angles[4] = math.degrees(final_point.positions[i])
            elif "joint_6" in name: target_angles[5] = math.degrees(final_point.positions[i])

        target = Axis(
            a1=target_angles[0], a2=target_angles[1], a3=target_angles[2],
            a4=target_angles[3], a5=target_angles[4], a6=target_angles[5]
        )
        self.get_logger().info(f"Sending: {target}")

        try:
            self.motion_client.ptp(target, max_velocity_scaling=0.1)
            self.get_logger().info("Command sent.")
        except Exception as e:
            self.get_logger().error(f"Transmission failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    bridge = VisionGripperBridge()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        bridge.get_logger().info("Shutting down bridge.")
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
