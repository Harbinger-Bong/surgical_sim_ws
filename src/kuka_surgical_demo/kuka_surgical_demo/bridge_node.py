import rclpy
from rclpy.node import Node
from moveit_msgs.msg import DisplayTrajectory
from geometry_msgs.msg import PointStamped
import math
from kuka_eki.eki import EkiMotionClient
from kuka_eki.krl import Axis, Pos

# ── Palm approach constants (base frame, mm) ──────────────────────
Z_TARGET = -163.18   # 20mm above mat surface
ORI_A    =  174.91
ORI_B    =   38.78
ORI_C    =  176.81


class MoveItEkiBridge(Node):
    def __init__(self):
        super().__init__('moveit_eki_bridge')
        self.kuka_ip = "192.168.1.147"
        self.get_logger().info(f"Connecting to KUKA at {self.kuka_ip}...")

        self.motion_client = EkiMotionClient(self.kuka_ip)
        self.motion_client.connect()
        self.get_logger().info("--- EKI BRIDGE CONNECTED ---")

        # Existing: MoveIt planned path → joint PTP
        self.subscription = self.create_subscription(
            DisplayTrajectory,
            '/display_planned_path',
            self.display_trajectory_callback,
            10
        )
        self.get_logger().info("Listening on /display_planned_path ...")

        # New: palm approach target → Cartesian PTP
        self.palm_sub = self.create_subscription(
            PointStamped,
            '/palm_target',
            self.palm_target_callback,
            10
        )
        self.get_logger().info("Listening on /palm_target ...")

    # ── Existing MoveIt callback ──────────────────────────────────
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

    # ── New palm target callback ──────────────────────────────────
    def palm_target_callback(self, msg: PointStamped):
        """
        Receives palm XY position (mm, base frame) from palm_approach_node.
        msg.point.x = target X (mm)
        msg.point.y = target Y (mm)
        msg.point.z = ignored (Z_TARGET constant used instead)
        """
        target_x = msg.point.x
        target_y = msg.point.y
        target_z = Z_TARGET

        self.get_logger().info(
            f"Palm target received: X={target_x:.1f} Y={target_y:.1f} Z={target_z:.1f} mm"
        )

        try:
            target = Pos(
                x=target_x, y=target_y, z=target_z,
                a=ORI_A, b=ORI_B, c=ORI_C,
                s=10, t=1010,
            )
            self.motion_client.ptp(target, max_velocity_scaling=0.08)
            self.get_logger().info("Palm approach PTP sent — robot moving.")
        except Exception as e:
            self.get_logger().error(f"Palm PTP failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    bridge = MoveItEkiBridge()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        bridge.get_logger().info("Shutting down bridge.")
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
