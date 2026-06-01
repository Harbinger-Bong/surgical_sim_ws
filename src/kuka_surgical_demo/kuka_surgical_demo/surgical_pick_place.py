#!/usr/bin/env python3
"""
Surgical pick and place — KR6 R900 sixx + vacuum gripper
Launch: kuka_kr_moveit_config moveit_planning_fake_hardware.launch.py
        robot_model:=kr6_r900_sixx_with_gripper robot_family:=agilus

SRDF chain tip: tool0  (gripper is fixed beyond, not in planning chain)
Root frame: world  (URDF has world→base_link fixed joint)
TCP offset: 0.076m (tool0 → gripper_tcp along Z)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    CollisionObject, PlanningScene,
    MotionPlanRequest, Constraints,
    PositionConstraint, OrientationConstraint, BoundingVolume,
)
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from rclpy.action import ActionClient
import time

# Distance from tool0 to suction contact point along Z
# gripper_base (60mm) + suction_cup to tcp (16mm) = 76mm
TCP_OFFSET = 0.076

# Root frame — matches world→base_link joint in the URDF
FRAME = 'world'

# IK target link — SRDF chain tip
TIP = 'tool0'


class SurgicalPickPlace(Node):

    def __init__(self):
        super().__init__('surgical_pick_place')
        self._move_client = ActionClient(self, MoveGroup, '/move_action')
        self._scene_client = self.create_client(
            ApplyPlanningScene, '/apply_planning_scene')

        self.get_logger().info('Waiting for MoveGroup action server...')
        self._move_client.wait_for_server()
        self.get_logger().info('Waiting for ApplyPlanningScene service...')
        self._scene_client.wait_for_service()
        self.get_logger().info('Ready.')

    def add_box(self, name, xyz, size):
        p = Pose()
        p.position.x, p.position.y, p.position.z = xyz
        p.orientation.w = 1.0

        co = CollisionObject()
        co.header.frame_id = FRAME
        co.id = name
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(size)
        co.primitives = [box]
        co.primitive_poses = [p]
        co.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [co]

        req = ApplyPlanningScene.Request()
        req.scene = scene
        future = self._scene_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info(f'  Scene: added {name}')

    def move_to(self, x, y, z, planner='LIN'):
        """
        Move tool0 to (x, y, z) in world frame.
        Z values are for tool0 — already include TCP_OFFSET compensation.
        Orientation: tool pointing straight down.
        """
        target = Pose()
        target.position.x = x
        target.position.y = y
        target.position.z = z
        target.orientation.x = 0.0
        target.orientation.y = 0.707
        target.orientation.z = 0.0
        target.orientation.w = 0.707

        req = MotionPlanRequest()
        req.group_name = 'manipulator'
        req.planner_id = planner
        req.pipeline_id = 'pilz_industrial_motion_planner'
        req.num_planning_attempts = 3
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = 0.15
        req.max_acceleration_scaling_factor = 0.15

        pos = PositionConstraint()
        pos.header.frame_id = FRAME
        pos.link_name = TIP
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.002]
        bv = BoundingVolume()
        bv.primitives = [sphere]
        bv.primitive_poses = [target]
        pos.constraint_region = bv
        pos.weight = 1.0

        ori = OrientationConstraint()
        ori.header.frame_id = FRAME
        ori.link_name = TIP
        ori.orientation = target.orientation
        ori.absolute_x_axis_tolerance = 0.05
        ori.absolute_y_axis_tolerance = 0.05
        ori.absolute_z_axis_tolerance = 0.05
        ori.weight = 1.0

        goal_con = Constraints()
        goal_con.position_constraints = [pos]
        goal_con.orientation_constraints = [ori]
        req.goal_constraints = [goal_con]

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.replan = False

        self.get_logger().info(f'  [{planner}] → ({x:.3f}, {y:.3f}, {z:.3f})')

        future = self._move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('  Goal REJECTED')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        code = result_future.result().result.error_code.val
        if code == 1:
            self.get_logger().info('  ✓ SUCCESS')
            return True
        else:
            self.get_logger().error(f'  ✗ FAILED (error_code={code})')
            return False

    def run(self):
        # ── Scene ─────────────────────────────────────────────────────────────
        self.get_logger().info('=== Building surgical scene ===')

        # Surgical table — top face at z=0
        self.add_box('surgical_table', (0.50, 0.00, -0.025), (0.8, 0.6, 0.05))

        # Instrument tray — sits on table, top face at z=0.01
        self.add_box('instrument_tray', (0.50, 0.00, 0.005), (0.3, 0.2, 0.01))

        # Scalpel — top face at z=0.0275
        self.add_box('scalpel', (0.48, -0.05, 0.020), (0.18, 0.02, 0.015))

        # Handoff zone marker
        self.add_box('handoff_zone', (0.35, 0.25, 0.151), (0.08, 0.08, 0.002))

        self.get_logger().info('=== Scene ready ===')
        time.sleep(1.0)  # let MoveIt digest the scene

        # ── Coordinate logic ──────────────────────────────────────────────────
        # We command tool0 positions. Since TCP is 76mm below tool0:
        #   tool0_z = desired_tcp_z + TCP_OFFSET
        #
        # Scalpel top face: z = 0.0275
        # To touch with suction cup: tcp_z = 0.0275 → tool0_z = 0.0275 + 0.076 = 0.1035
        # Approach (15cm clearance above scalpel): tcp_z = 0.0275 + 0.15 → tool0_z = 0.254

        pick_x,  pick_y  = 0.48, -0.05
        place_x, place_y = 0.35,  0.25

        scalpel_top  = 0.0275
        pick_approach_z = scalpel_top + TCP_OFFSET + 0.15   # = 0.254
        pick_contact_z  = scalpel_top + TCP_OFFSET + 0.002  # = 0.106  (2mm gap)

        place_approach_z = 0.30 + TCP_OFFSET   # = 0.376
        place_contact_z  = 0.16 + TCP_OFFSET   # = 0.236  (just above handoff marker)

        # ── Motion sequence ───────────────────────────────────────────────────
        self.get_logger().info('=== Starting pick and place ===')

        self.get_logger().info('1. PTP → above scalpel')
        if not self.move_to(pick_x, pick_y, pick_approach_z, 'PTP'):
            return

        self.get_logger().info('2. LIN → descend to pick contact')
        if not self.move_to(pick_x, pick_y, pick_contact_z, 'LIN'):
            return

        self.get_logger().info('   [VACUUM ON]')
        time.sleep(0.5)

        self.get_logger().info('3. LIN → retract with instrument')
        if not self.move_to(pick_x, pick_y, pick_approach_z, 'LIN'):
            return

        self.get_logger().info('4. PTP → swing to handoff zone')
        if not self.move_to(place_x, place_y, place_approach_z, 'PTP'):
            return

        self.get_logger().info('5. LIN → descend to handoff')
        if not self.move_to(place_x, place_y, place_contact_z, 'LIN'):
            return

        self.get_logger().info('   [VACUUM OFF]')
        time.sleep(0.5)

        self.get_logger().info('6. LIN → retract from handoff')
        self.move_to(place_x, place_y, place_approach_z, 'LIN')

        self.get_logger().info('=== COMPLETE ===')


def main():
    rclpy.init()
    node = SurgicalPickPlace()
    time.sleep(1.0)
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
