#!/usr/bin/env python3
"""
Multi-Instrument Surgical Pick and Place
KR6 R900 sixx + vacuum gripper (kroshu fake hardware)

SRDF chain tip: tool0
Root frame: base_link
TCP offset: 0.076m (tool0 → suction cup contact)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    CollisionObject, AttachedCollisionObject, PlanningScene,
    MotionPlanRequest, Constraints,
    PositionConstraint, OrientationConstraint, BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
import time

FRAME      = 'base_link'
TIP        = 'tool0'
TCP_OFFSET = 0.076
ORI        = dict(qx=0.0, qy=0.7071068, qz=0.0, qw=0.7071068)


class MultiInstrumentPickPlace(Node):

    def __init__(self):
        super().__init__('multi_instrument_pick_place')
        self._move_client = ActionClient(self, MoveGroup, '/move_action')
        self._scene_client = self.create_client(
            ApplyPlanningScene, '/apply_planning_scene')
        self._move_client.wait_for_server()
        self._scene_client.wait_for_service()
        self.get_logger().info('Backend verified.')

    def _apply_scene(self, scene):
        req = ApplyPlanningScene.Request()
        req.scene = scene
        future = self._scene_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

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
        self._apply_scene(scene)

    def attach_object(self, object_id):
        aco = AttachedCollisionObject()
        aco.link_name = TIP
        aco.object.id = object_id
        aco.object.operation = CollisionObject.ADD
        aco.touch_links = [
            'tool0', 'gripper_gripper_base',
            'gripper_suction_cup', 'gripper_tcp',
        ]
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [aco]
        self._apply_scene(scene)
        self.get_logger().info(f'  [VACUUM ON]  {object_id}')

    def detach_object(self, object_id):
        aco = AttachedCollisionObject()
        aco.object.id = object_id
        aco.object.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [aco]
        self._apply_scene(scene)
        self.get_logger().info(f'  [VACUUM OFF] {object_id}')

    def move_to(self, x, y, z, planner='PTP', vel=0.15):
        target = Pose()
        target.position.x = x
        target.position.y = y
        target.position.z = z
        target.orientation.x = ORI['qx']
        target.orientation.y = ORI['qy']
        target.orientation.z = ORI['qz']
        target.orientation.w = ORI['qw']

        req = MotionPlanRequest()
        req.group_name = 'manipulator'
        req.planner_id = planner
        req.pipeline_id = 'pilz_industrial_motion_planner'
        req.num_planning_attempts = 5
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = vel
        req.max_acceleration_scaling_factor = vel

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
        self.get_logger().error(f'  ✗ FAILED (error_code={code})')
        return False

    def setup_surgical_scene(self):
        self.get_logger().info('=== Building surgical scene ===')

        # Tray: center z=0.0, half-height=0.0025, top face at z=0.0025
        self.add_box('instrument_tray', (0.68, 0.0, 0.0), (0.16, 0.10, 0.005))

        # Instruments: center z=0.010, bottom face at z=0.005 → 2.5mm above tray top
        inst_size = (0.04, 0.02, 0.010)
        for name, y_off in [('scalpel', -0.03), ('forceps', 0.0), ('retractor', 0.03)]:
            self.add_box(name, (0.68, y_off, 0.010), inst_size)

        # FIX: handoff_zone is NOT added as a collision object.
        # It is a logical drop target, not a physical obstacle.
        # Adding it as a collision box causes the attached instrument
        # to collide with it on descent. Leave it out of the scene.

        self.get_logger().info('=== Scene ready ===')

    def run_pick_and_place(self):
        # ── Z geometry (all tool0 positions) ──────────────────────────────────
        #
        # Instrument:
        #   top face at z = 0.015
        #   pick contact: TCP touches top → tool0_z = 0.015 + TCP_OFFSET + 0.002 = 0.093
        #   pick approach: 12cm above top → tool0_z = 0.015 + TCP_OFFSET + 0.12  = 0.211
        #
        # Handoff (drop target centre at z=0.20):
        #   We want to release instrument just above the surface, not push into it.
        #   place contact: tool0_z = 0.20 + TCP_OFFSET + 0.015 = 0.291
        #     (the +0.015 ensures the bottom of the 10mm instrument is
        #      at z=0.20, not crashing through)
        #   place approach: 12cm above → tool0_z = 0.20 + TCP_OFFSET + 0.12 = 0.396

        INST_TOP         = 0.015
        PICK_CONTACT_Z   = INST_TOP  + TCP_OFFSET + 0.002   # 0.093
        PICK_APPROACH_Z  = INST_TOP  + TCP_OFFSET + 0.12    # 0.211

        HANDOFF_Z        = 0.20                             # desired instrument bottom at release
        # tool0 must be high enough that instrument bottom clears handoff surface:
        # tool0_z = HANDOFF_Z + TCP_OFFSET + inst_half_height
        PLACE_CONTACT_Z  = HANDOFF_Z + TCP_OFFSET + 0.010  # 0.286
        PLACE_APPROACH_Z = HANDOFF_Z + TCP_OFFSET + 0.12   # 0.396

        queue = [
            {'id': 'scalpel',   'pick_x': 0.68, 'pick_y': -0.03, 'place_x': 0.35, 'place_y': 0.17},
            {'id': 'forceps',   'pick_x': 0.68, 'pick_y':  0.00, 'place_x': 0.35, 'place_y': 0.25},
            {'id': 'retractor', 'pick_x': 0.68, 'pick_y':  0.03, 'place_x': 0.35, 'place_y': 0.33},
        ]

        self.get_logger().info('=== Starting pick and place sequence ===')

        for item in queue:
            self.get_logger().info(f"--- {item['id']} ---")
            px, py = item['pick_x'], item['pick_y']
            dx, dy = item['place_x'], item['place_y']

            if not self.move_to(px, py, PICK_APPROACH_Z,  planner='PTP'): return
            if not self.move_to(px, py, PICK_CONTACT_Z,   planner='LIN'): return
            self.attach_object(item['id'])
            time.sleep(0.5)
            if not self.move_to(px, py, PICK_APPROACH_Z,  planner='LIN'): return
            if not self.move_to(dx, dy, PLACE_APPROACH_Z, planner='PTP'): return
            if not self.move_to(dx, dy, PLACE_CONTACT_Z,  planner='LIN'): return
            self.detach_object(item['id'])
            time.sleep(0.5)
            if not self.move_to(dx, dy, PLACE_APPROACH_Z, planner='LIN'): return

        self.get_logger().info('--- Returning to neutral ---')
        self.move_to(0.35, 0.0, PLACE_APPROACH_Z, planner='PTP')
        self.get_logger().info('=== SEQUENCE COMPLETE ===')


def main(args=None):
    rclpy.init(args=args)
    node = MultiInstrumentPickPlace()
    node.setup_surgical_scene()
    time.sleep(1.0)
    node.run_pick_and_place()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
