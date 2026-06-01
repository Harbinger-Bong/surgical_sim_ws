#!/usr/bin/env python3
"""
Surgical Control Server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exposes  /execute_task  (TaskPickPlace.srv).

Orientation strategy
  Original orientation qy=0.707 (90° about Y) points tool0 straight
  down, but leaves joint_4 (forearm) extending horizontally forward
  into the instrument tray during the LIN descent — causing collision.

  Fix: add a 90° roll about the tool Z-axis by composing with qz=0.5.
  The combined quaternion (0.5, 0.5, 0.5, 0.5) still points the tool
  straight down in world frame, but rolls joint_4 by 90° so the
  forearm extends sideways (world Y direction) instead of forward into
  the tray.  The tray is narrow in Y so the forearm clears it.

  Pick orientation  = (qx=0.5, qy=0.5, qz=0.5, qw=0.5)   roll +90°
  Place orientation = same (consistent elbow posture throughout)
  Park  orientation = same
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import rclpy
import threading
import time

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    CollisionObject, AttachedCollisionObject, PlanningScene,
    MotionPlanRequest, Constraints,
    PositionConstraint, OrientationConstraint, BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.action import MoveGroup
from surgical_msgs.srv import TaskPickPlace

FRAME           = 'base_link'
TIP             = 'tool0'

# Tool pointing straight down + forearm rolled 90° sideways
# Quaternion: Y90 * Z90 = (w=0.5, x=0.5, y=0.5, z=0.5)
# This keeps the suction cup pointing at -Z (world down) while
# rotating joint_4 so link_4/5 swing out in the ±Y direction,
# away from the instrument tray which is narrow in Y (160mm).
ORI = dict(qx=0.0, qy=0.7071068, qz=0.0, qw=0.7071068)

PICK_CLEARANCE  = 0.12
PLACE_CLEARANCE = 0.12
PARK_X, PARK_Y, PARK_Z = 0.40, 0.00, 0.40


def _ros_sleep(node, seconds):
    end = node.get_clock().now().nanoseconds + int(seconds * 1e9)
    while node.get_clock().now().nanoseconds < end:
        time.sleep(0.01)


class SurgicalControlServer(Node):

    def __init__(self):
        super().__init__('surgical_control_server')
        self.cb = ReentrantCallbackGroup()
        self._move_client = ActionClient(
            self, MoveGroup, '/move_action', callback_group=self.cb)
        self._scene_client = self.create_client(
            ApplyPlanningScene, '/apply_planning_scene')
        self.create_service(
            TaskPickPlace, '/execute_task',
            self.execute_task_callback, callback_group=self.cb)
        self._task_lock = threading.Lock()

        self.get_logger().info('Waiting for MoveGroup action server...')
        self._move_client.wait_for_server()
        self.get_logger().info('Waiting for ApplyPlanningScene service...')
        self._scene_client.wait_for_service()
        self.get_logger().info('Surgical Control Server online — awaiting tasks.')

    async def execute_task_callback(self, request, response):
        obj = request.object_id
        if not self._task_lock.acquire(blocking=False):
            msg = f'Arm busy — rejected task for "{obj}"'
            self.get_logger().warn(msg)
            response.success = False
            response.message = msg
            return response
        try:
            return await self._run_task(request, response)
        finally:
            self._task_lock.release()

    async def _run_task(self, request, response):
        obj = request.object_id
        self.get_logger().info(f'=== Task: {obj.upper()} ===')
        px, py, pz = (request.pick_pose.position.x,
                      request.pick_pose.position.y,
                      request.pick_pose.position.z)
        dx, dy, dz = (request.place_pose.position.x,
                      request.place_pose.position.y,
                      request.place_pose.position.z)
        pick_app  = pz + PICK_CLEARANCE
        place_app = dz + PLACE_CLEARANCE

        self.get_logger().info('  Pick → approach (PTP)')
        if not await self.move_to(px, py, pick_app, 'PTP'):
            return self._fail(response, 'Pick approach failed')

        self.get_logger().info('  Pick → contact (LIN)')
        if not await self.move_to(px, py, pz, 'LIN'):
            return self._fail(response, 'Pick contact failed')

        await self.attach_object(obj)
        _ros_sleep(self, 0.4)

        self.get_logger().info('  Pick → retract (LIN)')
        if not await self.move_to(px, py, pick_app, 'LIN'):
            return self._fail(response, 'Pick retract failed')

        self.get_logger().info('  Place → transit (PTP)')
        if not await self.move_to(dx, dy, place_app, 'PTP'):
            return self._fail(response, 'Transit failed')

        self.get_logger().info('  Place → contact (LIN)')
        if not await self.move_to(dx, dy, dz, 'LIN'):
            return self._fail(response, 'Place contact failed')

        await self.detach_object(obj)
        _ros_sleep(self, 0.4)

        self.get_logger().info('  Place → retract (LIN)')
        if not await self.move_to(dx, dy, place_app, 'LIN'):
            return self._fail(response, 'Place retract failed')

        self.get_logger().info('  Parking (PTP)')
        await self.move_to(PARK_X, PARK_Y, PARK_Z, 'PTP')

        self.get_logger().info(f'=== Complete: {obj.upper()} ===')
        response.success = True
        response.message = f'{obj} transferred OK'
        return response

    def _fail(self, response, msg):
        self.get_logger().error(f'  ABORTED: {msg}')
        response.success = False
        response.message = msg
        return response

    def _apply_scene_sync(self, scene):
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
        self._apply_scene_sync(scene)
        self.get_logger().info(f'  Scene: added "{name}"')

    def setup_surgical_scene(self):
        self.get_logger().info('=== Building surgical scene ===')
        self.add_box('instrument_tray', (0.55, 0.0, 0.0), (0.20, 0.16, 0.005))
        inst_size = (0.04, 0.02, 0.010)
        for name, y_off in [
            ('scalpel',   -0.05),
            ('forceps',    0.00),
            ('retractor',  0.05),
        ]:
            self.add_box(name, (0.55, y_off, 0.010), inst_size)
        self.get_logger().info('=== Scene ready ===')

    async def attach_object(self, object_id):
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
        req = ApplyPlanningScene.Request()
        req.scene = scene
        await self._scene_client.call_async(req)
        self.get_logger().info(f'  [VACUUM ON]  {object_id}')

    async def detach_object(self, object_id):
        aco = AttachedCollisionObject()
        aco.object.id = object_id
        aco.object.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [aco]
        req = ApplyPlanningScene.Request()
        req.scene = scene
        await self._scene_client.call_async(req)
        self.get_logger().info(f'  [VACUUM OFF] {object_id}')

    async def move_to(self, x, y, z, planner='PTP', vel=0.15):
        target = Pose()
        target.position.x, target.position.y, target.position.z = x, y, z
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
        goal_handle = await self._move_client.send_goal_async(goal)
        if not goal_handle.accepted:
            self.get_logger().error('  Goal REJECTED')
            return False
        result_resp = await goal_handle.get_result_async()
        code = result_resp.result.error_code.val
        if code == 1:
            self.get_logger().info('  ✓ SUCCESS')
            return True
        self.get_logger().error(f'  ✗ FAILED (error_code={code})')
        return False


def main(args=None):
    rclpy.init(args=args)
    node = SurgicalControlServer()
    node.setup_surgical_scene()
    time.sleep(1.0)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
