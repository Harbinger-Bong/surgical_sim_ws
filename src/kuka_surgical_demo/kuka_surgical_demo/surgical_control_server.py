#!/usr/bin/env python3
"""
surgical_control_server.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exposes /execute_task (TaskPickPlace.srv).

Gripper architecture
────────────────────
  This node does NOT open an EKI socket. Instead it publishes
  gripper commands on /gripper_cmd (std_msgs/Int8: 1=ON, 0=OFF).
  gripper_bridge.py subscribes to that topic and fires the EKI
  packet through its own motion socket — the one that also sends
  robot motion commands. This avoids the EKI single-client-per-
  channel constraint that would otherwise block motion commands.

Calibration basis (2026-06-25, SmartPad ground truth — with gripper)
─────────────────────────────────────────────────────────────────────
  Suction tip on green mat corners (Tool 111 [1], Base $NULLFRAME [0]):
    Centre:        X=415.68  Y=-1.80   Z=53.50  A=174.00  B=44.72  C=175.09
    Top-right:     X=546.66  Y=-212.09 Z=55.36  A=173.95  B=44.72  C=175.03
    Bottom-right:  X=285.90  Y=-212.09 Z=53.71  A=173.95  B=44.72  C=175.03
    Bottom-left:   X=285.90  Y=208.45  Z=57.01  A=173.95  B=44.72  C=175.03
    Top-left:      X=544.88  Y=208.73  Z=54.28  A=174.01  B=44.69  C=175.07

  Corrected centre (without gripper):
    X=440.60  Y=-0.97  Z=2.95  A=-180.00  B=0.00  C=180.00

  Derived orientation quaternion (ZYX: A=174°, B=44.72°, C=175.09°):
    qx=0.0321  qy=0.9235  qz=0.0197  qw=0.3816

Z reference values (tool0, base_link frame, with gripper mounted)
──────────────────────────────────────────────────────────────────
  Z_TABLE      = +53.50 mm  (tip on mat = tool0 Z in working pose)
  INST_H       = +10 mm     (object height off mat surface)
  pick_z(H)    = Z_TABLE + H
  APPROACH_CLEARANCE = 120 mm
  Z_SAFE       = +250 mm    (must be above approach height of ~174mm)

Tray geometry (from physical corner measurements, base_link frame)
──────────────────────────────────────────────────────────────────
  X range: 285.9 → 546.7 mm  → centre X = 416.3 mm
  Y range: -212.1 → 208.7 mm → centre Y = -1.7 mm (≈ 0)
  Instruments spaced 60mm apart in Y, centred on tray.

Collision strategy
──────────────────
  Only the three instrument boxes (scalpel, forceps, retractor) are
  added to the MoveIt planning scene. The instrument_tray and
  table_surface slabs are NOT added — they caused link_5/6 collision
  during LIN descent and the remove/restore workaround caused an
  executor deadlock (blocking spin_until_future_complete inside async).
  The instrument boxes are still needed so MoveIt tracks them for the
  remove_object() → attach_object() pick sequence.

Motion sequence (6 steps)
──────────────────────────
  1. PTP to pick XY at transit_z
  2. Disable tray collisions → LIN down to contact
  3. Gripper ON → PTP arc to above place → Re-enable tray collisions
  4. LIN down to place contact
  5. Gripper OFF → LIN retract
  6. PTP to park

Velocity
────────
  VEL_TRANSIT = 5%  (long PTP transits, park)
  VEL_NEAR    = 3%  (approach, contact, retract near tray)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import rclpy
import threading
import time

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import Int8
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

# ── Frames ────────────────────────────────────────────────────────────────────
FRAME = 'base_link'
TIP   = 'tool0'

GRIPPER_CMD_TOPIC = '/gripper_cmd'

# ── Orientation (2026-06-25 ground truth: A=174°, B=44.72°, C=175.09°) ────────
ORI = dict(qx=0.0321, qy=0.9235, qz=0.0197, qw=0.3816)

# ── Z reference values (metres, tool0, base_link frame, gripper mounted) ──────
Z_TABLE            =  0.05350
INST_H             =  0.010
APPROACH_CLEARANCE =  0.120
Z_SAFE             =  0.250

def pick_z(obj_h: float) -> float:
    return Z_TABLE + obj_h

def approach_z(obj_h: float) -> float:
    return pick_z(obj_h) + APPROACH_CLEARANCE

# ── Workspace bounds ──────────────────────────────────────────────────────────
WS_X_MIN, WS_X_MAX =  0.250,  0.600
WS_Y_MIN, WS_Y_MAX = -0.860,  0.250
WS_Z_MIN, WS_Z_MAX =  0.030,  0.600

# ── Tray geometry ─────────────────────────────────────────────────────────────
TRAY_CENTRE_X =  0.4163
TRAY_CENTRE_Y = -0.0017
TRAY_Z        =  0.0535

# ── Park position ─────────────────────────────────────────────────────────────
PARK_X, PARK_Y, PARK_Z = 0.40, 0.00, 0.40

# ── Velocity scaling ──────────────────────────────────────────────────────────
VEL_TRANSIT = 0.05
VEL_NEAR    = 0.03


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

        self._gripper_pub = self.create_publisher(Int8, GRIPPER_CMD_TOPIC, 10)
        self.get_logger().info(f'Publishing gripper commands on {GRIPPER_CMD_TOPIC}')

        self.get_logger().info('Waiting for MoveGroup action server...')
        self._move_client.wait_for_server()
        self.get_logger().info('Waiting for ApplyPlanningScene service...')
        self._scene_client.wait_for_service()
        self.get_logger().info('Surgical Control Server online.')

    # ── Gripper ───────────────────────────────────────────────────────────────

    def _send_gripper(self, state: int):
        msg = Int8()
        msg.data = state
        self._gripper_pub.publish(msg)
        label = "ON  (pick)" if state else "OFF (place)"
        self.get_logger().info(f'  [GRIPPER CMD → {label}]')

    # ── Bounds check ──────────────────────────────────────────────────────────

    def _in_bounds(self, x, y, z) -> bool:
        return (WS_X_MIN <= x <= WS_X_MAX and
                WS_Y_MIN <= y <= WS_Y_MAX and
                WS_Z_MIN <= z <= WS_Z_MAX)

    # ── Service callback ──────────────────────────────────────────────────────

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

        px = request.pick_pose.position.x
        py = request.pick_pose.position.y
        pz = request.pick_pose.position.z
        dx = request.place_pose.position.x
        dy = request.place_pose.position.y
        dz = request.place_pose.position.z

        for label, x, y, z in [('pick', px, py, pz), ('place', dx, dy, dz)]:
            if not self._in_bounds(x, y, z):
                return self._fail(response,
                    f'{label} pose ({x:.3f},{y:.3f},{z:.3f}) outside workspace')

        pick_app  = pz + APPROACH_CLEARANCE
        place_app = dz + APPROACH_CLEARANCE
        transit_z = max(pick_app, place_app, Z_SAFE)

        # ── 1. Transit to above pick ──────────────────────────────────────────
        self.get_logger().info('  [1/6] Transit → above pick (PTP)')
        if not await self.move_to(px, py, transit_z, 'PTP', VEL_TRANSIT):
            return self._fail(response, 'Transit to pick column failed')

        # ── 2. LIN down to pick contact ───────────────────────────────────────
        self.get_logger().info('  [2/6] Pick → contact (LIN)')
        await self.remove_object(obj)   # remove before descent so no self-collision
        if not await self.move_to(px, py, pz, 'LIN', VEL_NEAR):
            return self._fail(response, 'Pick contact failed')

        self._send_gripper(1)
        await self.attach_object(obj)
        _ros_sleep(self, 0.5)

        # ── 3. PTP arc to above place ─────────────────────────────────────────
        self.get_logger().info('  [3/6] Retract + transit → above place (PTP)')
        if not await self.move_to(dx, dy, transit_z, 'PTP', VEL_TRANSIT):
            return self._fail(response, 'Transit to place column failed')

        # ── 4. LIN down to place contact ──────────────────────────────────────
        self.get_logger().info('  [4/6] Place → contact (LIN)')
        if not await self.move_to(dx, dy, dz, 'LIN', VEL_NEAR):
            return self._fail(response, 'Place contact failed')

        self._send_gripper(0)
        await self.detach_object(obj)
        _ros_sleep(self, 0.4)

        # ── 5. LIN retract ────────────────────────────────────────────────────
        self.get_logger().info('  [5/6] Place → retract (LIN)')
        if not await self.move_to(dx, dy, place_app, 'LIN', VEL_NEAR):
            return self._fail(response, 'Place retract failed')

        # ── 6. Park ───────────────────────────────────────────────────────────
        self.get_logger().info('  [6/6] Parking (PTP)')
        await self.move_to(PARK_X, PARK_Y, PARK_Z, 'PTP', VEL_TRANSIT)

        self.get_logger().info(f'=== Complete: {obj.upper()} ===')
        response.success = True
        response.message = f'{obj} transferred OK'
        return response

    def _fail(self, response, msg):
        self.get_logger().error(f'  ABORTED: {msg}')
        self._send_gripper(0)
        response.success = False
        response.message = msg
        return response

    # ── Planning scene ────────────────────────────────────────────────────────

    def _apply_scene_sync(self, scene):
        req = ApplyPlanningScene.Request()
        req.scene = scene
        future = self._scene_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    def add_box(self, name, xyz_m, size_m):
        p = Pose()
        p.position.x, p.position.y, p.position.z = xyz_m
        p.orientation.w = 1.0
        co = CollisionObject()
        co.header.frame_id = FRAME
        co.id = name
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(size_m)
        co.primitives = [box]
        co.primitive_poses = [p]
        co.operation = CollisionObject.ADD
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [co]
        self._apply_scene_sync(scene)
        self.get_logger().info(f'  Scene: added "{name}"')

    async def remove_object(self, object_id):
        co = CollisionObject()
        co.header.frame_id = FRAME
        co.id = object_id
        co.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [co]
        req = ApplyPlanningScene.Request()
        req.scene = scene
        await self._scene_client.call_async(req)
        self.get_logger().info(f'  Scene: removed world object "{object_id}"')

    async def attach_object(self, object_id):
        aco = AttachedCollisionObject()
        aco.link_name = TIP
        aco.object.id = object_id
        aco.object.operation = CollisionObject.ADD
        aco.touch_links = [
            'tool0', 'gripper_gripper_base',
            'gripper_suction_cup', 'gripper_tcp',
            'scalpel', 'forceps', 'retractor',
            'instrument_tray',
        ]
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [aco]
        req = ApplyPlanningScene.Request()
        req.scene = scene
        await self._scene_client.call_async(req)
        self.get_logger().info(f'  Scene: "{object_id}" attached to {TIP}')

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
        self.get_logger().info(f'  Scene: "{object_id}" detached')

    def setup_surgical_scene(self):
        self.get_logger().info('=== Building surgical scene ===')
        # No collision objects added to the planning scene.
        # The instrument boxes (scalpel, forceps, retractor) were causing
        # link_4/5/6 collision during every transit because the robot's
        # forearm geometry sweeps through the tray footprint in this pose.
        # The remove_object() / attach_object() / detach_object() sequence
        # still functions correctly with no world objects present.
        self.get_logger().info('=== Scene ready ===')

    # ── Motion ────────────────────────────────────────────────────────────────

    async def move_to(self, x, y, z, planner='PTP', vel=VEL_NEAR):
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

        self.get_logger().info(
            f'  [{planner} {int(vel*100)}%] → ({x:.4f}, {y:.4f}, {z:.4f}) m')
        goal_handle = await self._move_client.send_goal_async(goal)
        if not goal_handle.accepted:
            self.get_logger().error('  Goal REJECTED by MoveGroup')
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
