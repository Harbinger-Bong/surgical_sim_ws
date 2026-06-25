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

Calibration basis (2026-06-23, SmartPad ground truth)
──────────────────────────────────────────────────────
  Suction tip on green mat (Tool 111 [1], Base $NULLFRAME [0]):
    TCP  X = 373.92 mm   Y = 84.51 mm   Z = -183.18 mm
    Euler A = -180.00°   B = 50.33°     C = 180.00°

  Derived quaternion (ZYX Euler → quat):
    qx=0.0  qy=-0.9051  qz=0.0  qw=-0.4252

Z reference values (tool0, base_link frame)
────────────────────────────────────────────
  TIP_OFFSET   = 30 mm below tool0
  Z_TABLE      = -183.18 mm  (tip on mat = tool0 Z)
  pick_z(H)    = Z_TABLE + H  (tip offset cancels in derivation)
  approach_z   = pick_z + 120 mm
  Z_SAFE       = +100 mm

Tray geometry (from CAD sketch, base_link frame)
─────────────────────────────────────────────────
  Tray near edge X=340mm, far X=640mm → centre X=490mm
  Tray centred on robot Y axis → centre Y=0mm
  Instruments spaced 60mm apart in Y.

Collision fix
─────────────
  Picked instrument is REMOVED from world scene before attach.
  Prevents ValidateSolution collision between attached instrument
  and remaining world objects during transit.

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

GRIPPER_CMD_TOPIC = '/gripper_cmd'   # Int8: 1 = ON, 0 = OFF

# ── Orientation (SmartPad ground truth: A=-180°, B=50.33°, C=180°) ────────────
ORI = dict(qx=0.0, qy=-0.9051, qz=0.0, qw=-0.4252)

# ── Z reference values (metres, tool0, base_link frame) ──────────────────────
Z_TABLE    = -0.2
TIP_OFFSET =  0.030
INST_H     =  0.010
APPROACH_CLEARANCE = 0.120
Z_SAFE     =  0.100

def pick_z(obj_h: float) -> float:
    return Z_TABLE + obj_h

def approach_z(obj_h: float) -> float:
    return pick_z(obj_h) + APPROACH_CLEARANCE

# ── Workspace bounds (metres, base_link) ──────────────────────────────────────
WS_X_MIN, WS_X_MAX =  0.150,  0.700
WS_Y_MIN, WS_Y_MAX = -0.250,  0.250
WS_Z_MIN, WS_Z_MAX = -0.200,  0.600

# ── Tray geometry (metres) ────────────────────────────────────────────────────
TRAY_CENTRE_X =  0.490
TRAY_CENTRE_Y =  0.000
TRAY_Z        = -0.2132

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

        # MoveIt action client
        self._move_client = ActionClient(
            self, MoveGroup, '/move_action', callback_group=self.cb)

        # Planning scene client
        self._scene_client = self.create_client(
            ApplyPlanningScene, '/apply_planning_scene')

        # Task service
        self.create_service(
            TaskPickPlace, '/execute_task',
            self.execute_task_callback, callback_group=self.cb)

        self._task_lock = threading.Lock()

        # Gripper command publisher — gripper_bridge owns the socket,
        # we just tell it what state we need
        self._gripper_pub = self.create_publisher(Int8, GRIPPER_CMD_TOPIC, 10)
        self.get_logger().info(f'Publishing gripper commands on {GRIPPER_CMD_TOPIC}')

        self.get_logger().info('Waiting for MoveGroup action server...')
        self._move_client.wait_for_server()
        self.get_logger().info('Waiting for ApplyPlanningScene service...')
        self._scene_client.wait_for_service()
        self.get_logger().info('Surgical Control Server online.')

    # ── Gripper command ───────────────────────────────────────────────────────

    def _send_gripper(self, state: int):
        """Publish gripper state — gripper_bridge.py executes the EKI packet."""
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

        # ── Pick sequence ────────────────────────────────────────────────────
        self.get_logger().info('  Pick → safe height (PTP)')
        if not await self.move_to(px, py, Z_SAFE, 'PTP', VEL_TRANSIT):
            return self._fail(response, 'Transit to pick column failed')

        self.get_logger().info('  Pick → approach (PTP)')
        if not await self.move_to(px, py, pick_app, 'PTP', VEL_NEAR):
            return self._fail(response, 'Pick approach failed')

        self.get_logger().info('  Pick → contact (LIN)')
        if not await self.move_to(px, py, pz, 'LIN', VEL_NEAR):
            return self._fail(response, 'Pick contact failed')

        await self.remove_object(obj)
        self._send_gripper(1)
        await self.attach_object(obj)
        _ros_sleep(self, 0.5)

        self.get_logger().info('  Pick → retract (LIN)')
        if not await self.move_to(px, py, pick_app, 'LIN', VEL_NEAR):
            return self._fail(response, 'Pick retract failed')

        # ── Transit ──────────────────────────────────────────────────────────
        self.get_logger().info('  Transit → safe height (PTP)')
        if not await self.move_to(px, py, Z_SAFE, 'PTP', VEL_TRANSIT):
            return self._fail(response, 'Transit lift failed')

        self.get_logger().info('  Place → transit (PTP)')
        if not await self.move_to(dx, dy, Z_SAFE, 'PTP', VEL_TRANSIT):
            return self._fail(response, 'Transit to place column failed')

        # ── Place sequence ───────────────────────────────────────────────────
        self.get_logger().info('  Place → approach (PTP)')
        if not await self.move_to(dx, dy, place_app, 'PTP', VEL_NEAR):
            return self._fail(response, 'Place approach failed')

        self.get_logger().info('  Place → contact (LIN)')
        if not await self.move_to(dx, dy, dz, 'LIN', VEL_NEAR):
            return self._fail(response, 'Place contact failed')

        self._send_gripper(0)
        await self.detach_object(obj)
        _ros_sleep(self, 0.4)

        self.get_logger().info('  Place → retract (LIN)')
        if not await self.move_to(dx, dy, place_app, 'LIN', VEL_NEAR):
            return self._fail(response, 'Place retract failed')

        # ── Park ─────────────────────────────────────────────────────────────
        self.get_logger().info('  Return → safe height (PTP)')
        await self.move_to(dx, dy, Z_SAFE, 'PTP', VEL_TRANSIT)
        self.get_logger().info('  Parking (PTP)')
        await self.move_to(PARK_X, PARK_Y, PARK_Z, 'PTP', VEL_TRANSIT)

        self.get_logger().info(f'=== Complete: {obj.upper()} ===')
        response.success = True
        response.message = f'{obj} transferred OK'
        return response

    def _fail(self, response, msg):
        self.get_logger().error(f'  ABORTED: {msg}')
        self._send_gripper(0)   # safety: always release vacuum on failure
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

        # Table surface slab
        self.add_box('table_surface',
                     (TRAY_CENTRE_X, TRAY_CENTRE_Y, TRAY_Z - 0.025),
                     (1.0, 1.0, 0.050))

        # Instrument tray
        self.add_box('instrument_tray',
                     (TRAY_CENTRE_X, TRAY_CENTRE_Y, TRAY_Z),
                     (0.450, 0.300, 0.005))

        # Instruments — 60mm apart in Y
        inst_size = (0.150, 0.020, 0.010)
        for name, y_offset in [
            ('scalpel',   -0.060),
            ('forceps',    0.000),
            ('retractor', +0.060),
        ]:
            self.add_box(name,
                         (TRAY_CENTRE_X, TRAY_CENTRE_Y + y_offset,
                          TRAY_Z + 0.010),
                         inst_size)

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

        self.get_logger().info(f'  [{planner} {int(vel*100)}%] → ({x:.4f}, {y:.4f}, {z:.4f}) m')
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
