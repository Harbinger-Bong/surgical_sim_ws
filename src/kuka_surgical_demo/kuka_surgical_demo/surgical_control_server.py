#!/usr/bin/env python3
"""
surgical_control_server.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exposes /execute_task (TaskPickPlace.srv).

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
  Robot base centre = origin.
  Tray near edge at X=340mm, far at X=640mm → centre X=490mm
  Tray left edge at Y=-225mm, right at Y=+225mm → centre Y=0mm
  Tray is centred on the robot Y axis (from sketch geometry).
  Physical contact at (373.92, 84.51) was a specific jogged point,
  not the tray centre.

Collision fix
─────────────
  The picked instrument is REMOVED from the world scene before being
  attached to tool0. This prevents MoveIt from detecting a collision
  between the attached instrument and the world-object version of itself
  or its neighbours during transit.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import rclpy
import threading
import time
import socket

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

# ── Frames ────────────────────────────────────────────────────────────────────
FRAME = 'base_link'
TIP   = 'tool0'

# ── Orientation (SmartPad ground truth: A=-180°, B=50.33°, C=180°) ────────────
ORI = dict(qx=0.0, qy=-0.9051, qz=0.0, qw=-0.4252)

# ── Z reference values (metres, tool0, base_link frame) ──────────────────────
Z_TABLE    = -0.18318   # tool0 Z with suction tip on mat
TIP_OFFSET =  0.030     # suction tip 30mm below tool0
INST_H     =  0.010     # instrument height off tray (~10mm lying flat)
APPROACH_CLEARANCE = 0.120
Z_SAFE     =  0.100     # transit height

def pick_z(obj_h: float) -> float:
    """tool0 Z to contact object of height obj_h on mat. TIP_OFFSET cancels."""
    return Z_TABLE + obj_h

def approach_z(obj_h: float) -> float:
    return pick_z(obj_h) + APPROACH_CLEARANCE

# ── Workspace bounds (metres, base_link) ──────────────────────────────────────
WS_X_MIN, WS_X_MAX =  0.150,  0.700
WS_Y_MIN, WS_Y_MAX = -0.250,  0.250
WS_Z_MIN, WS_Z_MAX = -0.200,  0.600

# ── Tray geometry (metres, from CAD sketch) ───────────────────────────────────
TRAY_CENTRE_X =  0.490   # (340+640)/2 = 490mm
TRAY_CENTRE_Y =  0.000   # centred on robot Y axis per sketch
TRAY_Z        = -0.2132  # mat surface = Z_TABLE - TIP_OFFSET

# ── Park position ─────────────────────────────────────────────────────────────
PARK_X, PARK_Y, PARK_Z = 0.40, 0.00, 0.40

# ── Velocity scaling (fraction of max) ───────────────────────────────────────
VEL_TRANSIT = 0.05   # long-range PTP transits (safe height, park)
VEL_NEAR    = 0.03   # approach, contact, retract — anything near the tray

# ── EKI ───────────────────────────────────────────────────────────────────────
KUKA_IP  = "192.168.1.147"
EKI_PORT = 54600


def build_gripper_packet(state: int) -> bytes:
    return (
        b'<RobotCommand><Type>0</Type>'
        b'<Axis A1="0" A2="0" A3="0" A4="0" A5="0" A6="0"/>'
        b'<Cart X="0" Y="0" Z="0" A="0" B="0" C="0"/>'
        b'<Velocity>0.1</Velocity>'
        b'<Gripper>' + str(state).encode() + b'</Gripper>'
        b'</RobotCommand>'
    )


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

        # EKI gripper socket
        self._gripper_sock: socket.socket | None = None
        self._gripper_lock = threading.Lock()
        self._connect_gripper()

        self.get_logger().info('Waiting for MoveGroup action server...')
        self._move_client.wait_for_server()
        self.get_logger().info('Waiting for ApplyPlanningScene service...')
        self._scene_client.wait_for_service()
        self.get_logger().info('Surgical Control Server online.')

    # ── EKI gripper ───────────────────────────────────────────────────────────

    def _connect_gripper(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((KUKA_IP, EKI_PORT))
            sock.settimeout(None)
            self._gripper_sock = sock
            self.get_logger().info(f'Gripper socket connected to {KUKA_IP}:{EKI_PORT}')
        except OSError as e:
            self._gripper_sock = None
            self.get_logger().warn(
                f'Gripper socket failed ({e}). Vacuum will not fire.')

    def _send_gripper(self, state: int):
        if self._gripper_sock is None:
            return
        with self._gripper_lock:
            try:
                self._gripper_sock.sendall(build_gripper_packet(state))
                self.get_logger().info(
                    f'  [VACUUM {"ON" if state else "OFF"}]')
            except OSError as e:
                self.get_logger().error(f'Gripper send failed: {e}')

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

        # Remove world object BEFORE attaching — prevents collision check
        # between the attached instrument and its own world-object copy
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
        """Remove object from world scene before picking it up."""
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

        # Table surface — prevents planning through the table
        # 50mm thick slab sitting below mat surface
        self.add_box('table_surface',
                     (TRAY_CENTRE_X, TRAY_CENTRE_Y, TRAY_Z - 0.025),
                     (1.0, 1.0, 0.050))

        # Instrument tray — 450mm × 300mm mat, centred at (490, 0)
        self.add_box('instrument_tray',
                     (TRAY_CENTRE_X, TRAY_CENTRE_Y, TRAY_Z),
                     (0.450, 0.300, 0.005))

        # Instruments — 60mm apart in Y, centred on tray
        # 150mm long × 20mm wide × 10mm tall (lying flat)
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

        self.get_logger().info(f'  [{planner}] → ({x:.4f}, {y:.4f}, {z:.4f}) m')
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

    def destroy_node(self):
        self._send_gripper(0)
        if self._gripper_sock:
            self._gripper_sock.close()
        super().destroy_node()


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
