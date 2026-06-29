#!/usr/bin/env python3
"""
Vision & Logic Coordinator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Listens to /voice_command (std_msgs/String).
Maintains a toggle counter per instrument:
  odd call  → tray  → surgeon handoff
  even call → handoff → tray

Builds TaskPickPlace.srv requests and dispatches to /execute_task.
The surgical_control_server adds APPROACH_CLEARANCE above each
contact Z automatically, so the coordinates here are the actual
tool0 contact positions.

Coordinate conventions (base_link frame, 2026-06-25 calibration)
──────────────────────────────────────────────────────────────────
  Z_TABLE = +53.50 mm = +0.05350 m  (tip on mat, with gripper)
  INST_H  = +10 mm                   (object height off mat)
  pick_z  = Z_TABLE + INST_H = 0.0635 m

  Storage (tray):
    All instruments at X=0.4163m (tray centre X)
    Y spacing 60mm: scalpel=-0.060, forceps=0.000, retractor=+0.060
    Z contact = pick_z = 0.0635 m

  Handoff (surgeon's side):
    Measured 2026-06-25 (with gripper): X=347.62mm, Y=-769.03mm, Z=112.72mm
    Instruments spaced 60mm apart in Y around measured centre Y=-0.769m.
    Z contact = 0.1127 m (measured handoff height, higher than mat).

Phase upgrade path
  Vision AI  — replace STORAGE_COORDS / HANDOFF_COORDS with live
               perception output. Nothing else changes.
  Voice AI   — replace voice_terminal_mock with a real STT node
               publishing to /voice_command. Nothing else changes.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from surgical_msgs.srv import TaskPickPlace

# ── Calibrated Z contact (matches surgical_control_server constants) ──────────
Z_TABLE  = 0.05350   # m — tool0 Z with tip on mat, gripper mounted
INST_H   = 0.010     # m — object height off mat
PICK_Z   = Z_TABLE + INST_H   # = 0.0635 m

# ── Handoff Z — measured 2026-06-25 with gripper at surgeon side ──────────────
HANDOFF_Z = 0.1127   # m — tool0 Z at surgeon handoff position (higher surface)

# ── Handoff centre — measured 2026-06-25 ──────────────────────────────────────
HANDOFF_X =  0.3476   # m
HANDOFF_Y = -0.7690   # m (surgeon's side, negative Y)

# ── Storage positions (tray) — base_link frame ────────────────────────────────
# X = tray centre (416.3mm), Y = 60mm spacing, Z = tip contact
STORAGE_COORDS = {
    'scalpel':   (0.4163, -0.060, PICK_Z),
    'forceps':   (0.4163,  0.000, PICK_Z),
    'retractor': (0.4163, +0.060, PICK_Z),
}

# ── Handoff positions (surgeon's side) — base_link frame ─────────────────────
# Centre from physical measurement. Instruments spaced 60mm apart in Y.
# Y goes further negative for each instrument to keep them separated.
HANDOFF_COORDS = {
    'scalpel':   (HANDOFF_X, HANDOFF_Y - 0.060, HANDOFF_Z),
    'forceps':   (HANDOFF_X, HANDOFF_Y,          HANDOFF_Z),
    'retractor': (HANDOFF_X, HANDOFF_Y + 0.060,  HANDOFF_Z),
}

KNOWN_INSTRUMENTS = set(STORAGE_COORDS.keys())


class VisionLogicCoordinator(Node):

    def __init__(self):
        super().__init__('vision_logic_coordinator')

        self.cb = ReentrantCallbackGroup()

        # Counter per instrument — 0 = currently on tray
        self._call_count = {name: 0 for name in KNOWN_INSTRUMENTS}

        self._client = self.create_client(
            TaskPickPlace, '/execute_task',
            callback_group=self.cb)

        self.create_subscription(
            String, '/voice_command',
            self._voice_callback, 10)

        self.get_logger().info(
            'Vision/Logic Coordinator online. '
            f'Instruments: {sorted(KNOWN_INSTRUMENTS)}')

    # ── Subscriber ────────────────────────────────────────────────────────────

    def _voice_callback(self, msg: String) -> None:
        tool = msg.data.strip().lower()

        if tool not in KNOWN_INSTRUMENTS:
            self.get_logger().warn(
                f'Unknown instrument: "{tool}". '
                f'Valid: {sorted(KNOWN_INSTRUMENTS)}')
            return

        if not self._client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(
                '/execute_task service not available — '
                'is surgical_control_server running?')
            return

        self._call_count[tool] += 1
        odd_call = (self._call_count[tool] % 2 == 1)

        if odd_call:
            pick_xyz  = STORAGE_COORDS[tool]
            place_xyz = HANDOFF_COORDS[tool]
            direction = 'tray → surgeon'
        else:
            pick_xyz  = HANDOFF_COORDS[tool]
            place_xyz = STORAGE_COORDS[tool]
            direction = 'surgeon → tray'

        self.get_logger().info(
            f'Dispatching {tool.upper()} ({direction})  '
            f'[call #{self._call_count[tool]}]')

        req = TaskPickPlace.Request()
        req.object_id  = tool
        req.pick_pose  = self._make_pose(pick_xyz)
        req.place_pose = self._make_pose(place_xyz)

        future = self._client.call_async(req)
        future.add_done_callback(
            lambda f, t=tool, d=direction: self._on_done(f, t, d))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_pose(self, xyz: tuple) -> Pose:
        p = Pose()
        p.position.x, p.position.y, p.position.z = xyz
        # Orientation matches surgical_control_server ORI
        # (A=174°, B=44.72°, C=175.09° → quat)
        p.orientation.x = 0.0321
        p.orientation.y = 0.9235
        p.orientation.z = 0.0197
        p.orientation.w = 0.3816
        return p

    def _on_done(self, future, tool: str, direction: str) -> None:
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f'✓ {tool.upper()} {direction} complete')
            else:
                self.get_logger().error(
                    f'✗ {tool.upper()} failed: {result.message}')
        except Exception as exc:
            self.get_logger().error(f'Service call exception: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = VisionLogicCoordinator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
