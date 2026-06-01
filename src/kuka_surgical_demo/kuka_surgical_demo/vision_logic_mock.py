#!/usr/bin/env python3
"""
Vision & Logic Coordinator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Listens to /voice_command (std_msgs/String).
Maintains a toggle counter per instrument:
  odd call  → tray  → surgeon handoff
  even call → handoff → tray

Builds TaskPickPlace.srv requests and dispatches to /execute_task.
The surgical_control_server adds PICK_CLEARANCE / PLACE_CLEARANCE
above each contact Z automatically, so the coordinates here are
the actual tool0 contact positions.

Coordinate conventions (base_link frame)
  Storage (tray):
    Tray top face z = 0.0025 → instrument top face z = 0.015
    Contact: tool0 2 mm above instrument top → z = 0.017

  Handoff (surgeon's side):
    Desired instrument bottom z = 0.20
    Instrument half-height = 0.010
    → instrument centre z = 0.210
    → tool0 contact z = 0.210  (server moves to this, then clears)

Phase upgrade path
  Vision AI  — replace STORAGE_COORDS / HANDOFF_COORDS with live
               perception output.  Nothing else changes.
  Voice AI   — replace voice_terminal_mock with a real STT node
               publishing to /voice_command.  Nothing else changes.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from surgical_msgs.srv import TaskPickPlace

# ── World maps (Vision AI mock) ───────────────────────────────────
# All coordinates are tool0 contact positions in base_link frame.
# X=0.55 matches the tray in surgical_control_server.setup_surgical_scene().
# Y spacing of 50mm keeps gripper clear of adjacent instruments.

STORAGE_COORDS = {
    # (x, y, tool0_contact_z)
    'scalpel':   (0.55, -0.05, 0.093),
    'forceps':   (0.55,  0.00, 0.093),
    'retractor': (0.55,  0.05, 0.093),
}

HANDOFF_COORDS = {
    # Surgeon's side — spread out for unambiguous placement
    'scalpel':   (0.35,  0.15, 0.286),
    'forceps':   (0.35,  0.25, 0.286),
    'retractor': (0.35,  0.35, 0.286),
}

KNOWN_INSTRUMENTS = set(STORAGE_COORDS.keys())


class VisionLogicCoordinator(Node):

    def __init__(self):
        super().__init__('vision_logic_coordinator')

        self.cb = ReentrantCallbackGroup()

        # Counter per instrument.  0 = currently on tray.
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

    # ── Subscriber ────────────────────────────────────────────────

    def _voice_callback(self, msg: String) -> None:
        tool = msg.data.strip().lower()

        if tool not in KNOWN_INSTRUMENTS:
            self.get_logger().warn(
                f'Unknown instrument: "{tool}".  '
                f'Valid: {sorted(KNOWN_INSTRUMENTS)}')
            return

        if not self._client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(
                '/execute_task service not available — is the '
                'surgical_control_server running?')
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

    # ── Helpers ───────────────────────────────────────────────────

    def _make_pose(self, xyz: tuple) -> Pose:
        p = Pose()
        p.position.x, p.position.y, p.position.z = xyz
        # Tool pointing straight down in base_link frame
        p.orientation.x = 0.0
        p.orientation.y = 0.7071068
        p.orientation.z = 0.0
        p.orientation.w = 0.7071068
        return p

    def _on_done(self, future, tool: str, direction: str) -> None:
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(
                    f'✓ {tool.upper()} {direction} complete')
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
