#!/usr/bin/env python3
"""
Voice Terminal Mock
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reads instrument name from terminal stdin and publishes to
/voice_command (std_msgs/String).

Phase upgrade path:
  - Replace this entire node with a real STT node that publishes
    to the same /voice_command topic. Zero changes elsewhere.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VoiceTerminalMock(Node):

    def __init__(self):
        super().__init__('voice_terminal_mock')
        self._pub = self.create_publisher(String, '/voice_command', 10)
        self.get_logger().info(
            'Voice terminal ready.\n'
            'Commands: scalpel | forceps | retractor | quit')

    def loop(self):
        while rclpy.ok():
            try:
                cmd = input('\n[Surgeon] Instrument: ').strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if cmd in ('quit', 'exit', 'q'):
                break
            if not cmd:
                continue

            msg = String()
            msg.data = cmd
            self._pub.publish(msg)
            self.get_logger().info(f'Published: "{cmd}"')


def main(args=None):
    rclpy.init(args=args)
    node = VoiceTerminalMock()
    try:
        node.loop()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
