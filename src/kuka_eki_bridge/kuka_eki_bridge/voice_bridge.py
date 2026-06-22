#!/usr/bin/env python3
"""
Voice Grid Controller — KR6 R900 sixx (with embedded EKI client)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Grid layout (XZ vertical plane, Y lateral):
  1(hi-R)  2(hi-C)  3(hi-L)     Z=0.85m
  4(mi-R)  5(mi-C)  6(mi-L)     Z=0.55m
  7(lo-R)  8(lo-C)  9(lo-L)     Z=0.25m
  col:  Y=-0.35   Y=0.00   Y=+0.35
  all:  X=0.70m fixed

Flow:
  voice → /move_action (MoveIt plans + executes in sim)
        → on success, extract final joint angles from result
            → send PTP directly to KRC4 via EKI TCP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import math
import os
import threading

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

import sounddevice as sd
from vosk import Model, KaldiRecognizer

from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String

from kuka_eki.eki import EkiMotionClient
from kuka_eki.krl import Axis

# ── Constants ─────────────────────────────────────────────────────
FRAME   = 'base_link'
TIP     = 'tool0'
ORI     = dict(qx=0.0, qy=0.7071068, qz=0.0, qw=0.7071068)

KUKA_IP = "192.168.1.147"
EKI_VEL = 0.05          # 5% max velocity on hardware — keep this low

X_REACH = 0.70
Z_HI, Z_MID, Z_LO = 0.85, 0.55, 0.25
Y_R, Y_C, Y_L     = -0.35, 0.00, 0.35

GRID_POSITIONS = {
    '1': (X_REACH, Y_R, Z_HI),
    '2': (X_REACH, Y_C, Z_HI),
    '3': (X_REACH, Y_L, Z_HI),
    '4': (X_REACH, Y_R, Z_MID),
    '5': (X_REACH, Y_C, Z_MID),
    '6': (X_REACH, Y_L, Z_MID),
    '7': (X_REACH, Y_R, Z_LO),
    '8': (X_REACH, Y_C, Z_LO),
    '9': (X_REACH, Y_L, Z_LO),
}

WORD_TO_DIGIT = {
    'one': '1', 'two': '2', 'three': '3',
    'four': '4', 'five': '5', 'six': '6',
    'seven': '7', 'eight': '8', 'nine': '9',
}

STOP_WORDS = {'stop', 'cancel', 'halt', 'abort'}
PARK       = (0.40, 0.00, 0.40)
PLAN_VEL   = 0.10
PARK_VEL   = 0.08


# ── Helpers ───────────────────────────────────────────────────────

def _make_pose(x, y, z) -> Pose:
    p = Pose()
    p.position.x = float(x)
    p.position.y = float(y)
    p.position.z = float(z)
    p.orientation.x = ORI['qx']
    p.orientation.y = ORI['qy']
    p.orientation.z = ORI['qz']
    p.orientation.w = ORI['qw']
    return p


def _build_goal(x, y, z, vel: float) -> MoveGroup.Goal:
    target = _make_pose(x, y, z)

    req = MotionPlanRequest()
    req.group_name                      = 'manipulator'
    req.planner_id                      = 'PTP'
    req.pipeline_id                     = 'pilz_industrial_motion_planner'
    req.num_planning_attempts           = 5
    req.allowed_planning_time           = 10.0
    req.max_velocity_scaling_factor     = vel
    req.max_acceleration_scaling_factor = vel

    sphere = SolidPrimitive()
    sphere.type       = SolidPrimitive.SPHERE
    sphere.dimensions = [0.002]

    bv = BoundingVolume()
    bv.primitives      = [sphere]
    bv.primitive_poses = [target]

    pos = PositionConstraint()
    pos.header.frame_id   = FRAME
    pos.link_name         = TIP
    pos.constraint_region = bv
    pos.weight            = 1.0

    ori = OrientationConstraint()
    ori.header.frame_id           = FRAME
    ori.link_name                 = TIP
    ori.orientation               = target.orientation
    ori.absolute_x_axis_tolerance = 0.05
    ori.absolute_y_axis_tolerance = 0.05
    ori.absolute_z_axis_tolerance = 0.05
    ori.weight                    = 1.0

    gc = Constraints()
    gc.position_constraints    = [pos]
    gc.orientation_constraints = [ori]
    req.goal_constraints       = [gc]

    goal = MoveGroup.Goal()
    goal.request                    = req
    goal.planning_options.plan_only = False
    goal.planning_options.replan    = False
    return goal


def _extract_joint_angles(result_resp) -> Axis | None:
    """
    Pull the final joint angles (degrees) from a MoveGroup result.
    Returns an Axis, or None if the trajectory is missing.
    """
    try:
        traj = result_resp.result.planned_trajectory
        jt   = traj.joint_trajectory
        if not jt.points:
            return None

        final  = jt.points[-1]
        names  = jt.joint_names
        angles = [0.0] * 6

        for i, name in enumerate(names):
            deg = math.degrees(final.positions[i])
            if   "joint_1" in name: angles[0] = deg
            elif "joint_2" in name: angles[1] = deg
            elif "joint_3" in name: angles[2] = deg
            elif "joint_4" in name: angles[3] = deg
            elif "joint_5" in name: angles[4] = deg
            elif "joint_6" in name: angles[5] = deg

        return Axis(
            a1=angles[0], a2=angles[1], a3=angles[2],
            a4=angles[3], a5=angles[4], a6=angles[5]
        )
    except Exception:
        return None


# ── Node ──────────────────────────────────────────────────────────

class VoiceGridController(Node):

    def __init__(self):
        super().__init__('voice_grid_controller')
        self._cb = ReentrantCallbackGroup()

        # MoveIt action client
        self._move_client = ActionClient(
            self, MoveGroup, '/move_action',
            callback_group=self._cb)

        # EKI client — connect to physical hardware
        self._eki: EkiMotionClient | None = None
        self._connect_eki()

        # /voice_command subscription (also accepts external publishers)
        self.create_subscription(
            String, '/voice_command',
            self._voice_cb, 10,
            callback_group=self._cb)

        self._busy        = threading.Lock()
        self._cancel_flag = threading.Event()

        self.get_logger().info('Waiting for MoveGroup action server...')
        self._move_client.wait_for_server()
        self.get_logger().info('MoveGroup ready.')

        self._start_vosk()

        self.get_logger().info(
            'Voice Grid Controller online.\n'
            '  Say: one two three four five six seven eight nine\n'
            '  Say: stop / cancel  to abort')

    # ── EKI ───────────────────────────────────────────────────────

    def _connect_eki(self):
        self.get_logger().info(f'Connecting to KRC4 at {KUKA_IP}...')
        try:
            self._eki = EkiMotionClient(KUKA_IP)
            self._eki.connect()
            self.get_logger().info('EKI connection established.')
        except Exception as e:
            self.get_logger().warn(
                f'EKI connection failed: {e}\n'
                'Running in simulation-only mode — '
                'MoveIt will plan but no commands sent to hardware.')
            self._eki = None

    def _send_eki(self, axis: Axis):
        if self._eki is None:
            self.get_logger().warn(
                'EKI not connected — skipping hardware command.')
            return
        try:
            self._eki.ptp(axis, max_velocity_scaling=EKI_VEL)
            self.get_logger().info(f'EKI PTP sent: {axis}')
        except Exception as e:
            self.get_logger().error(f'EKI transmission failed: {e}')

    # ── Vosk STT ──────────────────────────────────────────────────

    def _start_vosk(self):
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'vosk-model-small-en-us')
        if not os.path.exists(model_path):
            self.get_logger().warn(
                f'Vosk model not found at {model_path}. '
                'STT disabled — publish to /voice_command manually.')
            return
        self._rec    = KaldiRecognizer(Model(model_path), 16000)
        self._stream = sd.RawInputStream(
            samplerate=16000, blocksize=8000,
            dtype='int16', channels=1,
            callback=self._audio_cb)
        self._stream.start()
        self.get_logger().info('Vosk STT listening...')

    def _audio_cb(self, indata, frames, time_info, status):
        if self._rec.AcceptWaveform(bytes(indata)):
            text = json.loads(self._rec.Result()).get('text', '').strip()
            if text:
                self.get_logger().info(f"Heard: '{text}'")
                self._process_text(text)
        else:
            pt = json.loads(self._rec.PartialResult()).get('partial', '')
            if pt:
                self.get_logger().info(
                    f"Partial: '{pt}'", throttle_duration_sec=1.0)

    def _process_text(self, text: str):
        msg      = String()
        msg.data = text
        self._voice_cb(msg)

    # ── Voice callback ────────────────────────────────────────────

    def _voice_cb(self, msg: String):
        words = msg.data.strip().lower().split()

        for w in words:
            if w in STOP_WORDS:
                self.get_logger().warn('STOP commanded.')
                self._cancel_flag.set()
                return

        digit = None
        for w in words:
            if w in WORD_TO_DIGIT:
                digit = WORD_TO_DIGIT[w]
                break
            if w in GRID_POSITIONS:
                digit = w
                break

        if digit is None:
            self.get_logger().info(f"No grid command in: '{msg.data}'")
            return

        if not self._busy.acquire(blocking=False):
            self.get_logger().warn(
                f'Busy — ignoring "{digit}". Say "stop" to cancel.')
            return

        threading.Thread(
            target=self._execute_move,
            args=(digit,),
            daemon=True
        ).start()

    # ── Motion execution ──────────────────────────────────────────

    def _execute_move(self, digit: str):
        try:
            self._cancel_flag.clear()
            x, y, z = GRID_POSITIONS[digit]
            self.get_logger().info(
                f'Moving to {digit}  ({x:.2f}, {y:.2f}, {z:.2f})')

            result_resp = self._send_goal(x, y, z, PLAN_VEL)

            if result_resp is None:
                self.get_logger().error(f'Motion to {digit} failed.')
                return

            self.get_logger().info(f'MoveIt: position {digit} reached.')

            # Extract joint angles from MoveIt result and send to KRC4
            axis = _extract_joint_angles(result_resp)
            if axis is not None:
                self._send_eki(axis)
            else:
                self.get_logger().warn(
                    'Could not extract joint angles from MoveIt result — '
                    'hardware command skipped.')

        finally:
            self._busy.release()

    def _send_goal(self, x, y, z, vel: float):
        """
        Send MoveGroup goal, block until result or cancel.
        Returns the full result response on success, None on failure.
        """
        goal = _build_goal(x, y, z, vel)

        send_future = self._move_client.send_goal_async(goal)
        if not self._spin_future(send_future):
            return None

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Goal REJECTED by MoveGroup.')
            return None

        self.get_logger().info('Goal accepted — waiting for result...')

        result_future = goal_handle.get_result_async()
        if not self._spin_future(result_future):
            return None

        result_resp = result_future.result()
        code = result_resp.result.error_code.val
        if code != 1:
            self.get_logger().error(f'MoveGroup error_code={code}')
            return None

        return result_resp

    def _spin_future(self, future, timeout_sec: float = 0.1) -> bool:
        while not future.done():
            if self._cancel_flag.is_set():
                self.get_logger().warn('Motion cancelled by user.')
                return False
            rclpy.spin_until_future_complete(
                self, future, timeout_sec=timeout_sec)
        return True

    # ── Park ──────────────────────────────────────────────────────

    def park(self):
        self.get_logger().info('Parking...')
        self._cancel_flag.clear()
        self._send_goal(*PARK, PARK_VEL)


# ── Entry point ───────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = VoiceGridController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.park()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
