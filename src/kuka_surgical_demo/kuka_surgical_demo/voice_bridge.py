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

Home: joint-space goal via MoveIt JointConstraints
  → MoveIt plans + executes in sim (RViz stays in sync)
  → extract final joints from result
  → send PTP to KRC4 via EKI

Stop:
  1. cancel flag → _spin_future() exits
  2. goal_handle.cancel_goal_async() → MoveIt cancels Pilz traj
  3. read current state via EkiStateClient → freeze PTP to KRC4

Park: MoveIt goal + EKI PTP (sim + hardware stay in sync)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import math
import os
import threading
import time

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
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String

from kuka_eki.eki import EkiMotionClient, EkiStateClient
from kuka_eki.krl import Axis

# ── Constants ─────────────────────────────────────────────────────
FRAME   = 'base_link'
TIP     = 'tool0'
ORI     = dict(qx=0.0, qy=0.7071068, qz=0.0, qw=0.7071068)

KUKA_IP         = "192.168.1.147"
EKI_VEL         = 0.4
EKI_RETRY_COUNT = 3
EKI_RETRY_DELAY = 2.0

X_REACH = 0.70
Z_HI, Z_MID, Z_LO = 0.85, 0.55, 0.25
Y_R, Y_C, Y_L     = -0.35, 0.00, 0.35

# Home position in joint space (degrees → radians for MoveIt)
# a1=0, a2=-105, a3=156, a4=0, a5=39, a6=0
HOME_JOINTS_DEG = [0.0, -105.0, 156.0, 0.0, 39.0, 0.0]
HOME_JOINTS_RAD = [math.radians(d) for d in HOME_JOINTS_DEG]

# Joint names must match your URDF exactly
JOINT_NAMES = [
    'joint_1', 'joint_2', 'joint_3',
    'joint_4', 'joint_5', 'joint_6',
]

# EKI home axis — used only as fallback if joint extraction fails
HOME_AXIS = Axis(a1=0.0, a2=-105.0, a3=156.0, a4=0.0, a5=39.0, a6=0.0)

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
    'for': '4', 'main': '9', 'the': '3',
}

STOP_WORDS = {'stop', 'cancel', 'halt', 'abort'}
PARK       = (0.40, 0.00, 0.40)
PARK_AXIS  = Axis(a1=0.0, a2=-90.0, a3=90.0, a4=0.0, a5=90.0, a6=0.0)
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


def _build_cartesian_goal(x, y, z, vel: float) -> MoveGroup.Goal:
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


def _build_joint_goal(
        joint_names: list,
        joint_angles_rad: list,
        vel: float) -> MoveGroup.Goal:
    """
    Build a MoveGroup goal from joint-space targets (radians).
    Uses JointConstraint with a tight tolerance so Pilz treats
    this as an exact joint-space PTP — no IK needed.
    """
    req = MotionPlanRequest()
    req.group_name                      = 'manipulator'
    req.planner_id                      = 'PTP'
    req.pipeline_id                     = 'pilz_industrial_motion_planner'
    req.num_planning_attempts           = 5
    req.allowed_planning_time           = 10.0
    req.max_velocity_scaling_factor     = vel
    req.max_acceleration_scaling_factor = vel

    jc_list = []
    for name, angle_rad in zip(joint_names, joint_angles_rad):
        jc = JointConstraint()
        jc.joint_name        = name
        jc.position          = angle_rad
        jc.tolerance_above   = 0.001   # ~0.057 degrees
        jc.tolerance_below   = 0.001
        jc.weight            = 1.0
        jc_list.append(jc)

    gc = Constraints()
    gc.joint_constraints = jc_list
    req.goal_constraints = [gc]

    goal = MoveGroup.Goal()
    goal.request                    = req
    goal.planning_options.plan_only = False
    goal.planning_options.replan    = False
    return goal


def _extract_joint_angles(result_resp) -> Axis | None:
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
            if   'joint_1' in name: angles[0] = deg
            elif 'joint_2' in name: angles[1] = deg
            elif 'joint_3' in name: angles[2] = deg
            elif 'joint_4' in name: angles[3] = deg
            elif 'joint_5' in name: angles[4] = deg
            elif 'joint_6' in name: angles[5] = deg

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

        self._move_client = ActionClient(
            self, MoveGroup, '/move_action',
            callback_group=self._cb)

        self._goal_handle      = None
        self._goal_handle_lock = threading.Lock()

        self._eki       = None
        self._eki_state = None
        self._eki_lock  = threading.Lock()
        self._connect_eki()

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
            '  Say: home  — return to home position\n'
            '  Say: stop / cancel  — abort and freeze arm')

    # ── EKI connection + reconnect ────────────────────────────────

    def _connect_eki(self) -> bool:
        try:
            self._eki = EkiMotionClient(KUKA_IP)
            self._eki.connect()
            self._eki_state = EkiStateClient(KUKA_IP)
            self._eki_state.connect()
            self.get_logger().info('EKI motion + state connected.')
            return True
        except Exception as e:
            self.get_logger().warn(
                f'EKI connection failed: {e}\n'
                'Running in simulation-only mode.')
            self._eki       = None
            self._eki_state = None
            return False

    def _reconnect_eki(self) -> bool:
        with self._eki_lock:
            for attempt in range(1, EKI_RETRY_COUNT + 1):
                self.get_logger().warn(
                    f'EKI reconnect attempt {attempt}/{EKI_RETRY_COUNT}...')
                try:
                    if self._eki is not None:
                        self._eki._tcp_client._socket.close()
                    if self._eki_state is not None:
                        self._eki_state._tcp_client._socket.close()
                except Exception:
                    pass
                self._eki       = None
                self._eki_state = None
                time.sleep(EKI_RETRY_DELAY)
                if self._connect_eki():
                    self.get_logger().info('EKI reconnected successfully.')
                    return True
            self.get_logger().error(
                f'EKI reconnect failed after {EKI_RETRY_COUNT} attempts.')
            return False

    # ── EKI send with auto-reconnect ──────────────────────────────

    def _send_eki(self, axis: Axis) -> bool:
        with self._eki_lock:
            if self._eki is None:
                self.get_logger().warn(
                    'EKI not connected — skipping hardware command.')
                return False
            try:
                self._eki.ptp(axis, max_velocity_scaling=EKI_VEL)
                self.get_logger().info(f'EKI PTP sent: {axis}')
                return True
            except Exception as e:
                self.get_logger().error(f'EKI transmission failed: {e}')

        if self._reconnect_eki():
            with self._eki_lock:
                try:
                    self._eki.ptp(axis, max_velocity_scaling=EKI_VEL)
                    self.get_logger().info(
                        f'EKI PTP sent after reconnect: {axis}')
                    return True
                except Exception as e:
                    self.get_logger().error(
                        f'EKI transmission failed after reconnect: {e}')
        return False

    def _stop_eki(self):
        with self._eki_lock:
            if self._eki is None or self._eki_state is None:
                self.get_logger().warn(
                    'EKI not connected — cannot send hardware stop.')
                return
            try:
                state = self._eki_state.state()
                current = Axis(
                    a1=float(state.axis.a1),
                    a2=float(state.axis.a2),
                    a3=float(state.axis.a3),
                    a4=float(state.axis.a4),
                    a5=float(state.axis.a5),
                    a6=float(state.axis.a6),
                )
                self.get_logger().info(
                    f'STOP: freeze PTP to current pose: {current}')
                self._eki.ptp(current, max_velocity_scaling=0.05)
            except Exception as e:
                self.get_logger().error(f'EKI stop failed: {e}')

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
        self._rec = KaldiRecognizer(
            Model(model_path), 16000,
            '["one", "two", "three", "four", "five", "six", '
            '"seven", "eight", "nine", "stop", "cancel", '
            '"halt", "abort", "home", "[unk]"]')
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
                self._do_stop()
                return

        for w in words:
            if w == 'home':
                if not self._busy.acquire(blocking=False):
                    self.get_logger().warn('Busy — ignoring "home".')
                    return
                threading.Thread(
                    target=self._execute_home,
                    daemon=True
                ).start()
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

    # ── Stop ──────────────────────────────────────────────────────

    def _do_stop(self):
        self._cancel_flag.set()

        with self._goal_handle_lock:
            gh = self._goal_handle
        if gh is not None:
            self.get_logger().warn('Cancelling MoveIt goal...')
            try:
                cancel_future = gh.cancel_goal_async()
                rclpy.spin_until_future_complete(
                    self, cancel_future, timeout_sec=1.0)
                self.get_logger().info('MoveIt goal cancel sent.')
            except Exception as e:
                self.get_logger().error(f'MoveIt cancel failed: {e}')

        self._stop_eki()

    # ── Motion execution ──────────────────────────────────────────

    def _execute_move(self, digit: str):
        try:
            self._cancel_flag.clear()
            x, y, z = GRID_POSITIONS[digit]
            self.get_logger().info(
                f'Moving to {digit}  ({x:.2f}, {y:.2f}, {z:.2f})')

            result_resp = self._send_moveit_goal(
                _build_cartesian_goal(x, y, z, PLAN_VEL))

            if result_resp is None:
                self.get_logger().error(f'Motion to {digit} failed.')
                return

            self.get_logger().info(f'MoveIt: position {digit} reached.')

            axis = _extract_joint_angles(result_resp)
            if axis is not None:
                self._send_eki(axis)
            else:
                self.get_logger().warn(
                    'Could not extract joint angles — hardware command skipped.')
        finally:
            with self._goal_handle_lock:
                self._goal_handle = None
            self._busy.release()

    def _execute_home(self):
        """
        Route home through MoveIt using joint constraints so that:
        - Pilz plans a valid PTP trajectory from current pose
        - RViz model moves and stays in sync
        - Final joint angles are extracted and sent to KRC4 via EKI
        """
        try:
            self._cancel_flag.clear()
            self.get_logger().info('Moving to HOME (via MoveIt joint goal)...')

            result_resp = self._send_moveit_goal(
                _build_joint_goal(JOINT_NAMES, HOME_JOINTS_RAD, PLAN_VEL))

            if result_resp is None:
                self.get_logger().error('Home motion failed.')
                return

            self.get_logger().info('MoveIt: HOME reached.')

            axis = _extract_joint_angles(result_resp)
            if axis is not None:
                self._send_eki(axis)
            else:
                # Fallback — send known home angles directly
                self.get_logger().warn(
                    'Joint extraction failed — sending HOME_AXIS directly.')
                self._send_eki(HOME_AXIS)

        finally:
            with self._goal_handle_lock:
                self._goal_handle = None
            self._busy.release()

    def _send_moveit_goal(self, goal: MoveGroup.Goal):
        """
        Send any MoveGroup goal, block until result or cancel.
        Returns full result response on success, None on failure.
        Stores goal_handle for stop cancellation.
        """
        send_future = self._move_client.send_goal_async(goal)
        if not self._spin_future(send_future):
            return None

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Goal REJECTED by MoveGroup.')
            return None

        with self._goal_handle_lock:
            self._goal_handle = goal_handle

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
        self.get_logger().info('Parking sim...')
        self._cancel_flag.clear()
        self._send_moveit_goal(
            _build_cartesian_goal(*PARK, PARK_VEL))
        self.get_logger().info('Parking hardware...')
        self._send_eki(PARK_AXIS)


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
