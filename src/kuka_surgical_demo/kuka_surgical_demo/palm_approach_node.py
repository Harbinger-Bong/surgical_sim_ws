"""
palm_approach_node.py  (ROS2 version)
──────────────────────────────────────
Detects palm via MediaPipe, publishes target XY to /palm_target.
bridge_node.py receives it and sends PTP over its persistent EKI socket.

Pipeline:
  palm_approach_node  →  /palm_target (PointStamped)  →  bridge_node  →  EKI  →  KRC4

Usage:
  # Terminal 1: bridge_node running (persistent EKI connection)
  ros2 run kuka_eki_bridge bridge_node

  # Terminal 2: palm detection
  python3 palm_approach_node.py --camera 2
"""

import sys
import math
import time
import argparse
import urllib.request
import os

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header

HOVER_X       =  294.24
HOVER_Y       =   86.15
Z_TABLE       = -183.18
Z_TARGET      = Z_TABLE + 20.0

CAM_HEIGHT_MM = 634.23
SCALE_MM_PX   = 2.0 * CAM_HEIGHT_MM * math.tan(math.radians(30.0)) / 640.0

WS_X_MIN, WS_X_MAX = 150.0, 500.0
WS_Y_MIN, WS_Y_MAX = -50.0, 250.0

MODEL_PATH = os.path.expanduser("~/hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model(path):
    if os.path.exists(path):
        return path
    print(f"[palm_approach] Downloading model → {path}")
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def build_landmarker(model_path):
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def detect_palm(frame_bgr, landmarker, frame_ms):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_img, frame_ms)
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    cx = int(((lm[0].x + lm[9].x) / 2) * w)
    cy = int(((lm[0].y + lm[9].y) / 2) * h)
    return (cx, cy)


def px_to_base_frame(dx_px, dy_px, scale):
    delta_x =  dy_px * scale
    delta_y = -dx_px * scale
    return delta_x, delta_y


def clamp_ws(x, y):
    xc = max(WS_X_MIN, min(WS_X_MAX, x))
    yc = max(WS_Y_MIN, min(WS_Y_MAX, y))
    return xc, yc, (xc == x and yc == y)


class PalmApproachNode(Node):
    def __init__(self, camera_idx, n_frames):
        super().__init__('palm_approach_node')

        self.publisher = self.create_publisher(PointStamped, '/palm_target', 10)
        self.get_logger().info("Publishing to /palm_target")

        self.cap = cv2.VideoCapture(camera_idx)
        if not self.cap.isOpened():
            self.get_logger().fatal(f"Cannot open camera {camera_idx}")
            raise RuntimeError("Camera open failed")

        self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.icx = self.img_w // 2
        self.icy = self.img_h // 2
        self.scale = 2.0 * CAM_HEIGHT_MM * math.tan(math.radians(30.0)) / self.img_w
        self.get_logger().info(
            f"Camera {camera_idx}: {self.img_w}x{self.img_h}  scale={self.scale:.3f} mm/px"
        )

        ensure_model(MODEL_PATH)
        self.landmarker = build_landmarker(MODEL_PATH)
        self.get_logger().info("MediaPipe ready. Place hand on mat...")

        self.n_frames = n_frames
        self.frame_ms = 0
        self.timer = self.create_timer(0.1, self._run_detection)

    def _run_detection(self):
        self.timer.cancel()

        detections = []
        attempts   = 0

        while len(detections) < self.n_frames and attempts < self.n_frames * 6:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.033)
                continue
            self.frame_ms += 33
            attempts += 1

            pt = detect_palm(frame, self.landmarker, self.frame_ms)
            if pt:
                detections.append(pt)
                dx, dy = pt[0] - self.icx, pt[1] - self.icy
                self.get_logger().info(
                    f"[{len(detections):2d}/{self.n_frames}] "
                    f"palm=({pt[0]},{pt[1]}) offset=({dx:+d},{dy:+d})px"
                )
            time.sleep(0.033)

        self.cap.release()
        self.landmarker.close()

        if len(detections) < 5:
            self.get_logger().error("Not enough detections. Aborting.")
            rclpy.shutdown()
            return

        avg_cx = sum(d[0] for d in detections) / len(detections)
        avg_cy = sum(d[1] for d in detections) / len(detections)
        dx_px  = avg_cx - self.icx
        dy_px  = avg_cy - self.icy

        ddx, ddy = px_to_base_frame(dx_px, dy_px, self.scale)
        target_x = HOVER_X + ddx
        target_y = HOVER_Y + ddy
        target_x, target_y, in_ws = clamp_ws(target_x, target_y)

        if not in_ws:
            self.get_logger().warn("Target clamped to workspace bounds.")

        self.get_logger().info(
            f"Target: X={target_x:.2f}  Y={target_y:.2f}  Z={Z_TARGET:.2f} mm"
        )

        msg = PointStamped()
        msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id='base')
        msg.point.x = target_x
        msg.point.y = target_y
        msg.point.z = Z_TARGET

        self.publisher.publish(msg)
        self.get_logger().info("Published /palm_target → bridge_node sending PTP.")

        time.sleep(0.5)
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--frames", type=int, default=15)
    args = parser.parse_args()

    rclpy.init()
    try:
        node = PalmApproachNode(args.camera, args.frames)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"[palm_approach] Fatal: {e}")


if __name__ == "__main__":
    main()
