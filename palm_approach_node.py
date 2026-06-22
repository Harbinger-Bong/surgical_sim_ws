"""
palm_approach_node.py
─────────────────────
Single-shot palm detection → PTP move to 20mm above palm.

Pipeline:
  1. Robot sits at detection pose (Z = 416.05mm, ~599mm above mat)
  2. Node captures frames, detects palm centre via MediaPipe
  3. Converts pixel offset → base-frame XY displacement
  4. Sends ONE PTP to (X_target, Y_target, Z_table+20mm)
     preserving the detection pose orientation
  5. Done — robot holds 20mm above palm

Calibrated constants (SmartPad, 2026-06-17, pose 3):
  Z_table   = -183.18 mm  (suction tip touching mat)
  Z_hover   =  416.05 mm  (detection pose TCP height)
  Z_target  = -163.18 mm  (20mm above mat)
  Hover XY  = (294.24, 86.15) mm  base-frame TCP during detection
  Orientation: A=174.91°  B=38.78°  C=176.81°

  Camera: Logitech C270, 60° HFOV, ~35mm above TCP
  Camera-to-mat = 416.05 - (-183.18) + 35 = 634mm
  Scale = 2 × 634 × tan(30°) / 640 = 1.146 mm/px

  Workspace safety bounds (base frame mm):
  X: 150–500,  Y: -50–250

Usage:
  source ~/surgical_sim_ws/install/setup.bash
  python3 palm_approach_node.py --dry-run   # detection only, no motion
  python3 palm_approach_node.py             # live
  python3 palm_approach_node.py --camera 1  # if Logitech not on index 0
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

try:
    from kuka_eki.eki import EkiMotionClient
    from kuka_eki.krl import Pos
    EKI_AVAILABLE = True
except ImportError:
    EKI_AVAILABLE = False

# ── Calibrated constants ──────────────────────────────────────────────────────
KUKA_IP       = "192.168.1.147"

Z_TABLE       = -183.18   # mm  suction tip on mat
Z_HOVER       =  416.05   # mm  TCP height at detection pose
Z_TARGET      = Z_TABLE + 20.0   # = -163.18 mm

HOVER_X       =  294.24   # mm  base-frame TCP X during detection
HOVER_Y       =   86.15   # mm  base-frame TCP Y during detection

ORI_A         =  174.91   # deg
ORI_B         =   38.78   # deg
ORI_C         =  176.81   # deg

# Camera height above mat = Z_hover - Z_table + lens_offset
# = 416.05 - (-183.18) + 35 = 634.23 mm
CAM_HEIGHT_MM = 634.23
# C270 HFOV = 60°, half-angle = 30°
# At 640px width: scale = 2 * h * tan(30°) / 640
SCALE_MM_PX   = 2.0 * CAM_HEIGHT_MM * math.tan(math.radians(30.0)) / 640.0
# ≈ 1.146 mm/px

# Workspace safety bounds
WS_X_MIN, WS_X_MAX = 150.0, 500.0
WS_Y_MIN, WS_Y_MAX = -50.0, 250.0

# MediaPipe model
MODEL_PATH = os.path.expanduser("~/hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_model(path):
    if os.path.exists(path):
        return path
    print(f"[palm_approach] Downloading MediaPipe model → {path}")
    urllib.request.urlretrieve(MODEL_URL, path)
    print("[palm_approach] Download complete.")
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
    """Returns (cx_px, cy_px) palm centre or None."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_img, frame_ms)
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    # Palm centre = midpoint wrist (0) and middle MCP (9)
    cx = int(((lm[0].x + lm[9].x) / 2) * w)
    cy = int(((lm[0].y + lm[9].y) / 2) * h)
    return (cx, cy)


def px_to_base_frame(dx_px, dy_px):
    """
    Pixel offset (from image centre) → base-frame mm displacement.

    At this pose (A1≈0°, arm reaching forward over mat, camera ~down):
      image right (+dx) → base frame  -Y
      image down  (+dy) → base frame  +X  (further from robot base)

    Flip signs here if robot moves wrong direction on first test.
    """
    delta_x = -dy_px * SCALE_MM_PX
    delta_y = -dx_px * SCALE_MM_PX
    return delta_x, delta_y


def clamp_ws(x, y):
    xc = max(WS_X_MIN, min(WS_X_MAX, x))
    yc = max(WS_Y_MIN, min(WS_Y_MAX, y))
    return xc, yc, (xc == x and yc == y)


def draw_debug(frame, cx, cy, icx, icy, target):
    vis = frame.copy()
    h, w = vis.shape[:2]
    cv2.line(vis, (icx, 0), (icx, h), (255, 60, 60), 1)
    cv2.line(vis, (0, icy), (w, icy), (255, 60, 60), 1)
    if cx is not None:
        cv2.circle(vis, (cx, cy), 10, (0, 255, 0), -1)
        cv2.line(vis, (icx, icy), (cx, cy), (0, 255, 255), 2)
        dx, dy = cx - icx, cy - icy
        cv2.putText(vis, f"dx={dx:+d}px  dy={dy:+d}px",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if target:
            cv2.putText(vis,
                f"Target  X={target[0]:.1f}  Y={target[1]:.1f}  Z={target[2]:.1f} mm",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2)
        cv2.putText(vis, f"scale={SCALE_MM_PX:.3f} mm/px",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)
    else:
        cv2.putText(vis, "NO HAND", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return vis


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--camera",     type=int, default=0)
    parser.add_argument("--frames",     type=int, default=15,
                        help="Detection frames to average (default 15)")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    global SCALE_MM_PX

    print(f"[palm_approach] Scale factor: {SCALE_MM_PX:.3f} mm/px")
    print(f"[palm_approach] Z_target: {Z_TARGET:.2f} mm  (20mm above mat)")

    # ── EKI ──────────────────────────────────────────────────────────────────
    motion_client = None
    if not args.dry_run:
        if not EKI_AVAILABLE:
            print("[ERROR] kuka_eki not importable.")
            print("        source ~/surgical_sim_ws/install/setup.bash first.")
            sys.exit(1)
        print(f"[palm_approach] Connecting EKI → {KUKA_IP} ...")
        motion_client = EkiMotionClient(KUKA_IP)
        motion_client.connect()
        print("[palm_approach] EKI connected.")
    else:
        print("[palm_approach] DRY RUN — no robot motion.")

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    ensure_model(MODEL_PATH)
    landmarker = build_landmarker(MODEL_PATH)
    print("[palm_approach] MediaPipe ready.")

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        sys.exit(1)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    icx, icy = img_w // 2, img_h // 2
    print(f"[palm_approach] Camera {args.camera}: {img_w}×{img_h}")

    # Recompute scale if camera is not 640px wide
    if img_w != 640:
        SCALE_MM_PX = 2.0 * CAM_HEIGHT_MM * math.tan(math.radians(30.0)) / img_w
        print(f"[palm_approach] Scale adjusted for {img_w}px: {SCALE_MM_PX:.3f} mm/px")

    # ── Detection loop ────────────────────────────────────────────────────────
    print(f"\n[palm_approach] Place hand flat on mat under the camera.")
    print(f"[palm_approach] Collecting {args.frames} detections...\n")

    detections = []
    frame_ms   = 0
    attempts   = 0
    last_frame = None
    last_cx = last_cy = None

    while len(detections) < args.frames and attempts < args.frames * 6:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.033)
            continue
        frame_ms += 33
        attempts += 1
        last_frame = frame.copy()

        pt = detect_palm(frame, landmarker, frame_ms)
        if pt:
            last_cx, last_cy = pt
            detections.append(pt)
            dx, dy = pt[0] - icx, pt[1] - icy
            print(f"  [{len(detections):2d}/{args.frames}] palm=({pt[0]},{pt[1]})  "
                  f"offset=({dx:+d},{dy:+d})px")
        else:
            print(f"  [frame {attempts:3d}] no hand detected")

        if not args.no_display and last_frame is not None:
            vis = draw_debug(last_frame, last_cx, last_cy, icx, icy, None)
            cv2.imshow("palm_approach — detecting", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Aborted.")
                cap.release(); cv2.destroyAllWindows(); landmarker.close()
                return
        time.sleep(0.033)

    cap.release()
    landmarker.close()

    if len(detections) < 5:
        print(f"\n[ERROR] Only {len(detections)} detections — hand not visible.")
        print("  Try: move hand to centre of mat, ensure good lighting.")
        print("  Try: --camera 1  if wrong camera is selected.")
        cv2.destroyAllWindows()
        return

    # ── Average ───────────────────────────────────────────────────────────────
    avg_cx = sum(d[0] for d in detections) / len(detections)
    avg_cy = sum(d[1] for d in detections) / len(detections)
    dx_px  = avg_cx - icx
    dy_px  = avg_cy - icy
    print(f"\n[palm_approach] Averaged over {len(detections)} frames:")
    print(f"  Palm pixel:  ({avg_cx:.1f}, {avg_cy:.1f})")
    print(f"  Px offset:   dx={dx_px:+.1f}  dy={dy_px:+.1f}")

    # ── Pixel → base frame ────────────────────────────────────────────────────
    ddx, ddy = px_to_base_frame(dx_px, dy_px)
    target_x = HOVER_X + ddx
    target_y = HOVER_Y + ddy
    target_z = Z_TARGET

    print(f"  Base Δ:      ΔX={ddx:+.1f}mm  ΔY={ddy:+.1f}mm")

    target_x, target_y, in_ws = clamp_ws(target_x, target_y)
    if not in_ws:
        print(f"  [WARN] Target clamped to workspace bounds.")

    print(f"\n[palm_approach] ════════════════════════════════════")
    print(f"  Target X = {target_x:.2f} mm")
    print(f"  Target Y = {target_y:.2f} mm")
    print(f"  Target Z = {target_z:.2f} mm  ← 20mm above mat")
    print(f"  Ori  A={ORI_A}°  B={ORI_B}°  C={ORI_C}°")
    print(f"[palm_approach] ════════════════════════════════════")

    # ── Final debug frame ─────────────────────────────────────────────────────
    if not args.no_display and last_frame is not None:
        vis = draw_debug(last_frame, int(avg_cx), int(avg_cy), icx, icy,
                         (target_x, target_y, target_z))
        cv2.imshow("palm_approach — target (any key to proceed)", vis)
        cv2.imwrite("/tmp/palm_approach_result.jpg", vis)
        print("\n[palm_approach] Press any key in image window to proceed...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.dry_run:
        print("\n[palm_approach] DRY RUN complete — no motion sent.")
        return

    # ── PTP ───────────────────────────────────────────────────────────────────
    print("\n[palm_approach] Sending PTP → KUKA ...")
    try:
        target_pos = Pos(
            x=target_x, y=target_y, z=target_z,
            a=ORI_A, b=ORI_B, c=ORI_C,
        )
        motion_client.ptp(target_pos, max_velocity_scaling=0.08)
        print("[palm_approach] ✓ Command sent — robot moving to 20mm above palm.")
    except Exception as e:
        print(f"[palm_approach] ✗ PTP failed: {e}")
        print("  → Confirm ros_eki.src is running (R = green) on SmartPad.")


if __name__ == "__main__":
    main()
