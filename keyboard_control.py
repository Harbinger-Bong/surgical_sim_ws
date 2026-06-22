import time
from pynput import keyboard
from kuka_eki.eki import EkiMotionClient, EkiStateClient
from kuka_eki.krl import Axis

ROBOT_IP = "192.168.1.147"
STEP_DEG = 1.0       # Move exactly 1 degree per tap
VEL_SCALE = 0.05     # 5% max velocity (very safe)

print(f"Connecting to State Server at {ROBOT_IP}...")
state_client = EkiStateClient(ROBOT_IP)
state_client.connect()

print(f"Connecting to Motion Server at {ROBOT_IP}...")
motion_client = EkiMotionClient(ROBOT_IP)
motion_client.connect()

print("\n--- CONNECTION SUCCESSFUL ---")
print("Controls:")
print("  LEFT/RIGHT -> Rotate Base (A1)")
print("  UP/DOWN    -> Pitch Arm (A2)")
print("  Q          -> Safe Disconnect")
print("\n[SAFETY ACTIVE] Key-holding is disabled. You must physically tap to move.")

# This set tracks keys that are currently pressed down to block OS auto-repeat
pressed_keys = set()
running = True

def on_press(key):
    global running
    
    # 1. ANTI-REPEAT: If key is already held down, ignore the OS spam completely
    if key in pressed_keys:
        return
    
    # Register the new physical key press
    pressed_keys.add(key)
    
    # Initialize a zeroed-out relative movement target
    target = Axis(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    moved = False

    try:
        if key == keyboard.Key.left:
            target.a1 = STEP_DEG
            moved = True
        elif key == keyboard.Key.right:
            target.a1 = -STEP_DEG
            moved = True
        elif key == keyboard.Key.up:
            target.a2 = -STEP_DEG
            moved = True
        elif key == keyboard.Key.down:
            target.a2 = STEP_DEG
            moved = True
        elif hasattr(key, 'char') and key.char:
            if key.char.lower() == 'q':
                print("\nInitiating safe disconnect...")
                running = False
                return False  # Kills the pynput listener
    except AttributeError:
        pass

    # 2. RELATIVE MOTION: Send exactly one relative step to the controller
    if moved and running:
        try:
            motion_client.ptp_rel(target, max_velocity_scaling=VEL_SCALE)
            print(f"Queued Step: {target}")
        except Exception as e:
            print(f"Transmission Error: {e}")

def on_release(key):
    # Free the key once it is physically released, allowing the next tap
    if key in pressed_keys:
        pressed_keys.remove(key)

# Start the listener with both press and release tracking
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

print("Bridge securely closed.")
