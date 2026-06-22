#!/usr/bin/env python3
import sys
import tty
import termios
import time
from kuka_eki.eki import EkiMotionClient, EkiStateClient
from kuka_eki.krl import Axis

ROBOT_IP = "192.168.1.147"

print(f"Connecting to Robot State Server at {ROBOT_IP}...")
state_client = EkiStateClient(ROBOT_IP)
state_client.connect()

print(f"Connecting to Robot Motion Server at {ROBOT_IP}...")
motion_client = EkiMotionClient(ROBOT_IP)
motion_client.connect()

print("\n--- CONNECTION SUCCESSFUL ---")
print("Controls:")
print(" [A] : Nudge Joint 1 Left (1 degree)")
print(" [D] : Nudge Joint 1 Right (1 degree)")
print(" [Q] : Quit")
print("-----------------------------\n")

def get_keypress():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

while True:
    char = get_keypress().lower()
    
    if char == 'q':
        print("\nDisconnecting...")
        break
        
    elif char == 'a':
        print("\nNudging A1 Left (+1.0 deg)...")
        nudge = Axis(a1=1.0, a2=0.0, a3=0.0, a4=0.0, a5=0.0, a6=0.0)
        motion_client.ptp_rel(nudge, 0.1) # 0.1 means 10% speed
        
    elif char == 'd':
        print("\nNudging A1 Right (-1.0 deg)...")
        nudge = Axis(a1=-1.0, a2=0.0, a3=0.0, a4=0.0, a5=0.0, a6=0.0)
        motion_client.ptp_rel(nudge, 0.1)
