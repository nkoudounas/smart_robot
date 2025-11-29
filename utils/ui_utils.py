"""
UI and keyboard control functions for robot interface
"""

import cv2 as cv
import time
from .robot_utils import cmd, check_connection


def setup_camera_window():
    """Setup OpenCV camera window with standard size and position"""
    cv.namedWindow('Camera', cv.WINDOW_NORMAL)
    cv.resizeWindow('Camera', 800, 600)
    cv.moveWindow('Camera', 0, 0)


def print_controls():
    """Print available keyboard controls to console"""
    print("\n=== CONTROLS ===")
    print("Autonomous Mode:")
    print("  p - Pause/Resume autonomous navigation")
    print("  k - Stop and exit program")
    print("  r - Restart navigation (reset iteration counter)")
    print("  m - Toggle navigation mode (sensor_fusion/vision_only)")
    print("\nManual Mode:")
    print("  Arrow keys for manual control:")
    print("    ↑ (Up)    - Move forward")
    print("    ↓ (Down)  - Move backward")
    print("    ← (Left)  - Turn left")
    print("    → (Right) - Turn right")
    print("  Space - Stop robot")
    print("  Ctrl+C - Emergency stop")


def handle_keyboard_input(sock, nav_mode='sensor_fusion'):
    """
    Handle keyboard input for manual control and mode switching.
    
    Args:
        sock: Socket connection to robot
        nav_mode: Current navigation mode string
        
    Returns:
        tuple: (command string or None, navigation_mode)
               Commands: 'exit', 'restart', 'pause', 'manual', 'stop', None
    """
    key = cv.waitKey(1) & 0xFF
    
    # Exit command
    if key == ord('k'):
        print("\n'k' pressed - Stopping and exiting...")
        return 'exit', nav_mode
    
    # Restart command
    elif key == ord('r'):
        print("\n'r' pressed - Restarting navigation...")
        if check_connection(sock):
            cmd(sock, 'stop')
            time.sleep(0.5)
        return 'restart', nav_mode
    
    # Pause/resume command
    elif key == ord('p'):
        if check_connection(sock):
            cmd(sock, 'stop')
        return 'pause', nav_mode
    
    # Toggle navigation mode
    elif key == ord('m'):
        if nav_mode == 'sensor_fusion':
            nav_mode = 'vision_only'
            print("\n'm' pressed - Switched to VISION ONLY mode")
        else:
            nav_mode = 'sensor_fusion'
            print("\n'm' pressed - Switched to SENSOR FUSION mode")
        return 'mode_change', nav_mode
    
    # Manual controls with arrow keys
    elif key == 82 or key == 0:  # Up arrow
        print("\n↑ Manual: Moving forward")
        if check_connection(sock):
            cmd(sock, 'move', where='forward', at=70)
        time.sleep(0.1)
        return 'manual', nav_mode
        
    elif key == 84 or key == 1:  # Down arrow
        print("\n↓ Manual: Moving backward")
        if check_connection(sock):
            cmd(sock, 'move', where='back', at=70)
        time.sleep(0.1)
        return 'manual', nav_mode
        
    elif key == 81 or key == 2:  # Left arrow
        print("\n← Manual: Turning left")
        if check_connection(sock):
            cmd(sock, 'move', where='left', at=70)
        time.sleep(0.1)
        return 'manual', nav_mode
        
    elif key == 83 or key == 3:  # Right arrow
        print("\n→ Manual: Turning right")
        if check_connection(sock):
            cmd(sock, 'move', where='right', at=70)
        time.sleep(0.1)
        return 'manual', nav_mode
    
    # Space to stop
    elif key == 32:  # Space bar
        print("\nSpace pressed - Stopping robot")
        if check_connection(sock):
            cmd(sock, 'stop')
        return 'stop', nav_mode
    
    # Show key code for debugging
    elif key != 255:
        print(f"\nKey pressed: {key} (press 'k' to exit)")
    
    return None, nav_mode
