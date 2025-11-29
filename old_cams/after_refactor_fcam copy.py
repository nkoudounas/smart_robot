"""
Functional version of cam.py - Robot navigation with YOLO object detection
All code organized into functions for better readability
Refactored with utils package structure
"""

import numpy as np
import cv2 as cv
import time

# Import from utils package
from utils import (
    # Robot communication
    capture, cmd, check_connection, reconnect_robot,
    
    # Detection
    initialize_yolo_model,
    
    # Navigation
    navigate_with_yolo, navigate_with_sensor_fusion,
    
    # UI
    print_controls, handle_keyboard_input, setup_camera_window,
    
    # Connection
    connect_to_robot, periodic_reconnect
)

# Global variables
iteration_count = 0
paused = False
manual_mode = False


def run_navigation_loop(sock, model, navigation_mode='sensor_fusion'):
    """
    Main navigation loop with YOLO object detection and sensor fusion.
    
    Args:
        sock: Socket connection to robot
        model: YOLO model instance
        navigation_mode: 'sensor_fusion' or 'vision_only'
        
    Returns:
        final iteration count
    """
    global iteration_count, paused, manual_mode
    
    print(f"\nStarting autonomous navigation in {navigation_mode.upper()} mode...")
    print("\nNOTE: Robot firmware closes connection after 8 commands - auto-reconnecting every 3 iterations")
    print_controls()
    
    # Setup camera window
    setup_camera_window()
    
    iteration_count = 0
    
    try:
        while True:
            # Handle keyboard input
            key_result, navigation_mode = handle_keyboard_input(sock, navigation_mode)
            
            if key_result == 'exit':
                break
            elif key_result == 'restart':
                iteration_count = 0
                paused = False
                manual_mode = False
                continue
            elif key_result == 'pause':
                paused = not paused
                manual_mode = False
                if paused:
                    print("\nPAUSED (Autonomous mode disabled)")
                else:
                    print("RESUMED (Autonomous mode enabled)")
                continue
            elif key_result == 'manual':
                manual_mode = True
                paused = True
                continue
            elif key_result == 'stop':
                if not paused:
                    paused = True
                    print("(Paused - press 'p' to resume autonomous mode)")
                continue
            elif key_result == 'mode_change':
                continue
            
            # Skip iteration if paused
            if paused:
                time.sleep(0.1)
                continue
            
            # Check connection
            if not check_connection(sock):
                print("\nERROR: Lost connection to robot!")
                break
            
            # Increment iteration
            iteration_count += 1
            
            # Periodic reconnect BEFORE sending commands to prevent firmware timeout
            # Reconnect every 3 iterations (at iterations 3, 6, 9...)
            if iteration_count % 3 == 0 and iteration_count > 0:
                new_sock = periodic_reconnect(sock, iteration_count)
                if new_sock is None:
                    print("Press 'r' to retry or 'k' to exit")
                    paused = True
                    continue
                sock = new_sock
            
            # Show progress
            if iteration_count % 3 == 0:
                print(f"\n--- Iteration {iteration_count} ({navigation_mode}) ---")
            
            # Capture image
            img = capture()
            if img is None:
                print("ERROR: Failed to capture image, retrying...")
                time.sleep(0.5)
                continue
            
            # Navigate based on selected mode
            try:
                if navigation_mode == 'sensor_fusion':
                    # Alternate distance checks to stay under command limit
                    check_distance = (iteration_count % 2 == 0)
                    result = navigate_with_sensor_fusion(sock, img, model, check_distance)
                else:  # vision_only
                    result = navigate_with_yolo(sock, img, model, target_class=None, avoid_classes=None)
            except Exception as e:
                print(f"\n!!! ERROR in navigation at iteration {iteration_count}: {e}")
                result = None
            
            # Handle navigation failure
            if result is None:
                print(f"\nERROR: Navigation command failed at iteration {iteration_count}!")
                print("Attempting to recover...")
                
                new_sock = reconnect_robot(sock)
                if new_sock is None:
                    print("✗ Failed to reconnect. Press 'r' to retry or 'k' to exit")
                    paused = True
                    continue
                sock = new_sock
                time.sleep(0.5)
                continue
            
            # Delay between commands
            time.sleep(0.3)
            
    except KeyboardInterrupt:
        print("\n\nEmergency stop - Ctrl+C pressed!")
        if check_connection(sock):
            cmd(sock, 'stop')
        else:
            print("Socket already disconnected")
    except Exception as e:
        print(f"\n\nUnexpected error in main loop at iteration {iteration_count}: {e}")
        import traceback
        traceback.print_exc()
        if check_connection(sock):
            cmd(sock, 'stop')
    finally:
        sock.close()
        cv.destroyAllWindows()
    
    return iteration_count


def main():
    """Main entry point"""
    # Initialize YOLO model
    model = initialize_yolo_model('yolov8n.pt')
    
    # Connect to robot
    sock = connect_to_robot()
    
    # Run navigation loop
    final_iterations = run_navigation_loop(sock, model)
    
    print(f"\nTotal iterations completed: {final_iterations}")
    print("Program ended")


if __name__ == '__main__':
    main()
