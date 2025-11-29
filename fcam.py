"""
Functional version of cam.py - Robot navigation with YOLO object detection
All code organized into functions for better readability
"""

import numpy as np
import cv2 as cv
import socket
import sys
import struct
import time
import colorlog

# Setup colored logging
logger = colorlog.getLogger()
logger.setLevel(colorlog.INFO)
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))
logger.addHandler(handler)

# Import robot utility functions
from utils.robot_utils import capture, cmd, check_connection, setup_socket_options, reconnect_robot

# Import detection utility functions
from utils.detection_utils import (
    initialize_yolo_model, 
    detect_objects_yolo,
    get_largest_object
)

# Import navigation functions
from utils.navigation_utils import navigate_with_yolo, thinking_mode

# Global variables
iteration_count = 0
paused = False
manual_mode = False


def connect_to_robot():
    """Establish connection to robot"""
    logger.info("Connecting to robot at 192.168.4.1:100...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        sock.connect(('192.168.4.1', 100))
        
        # Apply socket options
        setup_socket_options(sock)
        
        logger.info("✓ Connected to robot successfully!")
        logger.info(f"Local address: {sock.getsockname()}")
        logger.info(f"Remote address: {sock.getpeername()}")
        return sock
    except socket.timeout:
        logger.error("ERROR: Connection timeout. Is the robot powered on?")
        sys.exit(1)
    except socket.error as e:
        logger.error(f"ERROR: Connection failed: {e}")
        sys.exit(1)


def print_controls():
    """Print keyboard control instructions"""
    logger.info("\nKeyboard Controls:")
    logger.info("  'k' - Stop and exit")
    logger.info("  'r' - Restart navigation")
    logger.info("  'p' - Pause/resume autonomous mode")
    logger.info("  Arrow Keys - Manual control:")
    logger.info("    ↑ (Up)    - Move forward")
    logger.info("    ↓ (Down)  - Move backward")
    logger.info("    ← (Left)  - Turn left")
    logger.info("    → (Right) - Turn right")
    logger.info("  Space - Stop robot")
    logger.info("  Ctrl+C - Emergency stop")


def handle_keyboard_input(sock):
    """
    Handle keyboard input for manual control.
    Returns: command string or None to continue autonomous mode
    """
    global paused, manual_mode, iteration_count
    
    key = cv.waitKey(1) & 0xFF
    
    # Exit command
    if key == ord('k'):
        logger.warning("\n'k' pressed - Stopping and exiting...")
        return 'exit'
    
    # Restart command
    elif key == ord('r'):
        logger.info("\n'r' pressed - Restarting navigation...")
        iteration_count = 0
        manual_mode = False
        paused = False
        if check_connection(sock):
            cmd(sock, 'stop')
            time.sleep(0.5)
        return 'restart'
    
    # Pause/resume command
    elif key == ord('p'):
        paused = not paused
        manual_mode = False
        if paused:
            logger.warning("\n'p' pressed - PAUSED (Autonomous mode disabled)")
            if check_connection(sock):
                cmd(sock, 'stop')
        else:
            logger.info("'p' pressed - RESUMED (Autonomous mode enabled)")
        return 'pause'
    
    # Manual controls with arrow keys
    elif key == 82 or key == 0:  # Up arrow
        manual_mode = True
        paused = True
        logger.info("\n↑ Manual: Moving forward")
        if check_connection(sock):
            cmd(sock, 'move', where='forward', at=70)
        time.sleep(0.1)
        return 'manual'
        
    elif key == 84 or key == 1:  # Down arrow
        manual_mode = True
        paused = True
        logger.info("\n↓ Manual: Moving backward")
        if check_connection(sock):
            cmd(sock, 'move', where='back', at=70)
        time.sleep(0.1)
        return 'manual'
        
    elif key == 81 or key == 2:  # Left arrow
        manual_mode = True
        paused = True
        logger.info("\n← Manual: Turning left")
        if check_connection(sock):
            cmd(sock, 'move', where='left', at=70)
        time.sleep(0.1)
        return 'manual'
        
    elif key == 83 or key == 3:  # Right arrow
        manual_mode = True
        paused = True
        logger.info("\n→ Manual: Turning right")
        if check_connection(sock):
            cmd(sock, 'move', where='right', at=70)
        time.sleep(0.1)
        return 'manual'
    
    # Space to stop
    elif key == 32:  # Space bar
        logger.warning("\nSpace pressed - Stopping robot")
        if check_connection(sock):
            cmd(sock, 'stop')
        if not paused:
            paused = True
            logger.info("(Paused - press 'p' to resume autonomous mode)")
        return 'stop'
    
    # Show key code for debugging
    elif key != 255:
        logger.debug(f"\nKey pressed: {key} (press 'k' to exit)")
    
    return None


def periodic_reconnect(sock, iteration,stop_n_think = False):
    """
    Perform periodic reconnection to prevent robot firmware timeout.
    Robot firmware closes connection after 8 commands.
    Returns: new socket or None if failed
    """
    if iteration % 3 == 0 and iteration > 0:
        logger.warning("\n[Periodic maintenance: reconnecting before robot limit]")
        
        # Reconnect FIRST to reset command buffer
        sock = reconnect_robot(sock)
        
        if sock is None:
            return None
        
        if stop_n_think:
            # Now do maintenance with fresh connection - check if thinking mode needed
            try:
                thinking_activated = thinking_mode(sock)
                if not thinking_activated:
                    # Distance was safe, just stop
                    cmd(sock, 'stop')
                    time.sleep(0.2)
                else:
                    # Thinking mode used 3 commands - reconnect immediately
                    logger.warning("[Thinking mode active - reconnecting to reset buffer]")
                    sock = reconnect_robot(sock)
                    if sock is None:
                        return None
            except Exception as e:
                logger.error(f"Error in periodic thinking mode: {e}")
                pass
        
        return sock
    
    return sock


def run_navigation_loop(sock, model,stop_n_think):
    """
    Main navigation loop with YOLO object detection.
    Returns: final iteration count
    """
    global iteration_count, paused
    
    logger.info("\nStarting autonomous navigation with YOLO...")
    logger.warning("\nNOTE: Robot firmware closes connection after 8 commands - auto-reconnecting every 3 iterations")
    print_controls()
    
    # Setup camera window
    cv.namedWindow('Camera', cv.WINDOW_NORMAL)
    cv.resizeWindow('Camera', 800, 600)
    cv.moveWindow('Camera', 0, 0)
    
    iteration_count = 0
    
    try:
        while True:
            # Handle keyboard input
            key_result = handle_keyboard_input(sock)
            
            if key_result == 'exit':
                break
            elif key_result in ['restart', 'pause', 'manual', 'stop']:
                if key_result == 'restart':
                    iteration_count = 0
                continue
            
            # Skip iteration if paused
            if paused:
                time.sleep(0.1)
                continue
            
            # Check connection
            if not check_connection(sock):
                logger.error("\nERROR: Lost connection to robot!")
                break
            
            # Increment iteration
            iteration_count += 1
            
            # Show progress
            if iteration_count % 3 == 0:
                logger.info(f"\n--- Iteration {iteration_count} ---")
            
            # Periodic reconnect to prevent firmware timeout
            new_sock = periodic_reconnect(sock, iteration_count,stop_n_think)
            if new_sock is None:
                logger.error("Press 'r' to retry or 'k' to exit")
                paused = True
                continue
            sock = new_sock
            
            # Capture image
            img = capture()
            if img is None:
                logger.error("ERROR: Failed to capture image, retrying...")
                time.sleep(0.5)
                continue
            
            # Navigate using YOLO
            try:
                result = navigate_with_yolo(sock, img, model, target_class=None, avoid_classes=None)
            except Exception as e:
                logger.error(f"\n!!! ERROR in navigation at iteration {iteration_count}: {e}")
                result = None
            
            # Handle navigation failure
            if result is None:
                logger.error(f"\nERROR: Navigation command failed at iteration {iteration_count}!")
                logger.warning("Attempting to recover...")
                
                new_sock = reconnect_robot(sock)
                if new_sock is None:
                    logger.error("✗ Failed to reconnect. Press 'r' to retry or 'k' to exit")
                    paused = True
                    continue
                sock = new_sock
                time.sleep(0.5)
                continue
            
            # Delay between commands
            time.sleep(0.3)
            
    except KeyboardInterrupt:
        logger.critical("\n\nEmergency stop - Ctrl+C pressed!")
        if check_connection(sock):
            cmd(sock, 'stop')
        else:
            logger.warning("Socket already disconnected")
    except Exception as e:
        logger.critical(f"\n\nUnexpected error in main loop at iteration {iteration_count}: {e}")
        import traceback
        traceback.print_exc()
        if check_connection(sock):
            cmd(sock, 'stop')
    finally:
        sock.close()
        cv.destroyAllWindows()
    
    return iteration_count


def main(stop_n_think):
    """Main entry point"""
    # Initialize YOLO model
    model = initialize_yolo_model('yolov8s.pt')
    
    # Connect to robot
    sock = connect_to_robot()
    
    # Run navigation loop
    final_iterations = run_navigation_loop(sock, model,stop_n_think)
    
    logger.info(f"\nTotal iterations completed: {final_iterations}")
    logger.info("Program ended")


if __name__ == '__main__':
    stop_n_think = False
    main(stop_n_think)
