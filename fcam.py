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
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
from utils.robot_utils import capture, cmd, check_connection, setup_socket_options, reconnect_robot, should_reconnect, get_commands_sent

# Import detection utility functions
from utils.detection_utils import (
    initialize_yolo_model, 
    detect_objects_yolo,
    get_largest_object
)

# Import navigation functions
from utils.navigation_utils import navigate_with_yolo, movement_delay, movement_speed, MOVEMENT_DELAYS, MOVEMENT_SPEEDS
from utils.depth_estimation import DepthEstimator

from utils.connection_utils import connect_to_robot,periodic_reconnect

# Global variables
iteration_count = 0
paused = False
manual_mode = False

# Robot position tracking (2D estimated position)
robot_x = 0.0
robot_y = 0.0
robot_angle = 90.0  # degrees, 0=right, 90=up, 180=left, 270=down
path_x = [0.0]
path_y = [0.0]
last_command = None

# Obstacle tracking (objects detected by ultrasonic sensor)
obstacle_x = []
obstacle_y = []

# Detected objects tracking (vision-based with depth estimation)
# Each entry: {'x': x, 'y': y, 'class': class_name, 'age': frames_old, 'confidence': float}
detected_objects = []

# Depth estimator
depth_estimator = DepthEstimator()

# Stuck detection (frame comparison)
previous_frame = None
stuck_counter = 0
STUCK_THRESHOLD = 3  # Number of similar frames before considering stuck
PIXEL_DIFF_THRESHOLD = 0.02  # 2% pixel difference threshold


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
            cmd(sock, 'move', where='forward', at=movement_speed('manual'))
        movement_delay('manual')
        return 'manual'
        
    elif key == 84 or key == 1:  # Down arrow
        manual_mode = True
        paused = True
        logger.info("\n↓ Manual: Moving backward")
        if check_connection(sock):
            cmd(sock, 'move', where='back', at=movement_speed('manual'))
        movement_delay('manual')
        return 'manual'
        
    elif key == 81 or key == 2:  # Left arrow
        manual_mode = True
        paused = True
        logger.info("\n← Manual: Turning left")
        if check_connection(sock):
            cmd(sock, 'move', where='left', at=movement_speed('manual'))
        movement_delay('manual')
        return 'manual'
        
    elif key == 83 or key == 3:  # Right arrow
        manual_mode = True
        paused = True
        logger.info("\n→ Manual: Turning right")
        if check_connection(sock):
            cmd(sock, 'move', where='right', at=movement_speed('manual'))
        movement_delay('manual')
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


def update_robot_position(command, duration=0.5):
    """
    Update estimated robot position based on command.
    Simple dead-reckoning estimation.
    """
    global robot_x, robot_y, robot_angle, path_x, path_y, last_command
    
    last_command = command
    
    if command == 'forward':
        # Move forward in current direction (estimate ~20cm per 0.5s)
        distance = 0.20 * (duration / 0.5)
        robot_x += distance * np.cos(np.radians(robot_angle))
        robot_y += distance * np.sin(np.radians(robot_angle))
    elif command == 'back':
        # Move backward
        distance = 0.20 * (duration / 0.5)
        robot_x -= distance * np.cos(np.radians(robot_angle))
        robot_y -= distance * np.sin(np.radians(robot_angle))
    elif command == 'left':
        # Rotate left (estimate ~45 degrees per 0.5s)
        robot_angle += 45 * (duration / 0.5)
        robot_angle %= 360
    elif command == 'right':
        # Rotate right
        robot_angle -= 45 * (duration / 0.5)
        robot_angle %= 360
    
    # Record path
    path_x.append(robot_x)
    path_y.append(robot_y)


def mark_obstacle(distance_cm):
    """
    Mark obstacle position based on current robot position and direction.
    Distance is measured by ultrasonic sensor in cm.
    """
    global obstacle_x, obstacle_y
    
    # Convert distance to meters and calculate obstacle position
    distance_m = distance_cm / 100.0
    
    # Obstacle is in front of robot (in direction of robot_angle)
    obs_x = robot_x + distance_m * np.cos(np.radians(robot_angle))
    obs_y = robot_y + distance_m * np.sin(np.radians(robot_angle))
    
    # Add to obstacle list
    obstacle_x.append(obs_x)
    obstacle_y.append(obs_y)
    
    logger.info(f"📍 Obstacle marked at ({obs_x:.2f}, {obs_y:.2f}), distance: {distance_cm}cm")


def is_robot_stuck(current_frame):
    """
    Detect if robot is stuck by comparing current frame with previous frame.
    Returns: True if robot appears stuck (similar frames for multiple iterations)
    """
    global previous_frame, stuck_counter
    
    if previous_frame is None:
        previous_frame = current_frame.copy()
        return False
    
    # Resize both frames to same size for comparison (in case of any mismatch)
    h, w = current_frame.shape[:2]
    prev_resized = cv.resize(previous_frame, (w, h))
    
    # Convert to grayscale for comparison
    current_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
    prev_gray = cv.cvtColor(prev_resized, cv.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv.absdiff(current_gray, prev_gray)
    
    # Calculate percentage of pixels that changed significantly (threshold > 30)
    changed_pixels = np.sum(diff > 30)
    total_pixels = diff.size
    change_ratio = changed_pixels / total_pixels
    
    logger.debug(f"Frame change: {change_ratio*100:.2f}%")
    
    # If change is below threshold, increment stuck counter
    if change_ratio < PIXEL_DIFF_THRESHOLD:
        stuck_counter += 1
        logger.warning(f"⚠️ Low frame change detected ({stuck_counter}/{STUCK_THRESHOLD})")
    else:
        stuck_counter = 0  # Reset if significant change detected
    
    # Update previous frame
    previous_frame = current_frame.copy()
    
    # Return True if stuck for multiple frames
    if stuck_counter >= STUCK_THRESHOLD:
        logger.error(f"🚫 ROBOT STUCK DETECTED! ({stuck_counter} similar frames)")
        stuck_counter = 0  # Reset counter after detection
        return True
    
    return False


def mark_detected_objects(objects, image_width):
    """
    Mark detected objects on the map with estimated positions.
    Objects fade over time. Chair is green, others are black.
    """
    global detected_objects, depth_estimator
    
    # Age existing objects
    for obj in detected_objects:
        obj['age'] += 1
    
    # Remove very old objects (older than 10 frames)
    detected_objects = [obj for obj in detected_objects if obj['age'] < 10]
    
    # Estimate distances for new detections
    estimates = depth_estimator.estimate_all_objects(objects, image_width)
    
    for estimate in estimates:
        obj = estimate['object']
        distance = estimate['distance']
        confidence = estimate['confidence']
        
        if distance is not None:
            # Calculate object position relative to robot
            # If we don't have good confidence, place at edge of range (5m)
            if confidence < 0.5:
                distance = 5.0  # Place at border
            
            # Object is in direction of where it appears in frame
            # Center position -> straight ahead
            # Left position -> slightly left of robot angle
            # Right position -> slightly right of robot angle
            
            angle_offset = 0
            if obj['position'] == 'left':
                angle_offset = 20  # 20 degrees left
            elif obj['position'] == 'right':
                angle_offset = -20  # 20 degrees right
            
            obj_angle = robot_angle + angle_offset
            obj_x = robot_x + distance * np.cos(np.radians(obj_angle))
            obj_y = robot_y + distance * np.sin(np.radians(obj_angle))
            
            # Add to detected objects
            detected_objects.append({
                'x': obj_x,
                'y': obj_y,
                'class': obj['class'],
                'age': 0,
                'confidence': confidence
            })


def setup_live_plot():
    """Setup matplotlib figure for live robot tracking"""
    plt.ion()  # Interactive mode
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Robot Movement (Estimated)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Position the window
    mngr = plt.get_current_fig_manager()
    try:
        mngr.window.wm_geometry("+850+0")  # Position to right of camera window
    except:
        pass
    
    plt.show(block=False)
    plt.pause(0.1)
    
    return fig, ax


def update_plot(ax):
    """Update the live plot with current robot position"""
    ax.clear()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'Robot Movement - Angle: {robot_angle:.0f}° - Last: {last_command}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot detected objects with fading (vision-based)
    for obj in detected_objects:
        # Calculate alpha (transparency) based on age - newer = more opaque
        alpha = max(0.2, 1.0 - (obj['age'] / 10.0))
        
        # Chair is green, others are black
        color = 'green' if obj['class'] == 'chair' else 'black'
        marker = 's' if obj['class'] == 'chair' else 'o'  # square for chair, circle for others
        size = 150 if obj['class'] == 'chair' else 80
        
        ax.scatter(obj['x'], obj['y'], c=color, marker=marker, s=size, 
                  alpha=alpha, edgecolors='white', linewidths=1.5, zorder=4,
                  label=f"{obj['class']}" if obj['age'] == 0 else "")
    
    # Plot obstacles detected by ultrasonic sensor (red X)
    if len(obstacle_x) > 0:
        ax.scatter(obstacle_x, obstacle_y, c='red', marker='x', s=100, linewidths=3, 
                  label=f'Ultrasonic obstacles ({len(obstacle_x)})', zorder=5)
    
    # Plot path
    if len(path_x) > 1:
        ax.plot(path_x, path_y, 'b-', alpha=0.5, linewidth=2, label='Path')
    
    # Plot robot position with direction arrow
    ax.plot(robot_x, robot_y, 'ro', markersize=10, label='Robot')
    
    # Draw direction arrow
    arrow_length = 0.15
    dx = arrow_length * np.cos(np.radians(robot_angle))
    dy = arrow_length * np.sin(np.radians(robot_angle))
    ax.arrow(robot_x, robot_y, dx, dy, head_width=0.08, head_length=0.08, fc='red', ec='red')
    
    # Only show legend for unique labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    
    plt.draw()
    plt.pause(0.001)  # Very short pause to update display
    ax.arrow(robot_x, robot_y, dx, dy, head_width=0.08, head_length=0.08, fc='red', ec='red')
    
    ax.legend()
    plt.draw()
    plt.pause(0.001)  # Very short pause to update display


def run_navigation_loop(sock, model, use_ollama, ai_decide):
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
    
    # Setup live plot
    fig, ax = setup_live_plot()
    update_plot(ax)  # Draw initial plot
    
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
                update_plot(ax)  # Keep plot responsive
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
            # Use threshold=2 because navigation may use up to 2 commands (ultrasonic + move)
            new_sock = periodic_reconnect(sock, threshold=2)
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

            # Check if robot is stuck (same view for multiple frames)
            if is_robot_stuck(img):
                logger.error("🚨 Robot appears STUCK! Taking evasive action...")
                
                # Use vision-based recovery to intelligently choose escape direction
                from utils.navigation_utils import vision_based_stuck_recovery
                
                # Define capture callback to get fresh frames
                def capture_frame():
                    frame = capture()
                    if frame is None:
                        logger.error("Failed to capture frame during stuck recovery")
                        return None
                    return frame
                
                # Execute vision-based recovery with reconnection support
                try:
                    sock, chosen_dir, left_score, right_score = vision_based_stuck_recovery(
                        sock, capture_frame, reconnect_robot
                    )
                    if sock is None:
                        logger.error("Lost connection during stuck recovery")
                        paused = True
                        continue
                    logger.info(f"✅ Stuck recovery complete - chose {chosen_dir.upper()} path")
                except Exception as e:
                    logger.error(f"Vision recovery failed: {e}, using fallback (back + left)")
                    # Fallback to simple recovery
                    cmd(sock, 'move', where='back', at=movement_speed('back_avoid'))
                    movement_delay('unstuck_back')
                    cmd(sock, 'move', where='left', at=movement_speed('left_unstuck'))
                    movement_delay('unstuck_turn')
                    # Reconnect after fallback
                    sock = reconnect_robot(sock)
                    if sock is None:
                        logger.error("Failed to reconnect after fallback unstuck maneuver")
                        paused = True
                        continue
                
                # Skip navigation this iteration
                update_plot(ax)
                continue

            # First, detect objects with YOLO to get annotated image with bounding boxes
            from utils.detection_utils import detect_objects_yolo
            objects, annotated_img = detect_objects_yolo(img, model, target_class='chair')
            
            # Log all detected objects with their labels and probabilities
            if objects:
                logger.info(f"📸 Detected {len(objects)} object(s):")
                for i, obj in enumerate(objects, 1):
                    logger.info(f"  {i}. {obj['class']} - confidence: {obj['confidence']:.2%} - position: {obj['position']} - area: {obj['area']:.0f}px²")
            else:
                logger.info("📸 No objects detected in current frame")
            
            # Navigate using YOLO or Ollama
            try:
                if use_ollama:

                    
                    # Show YOLO detections briefly
                    cv.imshow('Camera', annotated_img)
                    cv.waitKey(1)
                    
                    # Now use Ollama with the ANNOTATED image (with bounding boxes)
                    from ollama.cam_ollama import navigate_with_ollama
                    result = navigate_with_ollama(sock, annotated_img, target_class='chair')
                    
                    # Wait for movement to execute before reconnecting
                    # Robot closes connection immediately after sending command
                    # but we need to give it time to execute the movement
                    time.sleep(0.3)  # Brief pause to let robot start movement
                    
                    # Ollama navigation sends 1 command, robot closes connection after - reconnect
                    logger.debug("Reconnecting after Ollama navigation command...")
                    sock = reconnect_robot(sock)
                    if sock is None:
                        logger.error("Failed to reconnect after Ollama navigation")
                        paused = True
                        continue
                else:
                    # Mark detected objects on the map
                    height, width = img.shape[:2]
                    mark_detected_objects(objects, width)
                    
                    # If AI decision mode is enabled, compute decision BEFORE calling navigate
                    ai_decision = None
                    if ai_decide:
                        # Reconnect before AI call to ensure fresh connection after AI finishes
                        # (AI takes 2-5+ seconds, robot firmware might close idle connection)
                        logger.debug("Reconnecting before AI decision call...")
                        sock = reconnect_robot(sock)
                        if sock is None:
                            logger.error("Failed to reconnect before AI decision")
                            paused = True
                            continue
                        
                        # Find target and compute AI decision while socket is idle
                        target_objects = [obj for obj in objects if obj['class'] == 'chair']
                        if target_objects:
                            target = max(target_objects, key=lambda x: x['area'])
                            other_objects = [obj for obj in objects if obj['class'] != 'chair']
                            
                            # Call AI decision function (takes 5+ seconds)
                            from utils.navigation_utils import ai_navigation_decision
                            ai_decision = ai_navigation_decision(annotated_img, target, other_objects, 'chair', width, height)
                        else:
                            # No target found - AI can still decide (search, explore, etc.)
                            logger.info("No chair detected, querying AI for search strategy...")
                            from utils.navigation_utils import ai_search_decision
                            ai_decision = ai_search_decision(annotated_img, 'chair', objects, width, height)
                    
                    # Now navigate with YOLO (with pre-computed AI decision if enabled)
                    result = navigate_with_yolo(sock, img, model, target_class='chair', avoid_classes=None, 
                                              ai_decide=ai_decide, ai_decision=ai_decision,
                                              objects=objects, annotated_img=annotated_img)
                    
                    # Extract duration from tuple (both AI and hardcoded modes return tuples now)
                    if isinstance(result, tuple):
                        _, sleep_duration = result
                        # Sleep for the full duration to let robot physically complete the movement
                        logger.debug(f"Waiting {sleep_duration:.1f}s for movement to complete...")
                        time.sleep(sleep_duration)
                    else:
                        # Fallback for non-tuple returns (e.g., 'reached', 'idle')
                        time.sleep(0.3)
                    
                    # Robot firmware closes connection after each command - must reconnect
                    # Reconnect AFTER movement completes (we already slept above)
                    logger.debug("Reconnecting after movement command (robot closes socket after each cmd)...")
                    sock = reconnect_robot(sock)
                    if sock is None:
                        logger.error("Failed to reconnect after YOLO navigation")
                        paused = True
                        continue
                
                # If ultrasonic triggered, we used 3 commands - reconnect immediately
                if result == 'ultrasonic_avoid':
                    logger.warning(f"\n[Ultrasonic avoid used 3 commands - reconnecting]")
                    sock = reconnect_robot(sock)
                    if sock is None:
                        logger.error("Failed to reconnect after ultrasonic avoid")
                        paused = True
                        continue
                
                # Update position estimate based on navigation result
                # AI mode returns tuple (result, duration), hardcoded mode returns string
                ai_duration = None
                if isinstance(result, tuple):
                    result, ai_duration = result
                
                if result and result != 'idle':
                    if result == 'ultrasonic_avoid':
                        # Ultrasonic avoid does back then right
                        # Duration uses the actual delays from MOVEMENT_DELAYS
                        update_robot_position('back', duration=MOVEMENT_DELAYS['back'])
                        update_robot_position('right', duration=MOVEMENT_DELAYS['right'])
                    elif result in ['forward', 'back']:
                        duration = ai_duration if ai_duration else MOVEMENT_DELAYS[result]
                        update_robot_position(result, duration=duration)
                    elif result in ['left', 'right', 'avoid_left', 'avoid_right', 'avoid_blocker', 'searching']:
                        cmd_type = result.replace('avoid_', '').replace('_blocker', '')
                        if cmd_type == 'searching':
                            # flip a coin to decide turn direction
                            if np.random.rand() < 0.5:
                                cmd_type = 'left'
                            else:
                                cmd_type = 'right'
                        duration = ai_duration if ai_duration else MOVEMENT_DELAYS.get(cmd_type, MOVEMENT_DELAYS['default'])
                        update_robot_position(cmd_type, duration=duration)
                
                # Update live plot
                update_plot(ax)
                
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
            # AI mode: no additional delay needed (already slept for ai_duration)
            # Hardcoded mode: use default movement delay
            if not (ai_decide and ai_duration is not None):
                movement_delay('default')
            
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
        plt.close('all')
    
    return iteration_count


def main(use_ollama=False, ai_decide=False):
    """Main entry point
    
    Args:
        use_ollama: If True, use Ollama for full navigation (deprecated - use ai_decide instead)
        ai_decide: If True, use AI (Ollama/Gemini) for decision making in YOLO navigation
    """
    # Initialize YOLO model
    model = initialize_yolo_model('yolo11l.pt')
    
    # Connect to robot
    sock = connect_to_robot()
    
    # Run navigation loop
    final_iterations = run_navigation_loop(sock, model, use_ollama, ai_decide)
    
    logger.info(f"\nTotal iterations completed: {final_iterations}")
    logger.info("Program ended")


if __name__ == '__main__':
    use_ollama = False  # Deprecated: use full Ollama navigation
    ai_decide = True    # NEW: Use AI for decision making with YOLO detection
    main(use_ollama, ai_decide)
