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
from utils.navigation_utils import ai_search_decision, navigate_with_yolo, movement_delay, movement_speed, MOVEMENT_DELAYS, MOVEMENT_SPEEDS
from utils.depth_estimation import DepthEstimator
from utils.video_logger import VideoLogger

from utils.connection_utils import connect_to_robot,periodic_reconnect
from utils.tts_utils import announce_target_found, speak

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
last_nav_command = 'none'  # Track last navigation command for stuck detection

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

# Stuck detection (consecutive forward movements with minimal area change)
last_forward_objects = None
last_forward_command_count = 0

# Rule 3: Target confirmation configuration
RULE3_ENABLED = True  # Enable/disable target confirmation
RULE3_AREA_THRESHOLD = 0.15  # Default minimum area ratio (15% of frame) to trigger confirmation

# Class-based area thresholds for Rule 3 (in pixels for 640x480 frame = 307,200 px²)
# Small objects need absolute pixel thresholds, large objects use percentage
RULE3_CLASS_THRESHOLDS = {
    'sports ball': 15000,      # ~5% of frame - ball at close range
    'cup': 20000,              # ~6.5% of frame
    'bottle': 25000,           # ~8% of frame
    'cell phone': 20000,       # ~6.5% of frame
    'mouse': 15000,            # ~5% of frame
    'remote': 20000,           # ~6.5% of frame
    'book': 30000,             # ~10% of frame
    'chair': 0.15,             # 15% ratio - use percentage for large objects
    'couch': 0.20,             # 20% ratio
    'bed': 0.20,               # 20% ratio
    'dining table': 0.18,      # 18% ratio
    'tv': 0.15,                # 15% ratio
}

# AI decision memory (track last N decisions)
ai_decision_history = []
AI_MEMORY_SIZE = 5  # Keep last 5 decisions

# Flag capture tracking (game ends after 3 flags)
flags_captured = 0
FLAGS_REQUIRED = 3  # Number of flags needed to win

# Movement history tracking (last 3 movements with target info)
movement_history = []
MOVEMENT_HISTORY_SIZE = 3

# Frame counter (skip target reached check on first frames)
frame_count = 0
MIN_FRAMES_BEFORE_TARGET_CHECK = 2  # Wait at least 2 frames before checking target reached

# Video recording
video_logger = None


def print_controls():
    """Print keyboard control instructions"""
    logger.info("\nKeyboard Controls:")
    logger.info("  'k' - Stop and exit")
    logger.info("  'v' - Save video and exit")
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
    global paused, manual_mode, iteration_count, video_logger
    
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
    
    # Save video and exit command
    elif key == ord('v'):
        logger.info("\n'v' pressed - Saving video and exiting...")
        return 'save_and_exit'
    
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





def is_robot_stuck(current_frame, last_nav_command='none', current_objects=None, target='chair', area_threshold=0.02):
    """
    Detect if robot is stuck using two rules:
    
    Rule 1 (Frame-based): Compare current frame with previous frame.
    Returns True if robot appears stuck (similar frames for multiple iterations)
    
    Rule 2 (Object-based): Check for 2 consecutive 'forward' commands with same target 
    object showing ±2% area change. This triggers vision-based stuck recovery.
    
    Args:
        current_frame: Current camera frame
        last_nav_command: Last navigation command executed
        current_objects: Currently detected objects with areas
        target: Target object class to track
        area_threshold: Maximum allowed area change ratio (default: 0.02 = ±2%)
    
    Returns:
        (is_stuck: bool, reason: str)
    """
    global previous_frame, stuck_counter
    global last_forward_objects, last_forward_command_count
    
    # Rule 2: Check consecutive forward commands with minimal area change
    if last_nav_command == 'forward':
        last_forward_command_count += 1
        
        # Check if we have 2 consecutive forwards with objects
        if last_forward_command_count >= 2 and last_forward_objects and current_objects:
            # Check if same objects detected (by class) with minimal area change
            target_objects_now = [obj for obj in current_objects if obj['class'] == target]
            target_objects_prev = [obj for obj in last_forward_objects if obj['class'] == target]
            
            if target_objects_now and target_objects_prev:
                # Get largest target object from each
                current_target = max(target_objects_now, key=lambda x: x['area'])
                prev_target = max(target_objects_prev, key=lambda x: x['area'])
                
                # Calculate area change percentage
                if prev_target['area'] > 0:
                    area_change = abs(current_target['area'] - prev_target['area']) / prev_target['area']
                    
                    if area_change <= area_threshold:
                        reason = f"2 consecutive forwards with {area_change*100:.1f}% area change (vision-based recovery)"
                        logger.error(f"🚫 ROBOT STUCK DETECTED! {reason}")
                        
                        # Reset state after detection
                        last_forward_objects = None
                        last_forward_command_count = 0
                        return True, reason
        
        # Update last forward objects for next check
        last_forward_objects = current_objects.copy() if current_objects else None
    else:
        # Reset counter if not forward
        last_forward_objects = None
        last_forward_command_count = 0
    
    # Rule 1: Frame comparison (existing logic)
    if previous_frame is None:
        previous_frame = current_frame.copy()
        return False, "Not stuck"
    
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
        reason = f"{stuck_counter} similar frames (frame-based)"
        logger.error(f"🚫 ROBOT STUCK DETECTED! {reason}")
        stuck_counter = 0  # Reset counter after detection
        return True, reason
    
    return False, "Not stuck"


def check_target_reached(current_frame, current_objects, target, last_nav_command='forward'):
    """
    Check if robot has reached the target object (Rule 3).
    Returns True if large target object is touching bottom of frame.
    Note: Skips check for first few frames to avoid false positives at startup.
    
    Args:
        current_frame: Current camera frame
        current_objects: Currently detected objects with areas
        target: Target object class to track
        last_nav_command: Last navigation command executed (unused, kept for compatibility)
    
    Returns:
        (reached: bool, reason: str)
    """
    global frame_count
    
    # Skip check on first few frames (e.g., if MIN=3, skip frames 0,1,2,3 and start checking at frame 4)
    if frame_count <= MIN_FRAMES_BEFORE_TARGET_CHECK:
        return False, "Warming up - not checking yet", None
    
    if not RULE3_ENABLED or not current_objects:
        return False, "Not at target", None
    
    h, w = current_frame.shape[:2]
    target_objects = [obj for obj in current_objects if obj['class'] == target]
    
    if not target_objects:
        return False, "No target detected", None
    
    largest_target = max(target_objects, key=lambda x: x['area'])
    
    # Get class-specific threshold or use default
    threshold = RULE3_CLASS_THRESHOLDS.get(target, RULE3_AREA_THRESHOLD)
    
    # Check if threshold is met (supports both absolute pixels and ratios)
    area_px = largest_target['area']
    area_ratio = area_px / (w * h)
    
    if isinstance(threshold, float):
        # Percentage-based threshold for large objects
        threshold_met = area_ratio > threshold
        threshold_desc = f"{threshold*100:.0f}%"
    else:
        # Absolute pixel threshold for small objects
        threshold_met = area_px > threshold
        threshold_desc = f"{threshold}px²"
    
    # Check if bottom edge touches or is very close to frame bottom
    touches_bottom = False
    distance_from_bottom = 0
    if 'bbox' in largest_target:
        # bbox format: [x1, y1, x2, y2]
        y2 = largest_target['bbox'][3]  # Bottom edge of object
        # Object must be within 5% of frame height from bottom (no gap)
        distance_from_bottom = h - y2
        gap_threshold = h * 0.05  # 5% of frame height
        touches_bottom = distance_from_bottom <= gap_threshold
        
        logger.debug(f"Rule 3 check: {target} area={area_px}px² ({area_ratio*100:.1f}%), threshold={threshold_desc}, y2={y2}, h={h}, gap={distance_from_bottom:.0f}px, touches={touches_bottom}")
    else:
        # Fallback: if no bbox, can't reliably check bottom position
        touches_bottom = False
    
    if threshold_met and touches_bottom:
        reason = f"{target} at target position ({area_px}px²/{area_ratio*100:.1f}% area, gap={distance_from_bottom:.0f}px, threshold={threshold_desc})"
        logger.warning(f"🎯 POSSIBLE TARGET REACHED! {reason}")
        return True, reason, largest_target
    
    return False, "Not at target", None
    
    return False, "Not at target", None



def add_movement_to_history(command, target_detected, target_position=None, target_area=0):
    """
    Track robot movements with target detection info.
    
    Args:
        command: Movement command ('forward', 'back', 'left', 'right', 'searching')
        target_detected: Boolean, was target detected in this frame
        target_position: Position of target ('left', 'center', 'right', None)
        target_area: Area of target object in pixels
    """
    global movement_history
    
    movement_entry = {
        'command': command,
        'target_detected': target_detected,
        'target_position': target_position,
        'target_area': target_area,
        'timestamp': time.time()
    }
    
    movement_history.append(movement_entry)
    
    # Keep only last N movements
    if len(movement_history) > MOVEMENT_HISTORY_SIZE:
        movement_history.pop(0)


def check_lost_target_after_turn(target_class):
    """
    Detect if robot turned toward target but then lost it.
    This suggests target is still in that direction but detection failed.
    Returns command to execute ('back') or None.
    
    Pattern to detect:
    1. Was searching (left or right turn)
    2. Found target in that direction
    3. Moved toward target (turned that direction)
    4. Now target is NOT detected
    
    Solution: Send 'back' command to retreat and re-acquire target
    """
    global movement_history
    
    if len(movement_history) < 2:
        return None
    
    # Get last 2 movements
    prev_move = movement_history[-2]
    current_move = movement_history[-1]
    
    # Check pattern: had target, turned toward it, now lost it
    if (prev_move['target_detected'] and 
        prev_move['command'] in ['left', 'right', 'searching'] and
        not current_move['target_detected']):
        
        logger.warning(f"⚠️ Target lost after turning {prev_move['command']}!")
        logger.info(f"Previous: {prev_move['command']} with target at {prev_move['target_position']}")
        logger.info(f"Current: {current_move['command']} with no target detected")
        logger.info("📍 Executing 'back' to re-acquire target")
        
        return 'back'
    
    return None


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


def update_plot(ax, target='chair'):
    """Update the live plot with current robot position"""
    # target parameter is now passed from run_navigation_loop
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
        
        # Target is green, others are black
        color = 'green' if obj['class'] == target else 'black'
        marker = 's' if obj['class'] == target else 'o'  # square for target, circle for others
        size = 150 if obj['class'] == target else 80
        
        ax.scatter(obj['x'], obj['y'], c=color, marker=marker, s=size, 
                  alpha=alpha, edgecolors='white', linewidths=1.5, zorder=4,
                  label=f"{obj['class']}" if obj['age'] == 0 else "")
    
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


def run_navigation_loop(sock, model, use_ollama, ai_decide, target='chair', capture_video=True, target_confidence=0.75):
    """
    Main navigation loop with YOLO object detection.
    Args:
        capture_video: If True, record navigation video with logs (default: True)
        target_confidence: Minimum confidence threshold for target detection (default: 0.75)
    Returns: final iteration count
    """
    # target: YOLO class name to search for (e.g., 'chair', 'ball', 'person')
    global iteration_count, paused, video_logger, last_nav_command, flags_captured, frame_count
    
    logger.info("\nStarting autonomous navigation with YOLO...")
    logger.warning("\nNOTE: Robot firmware closes connection after 8 commands - auto-reconnecting every 3 iterations")
    print_controls()
    
    # Setup camera window
    cv.namedWindow('Camera', cv.WINDOW_NORMAL)
    cv.resizeWindow('Camera', 800, 600)
    cv.moveWindow('Camera', 0, 0)
    
    # Setup live plot (DISABLED)
    # fig, ax = setup_live_plot()
    # update_plot(ax, target)  # Draw initial plot
    
    iteration_count = 0
    video_initialized = False  # Track if video has been initialized
    
    try:
        while True:
            # Handle keyboard input
            key_result = handle_keyboard_input(sock)
            
            if key_result == 'exit':
                break
            elif key_result == 'save_and_exit':
                # Save video and exit
                if video_logger and video_logger.is_recording:
                    video_logger.add_log("Saving video and exiting...", "INFO")
                    video_logger.stop_recording()
                    logger.info(f"✅ Video saved: {video_logger.output_path}")
                logger.info("Exiting program...")
                break
            elif key_result in ['restart', 'pause', 'manual', 'stop']:
                if key_result == 'restart':
                    iteration_count = 0
                continue
            
            # Skip iteration if paused
            if paused:
                time.sleep(0.1)
                # update_plot(ax, target)  # Keep plot responsive (DISABLED)
                continue
            
            # Check connection
            if not check_connection(sock):
                logger.error("\nERROR: Lost connection to robot!")
                logger.info("Attempting to reconnect...")
                
                # Try to reconnect (up to 3 attempts)
                reconnect_attempts = 0
                max_reconnect_attempts = 3
                
                while reconnect_attempts < max_reconnect_attempts:
                    reconnect_attempts += 1
                    logger.info(f"Reconnection attempt {reconnect_attempts}/{max_reconnect_attempts}...")
                    
                    new_sock = reconnect_robot(sock)
                    if new_sock is not None:
                        sock = new_sock
                        logger.info("✅ Reconnected successfully!")
                        break
                    else:
                        logger.warning(f"❌ Reconnection attempt {reconnect_attempts} failed")
                        if reconnect_attempts < max_reconnect_attempts:
                            time.sleep(2)  # Wait before retry
                
                # If all reconnection attempts failed, pause and wait for user
                if reconnect_attempts >= max_reconnect_attempts and not check_connection(sock):
                    logger.error(f"❌ Failed to reconnect after {max_reconnect_attempts} attempts")
                    logger.info("Press 'r' to retry connection or 'k' to exit")
                    paused = True
                    continue
            
            # Increment iteration
            iteration_count += 1
            
            # Show progress
            if iteration_count % 3 == 0:
                logger.info(f"\n--- Iteration {iteration_count} ---")
            
            # Periodic reconnect to prevent firmware timeout
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
            
            # Initialize video logger on first successful frame capture (if enabled)
            if not video_initialized and capture_video:
                video_logger = VideoLogger(fps=2.0, frame_width=640, frame_height=480)
                video_logger.start_recording()
                video_logger.add_log("Navigation started", "INFO")
                logger.info("📹 Video recording auto-started")
                video_initialized = True
            elif not video_initialized and not capture_video:
                # Mark as initialized even if not capturing video
                video_initialized = True
                logger.info("📹 Video recording disabled")

            # First, detect objects with YOLO to get annotated image with bounding boxes
            from utils.detection_utils import detect_objects_yolo
            objects, annotated_img = detect_objects_yolo(img, model, target_class=target)
            
            # Increment frame counter
            global frame_count
            frame_count += 1
            
            # Check if target is reached (Rule 3) - before stuck detection
            target_reached, reach_reason, target_object = check_target_reached(img, objects, target, last_nav_command)
            if target_reached:
                # Annotate image
                if target_object and 'bbox' in target_object:
                    x1, y1, x2, y2 = target_object['bbox']
                    cv.putText(annotated_img, "TARGET REACHED!", (x1, y1 - 40), 
                             cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                
                speak(f"Target reached! Type 'ball' to continue")
                # speak("Παρακαλώ πληκτρολογήστε τη μυστική λέξη για να επιβεβαιώσετε ότι φτάσατε στον στόχο")

                logger.info(f"\n{'='*60}")
                logger.info(f"🎯 TARGET REACHED!")
                logger.info(f"   Target: {target}")
                logger.info(f"   {reach_reason}")
                logger.info(f"   Iterations: {iteration_count}")
                logger.info(f"   Flags captured: {flags_captured}")
                logger.info(f"{'='*60}")
                logger.info(f"\n⌨️  Type 'ball' to continue or 'end' to end game (or press 'r' to resume navigation)\n")
                
                # Stop robot
                if check_connection(sock):
                    cmd(sock, 'stop')
                
                # Log to video
                if video_logger and video_logger.is_recording:
                    video_logger.add_log(f"🎯 Awaiting user confirmation for {target}", "INFO")
                
                # Wait for user input
                paused = True
                waiting_for_confirmation = True
                
                while waiting_for_confirmation:
                    cv.imshow('Camera', annotated_img)
                    key = cv.waitKey(100) & 0xFF
                    
                    # Check for 'r' to resume
                    if key == ord('r'):
                        logger.info("\n❌ User cancelled - resuming navigation")
                        if video_logger and video_logger.is_recording:
                            video_logger.add_log("User cancelled - resuming", "WARNING")
                        paused = False
                        waiting_for_confirmation = False
                        last_nav_command = 'none'
                        break
                    
                    # Check for 'k' to exit
                    elif key == ord('k'):
                        logger.warning("\n'k' pressed - Stopping and exiting...")
                        if check_connection(sock):
                            cmd(sock, 'stop')
                        waiting_for_confirmation = False
                        return iteration_count
                    
                    # Check terminal for secret word input
                    import sys
                    import select
                    if select.select([sys.stdin], [], [], 0)[0]:
                        user_input = sys.stdin.readline().strip().lower()
                        
                        # Check for "end" (end game immediately)
                        if user_input == 'end':
                            flags_captured += 1
                            logger.info(f"\n{'='*60}")
                            logger.info(f"🏁 GAME ENDED!")
                            logger.info(f"   Target: {target}")
                            logger.info(f"   Total iterations: {iteration_count}")
                            logger.info(f"   Flags captured: {flags_captured}")
                            logger.info(f"{'='*60}\n")
                            
                            speak("Game ended!")
                            
                            if video_logger and video_logger.is_recording:
                                video_logger.add_log(f"🏁 GAME ENDED!", "SUCCESS")
                            
                            logger.info("Press 'k' to exit")
                            waiting_for_confirmation = False
                            paused = True  # Stay paused until user exits
                        
                        # Check for "ball" (continue searching)
                        elif user_input == 'ball':
                            flags_captured += 1
                            logger.info(f"\n{'='*60}")
                            logger.info(f"🏆 FLAG {flags_captured} CAPTURED!")
                            logger.info(f"   Target: {target}")
                            logger.info(f"   Total iterations: {iteration_count}")
                            logger.info(f"   Flags: {flags_captured}")
                            logger.info(f"{'='*60}\n")
                            
                            # Announce via TTS
                            from utils.tts_utils import announce_target_reached
                            announce_target_reached(target)
                            
                            if video_logger and video_logger.is_recording:
                                video_logger.add_log(f"🏆 FLAG {flags_captured} CAPTURED!", "SUCCESS")
                            
                            # Always continue searching (game only ends with "end")
                            logger.info(f"🔍 Continuing search for flag {flags_captured + 1}...")
                            speak(f"Flag {flags_captured} captured! Searching for flag {flags_captured + 1}")
                            
                            # Capture fresh image (target should be removed by now)
                            logger.info("📸 Capturing fresh image...")
                            time.sleep(1.0)  # Give user time to remove target
                            fresh_img = capture()
                            if fresh_img is not None:
                                cv.imshow('Camera', fresh_img)
                                cv.waitKey(100)
                                logger.info("✅ Fresh image captured - resuming search")
                            
                            # Reset frame counter to avoid immediate re-trigger
                            frame_count = 0
                            
                            paused = False
                            waiting_for_confirmation = False
                        else:
                            logger.warning(f"❌ Incorrect input: '{user_input}' - Type 'ball' to continue or 'end' to end game (or press 'r' to resume)")
                
                # Reset last nav command
                last_nav_command = 'none'
                continue
            
            # Check if robot is stuck (Rules 1 & 2: frame-based and object-based)
            is_stuck, stuck_reason = is_robot_stuck(img, last_nav_command, objects, target)
            if is_stuck:
                # Real stuck detection - use vision-based recovery
                logger.error(f"🚨 Robot appears STUCK! Reason: {stuck_reason}")
                
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
                
                # Reset last nav command after recovery
                last_nav_command = 'none'
                
                # Skip navigation this iteration
                # update_plot(ax, target)  # (DISABLED)
                continue
            
            # Log all detected objects with their labels and probabilities
            if objects:
                logger.info(f"📸 Detected {len(objects)} object(s):")
                if video_logger and video_logger.is_recording:
                    video_logger.add_log(f"Detected {len(objects)} object(s)", "INFO")
                for i, obj in enumerate(objects, 1):
                    logger.info(f"  {i}. {obj['class']} - confidence: {obj['confidence']:.2%} - position: {obj['position']} - area: {obj['area']:.0f}px²")
                    if video_logger and video_logger.is_recording:
                        video_logger.add_log(f"{obj['class']} {obj['confidence']:.0%} {obj['position']}", "DEBUG")
            else:
                logger.info("📸 No objects detected in current frame")
                if video_logger and video_logger.is_recording:
                    video_logger.add_log("No objects detected", "WARNING")
            
            # Check if we lost target after turning toward it
            recovery_command = check_lost_target_after_turn(target)
            if recovery_command == 'back':
                logger.warning("🔙 Backing up to re-acquire lost target...")
                if check_connection(sock):
                    cmd(sock, 'move', where='back', at=movement_speed('back_avoid'))
                    movement_delay('back_avoid')
                    
                    # Reconnect after movement
                    sock = reconnect_robot(sock)
                    if sock is None:
                        logger.error("Failed to reconnect after recovery back command")
                        paused = True
                        continue
                
                # Add this recovery movement to history
                target_obj = next((obj for obj in objects if obj['class'] == target), None)
                add_movement_to_history(
                    'back',
                    target_obj is not None,
                    target_obj['position'] if target_obj else None,
                    target_obj['area'] if target_obj else 0
                )
                
                # Skip normal navigation this iteration
                # update_plot(ax, target)
                continue
            
            # Navigate using YOLO or Ollama
            try:
                if use_ollama:

                    
                    # Show YOLO detections briefly
                    cv.imshow('Camera', annotated_img)
                    cv.waitKey(1)
                    
                    # Now use Ollama with the ANNOTATED image (with bounding boxes)
                    from ollama.cam_ollama import navigate_with_ollama
                    result = navigate_with_ollama(sock, annotated_img, target_class=target)
                    
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
                        # AI mode: NO confidence filtering - AI sees all detections and judges quality itself
                        target_objects = [obj for obj in objects if obj['class'] == target]
                        # Find target and compute AI decision
                        if target_objects:
                            target_obj = max(target_objects, key=lambda x: x['area'])
                            other_objects = [obj for obj in objects if obj['class'] != target]
                            
                            # Call AI decision function (takes 5+ seconds)
                            from utils.navigation_utils import ai_navigation_decision
                            ai_decision = ai_navigation_decision(annotated_img, target_obj, other_objects, target, width, height, ai_decision_history)
                            
                            # Log AI decision to video
                            if video_logger and video_logger.is_recording and ai_decision:
                                video_logger.add_log(f"AI Decision: {ai_decision}", "INFO")
                        else:
                            # No target found - use smart search instead of AI search
                            logger.info(f"No {target} detected, will use smart search...")
                            ai_decision = None  # Don't use AI for search, use smart search instead
                        
                        # Reconnect AFTER AI finishes (before sending movement command)
                        # Robot may have closed connection during long AI processing
                        if ai_decision:  # Only reconnect if AI was actually called
                            logger.debug("Reconnecting after AI decision (before sending command)...")
                            sock = reconnect_robot(sock)
                            if sock is None:
                                logger.error("Failed to reconnect after AI decision")
                                paused = True
                                continue
                    
                    # Now navigate with YOLO (with pre-computed AI decision if enabled)
                    # Pass target_confidence to navigate_with_yolo for hardcoded mode filtering
                    result = navigate_with_yolo(sock, annotated_img, model, target_class=target, avoid_classes=None, 
                                              ai_decide=ai_decide, ai_decision=ai_decision,
                                              objects=objects, annotated_img=annotated_img, video_logger=video_logger,
                                              target_confidence=target_confidence)
                    
                    # If searching (no target found), use smart search
                    if result and result[0] == 'searching':
                        logger.info("🔍 Initiating smart search...")
                        if video_logger and video_logger.is_recording:
                            video_logger.add_log("🔍 Initiating smart search", "INFO")
                        
                        if False and ai_decide: # AI search disabled - always use smart_search (AI search too slow)
                            # Use AI-based search decision
                            from utils.navigation_utils import ai_search_decision
                            sock, direction, duration = ai_search_decision(sock, annotated_img, model, target, capture, reconnect_robot, decision_history=ai_decision_history)
                            
                            if video_logger and video_logger.is_recording and direction:
                                video_logger.add_log(f"AI Search: {direction}", "INFO")
                        else:
                            # Use traditional smart search
                            from utils.navigation_utils import smart_search_for_target                        
                            sock, direction = smart_search_for_target(sock, annotated_img, model, target, capture, reconnect_robot)
                            duration = MOVEMENT_DELAYS.get(direction, MOVEMENT_DELAYS['default']) if direction else MOVEMENT_DELAYS['default']
                            
                            if video_logger and video_logger.is_recording and direction:
                                video_logger.add_log(f"Smart Search: {direction}", "INFO")
                        
                        if sock is None:
                            logger.error("Search failed - socket disconnected")
                            paused = True
                            continue
                        
                        # Set result based on direction for position tracking
                        if direction:
                            result = (direction, duration)
                        else:
                            result = ('forward', MOVEMENT_DELAYS['default'])
                    
                    # Store AI decision in history if AI mode is enabled
                    if ai_decide and ai_decision:
                        decision_str, speed, duration = ai_decision
                        target_pos = target_objects[0]['position'] if target_objects else 'none'
                        target_area = target_objects[0]['area'] if target_objects else 0
                        
                        # Store full object information for each detected object
                        objects_snapshot = []
                        for obj in objects:
                            objects_snapshot.append({
                                'class': obj['class'],
                                'position': obj['position'],
                                'area': obj['area'],
                                'confidence': obj['confidence']
                            })
                        
                        ai_decision_history.append({
                            'decision': decision_str,
                            'speed': speed,
                            'duration': duration,
                            'target_position': target_pos,
                            'target_area': target_area,
                            'all_objects': objects_snapshot
                        })
                        # Keep only last N decisions
                        if len(ai_decision_history) > AI_MEMORY_SIZE:
                            ai_decision_history.pop(0)
                        
                        # Log to video if recording
                        if video_logger and video_logger.is_recording:
                            video_logger.add_log(f"AI Decision: {decision_str} (speed={speed}, dur={duration:.1f}s)", "INFO")
                            video_logger.add_log(f"Target: {target_pos}, {len(objects)} objects detected", "DEBUG")
                    
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
                
                # Update position estimate based on navigation result
                # AI mode returns tuple (result, duration), hardcoded mode returns string
                ai_duration = None
                if isinstance(result, tuple):
                    result, ai_duration = result
                
                # Track last navigation command for stuck detection
                if result and result != 'idle':
                    # Extract base command (forward, back, left, right)
                    if result in ['forward', 'back', 'left', 'right']:
                        last_nav_command = result
                    elif result == 'searching':
                        last_nav_command = 'searching'
                    else:
                        last_nav_command = 'none'
                    
                    # Track movement in history with target detection info
                    target_obj = next((obj for obj in objects if obj['class'] == target), None)
                    add_movement_to_history(
                        result if result != 'searching' else last_nav_command,
                        target_obj is not None,
                        target_obj['position'] if target_obj else None,
                        target_obj['area'] if target_obj else 0
                    )
                
                if result and result != 'idle': # update 2D plot
                    if result in ['forward', 'back']:
                        duration = ai_duration if ai_duration else MOVEMENT_DELAYS[result]
                        update_robot_position(result, duration=duration)
                    elif result in ['left', 'right', 'searching']:
                        cmd_type = result
                        # if cmd_type == 'searching':
                        #     # flip a coin to decide turn direction
                        #     if np.random.rand() < 0.5:
                        #         cmd_type = 'left'
                        #     else:
                        #         cmd_type = 'right'
                        duration = ai_duration if ai_duration else MOVEMENT_DELAYS.get(cmd_type, MOVEMENT_DELAYS['default'])
                        update_robot_position(cmd_type, duration=duration)
                
                # Update live plot (DISABLED)
                # update_plot(ax, target)
                
                # Write frame to video if recording
                if video_logger and video_logger.is_recording:
                    combined_frame = video_logger.write_frame(annotated_img)
                    # Optionally show combined frame instead of just camera
                    # cv.imshow('Camera', combined_frame)
                
            except Exception as e:
                logger.error(f"\n!!! ERROR in navigation at iteration {iteration_count}: {e}")
                if video_logger and video_logger.is_recording:
                    video_logger.add_log(f"ERROR: {str(e)[:50]}", "ERROR")
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
        # Stop video recording if active
        if video_logger and video_logger.is_recording:
            logger.info("Stopping video recording...")
            video_logger.stop_recording()
        
        sock.close()
        cv.destroyAllWindows()
        plt.close('all')
    
    return iteration_count


def main(use_ollama=False, ai_decide=False, target='chair', use_segmentation=False, capture_video=True, target_confidence=0.75, rule3_enabled=True, rule3_area_threshold=0.15, robot_ip='192.168.4.1', robot_port=100):
    """Main entry point
    
    Args:
        use_ollama: If True, use Ollama for full navigation (deprecated - use ai_decide instead)
        ai_decide: If True, use AI (Ollama/Gemini) for decision making in YOLO navigation
        target: YOLO class name to search for (e.g., 'chair', 'ball', 'person')
        use_segmentation: If True, use YOLO segmentation model (yolo11l-seg.pt)
        capture_video: If True, record navigation video with logs (default: True)
        target_confidence: Minimum confidence threshold for target detection (default: 0.75)
        rule3_enabled: If True, enable Rule 3 target confirmation (default: True)
        rule3_area_threshold: Minimum area ratio (0.0-1.0) to trigger Rule 3 (default: 0.15 = 15%)
        robot_ip: IP address of robot (default: '192.168.4.1')
        robot_port: Port number of robot (default: 100)
    """
    # Configure Rule 3
    global RULE3_ENABLED, RULE3_AREA_THRESHOLD
    RULE3_ENABLED = rule3_enabled
    RULE3_AREA_THRESHOLD = rule3_area_threshold
    
    logger.info(f"Rule 3 configuration: enabled={RULE3_ENABLED}, area_threshold={RULE3_AREA_THRESHOLD*100:.0f}%")
    
    # Initialize YOLO model - use segmentation if requested
    if use_segmentation:
        model_name = 'yolo11x-seg.pt'  # Large segmentation model
    else:
        model_name = 'yolo11x.pt'  # Extra large detection model
    
    model = initialize_yolo_model(model_name, use_segmentation=use_segmentation)
    
    # Connect to robot
    sock = connect_to_robot(robot_ip, robot_port)
    
    # Run navigation loop
    final_iterations = run_navigation_loop(sock, model, use_ollama, ai_decide, target, capture_video=capture_video, target_confidence=target_confidence)
    
    logger.info(f"\nTotal iterations completed: {final_iterations}")
    logger.info("Program ended")


if __name__ == '__main__':
    # Robot connection settings
    robot_ip = '192.168.4.1'
    robot_port = 100
    
    use_ollama = False  # Deprecated: use full Ollama navigation
    ai_decide =  False    # NEW: Use AI for decision making with YOLO detection
    # target = 'backpack'   
    # target = 'chair' 
    target = 'sports ball'   # Target object class to search for
    use_segmentation = True  # Use YOLO segmentation model for better object understanding
    capture_video = True  # Record navigation video with logs
    target_confidence = 0.1  # Minimum confidence threshold for target detection
    
    # Rule 3: Target confirmation configuration
    rule3_enabled = True  # Enable target confirmation when close to object
    rule3_area_threshold = 0.20  # 30% of frame area to trigger confirmation (closer proximity required)
    
    main(use_ollama, ai_decide, target, use_segmentation,
          capture_video, target_confidence,
          rule3_enabled, rule3_area_threshold,
          robot_ip, robot_port)
