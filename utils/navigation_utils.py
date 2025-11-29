"""  
Navigation functions for robot movement and obstacle avoidance
"""

import cv2 as cv
import colorlog
import time
import numpy as np
from .detection_utils import detect_objects_yolo, get_largest_object
from .robot_utils import cmd, read_distance, read_ir_sensors

# Setup logger
logger = colorlog.getLogger(__name__)

# Movement timing configuration
MOVEMENT_DELAYS = {
    'forward': 1.3,        # After autonomous forward movement in navigation
    'back': 0.8,           # After backing up during obstacle avoidance
    'left': 0.8,           # After left turns in navigation
    'right': 0.6,          # After right turns in navigation
    'manual': 0.1,         # After manual arrow key controls
    'unstuck_back': 0.8,   # Initial backup in vision-based stuck recovery
    'unstuck_turn': 0.8,   # Turns during vision-based stuck recovery scanning
    'default': 0.3         # Main loop delay between navigation iterations
}

# Movement speed configuration (0-100)
# Can be overridden by importing module
MOVEMENT_SPEEDS = {
    'forward_normal': 100,    # When exploring or no objects detected
    'forward_close': 60,     # When approaching target or avoiding obstacles
    'back_normal': 80,       # General backing up during obstacle avoidance
    'back_avoid': 80,        # Initial backup in vision-based stuck recovery
    'left_normal': 50,       # Normal left turns when target on left or avoiding obstacles
    'left_unstuck': 70,      # Left turns during vision-based stuck recovery scanning & positioning
    'right_normal': 50,      # Normal right turns when target on right or avoiding obstacles
    'right_avoid': 60,       # Right turns to avoid obstacles
    'right_search': 40,      # Slow rotation when searching for target
    'default': 50            # Fallback for any unspecified movement
}


def movement_delay(movement_type):
    """
    Configurable delay after robot movements.
    Use this to experiment with timing without changing code throughout.
    
    Args:
        movement_type: Type of movement ('forward', 'back', 'left', 'right', 'manual', etc.)
    
    Returns:
        Actual delay used
    """
    delay = MOVEMENT_DELAYS.get(movement_type, MOVEMENT_DELAYS['default'])
    time.sleep(delay)
    return delay


def movement_speed(speed_type):
    """
    Get configured speed for a movement type.
    
    Args:
        speed_type: Type of movement speed (e.g., 'forward_normal', 'back_avoid')
    
    Returns:
        Speed value (0-100)
    """
    return MOVEMENT_SPEEDS.get(speed_type, MOVEMENT_SPEEDS['default'])


def vision_based_stuck_recovery(sock, capture_callback, reconnect_callback):
    """
    Vision-based stuck recovery system using camera servo rotation.
    When robot is stuck, it:
    1. Backs up a bit to clear immediate obstacle
    2. Rotates camera left (90°) and captures image
    3. Rotates camera right (-90°) and captures image
    4. Returns camera to center (0°)
    5. Compares both images to find more open path
    6. Turns robot toward best direction
    
    Args:
        sock: Robot socket connection
        capture_callback: Function that returns current camera frame (no args)
        reconnect_callback: Function that reconnects socket (takes sock, returns new sock)
    
    Returns:
        tuple: (sock, chosen_direction: str, left_score: float, right_score: float)
               Returns updated socket and direction chosen ('left' or 'right')
    """
    logger.error("\n🧠 VISION-BASED STUCK RECOVERY ACTIVATED")
    
    # Reconnect first to ensure fresh command buffer
    logger.debug("Reconnecting before stuck recovery...")
    sock = reconnect_callback(sock)
    if sock is None:
        logger.error("Failed to reconnect before stuck recovery")
        return None, None, 0.0, 0.0
    
    # Step 1: Back up to clear immediate obstacle (2 commands)
    logger.info("Step 1: Backing up to clear obstacle...")
    cmd(sock, 'move', where='back', at=movement_speed('back_avoid'))
    time.sleep(movement_delay('unstuck_back'))
    cmd(sock, 'stop')
    time.sleep(0.3)  # Stabilize
    
    # Step 2: Rotate camera left and capture (1 command)
    logger.info("Step 2: Scanning LEFT (camera rotation 60°)...")
    cmd(sock, 'rotate', at=150)  # Rotate camera/ultrasonic servo left (safe range)
    time.sleep(0.4)  # Wait for servo to settle
    left_image = capture_callback()
    
    # Reconnect after 3 commands to stay under limit
    logger.debug("Reconnecting after left scan (3 commands used)...")
    sock = reconnect_callback(sock)
    if sock is None:
        logger.error("Failed to reconnect during stuck recovery")
        return None, None, 0.0, 0.0
    
    # Step 3: Rotate camera right and capture (1 command)
    logger.info("Step 3: Scanning RIGHT (camera rotation -60°)...")
    cmd(sock, 'rotate', at=30)  # Rotate camera/ultrasonic servo right (safe range)
    time.sleep(0.4)  # Wait for servo to settle
    right_image = capture_callback()
    
    # Step 4: Return camera to center (1 command)
    logger.info("Step 4: Returning camera to center...")
    cmd(sock, 'rotate', at=90)  # Return to center
    time.sleep(0.3)  # Wait for servo to settle
    
    # Total: 2 commands in this batch (rotate right, rotate center)
    
    # Step 5: Compare images to determine openness
    logger.info("Step 5: Analyzing path options...")
    left_score = calculate_path_openness(left_image)
    right_score = calculate_path_openness(right_image)
    
    logger.info(f"📊 Path Analysis - Left: {left_score:.2f}, Right: {right_score:.2f}")
    
    # Reconnect after 2 more commands before final turn
    logger.debug("Reconnecting before final turn (2 commands used)...")
    sock = reconnect_callback(sock)
    if sock is None:
        logger.error("Failed to reconnect during stuck recovery")
        return None, None, left_score, right_score
    
    # Step 6: Turn robot toward best direction (2 commands)
    if left_score > right_score:
        chosen = 'left'
        logger.info(f"✅ LEFT path more open ({left_score:.2f} > {right_score:.2f})")
        logger.info("Turning robot LEFT...")
        cmd(sock, 'move', where='left', at=movement_speed('left_unstuck'))
        time.sleep(1.2)  # Turn left
        cmd(sock, 'stop')
    else:
        chosen = 'right'
        logger.info(f"✅ RIGHT path more open ({right_score:.2f} >= {left_score:.2f})")
        logger.info("Turning robot RIGHT...")
        cmd(sock, 'move', where='right', at=movement_speed('right_normal'))
        time.sleep(1.2)  # Turn right
        cmd(sock, 'stop')
    
    logger.info(f"🎯 Proceeding {chosen.upper()}")
    
    return sock, chosen, left_score, right_score


def calculate_path_openness(image):
    """
    Calculate how "open" a path appears in an image.
    Higher score = more navigable space.
    
    Uses multiple metrics:
    - Edge density (fewer edges = more open)
    - Brightness in bottom half (brighter ground = clearer path)
    - Vertical line detection (fewer vertical lines = less obstacles)
    
    Args:
        image: BGR image from camera
    
    Returns:
        float: Openness score (0-100, higher is better)
    """
    if image is None:
        return 0.0
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Focus on bottom 60% of image (robot's path)
    roi_start = int(height * 0.4)
    roi = gray[roi_start:, :]
    
    # Metric 1: Edge density (lower is better - less obstacles)
    edges = cv.Canny(roi, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    edge_score = (1 - edge_density) * 40  # Inverted, max 40 points
    
    # Metric 2: Average brightness in path area (brighter = clearer floor)
    brightness = np.mean(roi)
    brightness_score = (brightness / 255) * 30  # Max 30 points
    
    # Metric 3: Vertical edge count (fewer vertical lines = fewer obstacles)
    sobelx = cv.Sobel(roi, cv.CV_64F, 1, 0, ksize=3)
    vertical_edges = np.sum(np.abs(sobelx) > 50)
    vertical_density = vertical_edges / roi.size
    vertical_score = (1 - vertical_density) * 30  # Inverted, max 30 points
    
    total_score = edge_score + brightness_score + vertical_score
    
    logger.debug(f"  Edge: {edge_score:.1f}, Brightness: {brightness_score:.1f}, Vertical: {vertical_score:.1f}")
    
    return total_score


def check_ultrasonic_distance(sock):
    """
    Read ultrasonic distance sensor and back up if too close.
    Returns: distance in cm, or None if reading failed
    """
    distance = read_distance(sock)
    if distance is not None:
        logger.debug(f"\033[94mUltrasonic: {distance}cm\033[0m")  # Blue
        
        # If object is very close, back up
        if distance < 20:
            logger.warning(f"Object very close ({distance}cm), backing up!")
            cmd(sock, 'move', where='back', at=movement_speed('back_avoid'))
            import time
            time.sleep(1)
            cmd(sock, 'stop')
        
        return distance
    else:
        logger.warning("Failed to read ultrasonic sensor")
        return None


def ultrasonic_safety_check(sock, threshold=30):
    """
    Ultrasonic safety check before moving forward.
    Reads distance sensor and backs up + turns if obstacle too close.
    
    Args:
        sock: Robot socket connection
        threshold: Distance threshold in cm (default 30cm)
    
    Returns:
        tuple: (safe: bool, distance: float or None)
               safe=True means clear to proceed, safe=False means obstacle avoided
    """
    distance = read_distance(sock)
    logger.info(f"🔍 Ultrasonic reading: {distance}cm")
    
    # Treat 0 as "obstacle very close" - sensor might return 0 for objects too close to measure
    # If distance is 0-threshold, something is very close
    if distance is not None and 0 <= distance <= threshold:
        logger.error(f"⚠️ ULTRASONIC ALERT: Object at {distance}cm - backing up!")
        logger.debug("\033[93m↓ Moving BACKWARD\033[0m")  # Yellow
        cmd(sock, 'move', where='back', at=movement_speed('back_avoid'))
        time.sleep(0.5)  # Let it back up
        
        # Then turn to avoid
        logger.debug("\033[96m→ Turning RIGHT to avoid\033[0m")  # Cyan
        cmd(sock, 'move', where='right', at=movement_speed('right_avoid'))
        return False, distance  # Not safe, obstacle avoided
    
    return True, distance  # Safe to proceed


def navigate_with_yolo(sock, img, model, target_class=None, avoid_classes=None):
    """
    Navigate the robot based on YOLO object detection.
    - target_class: specific object class to follow (e.g., 'person', 'cup', 'chair')
    - avoid_classes: list of classes to avoid (e.g., ['person', 'chair', 'dog'])
    """
    # Vision-based navigation
    objects, annotated_img = detect_objects_yolo(img, model)
    cv.imshow('Camera', annotated_img)
    cv.waitKey(1)
    
    height, width = annotated_img.shape[:2]
    
    if len(objects) == 0:
        # No objects detected, move forward slowly
        logger.info(" ")
        logger.debug("\033[92m↑ Moving FORWARD\033[0m")  # Green
        cmd(sock, 'move', where='forward', at=movement_speed('forward_normal'))
        return 'forward'
    
    # Filter for target class if specified
    if target_class:
        target_objects = [obj for obj in objects if obj['class'] == target_class]
        if target_objects:
            # Find largest target object
            target = max(target_objects, key=lambda x: x['area'])
            
            # Check for obstacles in the way (objects that are not the target)
            other_objects = [obj for obj in objects if obj['class'] != target_class]
            
            # If there are other objects blocking the path, avoid them first
            if other_objects:
                blocking_objects = [obj for obj in other_objects 
                                  if obj['area'] > width * height * 0.08 and obj['position'] == 'center']
                if blocking_objects:
                    blocker = max(blocking_objects, key=lambda x: x['area'])
                    logger.warning(f"⚠️ {blocker['class']} blocking path to {target_class}! Avoiding...")
                    logger.debug("\033[96m→ Turning RIGHT\033[0m")  # Cyan
                    cmd(sock, 'move', where='right', at=movement_speed('right_avoid'))
                    return 'avoid_blocker'
            
            # Navigate toward target
            if target['position'] == 'center':
                # Before moving forward, check again for any blocking objects (more aggressive check)
                if other_objects:
                    blocking_objects = [obj for obj in other_objects 
                                      if obj['area'] > width * height * 0.05 and obj['position'] == 'center']
                    if blocking_objects:
                        blocker = max(blocking_objects, key=lambda x: x['area'])
                        logger.warning(f"⚠️ {blocker['class']} directly blocking forward path! Avoiding...")
                        logger.debug("\033[96m→ Turning RIGHT\033[0m")  # Cyan
                        cmd(sock, 'move', where='right', at=movement_speed('right_avoid'))
                        return 'avoid_blocker'
                
                # Check if close enough (object is large)
                if target['area'] > width * height * 0.3:
                    logger.info(f"Reached {target_class}, stopping")
                    logger.debug("\033[91m■ STOP\033[0m")  # Red
                    cmd(sock, 'stop')
                    return 'reached'
                else:
                    # ULTRASONIC SAFETY CHECK before moving forward (DISABLED - sensor not working)
                    # safe, distance = ultrasonic_safety_check(sock, threshold=30)
                    # if not safe:
                    #     return 'ultrasonic_avoid'
                    
                    logger.info(f"Target {target_class} centered, moving forward")
                    logger.debug("\033[92m↑ Moving FORWARD\033[0m")  # Green
                    cmd(sock, 'move', where='forward', at=movement_speed('forward_close'))
                    return 'forward'
            elif target['position'] == 'left':
                # If chair is large enough (reasonably close), just turn
                # If still small (far away), move forward while turning
                if target['area'] > width * height * 0.15:
                    logger.info(f"Target {target_class} on left (close), turning left")
                    logger.debug("\033[95m← Turning LEFT\033[0m")  # Magenta
                    cmd(sock, 'move', where='left', at=movement_speed('left_normal'))
                    return 'left'
                else:
                    logger.info(f"Target {target_class} on left (far), moving forward")
                    logger.debug("\033[92m↑ Moving FORWARD\033[0m")  # Green
                    cmd(sock, 'move', where='forward', at=movement_speed('forward_normal'))
                    return 'forward'
            else:  # right
                # If chair is large enough (reasonably close), just turn
                # If still small (far away), move forward while turning
                if target['area'] > width * height * 0.15:
                    logger.info(f"Target {target_class} on right (close), turning right")
                    logger.debug("\033[96m→ Turning RIGHT\033[0m")  # Cyan
                    cmd(sock, 'move', where='right', at=movement_speed('right_normal'))
                    return 'right'
                else:
                    logger.info(f"Target {target_class} on right (far), moving forward")
                    logger.debug("\033[92m↑ Moving FORWARD\033[0m")  # Green
                    cmd(sock, 'move', where='forward', at=movement_speed('forward_normal'))
                    return 'forward'
        else:
            # Target not found, search by rotating
            logger.warning(f"Searching for {target_class}...")
            cmd(sock, 'move', where='right', at=movement_speed('right_search'))
            return 'searching'
    
    # Avoid obstacles mode
    if avoid_classes:
        obstacles = [obj for obj in objects if obj['class'] in avoid_classes]
    else:
        obstacles = objects  # Avoid all detected objects
    
    if obstacles:
        # Find largest/closest obstacle
        obstacle = get_largest_object(obstacles)
        logger.warning(f"obstacle: {obstacle}\n")
        # If obstacle is too close (large area), take action
        if obstacle['area'] > width * height * 0.15:
            logger.error(f"Avoiding {obstacle['class']} ({obstacle['position']})")
            if obstacle['position'] == 'center':
                logger.debug("\033[96m→ Turning RIGHT\033[0m")  # Cyan
                cmd(sock, 'move', where='right', at=movement_speed('right_avoid'))
                return 'avoid_right'
            elif obstacle['position'] == 'left':
                logger.debug("\033[96m→ Turning RIGHT\033[0m")  # Cyan
                cmd(sock, 'move', where='right', at=movement_speed('right_normal'))
                return 'avoid_right'
            else:  # right
                logger.debug("\033[95m← Turning LEFT\033[0m")  # Magenta
                cmd(sock, 'move', where='left', at=movement_speed('left_normal'))
                return 'avoid_left'
        else:
            # Obstacles far enough, can move forward
            logger.debug("\033[92m↑ Moving FORWARD\033[0m")  # Green
            cmd(sock, 'move', where='forward', at=movement_speed('forward_close'))
            return 'forward'
    
    return 'idle'

