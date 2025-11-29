"""
Navigation functions for robot movement and obstacle avoidance
"""

import cv2 as cv
import colorlog
import time
from .detection_utils import detect_objects_yolo, get_largest_object
from .robot_utils import cmd, read_distance, read_ir_sensors

# Setup logger
logger = colorlog.getLogger(__name__)


def thinking_mode(sock):
    """
    Emergency thinking mode when object is very close (< 20cm).
    Uses exactly 3 commands to respect buffer limit:
    1. Read distance
    2. Move backward
    3. Turn based on IR sensors (left if blocked on right, right otherwise)
    
    Returns: tuple (activated: bool, movements: list) where movements are ('back', 0.8) and ('left'/'right', 0.6)
    """
    # Command 1: Check distance (uses 1 command)
    distance = read_distance(sock)
    
    if distance is None:
        logger.warning("⚠ Could not read distance sensor")
        return False, []
    
    logger.debug(f"Distance: {distance}cm")
    
    # Only activate thinking mode if too close
    if distance >= 20:
        return False, []
    
    # THINKING MODE ACTIVATED
    logger.error(f"\n🧠 THINKING MODE: Object at {distance}cm - backing up!")
    
    movements = []
    
    # Command 2: Back up (uses 1 command)
    cmd(sock, 'move', where='back', at=80)
    time.sleep(0.8)  # Let it back up
    movements.append(('back', 0.8))
    
    # Command 3: Check IR sensors and turn smartly (uses 1 command)
    ir_result = read_ir_sensors(sock)
    
    if ir_result is not None and ir_result == 0:
        # Edge detected or off ground - turn left (safer)
        logger.warning("⚠ Edge/ground issue detected, turning LEFT")
        cmd(sock, 'move', where='left', at=70)
        time.sleep(0.6)
        movements.append(('left', 0.6))
    else:
        # Normal operation - turn right to explore
        logger.info("🔄 Turning RIGHT to avoid obstacle")
        cmd(sock, 'move', where='right', at=70)
        time.sleep(0.6)
        movements.append(('right', 0.6))
    
    logger.info("✓ Thinking mode complete (used 3 commands)")
    return True, movements


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
            cmd(sock, 'move', where='back', at=80)
            import time
            time.sleep(1)
            cmd(sock, 'stop')
        
        return distance
    else:
        logger.warning("Failed to read ultrasonic sensor")
        return None


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
        cmd(sock, 'move', where='forward', at=50)
        return 'forward'
    
    # Filter for target class if specified
    if target_class:
        target_objects = [obj for obj in objects if obj['class'] == target_class]
        if target_objects:
            # Find largest target object
            target = max(target_objects, key=lambda x: x['area'])
            
            # Navigate toward target
            if target['position'] == 'center':
                # Check if close enough (object is large)
                if target['area'] > width * height * 0.3:
                    logger.info(f"Reached {target_class}, stopping")
                    logger.debug("\033[91m■ STOP\033[0m")  # Red
                    cmd(sock, 'stop')
                    return 'reached'
                else:
                    logger.info(f"Target {target_class} centered, moving forward")
                    logger.debug("\033[92m↑ Moving FORWARD\033[0m")  # Green
                    cmd(sock, 'move', where='forward', at=60)
                    return 'forward'
            elif target['position'] == 'left':
                logger.info(f"Target {target_class} on left, turning left")
                logger.debug("\033[95m← Turning LEFT\033[0m")  # Magenta
                cmd(sock, 'move', where='left', at=50)
                return 'left'
            else:  # right
                logger.info(f"Target {target_class} on right, turning right")
                logger.debug("\033[96m→ Turning RIGHT\033[0m")  # Cyan
                cmd(sock, 'move', where='right', at=50)
                return 'right'
        else:
            # Target not found, search by rotating
            logger.warning(f"Searching for {target_class}...")
            cmd(sock, 'move', where='right', at=40)
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
                cmd(sock, 'move', where='right', at=60)
                return 'avoid_right'
            elif obstacle['position'] == 'left':
                logger.debug("\033[96m→ Turning RIGHT\033[0m")  # Cyan
                cmd(sock, 'move', where='right', at=50)
                return 'avoid_right'
            else:  # right
                logger.debug("\033[95m← Turning LEFT\033[0m")  # Magenta
                cmd(sock, 'move', where='left', at=50)
                return 'avoid_left'
        else:
            # Obstacles far enough, can move forward
            logger.debug("\033[92m↑ Moving FORWARD\033[0m")  # Green
            cmd(sock, 'move', where='forward', at=60)
            return 'forward'
    
    return 'idle'

