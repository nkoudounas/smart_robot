"""
Navigation functions for robot movement and obstacle avoidance
"""

import cv2 as cv
from .detection_utils import detect_objects_yolo, get_largest_object
from .robot_utils import cmd, read_distance


def navigate_with_yolo(sock, img, model, target_class=None, avoid_classes=None):
    """
    Navigate the robot based on YOLO object detection.
    
    Args:
        sock: Socket connection to robot
        img: Camera image
        model: YOLO model instance
        target_class: specific object class to follow (e.g., 'person', 'cup', 'chair')
        avoid_classes: list of classes to avoid (e.g., ['person', 'chair', 'dog'])
        
    Returns:
        Navigation state string
    """
    objects, annotated_img = detect_objects_yolo(img, model)
    cv.imshow('Camera', annotated_img)
    cv.waitKey(1)
    
    height, width = annotated_img.shape[:2]
    
    if len(objects) == 0:
        # No objects detected, move forward slowly
        print("No objects detected, moving forward")
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
                    print(f"Reached {target_class}, stopping")
                    cmd(sock, 'stop')
                    return 'reached'
                else:
                    print(f"Target {target_class} centered, moving forward")
                    cmd(sock, 'move', where='forward', at=60)
                    return 'forward'
            elif target['position'] == 'left':
                print(f"Target {target_class} on left, turning left")
                cmd(sock, 'move', where='left', at=50)
                return 'left'
            else:  # right
                print(f"Target {target_class} on right, turning right")
                cmd(sock, 'move', where='right', at=50)
                return 'right'
        else:
            # Target not found, search by rotating
            print(f"Searching for {target_class}...")
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
        
        # If obstacle is too close (large area), take action
        if obstacle['area'] > width * height * 0.15:
            print(f"Avoiding {obstacle['class']} ({obstacle['position']})")
            if obstacle['position'] == 'center':
                cmd(sock, 'move', where='right', at=60)
                return 'avoid_right'
            elif obstacle['position'] == 'left':
                cmd(sock, 'move', where='right', at=50)
                return 'avoid_right'
            else:  # right
                cmd(sock, 'move', where='left', at=50)
                return 'avoid_left'
        else:
            # Obstacles far enough, can move forward
            print("Path clear, moving forward")
            cmd(sock, 'move', where='forward', at=60)
            return 'forward'
    
    return 'idle'


def navigate_with_sensor_fusion(sock, img, model, check_distance=True):
    """
    Hybrid navigation using both vision (YOLO) and distance sensors.
    Provides multiple layers of obstacle detection.
    
    Args:
        sock: Socket connection to robot
        img: Camera image
        model: YOLO model instance
        check_distance: Whether to check ultrasonic sensor (uses 1 extra command)
        
    Returns:
        Navigation state string
        
    Notes:
        Command count: 
        - With distance check: 2 commands (distance + move)
        - Without distance check: 1 command (move only)
        Recommended: Alternate distance checks every other iteration
    """
    height, width = img.shape[:2] if img is not None else (480, 640)
    
    # Layer 1: Ultrasonic distance check (only when enabled)
    # This adds 1 extra command, so we don't check every iteration
    if check_distance:
        distance = read_distance(sock)
        if distance is not None:
            if distance < 20:  # Less than 20cm - EMERGENCY
                print(f"⚠️  EMERGENCY! Object {distance}cm ahead - backing up")
                cmd(sock, 'move', where='back', at=60)
                return 'emergency_backup'
            elif distance < 35:  # Less than 35cm - turn away
                print(f"⚠️  Object {distance}cm ahead - turning right")
                cmd(sock, 'move', where='right', at=60)
                return 'avoiding_close'
            else:
                print(f"Distance: {distance}cm - clear")
    
    # Layer 2: Vision-based navigation (always check, no extra commands)
    if img is not None:
        objects, annotated_img = detect_objects_yolo(img, model)
        cv.imshow('Camera', annotated_img)
        cv.waitKey(1)
        
        # Check for obstacles in vision
        if len(objects) > 0:
            # Find closest/largest obstacle
            obstacle = get_largest_object(objects)
            area_ratio = obstacle['area'] / (width * height)
            
            if area_ratio > 0.20:  # Very close
                print(f"Vision: {obstacle['class']} too close - avoiding {obstacle['position']}")
                if obstacle['position'] == 'center':
                    cmd(sock, 'move', where='right', at=60)
                elif obstacle['position'] == 'left':
                    cmd(sock, 'move', where='right', at=50)
                else:
                    cmd(sock, 'move', where='left', at=50)
                return 'avoiding_vision'
            elif area_ratio > 0.12:
                print(f"Vision: {obstacle['class']} ahead - adjusting")
                if obstacle['position'] != 'right':
                    cmd(sock, 'move', where='right', at=45)
                else:
                    cmd(sock, 'move', where='left', at=45)
                return 'adjusting'
    
    # Layer 3: Clear path - move forward
    print("Path clear - moving forward")
    cmd(sock, 'move', where='forward', at=60)
    return 'forward'
