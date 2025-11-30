"""  
Navigation functions for robot movement and obstacle avoidance
"""

import cv2 as cv
import colorlog
import time
import re
import numpy as np
from .detection_utils import detect_objects_yolo, get_largest_object
from .robot_utils import cmd, read_distance, read_ir_sensors

# Setup logger
logger = colorlog.getLogger(__name__)


def parse_ai_movement_response(response):
    """
    Parse AI response in format: '<decision> <speed> <duration>'
    
    Args:
        response: AI response string (e.g., "forward 80 1.5")
    
    Returns:
        tuple: (decision: str, speed: int, duration: float)
    """
    # Extract decision, speed, duration using regex
    pattern = r'(\w+)\s+(\d+)\s+([\d.]+)'
    match = re.search(pattern, response.lower())
    
    if match:
        decision = match.group(1)
        speed = int(match.group(2))
        duration = float(match.group(3))
        
        # Clamp speed to 0-100 range
        speed = max(0, min(100, speed))
        
        # Clamp duration to 0.1-3.0 seconds
        duration = max(0.1, min(3.0, duration))
        
        return decision, speed, duration
    else:
        # Fallback: try to extract just the decision word
        words = response.lower().split()
        for word in words:
            if word in ['forward', 'left', 'right', 'stop', 'avoid', 'backward', 'rotate_left', 'rotate_right']:
                logger.warning(f"AI response didn't match format, extracted '{word}' with defaults")
                return word, 60, 1.0
        
        # Ultimate fallback
        logger.error(f"Could not parse AI response: {response}")
        return 'stop', 0, 0


def ai_navigation_decision(annotated_img, target, other_objects, target_class, img_width, img_height, decision_history=None):
    """
    Use AI (Ollama/Gemini) to decide robot movement based on detected objects.
    
    Args:
        annotated_img: Image with bounding boxes drawn
        target: Target object dict (with position, area, class, etc.)
        other_objects: List of other detected objects (obstacles)
        target_class: Name of target class (e.g., 'chair','ball')
        img_width: Image width in pixels
        img_height: Image height in pixels
        decision_history: List of recent AI decisions (for context)
    
    Returns:
        tuple: (decision: str, speed: int, duration: float)
               decision: 'forward', 'left', 'right', 'stop', 'avoid'
               speed: motor speed 0-100
               duration: movement duration in seconds
    """
    # Import here to avoid circular dependency
    from ollama.ollama_vision import query_ollama_vision
    
    # Build context prompt with object information
    target_info = f"Target {target_class}: position={target['position']}, size={target['area']/(img_width*img_height)*100:.1f}% of image"
    
    obstacles_info = ""
    if other_objects:
        obstacles_info = f"\nObstacles detected: {len(other_objects)} objects - "
        obstacles_info += ", ".join([f"{obj['class']} ({obj['position']})" for obj in other_objects[:3]])
    
    # Build decision history context
    history_info = ""
    if decision_history and len(decision_history) > 0:
        history_info = "\n\nRecent actions (newest last):\n"
        for i, dec in enumerate(decision_history[-3:], 1):  # Show last 3
            history_info += f"{i}. Action: {dec['decision']} (speed={dec['speed']}, {dec['duration']:.1f}s)\n"
            history_info += f"   Target was: {dec['target_position']} (area={dec.get('target_area', 0):.0f}px²)\n"
            if dec.get('all_objects'):
                history_info += f"   Objects seen: "
                history_info += ", ".join([f"{obj['class']}@{obj['position']}({obj['area']:.0f}px²)" for obj in dec['all_objects'][:4]])
                history_info += "\n"
        history_info += "\nAnalyze: Are you making progress? Is target getting larger/more centered? Try different approach if stuck!"
    
    # Build movement configuration reference for AI
    speed_config = "Available speeds (0-100):\n" + "\n".join([f"    - {k}: {v}" for k, v in MOVEMENT_SPEEDS.items()])
    delay_config = "Typical movement durations (seconds):\n" + "\n".join([f"    - {k}: {v}" for k, v in MOVEMENT_DELAYS.items()])
    
    prompt = f"""You are controlling a robot car. This image shows detected objects with bounding boxes.

                Your GOAL: Navigate toward the {target_class} (shown in GREEN bounding box).

                Current situation:
                {target_info}{obstacles_info}{history_info}

                Visual context:
                - GREEN box: your target {target_class}
                - Other colored boxes: obstacles to consider
                - Target position: {target['position']} (left/center/right of image)
                - Target size: {'LARGE (close)' if target['area'] > img_width * img_height * 0.3 else 'MEDIUM (approaching)' if target['area'] > img_width * img_height * 0.15 else 'SMALL (far away)'}

                {speed_config}
                {delay_config}

                Respond with EXACTLY this format: <decision> <speed> <duration>

                Example responses (use as format reference only):
                - "forward 100 2.0" = move forward at speed 100 for 2.0 seconds
                - "left 60 0.5" = turn left at speed 60 for 0.5 seconds  
                - "right 80 1.5" = turn right at speed 80 for 1.5 seconds
                - "stop 0 0" = stop moving

                Available decisions: forward, left, right, stop, back
                Speed range: 0-100
                Duration range: 0.1-3.0 seconds

                Analyze the image and your recent history to decide the best action."""

    # Query AI
    response = query_ollama_vision(annotated_img, prompt)
    
    if response is None:
        logger.warning("AI decision failed, defaulting to stop")
        return 'stop', 0, 0
    
    # Parse decision with parameters
    decision, speed, duration = parse_ai_movement_response(response)
    logger.info(f"🤖 AI: {decision} (speed={speed}, duration={duration:.1f}s) - reasoning: {response[:40]}...")
    
    return decision, speed, duration


def ai_search_decision(sock, og_annotated_img, model, target_class, capture_callback, reconnect_callback, decision_history=None):
    """
    Use AI to decide search strategy when target is not detected.
    Rotates camera left/right, captures images, and asks AI which direction to turn.
    
    Args:
        sock: Robot socket connection
        og_annotated_img: Original annotated image (to restore display)
        model: YOLO model
        target_class: Name of target class being searched for (e.g., 'chair','ball')
        capture_callback: Function to capture new frame after rotation
        reconnect_callback: Function that reconnects socket (takes sock, returns new sock)
        decision_history: List of recent AI decisions (for context)
    
    Returns:
        tuple: (sock, direction: str, duration: float)
               sock: Updated socket connection
               direction: 'left', 'right', 'forward'
               duration: movement duration in seconds
    """
    # Import here to avoid circular dependency
    from ollama.ollama_vision import query_ollama_vision
    
    logger.info(f"🤖 AI search: scanning for {target_class}...")
    
    # Rotate camera to left (150°) and check (1 command)
    logger.debug("AI search: Scanning left (150°)...")
    cmd(sock, 'rotate', at=150)
    time.sleep(0.5)  # Wait for servo to rotate
    
    left_img = capture_callback()
    if left_img is not None:
        objects_left, annotated_left = detect_objects_yolo(left_img, model, target_class=target_class)
        cv.imshow('Camera', annotated_left)  # Show left scan
        cv.waitKey(1)
    
    # Reconnect after 1 command
    logger.debug("Reconnecting after left scan...")
    sock = reconnect_callback(sock)
    if sock is None:
        logger.error("Failed to reconnect during AI search")
        return None, None, 1.0
    
    # Rotate camera to right (30°) and check (1 command)
    logger.debug("AI search: Scanning right (30°)...")
    cmd(sock, 'rotate', at=30)
    time.sleep(0.5)  # Wait for servo to rotate
    
    right_img = capture_callback()
    if right_img is not None:
        objects_right, annotated_right = detect_objects_yolo(right_img, model, target_class=target_class)
        cv.imshow('Camera', annotated_right)  # Show right scan
        cv.waitKey(1)
    
    # Reconnect after 1 more command
    logger.debug("Reconnecting after right scan...")
    sock = reconnect_callback(sock)
    if sock is None:
        logger.error("Failed to reconnect during AI search")
        return None, None, 1.0
    
    # Always return camera to center (90°) - MANDATORY (1 command)
    logger.debug("Returning camera to center (90°)...")
    cmd(sock, 'rotate', at=90)
    cv.imshow('Camera', og_annotated_img)  # Restore original view
    cv.waitKey(1)
    time.sleep(0.5)
    
    # Reconnect before final robot movement
    logger.debug("Reconnecting before robot movement...")
    sock = reconnect_callback(sock)
    if sock is None:
        logger.error("Failed to reconnect during AI search")
        return None, None, 1.0
    
    # Build decision history context
    history_info = ""
    if decision_history and len(decision_history) > 0:
        history_info = "\n\nRecent search actions:\n"
        for i, dec in enumerate(decision_history[-3:], 1):
            history_info += f"{i}. {dec['decision']} ({dec['duration']:.1f}s)\n"
            if dec.get('all_objects'):
                history_info += f"   Saw: {', '.join([obj['class'] for obj in dec['all_objects'][:3]])}\n"
    
    # Build movement configuration reference for AI
    speed_config = "Available speeds (0-100):\n" + "\n".join([f"    - {k}: {v}" for k, v in MOVEMENT_SPEEDS.items()])
    delay_config = "Typical movement durations (seconds):\n" + "\n".join([f"    - {k}: {v}" for k, v in MOVEMENT_DELAYS.items()])
    
    # Ask AI to decide based on both views
    prompt = f"""You are controlling a robot car searching for a {target_class}.

            The robot just scanned LEFT and RIGHT by rotating its camera. You have two images:
            - LEFT image shows what robot sees when looking left
            - RIGHT image shows what robot sees when looking right

            Current situation:
            - Target {target_class} is NOT visible in either direction{history_info}

            {speed_config}
            {delay_config}

            Respond with EXACTLY this format: <decision> <speed> <duration>

            Example responses (use as format reference only):
            - "left 60 1.0" = turn left at speed 60 for 1.0 seconds
            - "right 80 0.8" = turn right at speed 80 for 0.8 seconds
            - "forward 100 1.5" = move forward at speed 100 for 1.5 seconds

            Available search movements: left, right, forward
            Speed range: 0-100
            Duration range: 0.1-3.0 seconds

            Analyze both images and your search history to decide the best exploration direction."""

    # Combine both images side by side for AI to analyze
    import numpy as np
    if left_img is not None and right_img is not None:
        # Resize to same height if needed
        h1, w1 = annotated_left.shape[:2]
        h2, w2 = annotated_right.shape[:2]
        if h1 != h2:
            scale = h1 / h2
            annotated_right = cv.resize(annotated_right, (int(w2 * scale), h1))
        
        # Concatenate horizontally with labels
        combined = np.hstack([annotated_left, annotated_right])
        
        # Add text labels
        cv.putText(combined, "LEFT", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv.putText(combined, "RIGHT", (w1 + 50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Query AI with combined image
        response = query_ollama_vision(combined, prompt)
    else:
        logger.warning("Failed to capture both images for AI search")
        response = None
    
    if response is None:
        logger.warning("AI search decision failed, defaulting to right turn")
        direction, speed, duration = 'right', 80, 1.0
    else:
        # Parse decision with parameters
        direction, speed, duration = parse_ai_movement_response(response)
        logger.info(f"🤖 AI search: {direction} (speed={speed}, duration={duration:.1f}s)")
        
        # Validate decision is valid for search mode
        if direction not in ['left', 'right', 'forward']:
            logger.warning(f"Invalid search decision '{direction}', defaulting to right")
            direction, speed, duration = 'right', 80, 1.0
    
    # Execute the AI's decision (1 command: move with timed duration)
    logger.info(f"AI search: executing {direction.upper()} movement")
    cmd(sock, 'move', where=direction, at=speed)
    time.sleep(duration)  # Wait for movement to complete (timed movement)
    
    return sock, direction, duration


# Movement timing configuration
MOVEMENT_DELAYS = {
    'forward': 2,        # After autonomous forward movement in navigation
    'back': 0.8,           # After backing up during obstacle avoidance
    'left': 0.5,           # After left turns in navigation
    'right': 0.5,          # After right turns in navigation
    'manual': 0.1,         # After manual arrow key controls
    'unstuck_back': 0.8,   # Initial backup in vision-based stuck recovery
    'unstuck_turn': 0.8,   # Turns during vision-based stuck recovery scanning
    'default': 1.0,         # Main loop delay between navigation iterations (longer for AI decisions)
    'target_on_sides': 1.5   # when target on right or left, but far away (area based) !
}

# Movement speed configuration (0-100)
# Can be overridden by importing module
MOVEMENT_SPEEDS = {
    'forward_normal': 100,    # When exploring or no objects detected

    'back_avoid': 80,        # Initial backup in vision-based stuck recovery

    'left_normal': 60,       # Normal left turns  avoiding obstacles
    'right_normal': 60,      # Normal right turns  avoiding obstacles
    
    'left_unstuck': 70,      # Left turns during vision-based stuck recovery scanning & positioning
    'right_unstuck':70,       # Right turns during vision-based stuck recovery scanning & positioning

    'right_avoid': 60,       # Right turns to avoid obstacles

    'no_target_search': 60,     # Slow rotation when searching for target
    'default': 50,           # Fallback for any unspecified movement
    'target_on_sides': 80    # when target on right or left, but far away (area based) !
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
        cmd(sock, 'move', where='right', at=movement_speed('right_unstuck'))
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


def smart_search_for_target(sock, og_annotated_img, model, target_class, capture_callback, reconnect_callback):
    """
    Intelligently search for target by scanning left, center, right with camera rotation.
    Then moves robot based on where target was found.
    
    Args:
        sock: Robot socket connection
        img: Current image
        model: YOLO model
        target_class: Target class to search for
        capture_callback: Function to capture new frame after rotation
        reconnect_callback: Function that reconnects socket (takes sock, returns new sock)
        
    Returns:
        tuple: (sock, direction) where direction is 'left', 'right', 'center', or None
    """
    logger.info(f"🔍 Smart search: scanning for {target_class}...")
    
    # Rotate camera to left (150°) and check (1 command)
    logger.debug("Scanning left (150°)...")
    cmd(sock, 'rotate', at=150)
    time.sleep(0.5)  # Wait for servo to rotate
    
    left_img = capture_callback()
    chair_on_left = False
    if left_img is not None:
        objects, annotated_left = detect_objects_yolo(left_img, model, target_class=target_class)
        cv.imshow('Camera', annotated_left)  # Show left scan
        cv.waitKey(1)
        target_objects = [obj for obj in objects if obj['class'] == target_class]
        if target_objects:
            logger.info(f"✓ Target {target_class} found on LEFT!")
            chair_on_left = True
    
    # Reconnect after 1 command
    logger.debug("Reconnecting after left scan...")
    sock = reconnect_callback(sock)
    if sock is None:
        logger.error("Failed to reconnect during smart search")
        return None, None
    
    # Rotate camera to right (30°) and check (1 command)
    logger.debug("Scanning right (30°)...")
    cmd(sock, 'rotate', at=30)
    time.sleep(0.5)  # Wait for servo to rotate
    
    right_img = capture_callback()
    chair_on_right = False
    if right_img is not None:
        objects, annotated_right = detect_objects_yolo(right_img, model, target_class=target_class)
        cv.imshow('Camera', annotated_right)  # Show right scan
        cv.waitKey(1)
        target_objects = [obj for obj in objects if obj['class'] == target_class]
        if target_objects:
            logger.info(f"✓ Target {target_class} found on RIGHT!")
            chair_on_right = True
    
    # Reconnect after 1 more command
    logger.debug("Reconnecting after right scan...")
    sock = reconnect_callback(sock)
    if sock is None:
        logger.error("Failed to reconnect during smart search")
        return None, None
    
    # Always return camera to center (90°) - MANDATORY (1 command)
    logger.debug("Returning camera to center (90°)...")
    cmd(sock, 'rotate', at=90)
    cv.imshow('Camera', og_annotated_img)  # Show left scan
    cv.waitKey(1)
    time.sleep(0.5)
    
    # Reconnect before final robot movement
    logger.debug("Reconnecting before robot movement...")
    sock = reconnect_callback(sock)
    if sock is None:
        logger.error("Failed to reconnect during smart search")
        return None, None
    
    # Decide robot movement based on scan results (1 command: move with timed duration)
    if chair_on_left:
        logger.info("Smart search: chair on left, turning robot left")
        cmd(sock, 'move', where='left', at=movement_speed('left_normal'))
        time.sleep(MOVEMENT_DELAYS['left'])  # Wait for turn to complete (timed movement)
        return sock, 'left'
    elif chair_on_right:
        logger.info("Smart search: chair on right, turning robot right")
        cmd(sock, 'move', where='right', at=movement_speed('right_normal'))
        time.sleep(MOVEMENT_DELAYS['right'])  # Wait for turn to complete (timed movement)
        return sock, 'right'
    else:
        # Not found anywhere (center was already checked before smart search)
        logger.warning(f"✗ Target {target_class} not found in any direction")
        direction = 'right'
        logger.info(f"Smart search: defaulting to turn {direction.upper()} to continue search")
        cmd(sock, 'move', where=direction, at=movement_speed('no_target_search'))
        time.sleep(MOVEMENT_DELAYS.get(direction, MOVEMENT_DELAYS['default']))  # Timed movement
        return sock, direction


def navigate_with_yolo(sock, img, model, target_class=None, avoid_classes=None, ai_decide=False, ai_decision=None, objects=None, annotated_img=None, video_logger=None):
    """
    Navigate the robot based on YOLO object detection.
    - target_class: specific object class to follow (e.g., 'person', 'cup', 'chair','ball')
    - avoid_classes: list of classes to avoid (e.g., ['person', 'chair', 'dog','ball'])
    - ai_decide: if True, use AI decision (must provide ai_decision parameter)
    - ai_decision: Pre-computed AI decision string (if ai_decide=True)
    - objects: Pre-detected objects list (optional, will detect if not provided)
    - annotated_img: Pre-annotated image (optional, will annotate if not provided)
    - video_logger: VideoLogger instance for recording logs to video (optional)
    """
    # Vision-based navigation
    if objects is None or annotated_img is None:
        objects, annotated_img = detect_objects_yolo(img, model, target_class=target_class)
    cv.imshow('Camera', annotated_img)
    cv.waitKey(1)
    
    height, width = annotated_img.shape[:2]
    
    if len(objects) == 0: # No objects detected --> trigger smart_search in fcam.py
        logger.info("No objects detected - will use smart search")
        # Don't send movement command here - smart_search will handle it
        # Just return signal to trigger smart search in fcam.py
        return ('searching', MOVEMENT_DELAYS['default'])
    
    # Filter for target class if specified
    if target_class:
        target_objects = [obj for obj in objects if obj['class'] == target_class]
        logger.debug(f"🔍 DEBUG: Looking for '{target_class}', found {len(objects)} total objects")
        logger.debug(f"🔍 DEBUG: Object classes: {[obj['class'] for obj in objects]}")
        logger.debug(f"🔍 DEBUG: Matching target objects: {len(target_objects)}")
        if target_objects:
            # Find largest target object
            target = max(target_objects, key=lambda x: x['area'])
            
            # Check for obstacles in the way (objects that are not the target)
            other_objects = [obj for obj in objects if obj['class'] != target_class]
            
            # AI Decision Mode
            if ai_decide and ai_decision:
                # Unpack AI decision tuple: (decision, speed, duration)
                if isinstance(ai_decision, tuple):
                    decision, speed, duration = ai_decision
                else:
                    # Fallback for old-style string-only decisions
                    decision = ai_decision
                    speed = 70
                    duration = 1.0
                
                # Execute the AI's decision with specified speed and duration
                if decision == 'stop':
                    logger.info(f"AI: Reached {target_class}, stopping")
                    logger.debug("\033[91m■ STOP\033[0m")
                    cmd(sock, 'stop')
                    return ('reached', 0.3)
                elif decision == 'forward':
                    logger.info(f"AI: Moving forward toward {target_class} (speed={speed}, duration={duration:.1f}s)")
                    logger.debug(f"\033[92m↑ Moving FORWARD (speed={speed}, delay={duration:.1f}s)\033[0m")
                    cmd(sock, 'move', where='forward', at=speed)
                    return ('forward', duration)
                elif decision == 'left':
                    logger.info(f"AI: Turning left toward {target_class} (speed={speed}, duration={duration:.1f}s)")
                    logger.debug(f"\033[95m← Turning LEFT (speed={speed}, delay={duration:.1f}s)\033[0m")
                    cmd(sock, 'move', where='left', at=speed)
                    return ('left', duration)
                elif decision == 'right':
                    logger.info(f"AI: Turning right toward {target_class} (speed={speed}, duration={duration:.1f}s)")
                    logger.debug(f"\033[96m→ Turning RIGHT (speed={speed}, delay={duration:.1f}s)\033[0m")
                    cmd(sock, 'move', where='right', at=speed)
                    return ('right', duration)
                elif decision == 'avoid':
                    logger.warning(f"AI: Avoiding obstacle (speed={speed}, duration={duration:.1f}s)")
                    logger.debug(f"\033[96m→ Avoiding (speed={speed}, delay={duration:.1f}s)\033[0m")
                    cmd(sock, 'move', where='right', at=speed)
                    return ('avoid_blocker', duration)
                else:
                    # Fallback to stop if AI returns unexpected decision
                    logger.warning(f"AI returned unexpected decision: {decision}, stopping")
                    cmd(sock, 'stop')
                    time.sleep(0.5)
                    return ('stop', 0.3)
            
            # Hardcoded Logic Mode (original behavior)
            else:
                # If there are other objects blocking the path, avoid them first
                if other_objects:
                    blocking_objects = [obj for obj in other_objects 
                                      if obj['area'] > width * height * 0.08 and obj['position'] == 'center']
                    if blocking_objects:
                        blocker = max(blocking_objects, key=lambda x: x['area'])
                        logger.warning(f"⚠️ {blocker['class']} blocking path to {target_class}! Avoiding...")
                        if video_logger and video_logger.is_recording:
                            video_logger.add_log(f"⚠️ {blocker['class']} blocking path! Avoiding", "WARNING")
                        logger.debug("\033[96m→ Turning RIGHT\033[0m")  # Cyan
                        cmd(sock, 'move', where='right', at=movement_speed('right_avoid'))
                        return ('avoid_blocker', MOVEMENT_DELAYS['right'])
                
                # Navigate toward target
                if target['position'] == 'center':
                    # Before moving forward, check again for any blocking objects (more aggressive check)
                    if other_objects:
                        blocking_objects = [obj for obj in other_objects 
                                          if obj['area'] > width * height * 0.05 and obj['position'] == 'center']
                        if blocking_objects:
                            blocker = max(blocking_objects, key=lambda x: x['area'])
                            logger.warning(f"⚠️ {blocker['class']} directly blocking forward path! Avoiding...")
                            if video_logger and video_logger.is_recording:
                                video_logger.add_log(f"⚠️ {blocker['class']} directly blocking! Avoiding", "WARNING")
                            logger.debug("\033[96m→ Turning RIGHT\033[0m")  # Cyan
                            cmd(sock, 'move', where='right', at=movement_speed('right_avoid'))
                            return ('avoid_blocker', MOVEMENT_DELAYS['right'])
                    
                    # Check if close enough (object is large)
                    if target['area'] > width * height * 0.3:
                        logger.info(f"Reached {target_class}, stopping")
                        if video_logger and video_logger.is_recording:
                            video_logger.add_log(f"Reached {target_class}, stopping", "INFO")
                        logger.debug("\033[91m■ STOP\033[0m")  # Red
                        cmd(sock, 'stop')
                        return ('reached', 0.3)
                    else:
                        # ULTRASONIC SAFETY CHECK before moving forward (DISABLED - sensor not working)
                        # safe, distance = ultrasonic_safety_check(sock, threshold=30)
                        # if not safe:
                        #     return 'ultrasonic_avoid'
                        
                        logger.info(f"Target {target_class} centered, moving forward")
                        if video_logger and video_logger.is_recording:
                            video_logger.add_log(f"Target {target_class} centered, moving forward", "INFO")
                        logger.debug("\033[92m↑ Moving FORWARD\033[0m")  # Green
                        cmd(sock, 'move', where='forward', at=movement_speed('forward_normal'))
                        return ('forward', MOVEMENT_DELAYS['forward'])
                elif target['position'] == 'left':
                    # Turn left to face target - adjust speed/duration based on distance
                    area_ratio = target['area'] / (width * height)
                    where='left'

                    if area_ratio > 0.15:  # Close - slow, short turn
                        speed = movement_speed('left_normal')
                        duration = MOVEMENT_DELAYS['left'] * 0.5  # Shorter
                        logger.info(f"Target {target_class} on left (close), turning left slowly")
                        if video_logger and video_logger.is_recording:
                            video_logger.add_log(f"Target {target_class} on left (close), turning left slowly", "INFO")

                    elif area_ratio > 0.05:  # Medium distance
                        speed = movement_speed('left_normal')
                        duration = MOVEMENT_DELAYS['left']
                        logger.info(f"Target {target_class} on left (medium), turning left")
                        if video_logger and video_logger.is_recording:
                            video_logger.add_log(f"Target {target_class} on left (medium), turning left", "INFO")

                    else:  # Far - faster, longer turn
                        speed = movement_speed('target_on_sides') 
                        duration = MOVEMENT_DELAYS['target_on_sides'] 
                        logger.info(f"Target {target_class} on left (far), going forward")
                        if video_logger and video_logger.is_recording:
                            video_logger.add_log(f"Target {target_class} on left (far), going forward", "INFO")
                        where = 'forward'

                    logger.debug(f"\033[95m← Going {where}\033[0m")  # Magenta
                    cmd(sock, 'move', where=where, at=speed)
                    return ('left', duration)
                else:  # right
                    # Turn right to face target - adjust speed/duration based on distance
                    area_ratio = target['area'] / (width * height)
                    where='right'
                    if area_ratio > 0.15:  # Close - slow, short turn
                        speed = movement_speed('right_normal')
                        duration = MOVEMENT_DELAYS['right'] * 0.5  # Shorter
                        logger.info(f"Target {target_class} on right (close), turning right slowly")
                        if video_logger and video_logger.is_recording:
                            video_logger.add_log(f"Target {target_class} on right (close), turning right slowly", "INFO")

                    elif area_ratio > 0.05:  # Medium distance
                        speed = movement_speed('right_normal')
                        duration = MOVEMENT_DELAYS['right']
                        logger.info(f"Target {target_class} on right (medium), turning right")
                        if video_logger and video_logger.is_recording:
                            video_logger.add_log(f"Target {target_class} on right (medium), turning right", "INFO")

                    else:  # Far - faster, longer turn
                        speed = movement_speed('target_on_sides')
                        duration = MOVEMENT_DELAYS['target_on_sides']
                        logger.info(f"Target {target_class} on right (far), going forward")
                        if video_logger and video_logger.is_recording:
                            video_logger.add_log(f"Target {target_class} on right (far), going forward", "INFO")
                        where = 'forward'

                    logger.debug(f"\033[96m→ Going {where}\033[0m")  # Cyan
                    cmd(sock, 'move', where=where, at=speed)
                    return ('right', duration)
        else:
            # Target not found - trigger smart search in fcam.py
            logger.warning(f"Target {target_class} not in current view")
            
            # Don't send movement command here - smart_search will handle it
            # Just return signal to trigger smart search in fcam.py
            return ('searching', MOVEMENT_DELAYS['default'])
    
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
                return ('avoid_right', MOVEMENT_DELAYS['right'])
            elif obstacle['position'] == 'left':
                logger.debug("\033[96m→ Turning RIGHT\033[0m")  # Cyan
                cmd(sock, 'move', where='right', at=movement_speed('right_normal'))
                return ('avoid_right', MOVEMENT_DELAYS['right'])
            else:  # right
                logger.debug("\033[95m← Turning LEFT\033[0m")  # Magenta
                cmd(sock, 'move', where='left', at=movement_speed('left_normal'))
                return ('avoid_left', MOVEMENT_DELAYS['left'])
        else:
            # Obstacles far enough, can move forward
            logger.debug("\033[92m↑ Moving FORWARD\033[0m")  # Green
            cmd(sock, 'move', where='forward', at=movement_speed('forward_normal'))
            return ('forward', MOVEMENT_DELAYS['forward'])
    
    return ('idle', 0.3)

