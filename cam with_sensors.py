# from pyjoystick import get_joystick
import numpy as np
import cv2 as cv
from urllib.request import urlopen
import socket
import sys
import json
import re
import struct
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO

### clean working space
# get_joystick().magic('clear')
# get_joystick().magic('reset -f')

### image from camera
cam_no = 0
#%% Capture image from camera
cv.namedWindow('Camera', cv.WINDOW_NORMAL)
cv.resizeWindow('Camera', 800, 600)
cv.moveWindow('Camera', 0, 0)
cmd_no = 0
def capture():
    global cmd_no
    cmd_no += 1
    print(str(cmd_no) + ': capture image', end=' ')
    try:
        cam = urlopen('http://192.168.4.1/capture')
        img = cam.read()
        img = np.asarray(bytearray(img), dtype = 'uint8')
        img = cv.imdecode(img, cv.IMREAD_UNCHANGED)
        if img is None or img.size == 0:
            print('ERROR: Failed to decode image')
            return None
        print(f'[OK: {img.shape}]')
        return img
    except Exception as e:
        print(f'\nERROR capturing image: {e}')
        return None

#%% Send a command and receive a response
off = [0.007,  0.022,  0.091,  0.012, -0.011, -0.05]

def check_connection(sock):
    """Check if socket is still connected"""
    try:
        # This will raise an error if socket is closed
        sock.getpeername()
        return True
    except:
        return False

def cmd(sock, do, what = '', where = '', at = ''):
    global cmd_no
    cmd_no += 1
    
    # Check connection before sending
    if not check_connection(sock):
        print(f"ERROR: Socket disconnected before command {cmd_no}")
        return None
    
    msg = {"H":str(cmd_no)} # dictionary
    if do == 'move':
        msg["N"] = 3
        what = ' car '
        if where == 'forward':
            msg["D1"] = 3
        elif where == 'back':
            msg["D1"] = 4
        elif where == 'left':
            msg["D1"] = 1
        elif where == 'right':
            msg["D1"] = 2
        msg["D2"] = at # at is speed here
        where = where + ' '
    elif do == 'stop':
        msg.update({"N":1,"D1":0,"D2":0,"D3":1})
        what = ' car'
    elif do == 'rotate':
        msg.update({"N":5,"D1":1,"D2":at}) # at is an angle here
        what = ' ultrasonic unit'
        where = ' '
    elif do == 'measure':
        if what == 'distance':
            msg.update({"N":21,"D1":2})
        elif what == 'motion':
            msg["N"] = 6
        what = ' ' + what
    elif do == 'check':
        msg["N"] = 23
        what = ' off the ground'
    msg_json = json.dumps(msg)
    print(str(cmd_no) + ': ' + do + what + where + str(at), end = ': ')
    
    # Don't clear buffer - it causes race conditions where we delete the response we're waiting for
    # Only clear on explicit error recovery
    
    try:
        sock.send(msg_json.encode())
        print("[SENT]", end=' ')
    except socket.error as e:
        print(f'\nERROR sending command: {e}')
        print(f'Socket state: {"connected" if check_connection(sock) else "disconnected"}')
        return None
    except Exception as e:
        print(f'\nUnexpected error: {e}')
        return None
    
    try:
        response_buffer = ""
        start_time = time.time()
        sock.settimeout(5.0)  # Longer timeout for recv
        
        while True:
            if time.time() - start_time > 5.0:
                print(f'\nERROR: Timeout waiting for response (buffer: {response_buffer[:50]}...)')
                return None
            
            try:
                chunk = sock.recv(512).decode()  # Increased buffer for faster reads
            except socket.timeout:
                # Check if we have a complete response
                if '_' in response_buffer:
                    break
                print(f'\nERROR: Recv timeout (partial: {response_buffer[:50]}...)')
                return None
            except Exception as e:
                print(f'\nERROR: Recv failed: {e}')
                return None
            
            if not chunk:
                print(f'\nERROR: Connection closed by robot')
                return None
            
            response_buffer += chunk
            
            if '_' in response_buffer:
                print(f"[RECV: {len(response_buffer)} bytes]", end=' ')
                break
        
        res = response_buffer
        
        # Longer delay after receiving to let robot fully process
        time.sleep(0.1)
        
    except socket.timeout:
        print('\nERROR: Socket timeout waiting for response')
        # Try to flush any remaining data
        try:
            sock.setblocking(False)
            sock.recv(1024)
            sock.setblocking(True)
        except:
            pass
        return None
    except socket.error as e:
        print(f'\nERROR receiving response: {e}')
        return None
    
    try:
        res = re.search('_(.*)}', res).group(1)
    except (AttributeError, IndexError) as e:
        print(f'\nERROR: Could not parse response: {res[:100]}')
        return None
    if res == 'ok' or res == 'true':
        res = 1
    elif res == 'false':
        res = 0
    elif msg.get("N") == 6:
        res = res.split(",")
        res = [int(x)/16384 for x in res] # convert to units of g
        res[2] = res[2] - 1 # subtract 1G from az
        res = [round(res[i] - off[i], 4) for i in range(6)]
    else:
        res = int(res)
    print(res)
    return res

#%% Sensor reading functions
def read_distance(sock):
    """Read ultrasonic distance sensor (returns distance in cm)"""
    result = cmd(sock, 'measure', what='distance')
    if result is not None and isinstance(result, int):
        return result
    return None

def read_ir_sensors(sock):
    """Read IR line sensors - checks if robot is off the ground or at edge"""
    result = cmd(sock, 'check')
    if result is not None:
        # Result: 1 = on ground, 0 = off ground/edge detected
        return result
    return None

def scan_environment(sock):
    """
    Scan environment using ultrasonic sensor by rotating it.
    Returns dict with distances at different angles: {-90, -45, 0, 45, 90}
    """
    angles = [-90, -45, 0, 45, 90]
    distances = {}
    
    for angle in angles:
        # Rotate ultrasonic sensor to angle
        cmd(sock, 'rotate', at=angle)
        time.sleep(0.2)  # Wait for servo to rotate
        
        # Measure distance
        dist = read_distance(sock)
        if dist is not None:
            distances[angle] = dist
        else:
            distances[angle] = 999  # No reading = assume clear
    
    # Return sensor to center
    cmd(sock, 'rotate', at=0)
    time.sleep(0.1)
    
    return distances

#%% Initialize YOLO model
def download_yolo_model(model_name='yolov8n.pt'):
    """
    Download and initialize YOLO model.
    Returns: YOLO model instance
    """
    print(f"Initializing YOLO model: {model_name}")
    print("This may download the model on first run (~6MB for yolov8n)...")
    try:
        model = YOLO(model_name)
        print(f"✓ Model {model_name} loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

# Load YOLOv8 small model (better accuracy, still fast)
model = download_yolo_model('yolov8s.pt')  # Upgraded from nano for better detection

#%% Navigation configuration
# Define obstacle categories for navigation
NAVIGATION_CONFIG = {
    # Sensor thresholds
    'distance_thresholds': {
        'emergency': 20,   # cm - immediate stop
        'close': 40,       # cm - slow down and turn
        'moderate': 60,    # cm - start adjusting
        'far': 100         # cm - just monitoring
    },
    'obstacles_to_avoid': [
        # High priority obstacles (stop or turn immediately)
        'person', 'dog', 'cat', 'bird',  # Living beings
        'car', 'truck', 'bicycle', 'motorcycle',  # Vehicles
        
        # Medium priority (navigate around)
        'chair', 'couch', 'bed', 'dining table',  # Furniture
        'potted plant', 'vase', 'bottle',  # Knockable objects
        
        # Low priority (can get close but avoid)
        'book', 'clock', 'teddy bear', 'backpack'  # Small objects
    ],
    
    'cone_course_objects': [
        # Objects that typically mark a course
        'bottle',  # Can use bottles as cones
        'cup',     # Cups work too
        'vase',    # Or vases
        'potted plant',  # Or plants
        'sports ball'  # Actual cones might be detected as balls
    ],
    
    'size_thresholds': {
        'too_close': 0.20,   # Object fills >20% of frame - emergency avoid
        'close': 0.12,       # Object fills >12% of frame - turn away
        'moderate': 0.05,    # Object fills >5% of frame - adjust course
        'far': 0.02          # Object fills >2% of frame - monitor
    }
}

#%% Object detection with YOLO
def detect_objects_yolo(img):
    """
    Detect objects using YOLO model.
    Returns: list of detected objects with their positions, sizes, and classes
    """
    if img is None:
        print("ERROR: Cannot detect objects, image is None")
        return [], None
    
    try:
        height, width = img.shape[:2]
        
        # Run YOLO inference
        results = model(img, verbose=False, conf=0.5)
    except Exception as e:
        print(f"ERROR in YOLO detection: {e}")
        return [], img
    
    except Exception as e:
        print(f"ERROR in YOLO detection: {e}")
        return [], img
    
    objects = []
    
    try:
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate center and size
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1
                area = w * h
                
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                
                # Determine position
                position = 'center' if abs(center_x - width//2) < width//6 else 'left' if center_x < width//2 else 'right'
                
                # Draw bounding box
                color = (0, 255, 0)  # Green for all objects
                cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f'{class_name} {conf:.2f}'
                cv.putText(img, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                objects.append({
                    'class': class_name,
                    'confidence': conf,
                    'x': center_x,
                    'y': center_y,
                    'width': w,
                    'height': h,
                    'area': area,
                    'position': position
                })
    except Exception as e:
        print(f"ERROR processing YOLO results: {e}")
    
    return objects, img

def navigate_with_sensors(sock, img, check_distance=True, use_vision=True):
    """
    Hybrid navigation using both vision (YOLO) and distance sensors.
    Provides multiple layers of obstacle detection.
    
    NOTE: Only reads distance sensor when check_distance=True to avoid
    sending too many commands and hitting robot's 8-command buffer limit.
    """
    height, width = img.shape[:2]
    
    # Layer 1: Ultrasonic distance check (only when enabled)
    # This adds 1 extra command, so we don't check every iteration
    if check_distance:
        distance = read_distance(sock)
        if distance is not None:
            if distance < 20:  # Less than 20cm - EMERGENCY
                print(f"⚠️  EMERGENCY! Object {distance}cm ahead - backing up")
                # Just back up, don't stop first (saves 1 command)
                cmd(sock, 'move', where='back', at=60)
                return 'emergency_backup'
            elif distance < 35:  # Less than 35cm - turn away
                print(f"⚠️  Object {distance}cm ahead - turning right")
                cmd(sock, 'move', where='right', at=60)
                return 'avoiding_close'
            else:
                print(f"Distance: {distance}cm - clear")
    
    # Layer 2: Vision-based navigation (always check, no extra commands)
    if use_vision and img is not None:
        objects, annotated_img = detect_objects_yolo(img)
        cv.imshow('Camera', annotated_img)
        cv.waitKey(1)
        
        # Check for obstacles in vision
        if len(objects) > 0:
            # Find closest/largest obstacle
            obstacle = max(objects, key=lambda x: x['area'])
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

def navigate_with_full_scan(sock, img):
    """
    Advanced navigation with 3-point ultrasonic scan.
    NOTE: This mode uses MANY commands (7+ per iteration) and will
    reconnect very frequently. Use sparingly!
    """
    # Simplified 3-point scan (left, center, right) to reduce commands
    angles = [-45, 0, 45]  # Reduced from 5 points to 3
    distances = {}
    
    for angle in angles:
        cmd(sock, 'rotate', at=angle)
        time.sleep(0.15)  # Shorter wait
        dist = read_distance(sock)
        distances[angle] = dist if dist else 999
    
    # Return sensor to center
    cmd(sock, 'rotate', at=0)
    
    print(f"Scan: L={distances.get(-45, 'N/A')}cm F={distances.get(0, 'N/A')}cm R={distances.get(45, 'N/A')}cm")
    
    # Check for immediate danger
    front_dist = distances.get(0, 999)
    if front_dist < 25:
        print(f"⚠️  Object {front_dist}cm ahead - avoiding")
        # Find best direction
        left_clear = distances.get(-45, 0) > distances.get(45, 0)
        if left_clear:
            print("Turning left")
            cmd(sock, 'move', where='left', at=60)
        else:
            print("Turning right")
            cmd(sock, 'move', where='right', at=60)
        return 'avoiding_scan'
    
    # Path clear - move forward
    print("Scan clear - moving forward")
    cmd(sock, 'move', where='forward', at=60)
    return 'forward'

def navigate_cone_course(sock, img):
    """
    Navigate through a cone course - weave between obstacles.
    Uses cone_course_objects as waypoints to navigate between.
    """
    objects, annotated_img = detect_objects_yolo(img)
    cv.imshow('Camera', annotated_img)
    cv.waitKey(1)
    
    height, width = annotated_img.shape[:2]
    
    # Filter for cone-like objects
    cones = [obj for obj in objects if obj['class'] in NAVIGATION_CONFIG['cone_course_objects']]
    
    if len(cones) == 0:
        print("No cones detected, moving forward")
        cmd(sock, 'move', where='forward', at=50)
        return 'searching'
    
    # Sort cones by horizontal position
    cones.sort(key=lambda x: x['x'])
    
    # Find gaps between cones to navigate through
    if len(cones) == 1:
        # Single cone - go around it
        cone = cones[0]
        if cone['position'] == 'left':
            print(f"Single {cone['class']} on left, going right")
            cmd(sock, 'move', where='right', at=55)
            return 'weaving_right'
        else:
            print(f"Single {cone['class']} on right, going left")
            cmd(sock, 'move', where='left', at=55)
            return 'weaving_left'
    else:
        # Multiple cones - find the gap
        # Check if there's a cone in center
        center_cones = [c for c in cones if c['position'] == 'center']
        
        if center_cones:
            # Cone dead ahead - check which side has more space
            left_cones = [c for c in cones if c['x'] < width // 2]
            right_cones = [c for c in cones if c['x'] > width // 2]
            
            if len(left_cones) < len(right_cones):
                print(f"Cone ahead, more space on left - turning left")
                cmd(sock, 'move', where='left', at=60)
                return 'weaving_left'
            else:
                print(f"Cone ahead, more space on right - turning right")
                cmd(sock, 'move', where='right', at=60)
                return 'weaving_right'
        else:
            # No center obstacle - move forward through the gap
            print(f"Gap detected, moving through ({len(cones)} cones)")
            cmd(sock, 'move', where='forward', at=60)
            return 'through_gap'

def navigate_with_yolo(sock, img, target_class=None, avoid_classes=None, mode='avoid_all'):
    """
    Navigate the robot based on YOLO object detection.
    
    Modes:
    - 'avoid_all': Avoid all detected objects (default)
    - 'cone_course': Navigate through cone course
    - 'follow': Follow target_class
    - 'custom': Use avoid_classes list
    
    Parameters:
    - target_class: specific object class to follow (e.g., 'person', 'cup')
    - avoid_classes: list of classes to avoid (e.g., ['person', 'chair', 'dog'])
    - mode: navigation mode
    """
    
    if mode == 'cone_course':
        return navigate_cone_course(sock, img)
    objects, annotated_img = detect_objects_yolo(img)
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
        obstacle = max(obstacles, key=lambda x: x['area'])
        
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

#%% Color-based detection (legacy function)
def detect_objects_color(img):
    """
    Detect objects using color-based detection.
    Returns: list of detected objects with their positions and sizes
    """
    height, width = img.shape[:2]
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Define color ranges for different objects (adjust as needed)
    # Red objects
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv.inRange(hsv, lower_red1, upper_red1) | cv.inRange(hsv, lower_red2, upper_red2)
    
    # Green objects
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = cv.inRange(hsv, lower_green, upper_green)
    
    # Blue objects
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
    
    objects = []
    
    # Find contours for each color
    for mask, color_name, color_bgr in [(mask_red, 'red', (0, 0, 255)), 
                                         (mask_green, 'green', (0, 255, 0)), 
                                         (mask_blue, 'blue', (255, 0, 0))]:
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 500:  # Filter small noise
                x, y, w, h = cv.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Draw bounding box
                cv.rectangle(img, (x, y), (x + w, y + h), color_bgr, 2)
                cv.putText(img, color_name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                
                objects.append({
                    'color': color_name,
                    'x': center_x,
                    'y': center_y,
                    'width': w,
                    'height': h,
                    'area': area,
                    'position': 'center' if abs(center_x - width//2) < width//6 else 'left' if center_x < width//2 else 'right'
                })
    
    return objects, img

def navigate_with_detection(sock, img, target_color=None, avoid_obstacles=True):
    """
    Navigate the robot based on object detection.
    - target_color: 'red', 'green', or 'blue' to follow a specific color
    - avoid_obstacles: if True, avoid detected objects; if False, move toward target
    """
    objects, annotated_img = detect_objects(img)
    cv.imshow('Camera', annotated_img)
    cv.waitKey(1)
    
    height, width = annotated_img.shape[:2]
    
    if len(objects) == 0:
        # No objects detected, move forward slowly
        print("No objects detected, moving forward")
        cmd(sock, 'move', where='forward', at=50)
        return 'forward'
    
    # Filter for target color if specified
    if target_color:
        target_objects = [obj for obj in objects if obj['color'] == target_color]
        if target_objects:
            # Find largest target object
            target = max(target_objects, key=lambda x: x['area'])
            
            # Navigate toward target
            if target['position'] == 'center':
                # Check if close enough (object is large)
                if target['area'] > width * height * 0.3:
                    print(f"Reached {target_color} target, stopping")
                    cmd(sock, 'stop')
                    return 'reached'
                else:
                    print(f"Target {target_color} centered, moving forward")
                    cmd(sock, 'move', where='forward', at=60)
                    return 'forward'
            elif target['position'] == 'left':
                print(f"Target {target_color} on left, turning left")
                cmd(sock, 'move', where='left', at=50)
                return 'left'
            else:  # right
                print(f"Target {target_color} on right, turning right")
                cmd(sock, 'move', where='right', at=50)
                return 'right'
    
    if avoid_obstacles:
        # Avoid obstacles - find largest obstacle
        obstacle = max(objects, key=lambda x: x['area'])
        
        # If obstacle is too close (large area), take action
        if obstacle['area'] > width * height * 0.2:
            if obstacle['position'] == 'center':
                print("Obstacle ahead, turning right")
                cmd(sock, 'move', where='right', at=60)
                return 'avoid_right'
            elif obstacle['position'] == 'left':
                print("Obstacle on left, turning right")
                cmd(sock, 'move', where='right', at=50)
                return 'avoid_right'
            else:  # right
                print("Obstacle on right, turning left")
                cmd(sock, 'move', where='left', at=50)
                return 'avoid_left'
        else:
            # Obstacle far enough, can move forward
            print("Path clear, moving forward")
            cmd(sock, 'move', where='forward', at=60)
            return 'forward'
    
    return 'idle'

# Connect to robot
print("Connecting to robot at 192.168.4.1:100...")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)  # Longer timeout - 10 seconds
    sock.connect(('192.168.4.1', 100))
    
    # Enable TCP keepalive to detect dead connections
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    
    # Disable Nagle's algorithm for lower latency
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    # Set SO_LINGER to prevent abrupt connection termination
    # l_onoff=1 enables linger, l_linger=5 means wait 5 seconds on close
    linger = struct.pack('ii', 1, 5)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
    
    # Set socket buffer sizes (if supported)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16384)  # Increased to 16KB
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)
    except:
        pass  # Not critical if this fails
    
    print("✓ Connected to robot successfully!")
    print(f"Local address: {sock.getsockname()}")
    print(f"Remote address: {sock.getpeername()}")
except socket.timeout:
    print("ERROR: Connection timeout. Is the robot powered on?")
    sys.exit(1)
except socket.error as e:
    print(f"ERROR: Connection failed: {e}")
    sys.exit(1)

# Navigation loop with YOLO object detection
print("\nStarting autonomous navigation with YOLO...")
print("\nNOTE: Robot firmware closes connection after 8 commands - auto-reconnecting every 3 iterations")
print("\nKeyboard Controls:")
print("  'k' - Stop and exit")
print("  'r' - Restart navigation")
print("  'p' - Pause/resume autonomous mode")
print("  'm' - Cycle navigation mode:")
print("        • Sensor Fusion (vision + ultrasonic)")
print("        • Full Scan (180° ultrasonic sweep)")
print("        • Vision Only (YOLO obstacle avoidance)")
print("        • Cone Course (weave through markers)")
print("  Arrow Keys - Manual control:")
print("    ↑ (Up)    - Move forward")
print("    ↓ (Down)  - Move backward")
print("    ← (Left)  - Turn left")
print("    → (Right) - Turn right")
print("  Space - Stop robot")
print("  Ctrl+C - Emergency stop")

iteration_count = 0
paused = False
manual_mode = False
navigation_mode = 'sensor_fusion'  # Start with sensor fusion mode
# Modes: 'avoid_all', 'cone_course', 'sensor_fusion', 'full_scan'

def setup_socket_options(socket_obj):
    """Apply all socket options to a socket object"""
    try:
        socket_obj.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        socket_obj.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        linger = struct.pack('ii', 1, 5)
        socket_obj.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
        try:
            socket_obj.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16384)
            socket_obj.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)
        except:
            pass
    except Exception as e:
        print(f"Warning: Could not set all socket options: {e}")

# download_yolo_model()
try:
    while True:
        # Check for keyboard input (non-blocking)
        key = cv.waitKey(1) & 0xFF
        
        # Exit command
        if key == ord('k'):
            print("\n'k' pressed - Stopping and exiting...")
            break
        
        # Restart command
        elif key == ord('r'):
            print("\n'r' pressed - Restarting navigation...")
            iteration_count = 0
            manual_mode = False
            paused = False
            if check_connection(sock):
                cmd(sock, 'stop')
                time.sleep(0.5)
            continue
        
        # Pause/resume command
        elif key == ord('p'):
            paused = not paused
            manual_mode = False
            if paused:
                print("\n'p' pressed - PAUSED (Autonomous mode disabled)")
                if check_connection(sock):
                    cmd(sock, 'stop')
            else:
                print("'p' pressed - RESUMED (Autonomous mode enabled)")
            continue
        
        # Cycle through navigation modes
        elif key == ord('m'):
            modes = ['sensor_fusion', 'full_scan', 'vision_only', 'cone_course']
            current_idx = modes.index(navigation_mode) if navigation_mode in modes else 0
            navigation_mode = modes[(current_idx + 1) % len(modes)]
            
            mode_names = {
                'sensor_fusion': 'SENSOR FUSION (Vision + Ultrasonic)',
                'full_scan': 'FULL SCAN (180° Ultrasonic Sweep)',
                'vision_only': 'VISION ONLY (YOLO)',
                'cone_course': 'CONE COURSE'
            }
            print(f"\n'm' pressed - Switched to {mode_names[navigation_mode]}")
            continue
        
        # Manual controls with arrow keys
        elif key == 82 or key == 0:  # Up arrow (key code varies by system)
            manual_mode = True
            paused = True
            print("\n↑ Manual: Moving forward")
            if check_connection(sock):
                cmd(sock, 'move', where='forward', at=70)
            time.sleep(0.1)
            continue
            
        elif key == 84 or key == 1:  # Down arrow
            manual_mode = True
            paused = True
            print("\n↓ Manual: Moving backward")
            if check_connection(sock):
                cmd(sock, 'move', where='back', at=70)
            time.sleep(0.1)
            continue
            
        elif key == 81 or key == 2:  # Left arrow
            manual_mode = True
            paused = True
            print("\n← Manual: Turning left")
            if check_connection(sock):
                cmd(sock, 'move', where='left', at=70)
            time.sleep(0.1)
            continue
            
        elif key == 83 or key == 3:  # Right arrow
            manual_mode = True
            paused = True
            print("\n→ Manual: Turning right")
            if check_connection(sock):
                cmd(sock, 'move', where='right', at=70)
            time.sleep(0.1)
            continue
        
        # Space to stop
        elif key == 32:  # Space bar
            print("\nSpace pressed - Stopping robot")
            if check_connection(sock):
                cmd(sock, 'stop')
            if not paused:
                paused = True
                print("(Paused - press 'p' to resume autonomous mode)")
            continue
        
        # Show key code for debugging (useful for finding arrow key codes on your system)
        elif key != 255:
            print(f"\nKey pressed: {key} (press 'k' to exit)")
        
        if paused:
            time.sleep(0.1)
            continue
        
        # Check connection at start of loop
        if not check_connection(sock):
            print("\nERROR: Lost connection to robot!")
            break
        
        # Increment and log iteration count
        iteration_count += 1
        
        # Show progress with estimated command count
        if iteration_count % 3 == 0:
            mode_cmds = {'sensor_fusion': '2-3', 'full_scan': '7+', 'vision_only': '2', 'cone_course': '2'}
            print(f"\n--- Iter {iteration_count} | Mode: {navigation_mode} | Cmds/iter: ~{mode_cmds.get(navigation_mode, '2')} | Total cmd_no: {cmd_no} ---")
        
        # Periodic reconnect - frequency depends on navigation mode
        # Full scan mode: reconnect every iteration (uses 7+ commands)
        # Other modes: reconnect every 3 iterations (uses 2-3 commands)
        reconnect_interval = 1 if navigation_mode == 'full_scan' else 3
        
        if iteration_count % reconnect_interval == 0 and iteration_count > 0:
            print(f"\n[Periodic reconnection - iteration {iteration_count}]")
            try:
                cmd(sock, 'stop')
                time.sleep(0.2)
            except:
                pass
            
            # Close and reopen socket to prevent buildup
            try:
                sock.shutdown(socket.SHUT_RDWR)
                sock.close()
                print("[Closed socket, waiting...]", end=' ')
            except:
                print("[Socket already closed...]", end=' ')
            
            time.sleep(0.8)  # Longer delay to let robot fully clean up
            
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10.0)
                sock.connect(('192.168.4.1', 100))
                
                # Re-apply socket options
                setup_socket_options(sock)
                
                print("✓ Reconnected successfully")
                time.sleep(0.3)  # Let robot stabilize after reconnect
            except Exception as e:
                print(f"\nERROR: Failed to reconnect: {e}")
                print("Press 'r' to retry or 'k' to exit")
                paused = True
                continue
        
        img = capture()
        if img is None:
            print("ERROR: Failed to capture image, retrying...")
            time.sleep(0.5)
            continue
        
        # Choose navigation mode based on current setting
        # NOTE: Sensor fusion alternates distance checks to avoid buffer overflow
        try:
            if navigation_mode == 'sensor_fusion':
                # Only check distance every other iteration to reduce commands
                check_dist = (iteration_count % 2 == 0)
                result = navigate_with_sensors(sock, img, check_distance=check_dist, use_vision=True)
            elif navigation_mode == 'full_scan':
                # WARNING: This mode uses 7+ commands per iteration!
                # Will reconnect very frequently (every iteration)
                result = navigate_with_full_scan(sock, img)
            elif navigation_mode == 'vision_only':
                result = navigate_with_yolo(sock, img, 
                                           avoid_classes=NAVIGATION_CONFIG['obstacles_to_avoid'],
                                           mode='custom')
            elif navigation_mode == 'cone_course':
                result = navigate_with_yolo(sock, img, mode='cone_course')
            else:
                # Default to sensor fusion
                check_dist = (iteration_count % 2 == 0)
                result = navigate_with_sensors(sock, img, check_distance=check_dist, use_vision=True)
        except Exception as e:
            print(f"\n!!! ERROR in navigation at iteration {iteration_count}: {e}")
            result = None
        
        # Check if command failed
        if result is None:
            print(f"\nERROR: Navigation command failed at iteration {iteration_count}!")
            print("Attempting to recover...")
            
            # Try to reconnect
            try:
                sock.close()
            except:
                pass
            
            time.sleep(0.5)
            
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10.0)
                sock.connect(('192.168.4.1', 100))
                
                # Re-apply socket options
                setup_socket_options(sock)
                
                print("✓ Reconnected after error")
                time.sleep(0.5)  # Extra delay after error recovery
                continue  # Try again
            except:
                print("✗ Failed to reconnect. Press 'r' to retry or 'k' to exit")
                paused = True
                continue
        
        # Option 2: Follow a specific object (e.g., 'person', 'cup', 'ball')
        # navigate_with_yolo(sock, img, target_class='person')
        
        # Option 3: Avoid specific objects
        # navigate_with_yolo(sock, img, target_class=None, avoid_classes=['person', 'chair', 'dog'])
        
        # Option 4: Use color-based detection (legacy)
        # navigate_with_detection(sock, img, target_color=None, avoid_obstacles=True)
        
        # Delay between commands - can be shorter since we reconnect every 3 iterations
        # This keeps us well under the robot's 8-command limit
        time.sleep(0.3)
        
except KeyboardInterrupt:
    print("\n\nEmergency stop - Ctrl+C pressed!")
    if check_connection(sock):
        cmd(sock, 'stop')
    else:
        print("Socket already disconnected")
    sock.close()
    cv.destroyAllWindows()
    print("Navigation stopped")
except Exception as e:
    print(f"\n\nUnexpected error in main loop at iteration {iteration_count}: {e}")
    import traceback
    traceback.print_exc()
    if check_connection(sock):
        cmd(sock, 'stop')
    sock.close()
    cv.destroyAllWindows()

print(f"\nTotal iterations completed: {iteration_count}")
print("Program ended")