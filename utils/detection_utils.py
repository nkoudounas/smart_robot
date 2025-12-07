"""
Object detection utility functions - YOLO and color-based detection
"""

import numpy as np
import cv2 as cv
from ultralytics import YOLO
import sys


def initialize_yolo_model(model_name='yolov8n.pt', use_segmentation=False):
    """
    Download and initialize YOLO model.
    
    Args:
        model_name: Name of YOLO model ('yolov8n.pt', 'yolo11l-seg.pt', etc.)
        use_segmentation: If True, load a segmentation model (e.g., 'yolo11l-seg.pt')
        
    Returns: 
        YOLO model instance
    """
    # Auto-detect segmentation model from name if not specified
    if '-seg' in model_name:
        use_segmentation = True
    
    print(f"Initializing YOLO model: {model_name} (segmentation={use_segmentation})")
    print("This may download the model on first run...")
    try:
        model = YOLO(model_name)
        # Force CPU usage to avoid CUDA memory errors
        model.to('cpu')
        model_type = "segmentation" if use_segmentation else "detection"
        print(f"✓ Model {model_name} loaded successfully (using CPU, type={model_type})!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def detect_objects_yolo(img, model, confidence=0.5, target_class=None):
    """
    Detect objects using YOLO model (supports both detection and segmentation models).
    
    Args:
        img: Input image (BGR format)
        model: YOLO model instance
        confidence: Confidence threshold for detections (0.0-1.0)
        target_class: Optional target class name to highlight in green
        
    Returns: 
        tuple: (list of detected objects, annotated image)
        Each object dict contains: class, confidence, x, y, width, height, area, position
    """
    if img is None:
        print("ERROR: Cannot detect objects, image is None")
        return [], None
    
    # Generate distinct colors for different classes (using HSV for better distribution)
    def get_class_color(class_name, is_target=False):
        if is_target:
            return (0, 255, 0)  # Green for target class
        
        # Generate consistent color per class name using hash
        # Hash to hue (0-179 for OpenCV HSV), saturation and value
        hash_val = abs(hash(class_name))
        hue = (hash_val % 180)  # OpenCV uses 0-179 for hue
        saturation = 200
        value = 200
        hsv_color = np.uint8([[[hue, saturation, value]]])
        bgr_color = cv.cvtColor(hsv_color, cv.COLOR_HSV2BGR)[0][0]
        return tuple(map(int, bgr_color))
    
    try:
        height, width = img.shape[:2]
        
        # Run YOLO inference on CPU to avoid CUDA errors
        results = model(img, verbose=False, conf=confidence, device='cpu')
    except Exception as e:
        print(f"ERROR in YOLO detection: {e}")
        return [], img
    
    objects = []
    
    try:
        # Process detections
        for result in results:
            boxes = result.boxes
            
            # Check if model has segmentation masks
            has_masks = hasattr(result, 'masks') and result.masks is not None
            
            for idx, box in enumerate(boxes):
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
                
                # Determine position relative to image center
                position = 'center' if abs(center_x - width//2) < width//6 else 'left' if center_x < width//2 else 'right'
                
                # Determine color based on whether it's the target class
                is_target = (target_class is not None and class_name == target_class)
                color = get_class_color(class_name, is_target)
                
                # Draw segmentation mask if available
                if has_masks:
                    mask = result.masks.data[idx].cpu().numpy()
                    # Resize mask to image dimensions
                    mask_resized = cv.resize(mask, (width, height))
                    mask_resized = (mask_resized > 0.5).astype(np.uint8)
                    
                    # Create colored mask overlay
                    mask_colored = np.zeros_like(img)
                    mask_colored[mask_resized == 1] = color
                    
                    # Blend with original image (30% mask, 70% original)
                    img = cv.addWeighted(img, 0.7, mask_colored, 0.3, 0)
                    
                    # Draw contour around mask
                    contours, _ = cv.findContours(mask_resized, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    cv.drawContours(img, contours, -1, color, 2)
                
                # Draw bounding box with thicker line for target
                thickness = 3 if is_target else 2
                cv.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                
                # Build label with target indicator and area
                label = f'{class_name} {conf:.2f}'
                if is_target:
                    label = f'🎯 {label}'
                
                # Add area on second line below the bbox
                area_label = f'{area:.0f}px²'
                
                # Draw top label (class + confidence) background for better readability
                (label_w, label_h), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                cv.putText(img, label, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw bottom label (area) inside bbox at bottom
                (area_w, area_h), _ = cv.getTextSize(area_label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                area_y = y2 - 5  # Position inside bbox, 5px from bottom
                cv.rectangle(img, (x1, area_y - area_h - 5), (x1 + area_w + 10, area_y + 5), color, -1)
                cv.putText(img, area_label, (x1 + 5, area_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                objects.append({
                    'class': class_name,
                    'confidence': conf,
                    'x': center_x,
                    'y': center_y,
                    'width': w,
                    'height': h,
                    'area': area,
                    'position': position,
                    'bbox': (x1, y1, x2, y2),
                    'is_target': is_target,
                    'has_mask': has_masks
                })
    except Exception as e:
        print(f"ERROR processing YOLO results: {e}")
    
    # Count objects by class
    class_counts = {}
    for obj in objects:
        class_name = obj['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Draw object counts in top-right corner
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    padding = 10
    line_height = 25
    
    # Sort by count (descending) then alphabetically
    sorted_classes = sorted(class_counts.items(), key=lambda x: (-x[1], x[0]))
    
    # Calculate total height needed for background
    total_lines = len(sorted_classes)
    if total_lines == 0:
        return objects, img
    
    # Find maximum text width for background rectangle
    max_text_width = 0
    for class_name, count in sorted_classes:
        count_text = f"{count}x {class_name}"
        (text_width, text_height), _ = cv.getTextSize(count_text, font, font_scale, font_thickness)
        max_text_width = max(max_text_width, text_width)
    
    # Position in top-right corner
    x_pos = width - max_text_width - padding * 2
    y_start = padding
    
    # Draw semi-transparent black background for all text
    overlay = img.copy()
    bg_height = total_lines * line_height + padding
    cv.rectangle(overlay, 
                 (x_pos - 5, y_start),
                 (width - padding, y_start + bg_height),
                 (0, 0, 0), -1)
    img = cv.addWeighted(overlay, 0.6, img, 0.4, 0)
    
    # Draw each class count
    y_pos = y_start + 20
    for class_name, count in sorted_classes:
        count_text = f"{count}x {class_name}"
        cv.putText(img, count_text, (x_pos, y_pos), font, font_scale, (255, 255, 255), font_thickness)
        y_pos += line_height
    
    return objects, img


def detect_objects_color(img, min_area=500):
    """
    Detect objects using color-based HSV thresholding.
    Detects red, green, and blue objects.
    
    Args:
        img: Input image (BGR format)
        min_area: Minimum contour area to filter noise
        
    Returns:
        tuple: (list of detected objects, annotated image)
        Each object dict contains: color, x, y, width, height, area, position
    """
    if img is None:
        print("ERROR: Cannot detect objects, image is None")
        return [], None
    
    height, width = img.shape[:2]
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Define color ranges for different objects (HSV)
    color_ranges = {
        'red': [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([180, 255, 255]))
        ],
        'green': [
            (np.array([40, 50, 50]), np.array([80, 255, 255]))
        ],
        'blue': [
            (np.array([100, 50, 50]), np.array([130, 255, 255]))
        ]
    }
    
    color_bgr = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0)
    }
    
    objects = []
    
    # Process each color
    for color_name, ranges in color_ranges.items():
        # Create mask for this color
        mask = np.zeros((height, width), dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv.inRange(hsv, lower, upper)
        
        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv.contourArea(contour)
            if area > min_area:  # Filter small noise
                x, y, w, h = cv.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Determine position
                position = 'center' if abs(center_x - width//2) < width//6 else 'left' if center_x < width//2 else 'right'
                
                # Draw bounding box
                cv.rectangle(img, (x, y), (x + w, y + h), color_bgr[color_name], 2)
                cv.putText(img, color_name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr[color_name], 2)
                
                objects.append({
                    'color': color_name,
                    'x': center_x,
                    'y': center_y,
                    'width': w,
                    'height': h,
                    'area': area,
                    'position': position,
                    'bbox': (x, y, x + w, y + h)
                })
    
    return objects, img


def filter_objects_by_class(objects, target_classes):
    """
    Filter objects by class name(s).
    
    Args:
        objects: List of detected objects
        target_classes: Single class name (str) or list of class names
        
    Returns:
        List of filtered objects
    """
    if isinstance(target_classes, str):
        target_classes = [target_classes]
    
    return [obj for obj in objects if obj.get('class') in target_classes]


def filter_objects_by_color(objects, target_colors):
    """
    Filter objects by color(s).
    
    Args:
        objects: List of detected objects
        target_colors: Single color name (str) or list of color names
        
    Returns:
        List of filtered objects
    """
    if isinstance(target_colors, str):
        target_colors = [target_colors]
    
    return [obj for obj in objects if obj.get('color') in target_colors]


def get_largest_object(objects):
    """
    Find the largest object by area.
    
    Args:
        objects: List of detected objects
        
    Returns:
        Largest object dict or None if list is empty
    """
    if not objects:
        return None
    return max(objects, key=lambda x: x['area'])


def get_closest_object(objects):
    """
    Find the closest object (approximated by largest area).
    
    Args:
        objects: List of detected objects
        
    Returns:
        Closest object dict or None if list is empty
    """
    return get_largest_object(objects)


def get_centered_object(objects, width_threshold=0.2):
    """
    Find the most centered object.
    
    Args:
        objects: List of detected objects
        width_threshold: Fraction of image width to consider as "center"
        
    Returns:
        Most centered object dict or None if list is empty
    """
    if not objects:
        return None
    
    centered = [obj for obj in objects if obj['position'] == 'center']
    if centered:
        return get_largest_object(centered)
    
    return None


def annotate_image(img, objects, show_class=True, show_confidence=True, show_position=False):
    """
    Draw bounding boxes and labels on image.
    
    Args:
        img: Input image (BGR format)
        objects: List of detected objects
        show_class: Show class/color name
        show_confidence: Show confidence score (if available)
        show_position: Show position (left/center/right)
        
    Returns:
        Annotated image
    """
    annotated = img.copy()
    
    for obj in objects:
        # Get bbox
        if 'bbox' in obj:
            x1, y1, x2, y2 = obj['bbox']
        else:
            x1 = obj['x'] - obj['width'] // 2
            y1 = obj['y'] - obj['height'] // 2
            x2 = x1 + obj['width']
            y2 = y1 + obj['height']
        
        # Choose color
        if 'color' in obj:
            color_map = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)}
            color = color_map.get(obj['color'], (0, 255, 0))
        else:
            color = (0, 255, 0)
        
        # Draw box
        cv.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Build label
        label_parts = []
        if show_class:
            label_parts.append(obj.get('class', obj.get('color', 'object')))
        if show_confidence and 'confidence' in obj:
            label_parts.append(f"{obj['confidence']:.2f}")
        if show_position:
            label_parts.append(f"({obj['position']})")
        
        label = ' '.join(label_parts)
        
        # Draw label
        cv.putText(annotated, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated


def calculate_object_distance(obj, img_width, img_height, focal_length=500, real_object_width=0.3):
    """
    Estimate distance to object using pinhole camera model (rough approximation).
    
    Args:
        obj: Detected object dict
        img_width: Image width in pixels
        img_height: Image height in pixels
        focal_length: Camera focal length in pixels (calibrate for accuracy)
        real_object_width: Real-world object width in meters
        
    Returns:
        Estimated distance in meters
    """
    pixel_width = obj['width']
    distance = (real_object_width * focal_length) / pixel_width
    return distance


def objects_summary(objects):
    """
    Generate a summary of detected objects.
    
    Args:
        objects: List of detected objects
        
    Returns:
        Dictionary with summary statistics
    """
    if not objects:
        return {'count': 0, 'classes': [], 'colors': []}
    
    classes = [obj.get('class') for obj in objects if 'class' in obj]
    colors = [obj.get('color') for obj in objects if 'color' in obj]
    
    summary = {
        'count': len(objects),
        'classes': list(set(classes)),
        'colors': list(set(colors)),
        'positions': {
            'left': len([o for o in objects if o['position'] == 'left']),
            'center': len([o for o in objects if o['position'] == 'center']),
            'right': len([o for o in objects if o['position'] == 'right'])
        },
        'largest_object': get_largest_object(objects)
    }
    
    return summary
