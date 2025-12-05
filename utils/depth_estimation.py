"""
Depth estimation using object size tracking across frames
"""
import numpy as np
from collections import defaultdict
import time

# Typical real-world widths of objects in meters
OBJECT_SIZES = {
    'person': 0.45,      # Average shoulder width
    'chair': 0.50,  
    'sports ball': 0.24,        # Average ball diameter
    'couch': 1.80,       # Average couch width
    'dining table': 1.20,
    'tv': 1.00,
    'laptop': 0.35,
    'cell phone': 0.07,
    'bottle': 0.08,
    'cup': 0.08,
    'book': 0.15,
    'car': 1.80,
    'bicycle': 0.60,
    'dog': 0.40,
    'cat': 0.25,
    'bed': 1.50,
    'potted plant': 0.30,
    'clock': 0.30,
    'vase': 0.20,
    'backpack': 0.40,
    'handbag': 0.30,
    'suitcase': 0.50,
}

# Assumed camera parameters (can be calibrated)
FOCAL_LENGTH_PIXELS = 600  # Rough estimate for most webcams at 640x480


class DepthEstimator:
    """Estimates object depth using size tracking across frames"""
    
    def __init__(self):
        # Track object history: {object_id: [(timestamp, bbox_width, estimated_distance)]}
        self.object_history = defaultdict(list)
        self.max_history = 10  # Keep last 10 observations per object
        
    def _get_object_id(self, obj):
        """Create a unique ID for tracking an object across frames"""
        # Use class + approximate position to track same object
        pos_x = int(obj['x'] / 100) * 100  # Discretize position
        pos_y = int(obj['y'] / 100) * 100
        return f"{obj['class']}_{pos_x}_{pos_y}"
    
    def estimate_distance(self, obj, image_width):
        """
        Estimate distance to an object using its bounding box width.
        
        Formula: distance = (real_width × focal_length) / pixel_width
        
        Args:
            obj: Object dict with 'class', 'width', 'x', 'y', etc.
            image_width: Width of the image in pixels
            
        Returns:
            Estimated distance in meters, or None if cannot estimate
        """
        obj_class = obj['class']
        pixel_width = obj['width']
        
        # Check if we know the typical size of this object
        if obj_class not in OBJECT_SIZES:
            return None
        
        real_width = OBJECT_SIZES[obj_class]
        
        # Basic pinhole camera formula
        if pixel_width > 0:
            distance = (real_width * FOCAL_LENGTH_PIXELS) / pixel_width
            return distance
        
        return None
    
    def estimate_with_tracking(self, obj, image_width):
        """
        Estimate distance with temporal smoothing using object tracking.
        
        This uses previous observations to improve the estimate:
        - Smooth out noise by averaging recent measurements
        - Detect if object is moving closer/farther
        - Improve accuracy over time
        
        Returns:
            tuple: (distance_estimate, confidence, velocity)
        """
        # Get basic distance estimate
        distance = self.estimate_distance(obj, image_width)
        
        if distance is None:
            return None, 0.0, 0.0
        
        # Track this object
        obj_id = self._get_object_id(obj)
        timestamp = time.time()
        pixel_width = obj['width']
        
        # Add to history
        self.object_history[obj_id].append((timestamp, pixel_width, distance))
        
        # Keep only recent history
        if len(self.object_history[obj_id]) > self.max_history:
            self.object_history[obj_id] = self.object_history[obj_id][-self.max_history:]
        
        history = self.object_history[obj_id]
        
        # If we have multiple observations, smooth the estimate
        if len(history) >= 2:
            # Use weighted average (recent measurements weighted more)
            weights = np.linspace(0.5, 1.0, len(history))
            distances = [h[2] for h in history]
            smoothed_distance = np.average(distances, weights=weights)
            
            # Calculate velocity (is object approaching or receding?)
            time_diff = history[-1][0] - history[0][0]
            if time_diff > 0:
                distance_change = history[-1][2] - history[0][2]
                velocity = distance_change / time_diff  # meters per second
            else:
                velocity = 0.0
            
            # Confidence increases with more observations
            confidence = min(len(history) / self.max_history, 1.0)
            
            return smoothed_distance, confidence, velocity
        else:
            # Single observation, lower confidence
            return distance, 0.3, 0.0
    
    def estimate_all_objects(self, objects, image_width):
        """
        Estimate distances for all detected objects.
        
        Returns:
            list of dicts with object info + distance estimates
        """
        results = []
        
        for obj in objects:
            distance, confidence, velocity = self.estimate_with_tracking(obj, image_width)
            
            if distance is not None:
                result = {
                    'object': obj,
                    'distance': distance,
                    'confidence': confidence,
                    'velocity': velocity,
                    'approaching': velocity < -0.1,  # Moving closer
                    'receding': velocity > 0.1,      # Moving away
                }
                results.append(result)
        
        return results
    
    def get_closest_object(self, objects, image_width):
        """Find the closest detected object"""
        estimates = self.estimate_all_objects(objects, image_width)
        
        if not estimates:
            return None
        
        # Sort by distance
        estimates.sort(key=lambda x: x['distance'])
        return estimates[0]
    
    def cleanup_old_tracks(self, max_age=5.0):
        """Remove object tracks that haven't been updated recently"""
        current_time = time.time()
        to_remove = []
        
        for obj_id, history in self.object_history.items():
            if history:
                last_timestamp = history[-1][0]
                if current_time - last_timestamp > max_age:
                    to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.object_history[obj_id]
