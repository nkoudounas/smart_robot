"""
Video logger utility - Records frames with embedded log text overlay
Creates a split-screen video with camera feed on the left and logs on the right
"""

import cv2 as cv
import numpy as np
from datetime import datetime
from collections import deque
import colorlog

logger = colorlog.getLogger(__name__)


class VideoLogger:
    """
    Records video frames with embedded log messages in a side-by-side layout.
    Left side: Camera feed with annotations
    Right side: Scrolling text log of recent messages
    """
    
    def __init__(self, output_path=None, fps=10.0, frame_width=640, frame_height=480, max_log_lines=25):
        """
        Initialize video logger.
        
        Args:
            output_path: Path to save video file (default: auto-generated timestamp)
            fps: Frames per second for output video
            frame_width: Width of input camera frames
            frame_height: Height of input camera frames
            max_log_lines: Maximum number of log lines to show in text area
        """
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_log_lines = max_log_lines
        
        # Output video dimensions (double width for side-by-side)
        self.output_width = frame_width * 2
        self.output_height = frame_height
        
        # Generate output path if not provided
        if output_path is None:
            # Create videos directory if it doesn't exist
            import os
            videos_dir = "videos"
            os.makedirs(videos_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(videos_dir, f"robot_video_{timestamp}.mp4")
        self.output_path = output_path
        
        # Video writer (initialized on first frame)
        self.writer = None
        self.fourcc = cv.VideoWriter_fourcc(*'mp4v')
        
        # Log message buffer (fixed-size deque for scrolling effect)
        self.log_buffer = deque(maxlen=max_log_lines)
        
        # Text rendering settings
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.45
        self.font_thickness = 1
        self.line_height = 20  # Pixels between lines
        self.text_color = (255, 255, 255)  # White text
        self.bg_color = (30, 30, 30)  # Dark gray background
        self.padding = 10  # Padding from edges
        
        # Recording state
        self.is_recording = False
        self.frame_count = 0
        
        logger.info(f"VideoLogger initialized: {output_path}")
    
    def start_recording(self):
        """Start video recording"""
        self.is_recording = True
        logger.info(f"📹 Recording started: {self.output_path}")
    
    def stop_recording(self):
        """Stop video recording and release resources"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.is_recording = False
        logger.info(f"📹 Recording stopped: {self.output_path} ({self.frame_count} frames)")
        self.frame_count = 0
    
    def add_log(self, message, level="INFO"):
        """
        Add a log message to the buffer.
        
        Args:
            message: Log message text
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        # Add timestamp and level prefix
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color code based on level (using ANSI-like colors mapped to BGR)
        color_map = {
            "DEBUG": (255, 255, 0),    # Cyan
            "INFO": (255, 255, 255),   # White
            "WARNING": (0, 255, 255),  # Yellow
            "ERROR": (0, 0, 255),      # Red
            "CRITICAL": (0, 0, 255),   # Red
        }
        color = color_map.get(level.upper(), (255, 255, 255))
        
        # Store message with metadata
        log_entry = {
            "timestamp": timestamp,
            "level": level.upper(),
            "message": message,
            "color": color
        }
        self.log_buffer.append(log_entry)
    
    def _create_log_panel(self):
        """
        Create the right-side log panel with scrolling text.
        
        Returns:
            numpy array: Image of log panel
        """
        # Create blank panel
        panel = np.full((self.frame_height, self.frame_width, 3), self.bg_color, dtype=np.uint8)
        
        # Add title at top
        title = "ROBOT LOGS"
        cv.putText(panel, title, (self.padding, 30), self.font, 0.6, (100, 255, 100), 2)
        
        # Draw separator line
        cv.line(panel, (self.padding, 40), (self.frame_width - self.padding, 40), (100, 100, 100), 1)
        
        # Render log messages (newest at bottom)
        y_position = 60  # Start below title
        for log_entry in self.log_buffer:
            if y_position + self.line_height > self.frame_height - self.padding:
                break  # Stop if we run out of space
            
            # Format message: [HH:MM:SS] LEVEL: message
            text = f"[{log_entry['timestamp']}] {log_entry['level']}: {log_entry['message']}"
            
            # Wrap long messages to fit panel width
            wrapped_lines = self._wrap_text(text, self.frame_width - 2 * self.padding)
            
            for line in wrapped_lines:
                if y_position + self.line_height > self.frame_height - self.padding:
                    break
                cv.putText(panel, line, (self.padding, y_position), 
                          self.font, self.font_scale, log_entry['color'], self.font_thickness)
                y_position += self.line_height
        
        return panel
    
    def _wrap_text(self, text, max_width):
        """
        Wrap text to fit within max_width pixels.
        
        Args:
            text: Text to wrap
            max_width: Maximum width in pixels
        
        Returns:
            list: List of wrapped text lines
        """
        words = text.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            (text_width, _), _ = cv.getTextSize(test_line, self.font, self.font_scale, self.font_thickness)
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines if lines else [text]
    
    def write_frame(self, frame, auto_log=True):
        """
        Write a frame to the video with embedded logs.
        
        Args:
            frame: Camera frame (BGR image)
            auto_log: If True, automatically log frame number
        
        Returns:
            numpy array: Combined frame (camera + logs) for preview
        """
        if not self.is_recording:
            return frame
        
        # Resize frame if needed
        if frame.shape[:2] != (self.frame_height, self.frame_width):
            frame = cv.resize(frame, (self.frame_width, self.frame_height))
        
        # Create log panel
        log_panel = self._create_log_panel()
        
        # Combine side by side
        combined = np.hstack([frame, log_panel])
        
        # Initialize writer on first frame
        if self.writer is None:
            self.writer = cv.VideoWriter(
                self.output_path, 
                self.fourcc, 
                self.fps, 
                (self.output_width, self.output_height)
            )
            logger.info(f"Video writer initialized: {self.output_width}x{self.output_height} @ {self.fps}fps")
        
        # Write frame
        self.writer.write(combined)
        self.frame_count += 1
        
        # Auto-log frame number periodically
        if auto_log and self.frame_count % 10 == 0:
            self.add_log(f"Frame {self.frame_count} recorded", "DEBUG")
        
        return combined
    
    def __enter__(self):
        """Context manager entry"""
        self.start_recording()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_recording()
    
    def __del__(self):
        """Destructor - ensure writer is released"""
        if self.writer is not None:
            self.writer.release()


# Convenience function for quick usage
def create_video_logger(output_path=None, fps=10.0, frame_width=640, frame_height=480):
    """
    Create and return a VideoLogger instance.
    
    Args:
        output_path: Path to save video file (default: auto-generated timestamp)
        fps: Frames per second for output video
        frame_width: Width of input camera frames
        frame_height: Height of input camera frames
    
    Returns:
        VideoLogger instance
    """
    return VideoLogger(output_path, fps, frame_width, frame_height)
