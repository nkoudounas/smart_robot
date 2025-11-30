#!/usr/bin/env python3
"""
Interactive servo control script
Type commands like "rotate 60" or "r -45" to control the servo
Type "quit" or "q" to exit
"""

import socket
import sys
import time
import colorlog
from utils.robot_utils import cmd, check_connection, setup_socket_options, reconnect_robot

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


def connect_to_robot():
    """Establish connection to robot - identical to fcam.py"""
    logger.info("Connecting to robot at 192.168.4.1:100...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        sock.connect(('192.168.4.1', 100))
        
        # Apply socket options (same as fcam.py)
        setup_socket_options(sock)
        
        logger.info("✓ Connected to robot successfully!")
        logger.info(f"Local address: {sock.getsockname()}")
        logger.info(f"Remote address: {sock.getpeername()}")
        
        # Wait for robot to settle after connection
        time.sleep(0.5)
        
        # Clear any pending data in receive buffer
        sock.setblocking(False)
        try:
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                logger.debug(f"Cleared {len(data)} bytes from buffer")
        except:
            pass  # No more data to read
        sock.setblocking(True)
        sock.settimeout(10.0)
        
        return sock
    except socket.timeout:
        logger.error("ERROR: Connection timeout. Is the robot powered on?")
        sys.exit(1)
    except socket.error as e:
        logger.error(f"ERROR: Connection failed: {e}")
        sys.exit(1)


def reconnect_with_buffer_clear(sock):
    """
    Reconnect to robot using robot_utils.reconnect_robot() 
    and then clear the receive buffer
    """
    try:
        # Use the proper reconnect from robot_utils (resets cmd counters)
        new_sock = reconnect_robot(sock)  # This calls the imported function from robot_utils
        if new_sock is None:
            return None
        
        # Wait for robot to settle
        time.sleep(0.5)
        
        # Clear any pending data in receive buffer
        new_sock.setblocking(False)
        try:
            while True:
                data = new_sock.recv(4096)
                if not data:
                    break
                logger.debug(f"Cleared {len(data)} bytes from buffer")
        except:
            pass  # No more data to read
        new_sock.setblocking(True)
        new_sock.settimeout(10.0)
        
        return new_sock
    except Exception as e:
        logger.error(f"✗ Failed to reconnect: {e}")
        return None


def print_help():
    """Print available commands"""
    logger.info("\n" + "="*50)
    logger.info("Servo Control Commands:")
    logger.info("="*50)
    logger.info("  rotate <angle>  - Rotate servo to angle (-90 to 90)")
    logger.info("  r <angle>       - Short form of rotate")
    logger.info("  center          - Center servo (0 degrees)")
    logger.info("  c               - Short form of center")
    logger.info("  left            - Rotate servo left (60 degrees)")
    logger.info("  right           - Rotate servo right (-60 degrees)")
    logger.info("  scan            - Scan left to right")
    logger.info("  distance        - Read ultrasonic distance sensor")
    logger.info("  d               - Short form of distance")
    logger.info("  help            - Show this help message")
    logger.info("  quit / q        - Exit program")
    logger.info("="*50)
    logger.info("Examples:")
    logger.info("  > rotate 60")
    logger.info("  > r -45")
    logger.info("  > center")
    logger.info("  > scan")
    logger.info("  > distance")
    logger.info("="*50 + "\n")


def parse_command(user_input):
    """
    Parse user input and return command tuple (action, value)
    Returns: (action: str, value: int or None)
    """
    user_input = user_input.strip().lower()
    
    if not user_input:
        return None, None
    
    parts = user_input.split()
    action = parts[0]
    
    # Quit commands
    if action in ['quit', 'q', 'exit']:
        return 'quit', None
    
    # Help command
    if action in ['help', 'h', '?']:
        return 'help', None
    
    # Center commands
    if action in ['center', 'c']:
        return 'rotate', 0
    
    # Left command
    if action in ['left', 'l']:
        return 'rotate', 60
    
    # Right command
    # Scan command
    if action in ['scan', 's']:
        return 'scan', None
    
    # Distance command
    if action in ['distance', 'd']:
        return 'distance', None
    
    # Rotate commands
    if action in ['rotate', 'r']:
        if len(parts) < 2:
            logger.error("ERROR: rotate command requires an angle")
            logger.info("Usage: rotate <angle>  (e.g., rotate 45)")
            return None, None
            logger.error("ERROR: rotate command requires an angle")
            logger.info("Usage: rotate <angle>  (e.g., rotate 45)")
            return None, None
        
        try:
            angle = int(parts[1])
            if angle < -90 or angle > 90:
                logger.error(f"ERROR: Angle must be between -90 and 90 (got {angle})")
                return None, None
            return 'rotate', angle
        except ValueError:
            logger.error(f"ERROR: Invalid angle '{parts[1]}' - must be a number")
            return None, None
    
    # Unknown command
    logger.error(f"ERROR: Unknown command '{action}'")
    logger.info("Type 'help' for available commands")
    return None, None


def execute_command(sock, action, value):
    """
    Execute parsed command
    Returns: socket (may be reconnected), continue (bool)
    """
    if action == 'quit':
        logger.info("Exiting...")
        return sock, False
    
    if action == 'help':
        print_help()
        return sock, True
    
    if action == 'rotate':
        # Check connection
        if not check_connection(sock):
            logger.error("Connection lost, attempting to reconnect...")
            sock = reconnect_with_buffer_clear(sock)
            if sock is None:
                logger.error("Failed to reconnect. Exiting.")
                return None, False
        
        # Send rotate command
        logger.info(f"Rotating servo to {value} degrees...")
        result = cmd(sock, 'rotate', at=value)
        
        if result is not None:
            logger.info(f"✓ Servo rotated to {value}°")
        else:
            logger.error("✗ Failed to rotate servo")
            # Try to reconnect
            logger.warning("Attempting to reconnect...")
            sock = reconnect_with_buffer_clear(sock)
            if sock is None:
                logger.error("Failed to reconnect. Exiting.")
                return None, False
        
        return sock, True
    
    if action == 'scan':
        logger.info("Scanning left to right...")
        
        # Check connection
        if not check_connection(sock):
            logger.error("Connection lost, attempting to reconnect...")
            sock = reconnect_with_buffer_clear(sock)
            if sock is None:
                logger.error("Failed to reconnect. Exiting.")
                return None, False
        
        # Scan sequence: center -> left -> right -> center
        angles = [0, 60, -60, 0]
        for angle in angles:
            logger.info(f"  → {angle}°")
            result = cmd(sock, 'rotate', at=angle)
            if result is None:
                logger.error(f"✗ Failed at {angle}°")
                break
            time.sleep(0.5)  # Wait for servo to move
        
        logger.info("✓ Scan complete")
        return sock, True
    
    if action == 'distance':
        # Check connection
        if not check_connection(sock):
            logger.error("Connection lost, attempting to reconnect...")
            sock = reconnect_with_buffer_clear(sock)
            if sock is None:
                logger.error("Failed to reconnect. Exiting.")
                return None, False
        
        # Read distance
        logger.info("Reading ultrasonic distance sensor...")
        result = cmd(sock, 'measure', what='distance')
        
        if result is not None and isinstance(result, int):
            logger.info(f"✓ Distance: {result} cm")
        else:
            logger.error("✗ Failed to read distance sensor")
            # Try to reconnect
            logger.warning("Attempting to reconnect...")
            sock = reconnect_with_buffer_clear(sock)
            if sock is None:
                logger.error("Failed to reconnect. Exiting.")
                return None, False
        
        return sock, True
    
    return sock, True


def main():
    """Main interactive loop"""
    # Print banner
    logger.info("\n" + "="*50)
    logger.info("🤖 Robot Servo Interactive Control")
    logger.info("="*50)
    
    # Connect to robot
    sock = connect_to_robot()
    
    # Print help
    print_help()
    
    # DON'T center servo on startup - let user control it
    logger.info("Ready! Type commands (try 'center', 'rotate 45', or 'help')")
    logger.info("Note: First command after connection may be slow\n")
    
    # Main loop
    try:
        while True:
            # Prompt for input
            try:
                user_input = input("\n\033[1;36m> \033[0m")  # Cyan prompt
            except EOFError:
                logger.info("\nEOF detected, exiting...")
                break
            
            # Parse command
            action, value = parse_command(user_input)
            
            if action is None:
                continue
            
            # Execute command
            sock, should_continue = execute_command(sock, action, value)
            
            if not should_continue:
                break
            
            if sock is None:
                logger.error("Lost connection to robot")
                break
    
    except KeyboardInterrupt:
        logger.warning("\n\nCtrl+C pressed, exiting...")
    
    finally:
        # Center servo before exit
        if sock and check_connection(sock):
            logger.info("Centering servo before exit...")
            cmd(sock, 'rotate', at=0)
            time.sleep(0.3)
            sock.close()
        logger.info("Goodbye!")


if __name__ == '__main__':
    main()
