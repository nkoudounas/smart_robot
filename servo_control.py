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
from utils.robot_utils import cmd, check_connection, setup_socket_options

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
        return sock
    except socket.timeout:
        logger.error("ERROR: Connection timeout. Is the robot powered on?")
        sys.exit(1)
    except socket.error as e:
        logger.error(f"ERROR: Connection failed: {e}")
        sys.exit(1)


def reconnect_robot(sock):
    """Reconnect to robot if connection lost"""
    try:
        logger.warning("\n[Reconnecting to robot...]")
        try:
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        except:
            pass
        
        time.sleep(0.5)
        
        new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        new_sock.settimeout(10.0)
        new_sock.connect(('192.168.4.1', 100))
        setup_socket_options(new_sock)
        
        logger.info("✓ Reconnected")
        time.sleep(0.3)
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
    logger.info("  help            - Show this help message")
    logger.info("  quit / q        - Exit program")
    logger.info("="*50)
    logger.info("Examples:")
    logger.info("  > rotate 60")
    logger.info("  > r -45")
    logger.info("  > center")
    logger.info("  > scan")
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
    if action in ['right', 'r'] and len(parts) == 1:
        return 'rotate', -60
    
    # Scan command
    if action in ['scan', 's']:
        return 'scan', None
    
    # Rotate commands
    if action in ['rotate', 'r']:
        if len(parts) < 2:
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
            sock = reconnect_robot(sock)
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
            sock = reconnect_robot(sock)
            if sock is None:
                logger.error("Failed to reconnect. Exiting.")
                return None, False
        
        return sock, True
    
    if action == 'scan':
        logger.info("Scanning left to right...")
        
        # Check connection
        if not check_connection(sock):
            logger.error("Connection lost, attempting to reconnect...")
            sock = reconnect_robot(sock)
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
    
    # Center servo on startup
    logger.info("Centering servo...")
    cmd(sock, 'rotate', at=0)
    time.sleep(0.3)
    
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
