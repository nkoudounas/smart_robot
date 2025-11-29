"""
Connection management functions for robot socket communication
"""

import socket
import sys
import time
from .robot_utils import cmd, setup_socket_options,reconnect_robot, should_reconnect, get_commands_sent
import colorlog

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
    """Establish connection to robot"""
    logger.info("Connecting to robot at 192.168.4.1:100...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        sock.connect(('192.168.4.1', 100))
        
        # Apply socket options
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


def periodic_reconnect(sock, threshold=3):
    """
    Perform periodic reconnection to prevent robot firmware timeout.
    Robot firmware closes connection after 8 commands.
    Reconnects based on actual command count (not iteration count).
    
    Args:
        threshold: Number of commands before reconnecting (default 3)
                   Use lower threshold (2) when navigation may use multiple commands
    
    Returns: new socket or None if failed
    """
    # Check if we've sent enough commands to warrant reconnection
    if should_reconnect(threshold=threshold):
        cmds = get_commands_sent()
        logger.warning(f"\n[Maintenance: {cmds} commands sent - reconnecting before robot limit]")
        
        # Reconnect to reset command buffer
        sock = reconnect_robot(sock)
        
        if sock is None:
            return None
        
        return sock
    
    return sock
