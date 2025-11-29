"""
Connection management functions for robot socket communication
"""

import socket
import sys
import time
from ..utils.robot_utils import cmd, setup_socket_options


def connect_to_robot():
    """
    Establish TCP socket connection to robot.
    
    Returns:
        socket: Connected socket object
        
    Raises:
        SystemExit: If connection fails
    """
    print("Connecting to robot at 192.168.4.1:100...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        sock.connect(('192.168.4.1', 100))
        
        # Apply socket options
        setup_socket_options(sock)
        
        print("✓ Connected to robot successfully!")
        print(f"Local address: {sock.getsockname()}")
        print(f"Remote address: {sock.getpeername()}")
        return sock
    except socket.timeout:
        print("ERROR: Connection timeout. Is the robot powered on?")
        sys.exit(1)
    except socket.error as e:
        print(f"ERROR: Connection failed: {e}")
        sys.exit(1)


def periodic_reconnect(sock, iteration):
    """
    Perform periodic reconnection to prevent robot firmware timeout.
    Robot firmware closes connection after 8 commands.
    
    Args:
        sock: Current socket connection
        iteration: Current iteration number
        
    Returns:
        socket: New socket if reconnected, original socket otherwise
    """
    if iteration % 3 == 0 and iteration > 0:
        print(f"\n[Periodic reconnection - iteration {iteration}]")
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
            return sock
        except Exception as e:
            print(f"\nERROR: Failed to reconnect: {e}")
            return None
    
    return sock
