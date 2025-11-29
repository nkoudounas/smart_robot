"""
Robot utility functions - socket communication and camera capture
"""
import socket
import struct
import json
import re
import time
import cv2 as cv
from urllib.request import urlopen
import numpy as np

cmd_no = 0
commands_sent = 0

def capture():
    """Capture image from robot camera (HTTP, not socket command)"""
    global cmd_no
    cmd_no += 1
    print(f"cmd_no: {str(cmd_no)}" + ': capture image', end=' ')
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            cam = urlopen('http://192.168.4.1/capture', timeout=5)
            img = cam.read()
            img = np.asarray(bytearray(img), dtype='uint8')
            img = cv.imdecode(img, cv.IMREAD_UNCHANGED)
            if img is None or img.size == 0:
                print('ERROR: Failed to decode image')
                if attempt < max_retries - 1:
                    print(f' (retry {attempt + 1}/{max_retries})...', end='')
                    time.sleep(0.5)
                    continue
                return None
            print(f'[OK: {img.shape}]')
            return img
        except Exception as e:
            if attempt < max_retries - 1:
                print(f'\nERROR capturing image: {e} (retry {attempt + 1}/{max_retries})...', end='')
                time.sleep(1.0)
            else:
                print(f'\nERROR capturing image: {e}')
                return None
    return None

def check_connection(sock):
    """Check if socket is still connected"""
    try:
        sock.getpeername()
        return True
    except:
        return False

def cmd(sock, do, what='', where='', at=''):
    """Send command to robot"""
    global cmd_no, commands_sent
    cmd_no += 1
    commands_sent += 1
    
    if not check_connection(sock):
        print(f"ERROR: Socket disconnected before command {cmd_no}")
        return None
    
    msg = {"H": str(cmd_no)}
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
        msg["D2"] = at  # speed
        where = where + ' '
    elif do == 'stop':
        msg.update({"N": 1, "D1": 0, "D2": 0, "D3": 1})
        what = ' car'
    elif do == 'rotate':
        msg.update({"N": 5, "D1": 1, "D2": at})  # at is angle here
        what = ' ultrasonic unit'
        where = ' '
    elif do == 'measure':
        if what == 'distance':
            msg.update({"N": 21, "D1": 2})
        elif what == 'motion':
            msg["N"] = 6
        what = ' ' + what
    elif do == 'check':
        msg["N"] = 23
        what = ' off the ground'
    
    msg_json = json.dumps(msg)
    print(str(cmd_no) + ': ' + do + what + where + str(at), end=': ')
    
    try:
        sock.send(msg_json.encode())
        print("[SENT]", end=' ')
    except socket.error as e:
        print(f'\nERROR sending command: {e}')
        return None
    
    try:
        response_buffer = ""
        start_time = time.time()
        sock.settimeout(5.0)
        
        while True:
            if time.time() - start_time > 5.0:
                print(f'\nERROR: Timeout waiting for response')
                return None
            
            try:
                chunk = sock.recv(512).decode()
            except socket.timeout:
                if '_' in response_buffer:
                    break
                print(f'\nERROR: Recv timeout')
                return None
            
            if not chunk:
                print(f'\nERROR: Connection closed by robot (no response received)')
                return None
            
            response_buffer += chunk
            
            if '_' in response_buffer:
                print(f"[RECV: {len(response_buffer)} bytes]", end=' ')
                break
        
        res = response_buffer
        time.sleep(0.1)
        
    except Exception as e:
        print(f'\nERROR receiving response: {e}')
        return None
    
    try:
        res = re.search('_(.*)}', res).group(1)
    except (AttributeError, IndexError) as e:
        print(f'\nERROR: Could not parse response')
        return None
    
    if res == 'ok' or res == 'true':
        res = 1
    elif res == 'false':
        res = 0
    else:
        res = int(res)
    print(res)
    return res

def setup_socket_options(socket_obj):
    """Apply socket options"""
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

def reconnect_robot(sock):
    """Reconnect to robot and reset command counters"""
    global commands_sent, cmd_no
    try:
        print("\n[Reconnecting to robot...]", end=' ')
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
        
        # CRITICAL: Reset command counters for new connection
        commands_sent = 0
        cmd_no = 0
        
        print("✓ Reconnected")
        time.sleep(0.3)
        return new_sock
    except Exception as e:
        print(f"✗ Failed: {e}")
        return None


def read_distance(sock):
    """Read ultrasonic distance sensor (returns distance in cm). Uses 1 command."""
    result = cmd(sock, 'measure', what='distance')
    if result is not None and isinstance(result, int):
        return result
    return None


def rotate_sensor(sock, angle):
    """Rotate ultrasonic sensor to specified angle (-90 to 90). Uses 1 command."""
    result = cmd(sock, 'rotate', at=angle)
    return result


def read_ir_sensors(sock):
    """Read IR line sensors - checks if robot is off the ground or at edge. Uses 1 command."""
    result = cmd(sock, 'check')
    if result is not None:
        # Result: 1 = on ground, 0 = off ground/edge detected
        return result
    return None
