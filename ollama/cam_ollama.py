"""
Ollama AI-powered robot navigation
Imports utility functions for clean code organization
"""
import cv2 as cv
import socket
import sys
import time

# Import robot communication functions
from utils.robot_utils import (
    capture, cmd, check_connection, 
    setup_socket_options, reconnect_robot
)

# Import Ollama vision functions
from .ollama_vision import (
    query_ollama_vision, parse_ollama_decision,
    check_ollama_connection
)

# Setup camera window
cv.namedWindow('Camera', cv.WINDOW_NORMAL)
cv.resizeWindow('Camera', 800, 600)
cv.moveWindow('Camera', 0, 0)

def navigate_with_ollama(sock, img, target_class=None, avoid_classes=None):
    """
    Ask Ollama to decide robot movement and execute it.
    
    Args:
        sock: Robot socket connection
        img: Camera image
        target_class: Specific object to navigate toward (e.g., 'chair', 'person')
        avoid_classes: List of object classes to avoid (currently not used, reserved for future)
    
    Returns:
        Movement decision string
    """
    if img is None:
        print("ERROR: No image for navigation")
        return None
    
    # Show original image immediately for responsive feedback
    cv.imshow('Camera', img)
    cv.waitKey(1)
    
    # Build navigation prompt based on target
    if target_class:
        prompt = f"""You are controlling a robot car. Look at this image from the robot's camera.

Your GOAL: Navigate toward the {target_class}.

Available movements:
- forward: move straight ahead
- left: turn left
- right: turn right  
- back: move backward
- stop: stop moving

Instructions:
1. First, identify if there is a {target_class} in the image
2. If you see the {target_class}:
   - If it's in the CENTER and CLOSE (large in frame): respond with "stop" (you've reached it)
   - If it's in the CENTER but FAR: respond with "forward" (move toward it)
   - If it's on the LEFT side: respond with "left" (turn to face it)
   - If it's on the RIGHT side: respond with "right" (turn to face it)
3. If you DON'T see the {target_class}:
   - Respond with "right" (search by rotating)
4. If there are obstacles blocking your path to the {target_class}:
   - Respond with "right" or "left" to avoid them

Respond with ONLY ONE WORD from the available movements above."""
    else:
        # General navigation (no specific target)
        prompt = """You are controlling a robot car. Look at this image from the robot's camera.
        Available movements:
        - forward: move straight ahead
        - left: turn left
        - right: turn right  
        - back: move backward
        - stop: stop moving

        Based on what you see, choose the BEST and SAFEST movement. Consider:
        - Obstacles and objects in the path
        - Clear space to move
        - Safety first - avoid collisions

        Respond with ONLY ONE WORD from the available movements above."""

    # Query Ollama
    response = query_ollama_vision(img, prompt)
    
    if response is None:
        print("Ollama failed, defaulting to stop")
        cmd(sock, 'stop')
        return 'stop'
    
    # Parse decision
    decision = parse_ollama_decision(response)
    
    # Display results with target info
    if target_class:
        print(f"🎯 Target: {target_class}")
    print(f"🤖 Ollama says: {response[:60]}...")
    print(f"→ Decision: {decision}")
    
    # Show annotated image with Ollama's decision
    annotated_img = img.copy()
    
    # Show target if specified
    if target_class:
        cv.putText(annotated_img, f"Target: {target_class.upper()}", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv.putText(annotated_img, f"Decision: {decision.upper()}", (10, 65), 
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv.putText(annotated_img, response[:80], (10, 95), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv.putText(annotated_img, f"Decision: {decision.upper()}", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv.putText(annotated_img, response[:80], (10, 60), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Replace with annotated image
    cv.imshow('Camera', annotated_img)
    cv.waitKey(1)
    
    # Execute decision
    if decision == 'stop':
        cmd(sock, 'stop')
    elif decision in ['forward', 'back', 'left', 'right']:
        speed = 50 if decision in ['left', 'right'] else 55
        cmd(sock, 'move', where=decision, at=speed)
    
    return decision

def main():
    """Main navigation loop"""
    # Check Ollama connection
    print("=" * 60)
    print("OLLAMA VISION TEST - Image Description")
    print("=" * 60)
    print("\nTesting Ollama connection...")
    
    if not check_ollama_connection():
        sys.exit(1)

    # Connect to robot
    print("\nConnecting to robot at 192.168.4.1:100...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        sock.connect(('192.168.4.1', 100))
        setup_socket_options(sock)
        
        print("✓ Connected to robot successfully!")
    except socket.timeout:
        print("ERROR: Connection timeout. Is the robot powered on?")
        sys.exit(1)
    except socket.error as e:
        print(f"ERROR: Connection failed: {e}")
        sys.exit(1)

    print("\n✓ Setup complete! Starting AI navigation...")
    print("\n" + "=" * 60)
    print("Starting Ollama AI Navigation...")
    print("=" * 60)
    print("\nKeyboard Controls:")
    print("  'k' - Exit")
    print("  'p' - Pause/Resume autonomous navigation")
    print("  Space - Manual: Capture and describe (no auto-execute)")
    print("  Arrow Keys - Manual override (↑↓←→)")
    print("  's' - Stop robot")
    print("  Ctrl+C - Emergency stop")

    autonomous = True
    iteration_count = 0

    try:
        while True:
            key = cv.waitKey(1) & 0xFF
            
            if key == ord('k'):
                print("\n'k' pressed - Exiting...")
                cmd(sock, 'stop')
                break
            
            elif key == ord('p'):
                autonomous = not autonomous
                print(f"\n'p' pressed - {'AUTONOMOUS' if autonomous else 'PAUSED'}")
                if not autonomous:
                    cmd(sock, 'stop')
                continue
            
            elif key == ord('s'):
                print("\n's' pressed - Stopping robot")
                cmd(sock, 'stop')
                if autonomous:
                    autonomous = False
                    print("(Switched to manual mode)")
                continue
            
            # Manual controls
            elif key == 82 or key == 0:  # Up
                autonomous = False
                print("\n↑ Manual: Moving forward")
                cmd(sock, 'move', where='forward', at=60)
                time.sleep(0.1)
                continue
                
            elif key == 84 or key == 1:  # Down
                autonomous = False
                print("\n↓ Manual: Moving backward")
                cmd(sock, 'move', where='back', at=60)
                time.sleep(0.1)
                continue
                
            elif key == 81 or key == 2:  # Left
                autonomous = False
                print("\n← Manual: Turning left")
                cmd(sock, 'move', where='left', at=60)
                time.sleep(0.1)
                continue
                
            elif key == 83 or key == 3:  # Right
                autonomous = False
                print("\n→ Manual: Turning right")
                cmd(sock, 'move', where='right', at=60)
                time.sleep(0.1)
                continue
            
            # Space for description only
            elif key == 32:
                print("\n" + "=" * 60)
                img = capture()
                if img is None:
                    print("ERROR: Failed to capture image")
                    continue
                
                prompt = """Describe what you see in this image in 2-3 short sentences.
Focus on: objects, obstacles, people, and safe directions to move."""
                
                response = query_ollama_vision(img, prompt)
                
                if response:
                    print("\n📷 Ollama describes:")
                    print(f"   {response}")
                
                print("=" * 60)
                continue
            
            # Autonomous mode
            if autonomous:
                iteration_count += 1
                
                print(f"\n{'='*60}")
                print(f"Iteration {iteration_count}")
                print(f"{'='*60}")
                
                # Capture image
                img = capture()
                if img is None:
                    print("ERROR: Failed to capture image after retries")
                    print("⚠️  Possible issues:")
                    print("   - Robot WiFi disconnected")
                    print("   - Robot powered off")
                    print("   - Network unreachable")
                    print("\nStopping robot and pausing...")
                    try:
                        cmd(sock, 'stop')
                    except:
                        pass
                    autonomous = False
                    print("Press 'p' to retry or 'k' to exit")
                    time.sleep(2.0)
                    continue
                
                # Give camera time to stabilize
                time.sleep(0.3)
                
                # Check connection before movement
                if not check_connection(sock):
                    print("⚠️  Connection lost - reconnecting...")
                    new_sock = reconnect_robot(sock)
                    if new_sock:
                        sock = new_sock
                    else:
                        print("Failed to reconnect - pausing")
                        autonomous = False
                        continue
                
                # Let Ollama navigate
                try:
                    result = navigate_with_ollama(sock, img)
                    
                    # Always reconnect after movement
                    # Robot closes connection after each command execution
                    print("Reconnecting for next iteration...")
                    new_sock = reconnect_robot(sock)
                    if new_sock:
                        sock = new_sock
                    else:
                        print("Failed to reconnect - pausing")
                        autonomous = False
                        continue
                        
                except Exception as e:
                    print(f"\n!!! ERROR in navigation: {e}")
                    new_sock = reconnect_robot(sock)
                    if new_sock:
                        sock = new_sock
                        print("Reconnected after error")
                        continue
                    else:
                        print("Failed to recover - pausing")
                        autonomous = False
                
                # Wait before next iteration
                time.sleep(1.5)
            else:
                time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nCtrl+C pressed - Emergency stop")
        try:
            cmd(sock, 'stop')
        except:
            pass
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        try:
            cmd(sock, 'stop')
        except:
            pass
    finally:
        sock.close()
        cv.destroyAllWindows()
        print(f"\nTotal iterations: {iteration_count}")
        print("Program ended")

if __name__ == "__main__":
    main()

