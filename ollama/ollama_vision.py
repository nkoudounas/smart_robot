"""
Ollama vision integration - image encoding and AI decision making
"""
import cv2 as cv
import base64
import requests
import time

def encode_image_base64(img):
    """Encode OpenCV image to base64 string for Ollama"""
    img_small = cv.resize(img, (640, 480))
    _, buffer = cv.imencode('.jpg', img_small, [cv.IMWRITE_JPEG_QUALITY, 85])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def query_ollama_vision(img, prompt, model="gemma3:4b", enable_thinking=False):
    """Send image and prompt to Ollama vision model with optional thinking process"""
    try:
        img_base64 = encode_image_base64(img)
        
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False
        }
        
        # Enable thinking parameter if supported by model (disabled by default for moondream)
        if enable_thinking:
            payload["think"] = True
        
        print(f"Querying Ollama ({model})...", end=' ')
        start_time = time.time()
        
        response = requests.post(url, json=payload, timeout=60)  # Increased timeout for thinking
        
        if response.status_code == 200:
            result = response.json()
            elapsed = time.time() - start_time
            print(f"[OK: {elapsed:.1f}s]")
            
            # Print thinking process if available
            if enable_thinking and 'thinking' in result:
                print("\n" + "="*60)
                print("🧠 OLLAMA THINKING PROCESS:")
                print("="*60)
                print(result['thinking'])
                print("="*60 + "\n")
            
            return result.get('response', '')
        else:
            print(f"[ERROR: {response.status_code}]")
            print(f"Response: {response.text[:200]}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("[ERROR: Cannot connect to Ollama. Is it running?]")
        return None
    except requests.exceptions.Timeout:
        print("[ERROR: Ollama request timed out]")
        return None
    except Exception as e:
        print(f"[ERROR: {e}]")
        return None

def parse_ollama_decision(response_text):
    """Parse Ollama's response to extract movement decision"""
    if not response_text:
        return None
    
    response_lower = response_text.lower()
    
    # Look for movement keywords in priority order
    if 'stop' in response_lower or 'halt' in response_lower or 'wait' in response_lower:
        return 'stop'
    elif 'back' in response_lower or 'backward' in response_lower or 'reverse' in response_lower:
        return 'back'
    elif 'left' in response_lower or 'turn left' in response_lower:
        return 'left'
    elif 'right' in response_lower or 'turn right' in response_lower:
        return 'right'
    elif 'forward' in response_lower or 'ahead' in response_lower or 'straight' in response_lower:
        return 'forward'
    else:
        return 'stop'

def check_ollama_connection():
    """Check if Ollama is running and has required model"""
    try:
        test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if test_response.status_code == 200:
            models = test_response.json().get('models', [])
            model_names = [m['name'] for m in models]
            print(f"✓ Ollama is running with {len(models)} model(s): {', '.join(model_names)}")
            
            if not any('gemma3' in m for m in model_names):
                print("\n⚠️  WARNING: gemma3:4b not found!")
                print("Run: ollama pull gemma3:4b")
                return False
            return True
        else:
            print("⚠️  Warning: Ollama responded but with error")
            return False
    except:
        print("⚠️  WARNING: Cannot connect to Ollama at localhost:11434")
        print("Make sure Ollama is running: ollama serve")
        print("Then load model: ollama run gemma3:4b")
        return False
