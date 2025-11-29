#!/usr/bin/env python3
"""
Test Ollama vision API with a sample image
"""
import requests
import base64
import json
import sys
import cv2 as cv
import numpy as np

def test_ollama_connection():
    """Test basic Ollama connection"""
    print("Testing Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            print(f"✓ Ollama is running")
            print(f"  Available models: {', '.join(model_names)}")
            return True
        else:
            print(f"✗ Ollama responded with error: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return False

def create_test_image():
    """Create a simple test image with text"""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White background
    
    # Add some colored shapes
    cv.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv.circle(img, (400, 150), 80, (0, 255, 0), -1)  # Green circle
    cv.rectangle(img, (450, 300), (590, 430), (0, 0, 255), -1)  # Red rectangle
    
    # Add text
    cv.putText(img, "TEST IMAGE", (180, 400), 
               cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    return img

def encode_image_base64(img):
    """Encode image to base64"""
    # Encode to JPEG
    _, buffer = cv.imencode('.jpg', img, [cv.IMWRITE_JPEG_QUALITY, 85])
    # Convert to base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def test_ollama_generate_endpoint(img_base64, model="qwen3-vl:2b"):
    """Test /api/generate endpoint"""
    print(f"\nTesting /api/generate endpoint with {model}...")
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": "Describe what you see in this image in 1 sentence.",
        "images": [img_base64],
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"  ✓ Response received")
            print(f"  Keys in response: {list(result.keys())}")
            if 'response' in result:
                print(f"  Content: {result['response'][:100]}...")
            return True
        else:
            print(f"  ✗ Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False

def test_ollama_chat_endpoint(img_base64, model="qwen3-vl:2b"):
    """Test /api/chat endpoint"""
    print(f"\nTesting /api/chat endpoint with {model}...")
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Describe what you see in this image.",
                "images": [img_base64]
            }
        ],
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"  ✓ Response received")
            print(f"  Keys in response: {list(result.keys())}")
            if 'message' in result:
                content = result['message'].get('content', '')
                print(f"  Content: {content[:100]}...")
            return True
        else:
            print(f"  ✗ Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False

def main():
    print("=" * 60)
    print("OLLAMA VISION API TEST")
    print("=" * 60)
    
    # Test connection
    if not test_ollama_connection():
        sys.exit(1)
    
    # Create test image
    print("\nCreating test image...")
    img = create_test_image()
    print(f"  Image shape: {img.shape}")
    
    # Show the test image
    cv.imshow('Test Image', img)
    print("  Displaying test image (close window to continue)...")
    cv.waitKey(2000)  # Show for 2 seconds
    cv.destroyAllWindows()
    
    # Encode image
    print("\nEncoding image to base64...")
    img_base64 = encode_image_base64(img)
    print(f"  Base64 length: {len(img_base64)} characters")
    
    # Test both endpoints
    generate_works = test_ollama_generate_endpoint(img_base64)
    chat_works = test_ollama_chat_endpoint(img_base64)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"/api/generate endpoint: {'✓ Works' if generate_works else '✗ Failed'}")
    print(f"/api/chat endpoint:     {'✓ Works' if chat_works else '✗ Failed'}")
    
    if generate_works:
        print("\n→ Use /api/generate endpoint in cam_ollama.py")
    elif chat_works:
        print("\n→ Use /api/chat endpoint in cam_ollama.py")
    else:
        print("\n→ Neither endpoint works - check Ollama model installation")
        print("  Try: ollama pull qwen3-vl:2b")

if __name__ == "__main__":
    main()
