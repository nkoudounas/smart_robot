# Smart Robot - Autonomous Navigation with YOLO & AI

Autonomous robot navigation system using YOLO object detection and AI decision-making (Ollama/Gemini).

## Features

- 🤖 YOLO object detection (v8 & v11) with segmentation support
- 🧠 AI-powered decision making using Ollama (local LLM)
- 📹 Video recording with embedded navigation logs
- 🗺️ Real-time path tracking and visualization
- 🎯 Smart object search and navigation
- 🚧 Obstacle avoidance with depth estimation

## Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Ollama (for AI decision-making, optional)
- Robot hardware: Elegoo Robot Car V4 or compatible

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/nkoudounas/smart_robot.git
cd smart_robot
```

2. **Install dependencies**
```bash
uv pip install opencv-python ultralytics numpy colorlog matplotlib requests
```

3. **Install Ollama (optional, for AI features)**
```bash
# Install Ollama from https://ollama.ai
# Then pull the Gemma model:
ollama run gemma3:4b
```

## Quick Start

### Basic Navigation (without AI)

```bash
uv run fcam.py
```

The script will automatically:
- ✅ Download YOLO models on first run (yolo11x.pt)
- ✅ Connect to robot at 192.168.4.1
- ✅ Start object detection and navigation
- ✅ Record video with logs

### AI-Powered Navigation

1. **Start Ollama server** (in a separate terminal):
```bash
ollama serve
```

2. **Run with AI decision-making**:
```bash
uv run fcam.py
```

Edit `fcam.py` to enable AI:
```python
if __name__ == '__main__':
    use_ollama = False
    ai_decide = True      # Enable AI decision-making
    target = 'chair'      # Target object to search for
    use_segmentation = True  # Use segmentation for better detection
    capture_video = True  # Record navigation video
    main(use_ollama, ai_decide, target, use_segmentation, capture_video)
```

## Configuration

### Main Parameters (in `fcam.py`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ai_decide` | Use AI for decision-making | `True` |
| `target` | Object class to search for | `'chair'` |
| `use_segmentation` | Use YOLO segmentation model | `True` |
| `capture_video` | Record navigation video | `True` |

### YOLO Models

The script automatically downloads models on first run:
- **Detection**: `yolo11x.pt` (extra-large)
- **Segmentation**: `yolo11x-seg.pt` (with masks)

You can also manually download models:
```python
from ultralytics import YOLO
model = YOLO('yolo11x.pt')  # Auto-downloads if not present
```

### Supported Target Objects

Any COCO dataset class: `person`, `chair`, `cup`, `bottle`, `ball`, `car`, `dog`, `cat`, etc.

## Keyboard Controls

| Key | Action |
|-----|--------|
| `v` | Save video and exit |
| `k` | Kill/exit program |
| `p` | Pause/resume navigation |
| `r` | Restart (reset iteration count) |
| `m` | Toggle manual/auto mode |
| `s` | Stop robot |

## Video Recording

Videos are automatically saved to `videos/` folder with format:
```
videos/robot_video_YYYYMMDD_HHMMSS.mp4
```

Each video includes:
- 📹 Camera feed (left panel)
- 📝 Navigation logs (right panel)
- 🎯 Object detection info
- 🧠 AI decisions
- 🚗 Movement commands

## Project Structure

```
smart_robot/
├── fcam.py                 # Main navigation script
├── servo_control.py        # Servo testing utility
├── utils/
│   ├── robot_utils.py      # Robot communication
│   ├── detection_utils.py  # YOLO object detection
│   ├── navigation_utils.py # Navigation algorithms
│   ├── video_logger.py     # Video recording
│   ├── depth_estimation.py # Depth calculation
│   └── connection_utils.py # Network handling
├── ollama/
│   ├── ollama_vision.py    # Ollama API integration
│   ├── cam_ollama.py       # Ollama navigation
│   └── test_ollama.py      # Test Ollama connection
└── videos/                 # Recorded navigation videos
```

## Troubleshooting

### YOLO Models Not Downloading
Models auto-download on first run. If it fails:
```python
from ultralytics import YOLO
YOLO('yolo11x.pt')  # Manually trigger download
```

### Ollama Connection Issues
```bash
# Check if Ollama is running:
curl http://localhost:11434/api/tags

# Start Ollama server:
ollama serve

# Pull model if missing:
ollama pull gemma3:4b
```

### Robot Connection Failed
- Ensure robot is powered on
- Connect to robot's WiFi (usually `ElegooCarXXX`)
- Robot IP should be `192.168.4.1:100`

### Video Not Recording
Set `capture_video = True` in `fcam.py`:
```python
main(use_ollama, ai_decide, target, use_segmentation, capture_video=True)
```

## Hardware Setup

1. **Robot**: Elegoo Robot Car V4 with ESP32-CAM
2. **Network**: Robot creates WiFi AP at 192.168.4.1
3. **Connection**: Socket-based control on port 100
4. **Camera**: HTTP stream at http://192.168.4.1/capture

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - feel free to use and modify!

## Acknowledgments

- YOLO models by Ultralytics
- Ollama for local LLM inference
- Elegoo for robot hardware platform
