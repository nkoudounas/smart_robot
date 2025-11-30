# Capture The Flag (CTF) - Autonomous Navigation Challenge

## Overview

This project simulates a **Capture The Flag** challenge from a self-driving vehicle perspective. The robot must autonomously navigate to find and reach a target object (the "flag") using computer vision and AI decision-making, similar to how autonomous vehicles navigate to destinations.

**Goal:** Navigate to and "capture" a target object (e.g., chair, ball, person) using only camera vision and AI/rule-based decision making.

---

## System Architecture

The navigation system consists of **3 main components** that work together to achieve autonomous navigation:

### 1. **Navigation** (`navigate_with_yolo()`)
**Purpose:** Main navigation system toward detected target objects

**Two Operating Modes:**

#### Rule-Based Navigation (Default)
Hardcoded logic based on object detection:
- ✅ Target centered → move forward
- ✅ Target left → turn left  
- ✅ Target right → turn right
- ✅ Target too close (large area) → stop (FLAG CAPTURED!)
- ✅ Blocker detected → avoid right

#### AI-Based Navigation (`ai_decide=True`)
Uses Ollama LLM for intelligent decision-making:
- 🧠 Sends annotated image to AI with context
- 🧠 AI analyzes: target position, obstacles, history
- 🧠 AI decides: forward/left/right/stop
- 🧠 Uses decision history for context-aware navigation
- 🧠 Function: `ai_navigation_decision()`

**Location:** `utils/navigation_utils.py` lines 645-855

---

### 2. **Smart Search** (Target Not Visible)
**Purpose:** Scan environment to locate the target when it's out of view

**Two Implementations:**

#### Active: `smart_search_for_target()` (Rule-Based Camera Scanning)
When target disappears from view:
1. 📷 Rotate camera left → capture → detect objects
2. 📷 Rotate camera right → capture → detect objects  
3. 🎯 Compare detections on both sides
4. 🔄 Robot turns toward side where target was found
5. 🔁 Return camera to center, continue navigation

**Location:** `utils/navigation_utils.py` lines 485-642

#### Disabled: `ai_search_decision()` (AI-Based Search - DEAD CODE)
Alternative AI-powered search approach:
- 📸 Captures 4 camera angles (left, center-left, center-right, right)
- 🖼️ Combines images into 2x2 grid
- 🧠 Asks Ollama AI which direction to search
- ⚠️ Currently disabled in `fcam.py` (line 669: `if ai_decide or False:`)

**Location:** `utils/navigation_utils.py` lines 166-350

---

### 3. **Vision-Based Stuck Recovery** 
**Purpose:** Detect and escape when robot gets stuck (obstacle collision, wall, etc.)

**How It Works:**
1. 🔍 **Stuck Detection:** Compares consecutive frames
   - If frames are >97% similar for 3+ frames → robot is stuck
   
2. 📷 **Environment Scanning:** Captures 3 views
   - Left angle
   - Center angle  
   - Right angle

3. 🎯 **Escape Direction Selection:** Chooses direction with most open space
   - Analyzes object count and sizes in each view
   - Selects direction with fewest/smallest obstacles

4. 🚗 **Evasive Maneuver:** Executes escape sequence
   - Move backward (clear immediate obstacle)
   - Turn toward open direction
   - Resume normal navigation

**Location:** `utils/navigation_utils.py` lines 352-483

---

## Call Hierarchy

```
run_navigation_loop() [fcam.py]
│
├─→ is_robot_stuck()                                   # Frame comparison
│   └─→ vision_based_stuck_recovery()                  # If stuck detected
│       ├─→ Capture 3 angles
│       ├─→ Analyze each direction
│       └─→ Execute backward + turn
│
├─→ detect_objects_yolo()                              # YOLO detection
│   └─→ Returns: objects list + annotated image
│
├─→ navigate_with_yolo()                               # Main navigation
│   │
│   ├─→ AI Mode (ai_decide=True):
│   │   └─→ ai_navigation_decision()                   # Ollama decision
│   │       ├─→ Send image + context to AI
│   │       ├─→ Parse AI response
│   │       └─→ Return: forward/left/right/stop
│   │
│   └─→ Rule-Based Mode (ai_decide=False):
│       └─→ Hardcoded logic (position-based turns)
│
└─→ smart_search_for_target()                          # If target not found
    │
    ├─→ Active: Rule-based camera scan
    │   ├─→ Rotate camera left + detect
    │   ├─→ Rotate camera right + detect
    │   └─→ Turn robot toward target side
    │
    └─→ DEAD CODE: ai_search_decision()                # Disabled
        ├─→ Capture 4-angle grid
        └─→ Ask AI for search direction
```

---

## Capture The Flag Workflow

### Phase 1: Target Acquisition
1. 📷 **Vision:** Capture camera frame
2. 🎯 **Detection:** YOLO identifies objects in view
3. 🔍 **Search:** If target not found → `smart_search_for_target()`

### Phase 2: Navigation
1. 🧠 **Decision:** AI or rule-based decides movement
2. 🚗 **Execution:** Send command to robot (forward/left/right)
3. 🚧 **Obstacle Avoidance:** Avoid blockers while navigating

### Phase 3: Stuck Recovery
1. 🔍 **Detection:** Monitor frame similarity
2. 🚨 **Trigger:** 3+ identical frames → stuck
3. 🔄 **Recovery:** `vision_based_stuck_recovery()` escapes
4. ▶️ **Resume:** Return to navigation

### Phase 4: Flag Capture
1. 🎯 **Approach:** Target centered, moving forward
2. 📏 **Distance Check:** Target area > 30% of frame
3. 🏁 **SUCCESS:** Stop robot → FLAG CAPTURED!

---

## Configuration

Edit `fcam.py` to configure CTF parameters:

```python
if __name__ == '__main__':
    use_ollama = False       # Deprecated full Ollama mode
    ai_decide = True         # Enable AI decision-making
    target = 'chair'         # The "flag" to capture
    use_segmentation = True  # Better object detection
    capture_video = True     # Record navigation session
    
    main(use_ollama, ai_decide, target, use_segmentation, capture_video)
```

### Supported "Flags" (COCO Classes)
- `chair`, `person`, `cup`, `bottle`, `ball`, `car`, `dog`, `cat`, etc.

---

## Self-Driving Vehicle Parallels

This CTF challenge mirrors real autonomous vehicle problems:

| Robot Challenge | Autonomous Vehicle Equivalent |
|----------------|------------------------------|
| Find target object | Navigate to GPS destination |
| Obstacle avoidance | Pedestrian/vehicle detection |
| Stuck recovery | Handle deadlock situations |
| Smart search | Re-routing when lost |
| AI decision-making | Path planning algorithms |
| Vision-based navigation | Camera-based SLAM |

---

## Video Recording

All navigation sessions are recorded with embedded logs showing:
- 🎯 Object detections
- 🧠 AI decisions (if enabled)
- 🚗 Movement commands
- 🔍 Search actions
- 🚨 Stuck recovery events

Videos saved to: `videos/robot_video_YYYYMMDD_HHMMSS.mp4`

---

## Technical Stack

- **Vision:** YOLO v8/v11 (detection + segmentation)
- **AI:** Ollama + Gemma3:4b (local LLM)
- **Control:** Socket-based robot communication
- **Recording:** OpenCV video writer with split-screen layout
- **Visualization:** Matplotlib real-time path tracking

---

## Success Metrics

- ✅ **Flag Captured:** Robot reaches target and stops
- 📏 **Distance:** Minimize path length to target
- ⏱️ **Time:** Minimize time to capture
- 🚧 **Collisions:** Zero collisions (stuck events)
- 🧠 **AI Decisions:** Quality of AI reasoning (if enabled)
