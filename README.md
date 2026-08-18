# PathNav

Assistive walkable-path navigation for the visually impaired, running on a Raspberry Pi 5 with an ESP32-S3 camera. It finds the walkable path ahead, estimates obstacle distances in metres, and reports steering cues (`STRAIGHT` / `BEAR LEFT` / `BEAR RIGHT`) plus warnings (`FORK`, `CROSSING AHEAD`, on-path obstacles).

## Overview

A chest-mounted camera watches the ground; a Raspberry Pi 5 runs two ONNX models (SegFormer for the walkable mask, YOLO for obstacles), ranges obstacles with a monocular ground-plane model, and plans a route on a bird's-eye costmap. Output is text over SSH for now (audio/TTS is future work). Everything runs on-device

## System architecture

```
ESP32-S3 camera (USB UVC)
   -> Raspberry Pi 5, downscale 320x240
      -> SegFormer-B0 ONNX (stride 2) + HSV mask + temporal EMA = walkable mask
      -> YOLOv8n ONNX -> IoU tracker
      -> GroundPlane ranging (height + adaptive tilt) = metres / bearing / size
      -> BevPlanner (10 cm grid, 31-arc fan) = steer around obstacles
   -> steering cue + warnings (SSH text; annotated video + JSONL)
```

## Hardware

- **Raspberry Pi 5** — on-device inference + planning
- **ESP32-S3 UVC webcam** — USB (`/dev/video0`, MJPG 640×480)
- Ranging assumes camera height ≈ 1.41 m, tilt ≈ 18° (adapted online), FOV ≈ 70°

## Software architecture

| Module | Responsibility |
|--------|----------------|
| `path_nav.py` | Orchestrator: models, hybrid mask, horizon/tilt, guidance, I/O |
| `ground_plane.py` | Monocular pixel → metric distance / bearing / size |
| `planner.py` | Image→BEV warp, cost layers, arc scoring, fork/crossing |
| `tracking.py` | IoU multi-object tracker, heading, time-to-contact |


## How to run

```bash
git clone https://github.com/jinzhu-zhang/vision_project.git
cd vision_project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Export / place ONNX models first (see models/README.md), then:
python3 path_nav.py                            # /dev/video0
python3 path_nav.py http://ESP32_IP:81/stream  # network camera
python3 path_nav.py --video walk.mp4 --bev     # replay a clip + costmap panel
```


## Repository layout

```
path_nav.py  ground_plane.py  planner.py  tracking.py
tools/   export, bench, eval, batch render, log analysis
tests/   planner geometry unit tests
docs/    architecture diagram + HANDOFF design notes
models/  how to obtain ONNX weights (weights are gitignored)
media/   small demo stills
```

