# Vision Project — Conversation Handoff

> Purpose: seed a fresh Cursor chat in the **Remote-SSH window (`jinzhu@visionpi.local`)**
> with the full context from a planning conversation that happened in a separate
> (local) Cursor window. Cursor chat history is per-workspace and does not transfer
> automatically, so `@`-reference this file (or paste it) in the new chat.

## End goal
An **assistive navigation aid for visually impaired users**. The device watches the
ground ahead and tells the user how to follow a **walkable path** — e.g. an outdoor
**paved trail with irregular twists/turns**, or an **indoor hallway** (less twisty but
still irregular). It computes a virtual "guide line" down the path and reports
steering cues ("straight / bear left / bear right", plus fork detection).

- **Hardware:** Raspberry Pi 5. Camera is an **ESP32-S3 UVC webcam** — used two ways
  in existing code: USB (`/dev/video0`, MJPG 640x480) and a **network MJPEG stream**
  (`http://<esp32-ip>:81/stream`).
- **Role of the Pi:** **detect + report only** (no motors / no robot control).
- **v1 output:** **printed text over SSH** is fine for now (TTS/audio comes later).

## Key technical decisions (the "why")

1. **OS direction: Yocto (eventually), not Ubuntu.** Goal is a minimal embedded image.
   - IMPORTANT: **Yocto cannot be built natively on macOS** (needs a Linux host +
     case-sensitive FS). Build it inside a **Linux VM (UTM/VMware, Ubuntu 22.04)** or
     via **CROPS/Docker**. The build command is **`bitbake <image>`**, not `make`.
   - Pi BSP layer: **`meta-raspberrypi`**; `MACHINE = "raspberrypi5"`.
   - Keep minimal: start from `core-image-minimal`, then ADD only needed packages
     (openssh, python3, v4l-utils, OpenCV via `meta-openembedded`, etc.).

2. **Decouple the two hard problems.** Yocto is hard AND the vision pipeline is hard.
   **Prototype the vision app on the CURRENT Ubuntu Pi first** (where `pip` just works),
   get it working, THEN port the finished, dependency-frozen app to a minimal Yocto image.
   **We are now in the "build v1 vision app on Ubuntu" phase.**

3. **Drop `ultralytics` + PyTorch on-device; use ONNX.** Ditching ultralytics does NOT
   mean ditching the YOLO model — export the same weights to ONNX and run via
   **OpenCV-DNN** or **ONNX Runtime**. Detections are identical; we just lose the
   convenience wrapper (~30–40 lines of pre/post-processing to re-add). This also makes
   the app Yocto-friendly (OpenCV has a real recipe; PyTorch/ultralytics do not).

4. **Path-finding = semantic segmentation, NOT classical CV.** For irregular outdoor
   trails / hallways, color/edge/Hough approaches are too brittle. Use a small
   segmentation model that labels "walkable" pixels.
   - Recommended starting model: **SegFormer-B0** (`nvidia/segformer-b0-finetuned-ade-512-512`),
     trained on **ADE20K**, which has `floor / road / sidewalk / path / earth` classes —
     covers both indoor floor and outdoor trail. Export to **ONNX**, run via ONNX Runtime
     on the Pi 5 CPU (expect ~1–4 FPS at 256–512px; fine for walking-pace reporting).
   - Lighter alternatives if too slow: PP-LiteSeg, Fast-SCNN, DeepLabV3-MobileNetV2.

5. **Obstacles = YOLOv8n exported to ONNX.** "Obstacle ON the path" = a detection box
   whose **bottom-center pixel falls inside the path mask** (simple geometry).

## Target pipeline
```
camera frame
  ├─► SegFormer (ONNX)  ──► walkable-path pixel mask
  │        └─► per-row center of mask ──► fit curve = virtual guide line
  │                 └─► heading offset ──► text cue ("bear left/straight/right", fork)
  └─► YOLOv8n (ONNX)    ──► obstacle boxes
           └─► box bottom-center inside path mask? ──► "obstacle ahead"
```

## Existing files in the repo (already written, untracked in git)
- **`path_nav.py`** — Classical-CV walkable-path **scaffold**. Seed-based Lab-color
  segmentation + morphology + connected-components, then `analyze()` does multi-row
  centerline scan → steering cue + fork detection, and `draw()` makes an annotated
  `annotated_path.jpg`. **The guidance logic (`analyze`/`draw`) is reusable as-is** —
  v1 just replaces `segment_walkable()` with a SegFormer-ONNX mask.
- **`detect.py`** — current YOLO+ultralytics demo over USB UVC cam (`/dev/video0`).
- **`detect_pi.py`** — YOLO+ultralytics over the ESP32 **network** MJPEG stream, with a
  Flask MJPEG viewer on port 5000.
- **`camera_test.py`** — camera benchmark (negotiated resolution, FPS, read latency).
- `venv/` (python3.8) and `yolov8n.pt` are present and **should be gitignored**.

## Why ONNX export (and how it helps Yocto)
ONNX export is not *only* for Yocto — its main purpose is to **decouple the model from
PyTorch**:
- An `.onnx` file is just the frozen model graph + weights. Run it with a small runtime
  (**ONNX Runtime** or **OpenCV-DNN**) instead of the ~2GB PyTorch framework. This is
  exactly what makes Yocto feasible (PyTorch has no Yocto recipe; ONNX Runtime is small).
- It also helps immediately on Ubuntu: faster CPU inference on ARM, framework-independent,
  and a reproducible/version-pinned artifact.
- **Rule of thumb:** export on a machine that HAS PyTorch (Mac or current Ubuntu Pi),
  ONCE. The device that *runs* the model does not need PyTorch — only the light runtime.

## ONNX export steps (run once, on a PyTorch machine)
Do this on the Mac or the current Ubuntu Pi (anything with PyTorch). Easiest is the
Ubuntu Pi, so the `.onnx` lands right where v1 will run.

```bash
# 1. SegFormer-B0 (path segmentation) -> ONNX, via HuggingFace Optimum
pip install "optimum[exporters]" transformers
optimum-cli export onnx \
  --model nvidia/segformer-b0-finetuned-ade-512-512 \
  --task semantic-segmentation \
  segformer_b0_ade_onnx/            # produces segformer_b0_ade_onnx/model.onnx

# 2. YOLOv8n (obstacles) -> ONNX, via ultralytics (uses existing yolov8n.pt)
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12   # -> yolov8n.onnx
```
Notes:
- `.onnx` files are gitignored; move them via scp if you export on the Mac, or just
  export directly on the Pi.
- Verify a model loads with: `python3 -c "import onnxruntime as ort; ort.InferenceSession('segformer_b0_ade_onnx/model.onnx')"`

## Immediate next step (v1)
Build the ONNX path-segmentation prototype on the **Ubuntu Pi**:
1. Export the two models to ONNX (see "ONNX export steps" above) — at minimum
   `segformer_b0_ade_onnx/model.onnx` for v1.
2. `pip install onnxruntime numpy opencv-python` on the Pi (the light runtime — no torch).
3. New `navigate.py` (or refactor `path_nav.py`): run SegFormer via ONNX Runtime to
   produce the walkable mask (map ADE20K floor/road/sidewalk/path classes → "walkable"),
   then feed that mask into the EXISTING `analyze()` + `draw()` guidance logic.
4. Print steering cues over SSH; save `annotated_path.jpg` to inspect over scp.
5. (Later) add YOLOv8n-ONNX obstacle detection + on-path overlap test.
6. (Later) port the frozen app (`.onnx` + onnxruntime + opencv) onto a minimal Yocto image.

## Git state
- Remote: `origin = https://github.com/jinzhu-zhang/vision_project.git`, branch `main`.
- Nothing committed yet (all files untracked). No `.gitignore` (add one for `venv/`, `*.pt`, `*.onnx`, `annotated*.jpg`).
