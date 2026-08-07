import time

import cv2
from ultralytics import YOLO

# --- Camera (ESP32-S3 UVC webcam over USB) ---
DEVICE = 0              # /dev/video0
WIDTH, HEIGHT = 640, 480
CONF = 0.40             # confidence threshold

# --- Headless output ---
# We run over SSH with no display, so instead of cv2.imshow() we periodically
# save the latest annotated frame to a file you can scp over and inspect.
SAVE_PATH = "annotated.jpg"
SAVE_EVERY = 15         # save roughly once per second at ~15 FPS

model = YOLO("yolov8n.pt")          # downloads automatically on first run

cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
if not cap.isOpened():
    print(f"Could not open /dev/video{DEVICE}. Is the ESP32 UVC camera plugged in?")
    raise SystemExit
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always grab the freshest frame

print("Camera opened. Running YOLO (Ctrl+C to quit).")
frame_idx = 0
t_last = time.time()
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("...dropped frame, retrying")
            continue

        results = model(frame, conf=CONF, verbose=False)
        annotated = results[0].plot()          # draws boxes + labels

        labels = [model.names[int(b.cls[0])] for b in results[0].boxes]
        if labels:
            print("Detected:", ", ".join(sorted(set(labels))))

        frame_idx += 1
        if frame_idx % SAVE_EVERY == 0:
            now = time.time()
            fps = SAVE_EVERY / (now - t_last)
            t_last = now
            cv2.imwrite(SAVE_PATH, annotated)
            print(f"[{fps:.1f} FPS] saved {SAVE_PATH}")
except KeyboardInterrupt:
    print("\nStopping.")
finally:
    cap.release()
