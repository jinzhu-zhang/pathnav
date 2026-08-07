import threading

import cv2
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO

STREAM_URL = "http://192.168.3.231:81/stream"   # <-- ESP32 IP; update if it changed
CONF = 0.40                                       # confidence threshold
MODEL = "yolov8n.pt"                              # nano = right size for the Pi 5
VIEW_PORT = 5000                                  # open http://visionpi.local:5000 on your Mac

model = YOLO(MODEL)                                # downloads automatically on first run

cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    raise SystemExit(
        "Could not open stream. Check the ESP32 IP and that the Pi is on the same WiFi."
    )

print("Stream opened. Detection running on the Pi.", flush=True)

# Latest annotated frame, shared between the detection thread and the web viewer.
_latest_jpeg = None
_lock = threading.Lock()


def detection_loop():
    global _latest_jpeg
    while True:
        ok, frame = cap.read()
        if not ok:
            print("...dropped frame, retrying", flush=True)
            continue

        results = model(frame, conf=CONF, imgsz=640, verbose=False)
        annotated = results[0].plot()             # draws boxes + labels

        labels = [model.names[int(b.cls[0])] for b in results[0].boxes]
        if labels:
            print("Detected:", ", ".join(sorted(set(labels))), flush=True)

        ok_enc, buf = cv2.imencode(".jpg", annotated)
        if ok_enc:
            with _lock:
                _latest_jpeg = buf.tobytes()


app = Flask(__name__)


@app.route("/")
def index():
    return '<h1>Vision Pi</h1><img src="/stream" style="max-width:100%">'


def _mjpeg():
    while True:
        with _lock:
            frame = _latest_jpeg
        if frame is None:
            continue
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.route("/stream")
def stream():
    return Response(_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    threading.Thread(target=detection_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=VIEW_PORT, threaded=True)
