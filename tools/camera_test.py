#!/usr/bin/env python3
"""Benchmark a camera source: negotiated resolution, throughput (FPS), and
per-frame read latency. Headless-friendly (no GUI window): it prints stats and
saves one sample frame to disk so you can eyeball quality over scp.

Usage:
    python3 camera_test.py                       # /dev/video0 @ 640x480 MJPG
    python3 camera_test.py 0 1280 720            # device index 0 @ 1280x720
    python3 camera_test.py 0 320 240             # device index 0 @ 320x240
    python3 camera_test.py http://HOST:81/stream # a network MJPEG stream

Notes:
- For the ESP32-S3 UVC cam we force the MJPG pixel format; it only offers MJPG,
  and forcing it avoids OpenCV falling back to a slow/raw mode.
- "read latency" here is the time cap.read() blocks; for a steady stream that's
  essentially the inter-frame interval (~1000/FPS), so treat it as a throughput
  proxy and a stall detector, not true glass-to-glass latency.
"""
import statistics
import sys
import time

import cv2

N_WARMUP = 10
N_FRAMES = 150


def parse_args(argv):
    src = argv[1] if len(argv) > 1 else "0"
    width = int(argv[2]) if len(argv) > 2 else 640
    height = int(argv[3]) if len(argv) > 3 else 480
    if src.isdigit():
        src = int(src)
    return src, width, height


def main():
    source, width, height = parse_args(sys.argv)
    is_device = isinstance(source, int)

    cap = cv2.VideoCapture(source, cv2.CAP_V4L2) if is_device else cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: could not open source {source!r}")
        sys.exit(1)

    if is_device:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize stale-frame backlog

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source:                {source!r}")
    print(f"Requested resolution:  {width}x{height}")
    print(f"Negotiated resolution: {actual_w}x{actual_h}")

    for _ in range(N_WARMUP):
        cap.read()

    latencies = []
    frames_ok = 0
    last_frame = None
    t_start = time.time()
    for _ in range(N_FRAMES):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            print("...dropped frame")
            continue
        latencies.append((t1 - t0) * 1000.0)
        frames_ok += 1
        last_frame = frame
    t_total = time.time() - t_start
    cap.release()

    if frames_ok == 0:
        print("ERROR: no frames captured")
        sys.exit(1)

    fps = frames_ok / t_total
    print()
    print(f"Frames captured:  {frames_ok}/{N_FRAMES}")
    print(f"Wall time:        {t_total:.2f}s")
    print(f"Throughput:       {fps:.1f} FPS")
    print(
        "Read latency (ms): "
        f"avg {statistics.mean(latencies):.1f}, "
        f"min {min(latencies):.1f}, "
        f"max {max(latencies):.1f}, "
        f"median {statistics.median(latencies):.1f}"
    )

    out = "camera_test_frame.jpg"
    cv2.imwrite(out, last_frame)
    print(f"\nSaved a sample frame to {out} (scp it to your Mac to check quality).")


if __name__ == "__main__":
    main()
