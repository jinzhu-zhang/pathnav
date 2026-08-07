#!/usr/bin/env python3
"""Where the per-frame time goes, and what it would take to reach real time.

Renders currently run at roughly one frame per second, which is fine for producing
annotated clips offline but useless for guiding somebody. This measures each stage
separately at several input resolutions so the trade is explicit: both networks are
fed far more pixels than the 320x240 the guidance actually reasons over, and that is
the first place to look before reaching for an accelerator.

  ./.venv/bin/python tools/bench_speed.py [--reps 5]
"""
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

import _bootstrap  # noqa: F401
import path_nav as P

WALK_SIZES = [512, 384, 320, 256]
YOLO_SIZES = [640, 480, 384, 320]


def _session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(path, sess_options=opts,
                               providers=["CPUExecutionProvider"])


def _time(fn, reps):
    fn()                                   # warm up (allocations, thread pool)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / reps


def bench_net(path, sizes, reps, dynamic_name):
    """Time a square-input net at each size, resizing the input tensor only.

    A fixed-shape ONNX export will refuse anything but its own size; report that
    rather than pretending, since it tells you a re-export is the required step.
    """
    sess = _session(path)
    inp = sess.get_inputs()[0]
    out = []
    for s in sizes:
        x = np.random.rand(1, 3, s, s).astype(np.float32)
        try:
            ms = _time(lambda: sess.run(None, {inp.name: x}), reps)
            out.append((s, ms))
        except Exception:
            out.append((s, None))
    print(f"  {dynamic_name} input shape {inp.shape}")
    return out


def main(argv):
    reps = 5
    if "--reps" in argv:
        reps = int(argv[argv.index("--reps") + 1])

    frame = np.random.randint(0, 255, (P.PROC_H, P.PROC_W, 3), dtype=np.uint8)
    print(f"Raspberry Pi 5, 4 threads, {reps} reps per measurement\n")

    print("SegFormer-B0 (semantic segmentation)")
    seg = bench_net(P.MODEL_PATH, WALK_SIZES, reps, "segformer")
    for s, ms in seg:
        print(f"    {s:4d}px  {('%8.0f ms' % ms) if ms else '   fixed-shape export':>12s}")

    print("\nYOLOv8n (obstacle detection)")
    yolo = bench_net(P.YOLO_PATH, YOLO_SIZES, reps, "yolov8n")
    for s, ms in yolo:
        print(f"    {s:4d}px  {('%8.0f ms' % ms) if ms else '   fixed-shape export':>12s}")

    print("\nEverything else, per frame at 320x240")
    ground = P.make_ground(2160, 3840)
    planner = P.make_planner(ground)
    horizon = P.make_horizon(ground)
    segm = P.WalkableSegmenter()
    segm.reset()
    segm.set_horizon(ground.horizon_ny(), P.PROC_H)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    segm(frame)                                   # populate the SegFormer caches
    sf_walk, veto = segm._sf_walk, segm._sf_veto
    walk_mask = segm(frame)[0]

    stages = [
        ("horizon estimate", lambda: horizon.update(frame)),
        ("appearance mask", lambda: segm._appearance_walkable(frame, sf_walk, veto, kernel)),
        ("hole fill + morphology", lambda: segm._fill_small_holes(walk_mask)),
        ("BEV plan", lambda: planner.plan(walk_mask, [])),
    ]
    other = 0.0
    for name, fn in stages:
        ms = _time(fn, reps)
        other += ms
        print(f"    {name:24s} {ms:8.1f} ms")
    print(f"    {'---':24s} {other:8.1f} ms total")

    print("\nAchievable rate, taking the cheapest working size for each net")
    seg_ms = {s: ms for s, ms in seg if ms}
    yolo_ms = {s: ms for s, ms in yolo if ms}
    if not seg_ms or not yolo_ms:
        print("    (a net is fixed-shape; re-export at the target size to measure)")
        return
    for seg_stride in (1, 2, 4):
        for s_size in sorted(seg_ms):
            for y_size in sorted(yolo_ms):
                total = seg_ms[s_size] / seg_stride + yolo_ms[y_size] + other
                if total > 400:
                    continue
                print(f"    seg {s_size}px every {seg_stride} frame(s) + yolo {y_size}px "
                      f"-> {total:6.0f} ms/frame = {1000.0/total:4.1f} FPS")


if __name__ == "__main__":
    main(sys.argv)
