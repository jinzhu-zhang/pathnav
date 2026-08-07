#!/usr/bin/env python3
"""Compare RAW detector output against the warning the user actually hears.

The hold in StickySignal is an amplifier: a scattered raw signal becomes a warning
that covers most of a clip. This runs the real pipeline over a window of a clip,
records the raw fork/opening per frame, and reports what the latch makes of it -- so
a change to the rise/hold timings can be judged on both counts at once, on real
footage, without re-rendering a whole video.

  ./.venv/bin/python tools/check_warnings.py IMG_7198.mov --start 64 --frames 150
"""
import sys

import cv2
import numpy as np

import _bootstrap  # noqa: F401
import path_nav as P


def episodes(flags, fps):
    out, cur = [], 0
    for f in flags:
        if f:
            cur += 1
        elif cur:
            out.append(cur / fps)
            cur = 0
    if cur:
        out.append(cur / fps)
    return out


def run(clip, start, n_frames, stride=2):
    cap = cv2.VideoCapture(clip)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    fps = src_fps / stride

    ground = P.make_ground(ow, oh)
    planner = P.make_planner(ground)
    horizon = P.make_horizon(ground)
    seg = P.WalkableSegmenter()
    seg.reset()
    raw_f, raw_c, tilts = [], [], []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        for _ in range(stride - 1):
            cap.read()
        proc = cv2.resize(frame, (P.PROC_W, P.PROC_H))
        if horizon is not None and horizon.update(proc):
            ground.set_tilt(horizon.tilt_deg)
            planner.refresh_geometry()
        seg.set_horizon(ground.horizon_ny(), P.PROC_H)
        r = planner.plan(seg(proc)[0], [])
        raw_f.append(bool(r.fork))
        raw_c.append(bool(r.opening))
        tilts.append(horizon.tilt_deg if horizon else P.CAM_TILT_DEG)
    cap.release()
    return {"fps": fps, "raw_fork": raw_f, "raw_open": raw_c,
            "tilt": (min(tilts), max(tilts)) if tilts else (0, 0)}


def latch(raw, fps, rise_s):
    """Replay a recorded raw sequence through the warning latch."""
    sig = P.StickySignal(fps, rise_s=rise_s)
    return [sig.update(v) for v in raw]


def main(argv):
    clips, start, n = [], 0, 150
    args = argv[1:]
    while args:
        a = args.pop(0)
        if a == "--start":
            start = int(args.pop(0))
        elif a == "--frames":
            n = int(args.pop(0))
        else:
            clips.append(a if "/" in a else f"test_videos/{a}")
    if not clips:
        raise SystemExit(__doc__)
    rises = [0.0, 0.2, 0.4, 0.6, 0.9]
    print(f"hold {P.WARN_HOLD_S}s  clear {P.WARN_CLEAR_S}s   "
          f"(each cell: % of window warned / number of episodes)")
    for c in clips:
        r = run(c, start, n)
        print(f"\n{c.split('/')[-1]}  frames {start}..{start+n*2}  "
              f"tilt {r['tilt'][0]:.1f}-{r['tilt'][1]:.1f} deg  at {r['fps']:.0f} fps")
        for name, raw in (("fork", r["raw_fork"]), ("opening", r["raw_open"])):
            cells = []
            for rs in rises:
                held = latch(raw, r["fps"], rs)
                eps = episodes(held, r["fps"])
                cells.append(f"{100*sum(held)/max(1,len(held)):3.0f}%/{len(eps)}")
            print(f"  {name:8s} raw {100*sum(raw)/max(1,len(raw)):3.0f}%   "
                  + "   ".join(f"rise {rs:.1f}s: {c2:>7s}" for rs, c2 in zip(rises, cells)))


if __name__ == "__main__":
    main(sys.argv)
