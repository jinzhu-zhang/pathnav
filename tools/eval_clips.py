#!/usr/bin/env python3
"""Quick A/B evaluation over short windows of the test clips.

Rendering all eight clips takes over an hour, which is far too slow a loop for
tuning guidance thresholds. This processes a contiguous window from each clip
(contiguous matters: the mask carries temporal state, so sampled frames would not
reproduce the real behaviour) and reports the things we are trying to fix --
fork rate under both detectors, steering magnitude and how often the announced cue
would be STRAIGHT.

Usage (from repo root):
    python3 tools/eval_clips.py [--frames N] [--start F] [CLIP ...]
"""
import os
import sys

import cv2
import numpy as np

import _bootstrap  # noqa: F401
import path_nav as P
import planner as PL
from tracking import ObjectTracker

CLIP_DIR = "test_videos"
DEFAULT_CLIPS = ["IMG_7198.mov", "IMG_7200.mov", "IMG_7201.mov",
                 "IMG_7203.mov", "IMG_7204.mov", "IMG_7205.mov"]


def evaluate(seg, det, path, start, n_frames, stride=2):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  cannot open {path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    begin = max(0, min(start, max(0, total - n_frames * stride)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, begin)

    ground = P.make_ground(ow, oh)
    pl = P.make_planner(ground)
    seg.reset()
    if pl:
        pl.reset()
    guide = P.Guidance(fps=fps / stride)
    tr = ObjectTracker()

    fork_new, fork_old, raw, cues, conf, kap, ksign = [], [], [], [], [], [], []
    i = begin
    while len(raw) < n_frames:
        ok, f = cap.read()
        if not ok:
            break
        if (i - begin) % stride:
            i += 1
            continue
        proc = cv2.resize(f, (P.PROC_W, P.PROC_H))
        walkable = seg(proc)[0]
        objs = tr.update(det(proc) if det else [], i / fps)
        P.enrich_objects(objs, walkable, ground)
        r = pl.plan(walkable, objs)
        fork_new.append(bool(r.fork))
        fork_old.append(bool(P.detect_fork(walkable)))
        raw.append(r.offset_deg if r.valid else 0.0)
        kap.append(abs(r.kappa))
        ksign.append(r.kappa)
        conf.append(r.confidence)
        cue = guide.update(r.offset_deg, r.valid, r.curve_x, r.curve_ys, r.fork)[1]
        cues.append(cue)
        i += 1
    cap.release()
    if not raw:
        return None
    raw = np.abs(raw)
    ks = np.array(ksign)
    return {
        "n": len(raw), "fork_new": np.mean(fork_new) * 100,
        "fork_old": np.mean(fork_old) * 100, "raw_mean": raw.mean(),
        "raw_p90": np.percentile(raw, 90), "kap": np.mean(kap),
        "conf": np.mean(conf),
        # Two failure modes to tell apart: k_sd near zero means the plan is pinned
        # and no longer tracking the path, while a large frame-to-frame step means
        # the winner is hopping between near-tied arcs (the "messy line").
        "k_sd": float(ks.std()), "k_step": float(np.abs(np.diff(ks)).mean()),
        "straight": 100 * np.mean([c == "STRAIGHT" for c in cues]),
        "changes": sum(1 for a, b in zip(cues, cues[1:]) if a != b),
    }


def main(argv):
    args = argv[1:]
    n_frames, start = 120, 300
    if "--frames" in args:
        i = args.index("--frames")
        n_frames = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    if "--start" in args:
        i = args.index("--start")
        start = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    clips = args or DEFAULT_CLIPS

    seg = P.WalkableSegmenter()
    det = P.load_detector()
    print(f"window: {n_frames} processed frames from source frame {start}, stride 2")
    print(f"CLEAR_PREF_M={PL.CLEAR_PREF_M}  COST_PREV_KAPPA={PL.COST_PREV_KAPPA}  "
          f"LOOKAHEAD={PL.LOOKAHEAD_S_M}m  STRAIGHT_DEG={P.STRAIGHT_DEG}  "
          f"DECISION_PERIOD_S={P.DECISION_PERIOD_S}")
    print(f"\n{'clip':12s} {'n':>4s} {'fork NEW':>9s} {'fork OLD':>9s} | "
          f"{'mean|off|':>9s} {'p90':>5s} {'mean|k|':>8s} {'k sd':>6s} {'k step':>7s} {'conf':>5s} | "
          f"{'STRAIGHT':>9s} {'changes':>8s}")
    for c in clips:
        p = os.path.join(CLIP_DIR, c)
        r = evaluate(seg, det, p, start, n_frames)
        if not r:
            continue
        print(f"{c.replace('.mov',''):12s} {r['n']:4d} {r['fork_new']:8.0f}% "
              f"{r['fork_old']:8.0f}% | {r['raw_mean']:9.1f} {r['raw_p90']:5.1f} "
              f"{r['kap']:8.3f} {r['k_sd']:6.3f} {r['k_step']:7.3f} "
              f"{r['conf']:5.2f} | {r['straight']:8.0f}% "
              f"{r['changes']:8d}", flush=True)


if __name__ == "__main__":
    main(sys.argv)
