#!/usr/bin/env python3
"""Summarise the per-frame JSONL exports to check guidance quality.

Reads test_videos_annotated/*.jsonl (written by render_test_videos.py) and reports
the things that are hard to judge by watching a clip: how often each cue is
actually announced, how large the raw steering angles are versus the STRAIGHT
deadband, how often a fork is claimed and in what length of episode, and whether
the planner is reaching its full lookahead or being walled off early.

Usage (from repo root):
    python3 tools/analyze_logs.py [DIR]        # default: test_videos_annotated
"""
import glob
import json
import os
import sys

import numpy as np

import _bootstrap  # noqa: F401
import path_nav as P
import planner as PL

# The line's heading at the lookahead is what the announced bearing is derived from,
# so this is the only comparison between arrow and line that is apples to apples.
PLAN_LOOKAHEAD_M = PL.LOOKAHEAD_S_M


def load(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        rows = [json.loads(l) for l in open(p)]
        if rows:
            out.append((os.path.basename(p).replace(".jsonl", ""), rows))
    return out


def runs_of(flags):
    """Lengths of consecutive True runs."""
    res, cur = [], 0
    for v in flags:
        if v:
            cur += 1
        elif cur:
            res.append(cur)
            cur = 0
    if cur:
        res.append(cur)
    return res


def main(d):
    clips = load(d)
    if not clips:
        raise SystemExit(f"No .jsonl files in {d}")

    print("=" * 102)
    print("A. DIRECTIVES  -- what the user is actually told, vs the raw steering angle")
    print("=" * 102)
    print(f"{'clip':10s} {'frames':>6s} {'STRAIGHT':>9s} {'LEFT':>5s} {'RIGHT':>6s} "
          f"{'NOPATH':>7s} {'changes':>8s} | {'mean|off|':>9s} {'p90':>5s} {'max':>5s} "
          f"| {'mean|k|':>8s} {'p90|k|':>7s}")
    all_off, all_k = [], []
    for name, rows in clips:
        cues = [r["cue"] for r in rows]
        off = np.abs([r["offset_deg"] for r in rows])
        kap = np.abs([r.get("kappa", 0.0) for r in rows])
        all_off.append(off)
        all_k.append(kap)
        print(f"{name:10s} {len(rows):6d} {cues.count('STRAIGHT')/len(rows)*100:8.0f}% "
              f"{cues.count('BEAR LEFT'):5d} {cues.count('BEAR RIGHT'):6d} "
              f"{cues.count('NO PATH'):7d} "
              f"{sum(1 for a,b in zip(cues, cues[1:]) if a != b):8d} | "
              f"{off.mean():9.1f} {np.percentile(off,90):5.1f} {off.max():5.1f} | "
              f"{kap.mean():8.3f} {np.percentile(kap,90):7.3f}")

    off = np.concatenate(all_off)
    kap = np.concatenate(all_k)
    print(f"\n  all clips: mean|offset| {off.mean():.1f} deg, p90 {np.percentile(off,90):.1f}, "
          f"max {off.max():.1f}   (STRAIGHT deadband = {P.STRAIGHT_DEG:.0f} deg)")
    print(f"             {(off < P.STRAIGHT_DEG).mean()*100:.0f}% of frames fall inside "
          f"the deadband and are announced STRAIGHT")
    print(f"             mean|kappa| {kap.mean():.3f}, p90 {np.percentile(kap,90):.3f}, "
          f"max {kap.max():.3f}   (arc fan limit = {0.30:.2f})")

    print("\n" + "=" * 102)
    print("B. DOES THE ARROW AGREE WITH THE LINE?  The announced bearing and the drawn")
    print("   line must never point opposite ways -- a directive that contradicts the")
    print("   picture cannot be followed. Disagreement should be 0%.")
    print("=" * 102)
    print(f"{'clip':10s} {'opposed':>8s} {'worst lag':>10s}  {'when opposed':>12s}")
    for name, rows in clips:
        o = np.array([r["offset_deg"] for r in rows])
        pd = np.degrees(np.array([r.get("kappa", 0.0) for r in rows]) * PLAN_LOOKAHEAD_M)
        live = (np.abs(o) > 1e-6) & (np.abs(pd) > 1e-6)
        opposed = live & (np.sign(o) != np.sign(pd))
        lag = np.abs(np.abs(pd) - np.abs(o))[live]
        print(f"{name:10s} {opposed.mean()*100:7.1f}% {np.percentile(lag,90):9.1f}d  "
              f"{'; '.join(f'{rows[i]['t_sec']:.1f}s' for i in np.flatnonzero(opposed)[:4])}")

    print("\n" + "=" * 102)
    print("E. IS THE TEXT READABLE?  Warnings are replayed from the logged object flags")
    print("   through the same latch the renderer uses. An episode shorter than ~1.5 s")
    print("   is a flicker the user cannot read.")
    print("=" * 102)
    print(f"{'clip':10s} {'on-screen':>9s} {'episodes':>9s} {'median':>7s} {'shortest':>8s} "
          f"{'<1.5s':>6s} {'edits/10s':>9s}")
    for name, rows in clips:
        fps = 1.0 / max(1e-3, np.median(np.diff([r["t_sec"] for r in rows])))
        ann = P.ObstacleAnnouncer(fps)
        texts = [P._warnings_text(r["fork"], r["crossing"], ann.update(r["objects"]))
                 for r in rows]
        shown = [bool(t) for t in texts]
        eps = [n / fps for n in runs_of(shown)]
        edits = sum(1 for a, b in zip(texts, texts[1:]) if a != b)
        dur = rows[-1]["t_sec"] - rows[0]["t_sec"]
        print(f"{name:10s} {np.mean(shown)*100:8.0f}% {len(eps):9d} "
              f"{np.median(eps) if eps else 0:6.1f}s {min(eps) if eps else 0:7.1f}s "
              f"{sum(1 for e in eps if e < 1.5):6d} {edits/max(1e-6, dur)*10:9.1f}")

    print("\n" + "=" * 102)
    print("C. FORKS -- a real fork is a sustained episode, not a 1-2 frame flicker")
    print("=" * 102)
    for name, rows in clips:
        r = runs_of([bool(x["fork"]) for x in rows])
        frac = np.mean([bool(x["fork"]) for x in rows]) * 100
        print(f"{name:10s} fork on {frac:5.1f}% of frames, {len(r):3d} episodes, "
              f"median {int(np.median(r)) if r else 0:3d} frames, "
              f"longest {max(r) if r else 0:3d}, "
              f"{sum(1 for x in r if x <= 2):3d} episodes <=2 frames")

    print("\n" + "=" * 102)
    print("D. PLANNER HEALTH -- low reach means the costmap saw a wall early")
    print("=" * 102)
    for name, rows in clips:
        rc = np.array([x.get("reach_m", 0.0) for x in rows])
        cf = np.array([x.get("confidence", 0.0) for x in rows])
        pl = np.array([bool(x.get("planned")) for x in rows])
        print(f"{name:10s} planned {pl.mean()*100:3.0f}%  reach mean {rc.mean():4.1f}m "
              f"p10 {np.percentile(rc,10):4.1f}m  at-full-reach "
              f"{(rc > 5.7).mean()*100:3.0f}%  conf mean {cf.mean():.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_videos_annotated")
