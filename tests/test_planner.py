#!/usr/bin/env python3
"""Checks for the metric planner: geometry, avoidance, fork detection, timing.

Run after touching planner.py or the ground-plane model:
    python3 tests/test_planner.py
Exits non-zero if anything fails.
"""
import math
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import path_nav as P
import planner as PL

FAILED = []


def check(label, cond, detail=""):
    if not cond:
        FAILED.append(label)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail}")


def open_ground_mask(ground):
    """Everything below the horizon is walkable."""
    m = np.zeros((P.PROC_H, P.PROC_W), dtype=np.uint8)
    hy = int((ground.horizon_ny() + 0.5) * P.PROC_H)
    m[hy + 6:, :] = 255
    return m, hy


def obstacle_at(X, Z, w=0.6):
    """A ranged object shaped like what enrich_objects produces."""
    d = math.hypot(X, Z)
    return {"name": "person", "kind": "static", "moving": False,
            "distance_m": d, "bearing_deg": math.degrees(math.atan2(X, Z)),
            "size_m": (w, 1.7), "box": (0, 0, 1, 1), "cx": 0, "cy": 0}


def ground_strip(ground, pl, xs_by_z):
    """Paint a walkable mask from metric spans: xs_by_z(Z) -> list of (X0, X1)."""
    m = np.zeros((P.PROC_H, P.PROC_W), dtype=np.uint8)
    for zi in range(5, 90):
        Z = zi * 0.1
        for X0, X1 in xs_by_z(Z):
            n = 60
            for j in range(n + 1):
                X = X0 + (X1 - X0) * j / n
                p = ground.project(X, Z)
                if p is None:
                    continue
                x = int(round((p[0] + 0.5) * P.PROC_W))
                y = int(round((p[1] + 0.5) * P.PROC_H))
                if 0 <= x < P.PROC_W and 0 <= y < P.PROC_H:
                    cv2.circle(m, (x, y), 3, 255, -1)
    return m


def main():
    print("1. GroundPlane.project() inverts point()")
    g = P.make_ground(2160, 3840)
    for X, Z in [(0.0, 3.0), (-1.2, 2.0), (2.0, 6.0), (0.5, 1.0)]:
        nx, ny = g.project(X, Z)
        bx, bz = g.xz(nx, ny)
        err = math.hypot(bx - X, bz - Z)
        check(f"({X:+.1f},{Z:.1f}) m round trip", err < 1e-3, f"err={err:.1e}")

    print("\n2. Bird's-eye homography")
    pl = PL.BevPlanner(g, P.PROC_W, P.PROC_H)
    for X, Z in [(0.0, 2.0), (-1.5, 4.0), (1.0, 6.5)]:
        nx, ny = g.project(X, Z)
        px = np.array([[[(nx + .5) * P.PROC_W, (ny + .5) * P.PROC_H]]], dtype=np.float32)
        cell = cv2.perspectiveTransform(px, pl.H).reshape(2)
        want = PL.BevPlanner._to_cell(X, Z)
        err = math.hypot(cell[0] - want[0], cell[1] - want[1])
        check(f"({X:+.1f},{Z:.1f}) m -> correct cell", err < 0.5, f"err={err:.3f}")
    check("observed cells are a sane fraction", 0.3 < pl.valid.mean() < 0.95,
          f"valid={pl.valid.mean():.2f}")

    print("\n3. Obstacle avoidance")
    mask, hy = open_ground_mask(g)
    pl.reset()
    free = pl.plan(mask, [])
    check("open ground -> valid plan", free.valid)
    check("open ground -> straight", abs(free.offset_deg) < 4.0,
          f"offset={free.offset_deg:+.1f}")

    pl.reset()
    mid = pl.plan(mask, [obstacle_at(0.0, 3.0)])
    check("obstacle dead ahead -> still finds a way", mid.valid)
    check("obstacle dead ahead -> leaves the centre", abs(mid.offset_deg) > 6.0,
          f"offset={mid.offset_deg:+.1f}")
    pl.reset()
    lb = pl.plan(mask, [obstacle_at(-0.9, 3.0), obstacle_at(0.0, 3.5)])
    check("blocked ahead+left -> steers right", lb.offset_deg > 5.0,
          f"offset={lb.offset_deg:+.1f}")
    pl.reset()
    rb = pl.plan(mask, [obstacle_at(0.9, 3.0), obstacle_at(0.0, 3.5)])
    check("blocked ahead+right -> steers left", rb.offset_deg < -5.0,
          f"offset={rb.offset_deg:+.1f}")
    pl.reset()
    wall = pl.plan(mask, [obstacle_at(x / 10.0, 2.5) for x in range(-25, 26, 4)])
    check("full wall at 2.5 m -> no usable plan",
          (not wall.valid) or wall.reach_z_m < 2.6,
          f"valid={wall.valid} reach={wall.reach_z_m:.1f}m")

    print("\n4. Centring and the no-path case")
    # A 1.6 m corridor bending away at a rate the steering fan can actually follow
    # (it starts under your feet -- a corridor that is already offset at Z=0 is not
    # reachable by any arc, since every arc begins at X=0).
    def bend(rate):
        return lambda Z: [(rate * max(0.0, Z - 0.8) - 0.8,
                           rate * max(0.0, Z - 0.8) + 0.8)]

    pl.reset()
    corr = ground_strip(g, pl, bend(0.30))
    rc = pl.plan(corr, [])
    check("corridor bending right -> valid", rc.valid, f"reach={rc.reach_z_m:.1f}m")
    check("corridor bending right -> steers right", rc.offset_deg > 5.0,
          f"offset={rc.offset_deg:+.1f}")
    pl.reset()
    lc = pl.plan(ground_strip(g, pl, bend(-0.30)), [])
    check("corridor bending left -> steers left", lc.valid and lc.offset_deg < -5.0,
          f"offset={lc.offset_deg:+.1f}")
    # Standing on a mask hole must not report "no path".
    pl.reset()
    holed = mask.copy()
    holed[-40:, P.PROC_W // 2 - 25:P.PROC_W // 2 + 25] = 0
    check("hole underfoot -> still plans", pl.plan(holed, []).valid)

    print("\n5. Fork detection (metric)")
    pl.reset()
    check("open ground -> no fork", not pl.plan(mask, []).fork)
    pl.reset()
    check("single corridor -> no fork", not pl.plan(corr, []).fork)
    # A real fork: one corridor that divides at 2.5 m into two branches, each wide
    # enough to walk, separated by a wedge that widens with depth the way a real
    # island of vegetation between two paths does.
    def split(Z):
        if Z < 2.5:
            return [(-1.5, 1.5)]
        half = min(1.3, 0.15 + 0.45 * (Z - 2.5))     # inner edges, widening wedge
        outer = 1.5 + 0.35 * (Z - 2.5)               # outer edges diverge too
        return [(-outer, -half), (half, outer)]

    forked = ground_strip(g, pl, split)
    pl.reset()
    fr = pl.plan(forked, [])
    check("two branches split by a solid wedge -> FORK", fr.fork,
          f"valid={fr.valid} reach={fr.reach_z_m:.1f}m")
    # A narrow nick in one corridor is not a fork.
    nick = corr.copy()
    nick[hy + 40:hy + 48, :] = np.where(nick[hy + 40:hy + 48, :] > 0, 0,
                                        nick[hy + 40:hy + 48, :])
    pl.reset()
    check("thin horizontal nick -> no fork", not pl.plan(nick, []).fork)

    print("\n6. Timing on real frames")
    cap = cv2.VideoCapture("test_videos/IMG_7205.mov")
    frames = []
    for _ in range(4):
        ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(f, (P.PROC_W, P.PROC_H)))
    cap.release()
    if frames:
        seg = P.WalkableSegmenter()
        seg.reset()
        masks = [seg(f)[0] for f in frames]
        pl.plan(masks[0], [])
        t0 = time.perf_counter()
        N = 60
        for i in range(N):
            pl.plan(masks[i % len(masks)], [obstacle_at(0.4, 3.0)])
        ms = (time.perf_counter() - t0) * 1000.0 / N
        print(f"  plan() on real masks: {ms:.2f} ms/frame")
        check("planning stays under 15 ms", ms < 15.0, f"{ms:.2f} ms")
        r = pl.plan(masks[0], [])
        print(f"  real frame: valid={r.valid} offset={r.offset_deg:+.1f} "
              f"reach={r.reach_z_m:.1f}m conf={r.confidence:.2f} fork={r.fork}")

    print("\nALL PASS" if not FAILED else f"\n{len(FAILED)} FAILURE(S): {FAILED}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
