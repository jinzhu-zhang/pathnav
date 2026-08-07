#!/usr/bin/env python3
"""Dump the per-arc score breakdown for a few real frames of one clip.

Answers "why did the planner pick that arc?" -- which cost term is actually
deciding, and whether the costmap is responding to the scene or to fixed geometry
(the visibility cone, the grid, the near-field fill).

Usage (from repo root):
    python3 tools/debug_frame.py [CLIP] [--start F] [--n K]
"""
import math
import os
import sys

import cv2
import numpy as np

import _bootstrap  # noqa: F401
import path_nav as P
import planner as PL


def sweep(clip, start, n, slacks):
    """How much does the straight-preference tie-break cost us on real frames?

    One segmentation pass per frame, then every candidate TIE_SLACK is applied to
    the same scores -- so the whole sweep costs one pass, not one per value.
    """
    cap = cv2.VideoCapture(clip)
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    g = P.make_ground(ow, oh)
    pl = PL.BevPlanner(g, P.PROC_W, P.PROC_H)
    seg = P.WalkableSegmenter()
    seg.reset()
    seg.set_horizon(g.horizon_ny(), P.PROC_H)

    per = {s: [] for s in slacks}
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        cap.read()                       # stride 2, as the renderer uses
        proc = cv2.resize(frame, (P.PROC_W, P.PROC_H))
        _, t = pl.explain(seg(proc)[0], [])
        sc, ka = t["scores"], t["kappa"]
        if not np.isfinite(sc).any():
            continue
        b = int(np.argmin(sc))
        for s in slacks:
            near = np.flatnonzero(sc <= sc[b] + s)
            i = int(near[np.argmin(np.abs(ka[near]))])
            per[s].append(math.degrees(ka[i] * PL.LOOKAHEAD_S_M))
    cap.release()
    print(f"\n{os.path.basename(clip)}  {len(per[slacks[0]])} frames")
    print(f"  {'slack':>6s} {'mean|off|':>9s} {'p90':>6s} {'max':>6s} "
          f"{'%|off|>6deg':>12s}")
    for s in slacks:
        a = np.abs(per[s])
        if not len(a):
            continue
        print(f"  {s:6.2f} {a.mean():9.2f} {np.percentile(a,90):6.1f} "
              f"{a.max():6.1f} {100*np.mean(a > P.STRAIGHT_DEG):11.0f}%")


def fork_probe(clip, start, n):
    """Row-by-row branch structure in the bird's-eye grid, in metres.

    Shows why a fork was or was not called: the walkable runs at each depth, the
    gap between them, and how much of that gap is known-solid rather than unseen.
    """
    cap = cv2.VideoCapture(clip)
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    g = P.make_ground(ow, oh)
    pl = PL.BevPlanner(g, P.PROC_W, P.PROC_H)
    hz = P.make_horizon(g)
    seg = P.WalkableSegmenter()
    seg.reset()
    seg.set_horizon(g.horizon_ny(), P.PROC_H)

    for f_i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        cap.read()
        proc = cv2.resize(frame, (P.PROC_W, P.PROC_H))
        if hz is not None and hz.update(proc):
            g.set_tilt(hz.tilt_deg)
            pl.refresh_geometry()
        seg.set_horizon(g.horizon_ny(), P.PROC_H)
        r = pl.plan(seg(proc)[0], [])
        walk, solid = pl._walk, pl._solid
        print(f"\n--- {os.path.basename(clip)} frame {start + f_i*2}  "
              f"fork={r.fork}  (need {PL.FORK_DEPTH_M} m of contiguous split, "
              f"branches >= {PL.FORK_BRANCH_W_M} m, wedge >= {PL.FORK_WEDGE_W_M} m)")
        print(f"    {'Z':>5s} {'visible half-width':>18s}  runs of walkable ground (m)")
        for row in range(PL.GRID_D):
            Z = PL.BEV_DEPTH_M - row * PL.BEV_CELL_M
            if not (1.5 <= Z <= PL.FORK_Z_RANGE[1] + 1.0) or row % 5:
                continue
            runs, s = [], None
            for x in range(PL.GRID_W + 1):
                on = x < PL.GRID_W and walk[row, x]
                if on and s is None:
                    s = x
                elif not on and s is not None:
                    runs.append((s, x))
                    s = None
            desc = []
            for a, b in runs:
                wm = (b - a) * PL.BEV_CELL_M
                x0 = (a - PL.GRID_W / 2) * PL.BEV_CELL_M
                desc.append(f"[{x0:+.1f}..{x0+wm:+.1f}]{'*' if wm >= PL.FORK_BRANCH_W_M else ''}")
            gaps = []
            for (a, b), (c, d) in zip(runs, runs[1:]):
                gm = (c - b) * PL.BEV_CELL_M
                gaps.append(f"gap {gm:.1f}m solid {solid[row, b:c].mean()*100:.0f}%")
            print(f"    {Z:5.1f} {Z*math.tan(math.radians(21.5))*2:18.2f}  "
                  f"{' '.join(desc) if desc else '(none)'}   {'; '.join(gaps)}")
    cap.release()


def main(argv):
    clip = "test_videos/IMG_7199.mov"
    start, n = 400, 3
    args = argv[1:]
    if args and not args[0].startswith("--"):
        clip = args[0] if "/" in args[0] else f"test_videos/{args[0]}"
        args = args[1:]
    if "--start" in args:
        start = int(args[args.index("--start") + 1])
    if "--n" in args:
        n = int(args[args.index("--n") + 1])
    save = args[args.index("--save") + 1] if "--save" in args else None
    if save:
        os.makedirs(save, exist_ok=True)
    if "--fork" in args:
        fork_probe(clip, start, n)
        return
    if "--sweep" in args:
        sweep(clip, start, n, [0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
        return

    cap = cv2.VideoCapture(clip)
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    g = P.make_ground(ow, oh)
    pl = PL.BevPlanner(g, P.PROC_W, P.PROC_H)
    hz = P.make_horizon(g)
    seg = P.WalkableSegmenter()
    seg.reset()
    seg.set_horizon(g.horizon_ny(), P.PROC_H)

    print(f"{clip}  from frame {start}")
    print(f"grid {PL.GRID_W}x{PL.GRID_D} cell {PL.BEV_CELL_M} m   "
          f"visible cells {pl.valid.mean()*100:.0f}%   "
          f"z_near_visible {pl.z_near_visible:.2f} m   "
          f"min_reach {pl.min_reach_z_m:.2f} m")

    for f_i in range(n):
        for _ in range(2):
            ok, frame = cap.read()
        if not ok:
            break
        proc = cv2.resize(frame, (P.PROC_W, P.PROC_H))
        if hz is not None and hz.update(proc):
            g.set_tilt(hz.tilt_deg)
            pl.refresh_geometry()
        seg.set_horizon(g.horizon_ny(), P.PROC_H)
        walk_mask = seg(proc)[0]
        r, t = pl.explain(walk_mask, [])

        wk, sl = pl._walk, pl._solid
        print(f"\n--- frame {start + f_i*2}  mask walkable {walk_mask.mean()/255*100:.0f}% "
              f"of image | grid: known-walkable {wk.mean()*100:.0f}%, "
              f"known-solid {sl.mean()*100:.0f}%, "
              f"unseen {(~pl.valid).mean()*100:.0f}%")
        print(f"    chosen kappa {r.kappa:+.3f}  offset {r.offset_deg:+.1f} deg  "
              f"reach {r.reach_z_m:.1f} m  conf {r.confidence:.2f}  fork {r.fork}")
        best = int(np.argmin(np.where(t["viable"], 0, 1) * 1e9
                             + t["mean_cost"] + t["progress"] + t["curvature"]
                             + t["continuity"]))
        print(f"    {'kappa':>7s} {'total':>8s} {'meancost':>9s} {'progress':>9s} "
              f"{'curv':>6s} {'contin':>7s} {'z_end':>6s} {'ok':>3s}")
        for i in range(0, PL.N_ARCS, 2):
            tot = (t["mean_cost"][i] + t["progress"][i] + t["curvature"][i]
                   + t["continuity"][i])
            star = " <=" if i == best else ""
            print(f"    {t['kappa'][i]:+7.3f} {tot:8.2f} {t['mean_cost'][i]:9.2f} "
                  f"{t['progress'][i]:9.2f} {t['curvature'][i]:6.2f} "
                  f"{t['continuity'][i]:7.2f} {t['z_end'][i]:6.2f} "
                  f"{'y' if t['viable'][i] else 'n':>3s}{star}")

        # Is the deciding signal symmetric? A constant bias means fixed geometry.
        mc = t["mean_cost"]
        left, right = mc[:PL.N_ARCS // 2].mean(), mc[PL.N_ARCS // 2 + 1:].mean()
        print(f"    mean_cost left half {left:.2f} vs right half {right:.2f} "
              f"(spread across arcs {mc.max()-mc.min():.2f})")

        if save:
            # Re-run the frame through the real pipeline so the still shows exactly
            # what the annotated video would, rather than a lookalike drawn here.
            off, cue, pts, fk, _ = P.Guidance(fps=30).update(
                r.offset_deg, r.valid, r.curve_x, r.curve_ys, r.fork)
            vis = P.draw(proc, walk_mask, np.zeros_like(walk_mask), cue, off, pts,
                         fk, False, [], aim_frac=r.aim_frac)
            out = np.hstack([proc, vis, pl.render(r, proc.shape[0])])
            name = os.path.join(save, f"dbg_f{start + f_i*2:05d}.png")
            cv2.imwrite(name, out)
            print(f"    wrote {name}")
    cap.release()


if __name__ == "__main__":
    main(sys.argv)
