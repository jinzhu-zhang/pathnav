#!/usr/bin/env python3
"""Batch-render every clip in test_videos/ through the ground-plane nav pipeline.

Writes an annotated .mp4 and a per-frame .jsonl (metric distances, sizes, bearings,
collision flags) for each input into test_videos_annotated/. Designed to run
unattended in a tmux/background session.

With --bev each annotated frame gets the bird's-eye costmap the planner actually
steered by, side by side with the camera view. --stills additionally drops PNGs of
just the frames where an obstacle is on the path, so those moments can be reviewed
without scrubbing an hour of footage.

Usage (from repo root):
    python3 tools/render_test_videos.py [--stride N] [--bev] [--stills]
                                        [--only IMG_7198.mov ...]
"""
import glob
import os
import shutil
import sys
import time

import _bootstrap  # noqa: F401  — chdir to repo root, fix imports
import path_nav as P

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_DIR = os.path.join(HERE, "test_videos")
OUT_DIR = os.path.join(HERE, "test_videos_annotated")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    args = sys.argv[1:]
    stride = 2
    if "--stride" in args:
        i = args.index("--stride")
        stride = max(1, int(args[i + 1]))
        args = args[:i] + args[i + 2:]
    show_bev = "--bev" in args
    stills = "--stills" in args
    args = [a for a in args if a not in ("--bev", "--stills")]
    only = set(a for a in args if a != "--only")
    stills_dir = os.path.join(OUT_DIR, "stills") if stills else None

    vids = sorted(glob.glob(os.path.join(IN_DIR, "*.mov"))
                  + glob.glob(os.path.join(IN_DIR, "*.MOV"))
                  + glob.glob(os.path.join(IN_DIR, "*.mp4")))
    if only:
        vids = [v for v in vids if os.path.basename(v) in only]
    if not vids:
        raise SystemExit(f"No videos found in {IN_DIR}")

    print(f"Ground-plane render: height={P.CAM_HEIGHT_M:.3f} m, tilt={P.CAM_TILT_DEG} deg, "
          f"fov_long={P.CAM_FOV_LONG_DEG} deg. {len(vids)} clip(s), stride={stride}.",
          flush=True)
    print(f"Planner: {'BEV costmap + arcs' if P.USE_BEV_PLANNER else 'mask centre-line'}"
          f", bev panel={show_bev}, stills={'yes' if stills else 'no'}.", flush=True)

    segmenter = P.WalkableSegmenter()
    detector = P.load_detector()

    t_all = time.time()
    for n, vpath in enumerate(vids, 1):
        name = os.path.basename(vpath)
        stem = os.path.splitext(name)[0]
        print(f"\n=== [{n}/{len(vids)}] {name} ===", flush=True)
        t0 = time.time()
        export = os.path.join(OUT_DIR, stem + ".jsonl")
        P.run_video(segmenter, detector, vpath, stride, export_path=export,
                    show_bev=show_bev, stills_dir=stills_dir)
        # run_video writes annotated_<stem>.mp4/.avi in cwd; move into OUT_DIR
        moved = None
        for ext in (".mp4", ".avi"):
            src = "annotated_" + stem + ext
            if os.path.exists(src):
                dst = os.path.join(OUT_DIR, src)
                shutil.move(src, dst)
                moved = dst
                break
        print(f"  -> {moved}  (+ {os.path.basename(export)})  [{time.time()-t0:.0f}s]",
              flush=True)

    print(f"\nAll done in {time.time()-t_all:.0f}s. Outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
