#!/usr/bin/env python3
"""Run the path_nav pipeline on files in samples/ (if present).

Usage (from repo root):
    python3 tools/run_samples.py [--stride N] [filename ...]
"""
import glob
import os
import shutil
import sys
import time

import _bootstrap  # noqa: F401
import path_nav as P

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(HERE, "samples_annotated")
SAMPLES = os.path.join(HERE, "samples")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    args = sys.argv[1:]
    stride = 3
    if "--stride" in args:
        i = args.index("--stride")
        stride = max(1, int(args[i + 1]))
        args = args[:i] + args[i + 2:]
    only = args

    if not os.path.isdir(SAMPLES):
        print(f"No samples/ directory at {SAMPLES}. Put test images/videos there first.")
        return 1

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.mp4", "*.mov"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(SAMPLES, pat)))
    files = sorted(files)
    if only:
        want = set(only)
        files = [f for f in files if os.path.basename(f) in want]

    if not files:
        print("No sample media found.")
        return 1

    for path in files:
        name = os.path.basename(path)
        print(f"=== {name} ===")
        t0 = time.time()
        if path.lower().endswith((".mp4", ".mov")):
            out = os.path.join(OUT_DIR, f"annotated_{os.path.splitext(name)[0]}.mp4")
            # path_nav video entry expects argv-style usage; call process helpers if present
            sys.argv = ["path_nav.py", "--video", path, "--stride", str(stride)]
            P.main()
        else:
            sys.argv = ["path_nav.py", "--image", path]
            P.main()
            src = "annotated_path.jpg"
            if os.path.isfile(src):
                dst = os.path.join(OUT_DIR, f"annotated_{os.path.splitext(name)[0]}.jpg")
                shutil.copy2(src, dst)
                print(f"  -> {dst}")
        print(f"  done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
