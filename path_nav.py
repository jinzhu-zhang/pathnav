#!/usr/bin/env python3
"""Walkable-path navigation aid (SegFormer-B0 + YOLOv8n, both ONNX).

Pipeline:
  capture -> downscale
    |-> SegFormer-B0 semantic segmentation (ONNX Runtime)
    |     -> map ADE20K classes to a "walkable" mask (+ split road vs walkway)
    |     -> clean mask -> keep the region directly in front of you
    |     -> fork + "crossing/road ahead" detection
    |-> YOLOv8n object detection (ONNX Runtime, relevant COCO classes only)
    |     -> box bottom-centre inside the walkable mask => on path
    |     -> ground-plane ranging => metres, size, bearing per object
    |     -> close + on-path => urgent ("STOP") vs distant => informational
    -> planner.py: warp mask + ranged obstacles into a bird's-eye COSTMAP in
       metres, score a fan of walkable arcs, keep the cheapest
         -> guide line that steers AROUND obstacles, plus a steering cue and a
            confidence; falls back to the mask centre-line if nothing is viable
    -> temporal smoothing (steer EMA + cue hysteresis) for stable guidance
    -> save an annotated visualization (headless friendly).

Both models run via ONNX Runtime, so the on-device dependency footprint is just
onnxruntime + opencv + numpy (no PyTorch) -- portable to a minimal Yocto image.

Export the models once (on any machine with PyTorch), e.g. the Pi itself:
    python3 tools/export_segformer.py      # -> segformer_b0_ade_onnx/model.onnx
    pip install ultralytics
    yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12   # -> yolov8n.onnx

Run on the Pi (onnxruntime + apt OpenCV):
    python3 path_nav.py                          # /dev/video0 (ESP32 UVC cam)
    python3 path_nav.py http://HOST:81/stream    # ESP32 network MJPEG stream
    python3 path_nav.py --image frame.jpg        # single-image test
    python3 path_nav.py --video walk.mp4         # replay a recorded walk
    python3 path_nav.py --video walk.mp4 --stride 2   # process every 2nd frame
    python3 path_nav.py --video walk.mp4 --bev        # + bird's-eye costmap panel

Obstacle detection is optional: if yolov8n.onnx is absent the app runs
path-segmentation only (and says so once at startup).

Outputs:
  - prints steering cues + warnings (once per second live; on-change for video)
  - writes the latest annotated frame to annotated_path.jpg
  - --video also writes an annotated clip to annotated_video.mp4
"""
import json
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

from tracking import ObjectTracker
from ground_plane import GroundPlane
from planner import BevPlanner

# --- Camera ---
DEVICE = 0
CAP_W, CAP_H = 640, 480

# --- Processing ---
PROC_W, PROC_H = 320, 240        # guidance/visualization resolution
MIN_WALKABLE_FRAC = 0.04         # below this => assume the path is lost

# --- Segmentation model ---
MODEL_DIR = "segformer_b0_ade_onnx"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
INPUT_SIZE = 512                 # network input is INPUT_SIZE x INPUT_SIZE
# ADE20K labels considered "walkable ground" (matched as whole-word tokens
# against the model's id2label, e.g. "floor;flooring", "sidewalk;pavement").
# We further split them into a safe "walkway" surface vs "road" (a road ahead
# of a sidewalk implies an upcoming street crossing).
WALKWAY_NAMES = {"floor", "flooring", "sidewalk", "pavement", "earth",
                 "ground", "path"}
ROAD_NAMES = {"road", "route", "runway", "crosswalk"}
WALKABLE_NAMES = WALKWAY_NAMES | ROAD_NAMES
# Classes that are definitely NOT walkable ground and are tall/solid -- used to
# VETO appearance-based path pixels (so a grey wall/building/crate beside the
# path can't leak into the mask just because it's grey). We trust SegFormer's
# negative call here even when we don't trust its positive one.
#
# "rock" is deliberately NOT in this set, though it looks like it belongs. ADE20K uses
# it for both boulders and rocky ground, and on a gravel trail SegFormer calls the
# path itself rock -- 70% of the surface underfoot in IMG_7203 -- so vetoing it punched
# holes straight through the path wherever the gravel was coarsest. That is the mask
# "falling off" mid-clip: not flicker, but a confident label we were reading as a
# solid object. A boulder actually in the way is caught by YOLO and by the mask having
# no walkable surface there, neither of which needs this veto.
STRUCTURE_NAMES = {"wall", "building", "tree", "person", "mountain", "plant",
                   "car", "house", "fence", "column", "signboard",
                   "skyscraper", "grandstand", "hovel", "tower", "truck", "bus",
                   "railing", "bench", "van", "bicycle", "streetlight", "pole"}

# --- Appearance-based path mask (HSV) -------------------------------------
# SegFormer (ADE20K) badly mis-reads real trails: loose gravel scores as *not*
# walkable while the grass beside it scores as walkable, so the semantic mask
# gets holes / drops out / follows the grass. But a pedestrian path is almost
# always a low-saturation grey surface (asphalt, concrete, gravel) while the
# surroundings are vivid green grass, so colour separates them cleanly and
# cheaply. We build the path from appearance and only use SegFormer to add any
# path it *did* recognise and to veto structures. This runs every frame (cheap),
# even when SegFormer is strided.
GRASS_HUE_LO, GRASS_HUE_HI = 32, 95   # OpenCV H (0-179): green vegetation band
GRASS_SAT_MIN = 55                    # vivid-green if saturation above this
SKY_VAL_MIN, SKY_SAT_MAX = 145, 45    # bright + desaturated => sky/overcast
# A clear blue sky is bright but NOT desaturated -- measured at S 33-71 across these
# clips -- so the desaturated test alone matched almost none of it, which is why the
# walkable mask was free to climb into the sky. Hue catches what saturation misses:
# sky here sits at H 94-102, comfortably inside the blue band. Bright grey pavement
# can take on an arbitrary hue because hue is meaningless at low saturation, but that
# does not matter for either use of this mask: the walkable veto only applies above
# the horizon, and the horizon estimate only counts sky that runs unbroken from the
# top of the frame.
SKY_HUE_LO, SKY_HUE_HI = 90, 135      # OpenCV H (0-179): blue sky band
# ...but only ABOVE the horizon. "Bright and desaturated" also describes sunlit
# concrete and pale gravel exactly, so applying this test to the whole frame punched
# large holes in the path wherever the sun hit it -- which broke the corridor,
# dragged the line toward the shaded side, and manufactured forks. Sky cannot be
# below the horizon, and the ground plane already tells us where that is, so the
# test is restricted to where sky can actually occur. The margin keeps the veto
# active just past the horizon, where distant haze blends into the ground and is
# far beyond anything we plan through anyway.
SKY_HORIZON_MARGIN_PX = 5
GRAY_SAT_MAX = 60                     # low saturation => grey walkable surface
# ImageNet normalization (SegFormer image processor defaults), RGB order.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- Road-mask temporal stickiness (Fix B) ---
# Road vs walkway is a near tie for a B0 model, so shadows/texture flip the raw
# argmax frame-to-frame. Require a patch to look like road for ~2 frames before
# it turns blue (and stay clear ~2 frames before it turns back) instead of
# trusting each frame. This votes on the per-pixel *decision* over time; it does
# not blur frames together, so a moving camera is not smeared.
ROAD_EMA_ALPHA = 0.4             # weight of the current frame (lower = stickier)
ROAD_STICK_THRESH = 0.5          # road is "on" where the running vote >= this

# --- Walkable-mask temporal persistence (kill the green flicker) ---
# The raw walkable union (walkway OR road) is recomputed from scratch each frame,
# so a single bad SegFormer frame makes chunks of the path drop out -- or the
# whole path vanish ("no path detected"). We keep a per-pixel running vote so the
# mask is *sticky*: a pixel turns green fast (one confident frame) but only fades
# out after several frames of genuine absence. This keeps the whole path covered
# and stops the width from pulsing wide/narrow between frames. It votes on the
# decision, not the pixels, so forward motion is not smeared.
WALK_EMA_ALPHA = 0.40            # weight of the current frame (lower = stickier)
WALK_STICK_THRESH = 0.28         # stays green until the vote decays below this

# --- Small-hole fill (mask speckle) ---
# Shadows, seams, wet patches and leaf litter punch small holes in the walkable
# mask. Each one is a false obstruction: it breaks the corridor, drags the
# centre-line around, and used to manufacture forks. Interior holes below this
# fraction of the frame are filled in. The cap matters -- it must stay well under
# the silhouette of a real obstacle at a distance we care about (a person at 3 m is
# roughly 5% of the frame), so genuine blockages still punch through. Only holes
# fully enclosed by walkable pixels are filled; anything touching the frame edge is
# left alone, since that is the boundary with the grass/wall beside the path.
HOLE_FILL_MAX_FRAC = 0.015

# --- Whole-mask collapse guard (stop the path from vanishing entirely) ---
# The per-pixel vote above smooths edges, but a run of bad SegFormer frames can
# still make the ENTIRE path drop out for a moment -- then the guide line shrinks
# to nothing and snaps back, looking like it curves erratically. So we also watch
# the total walkable *area*: if it suddenly craters versus its running level, we
# reject that frame and reuse the last good mask for a short spell. A real path
# that genuinely ends shrinks gradually and/or stays gone past the hold window,
# so we still report NO PATH when the path truly runs out.
COLLAPSE_FRAC = 0.55             # frame area below this * running level = a collapse
COLLAPSE_MIN_AREA = 0.05         # only guard once we've actually had a real path
COLLAPSE_HOLD_FRAMES = 12        # hold the last good mask at most this many frames
AREA_EMA_ALPHA = 0.15            # how fast the "expected area" level tracks changes

# --- SegFormer frame-striding (the heavy model is the bottleneck) ---
# SegFormer is ~all of the per-frame cost, so we run it only every SEG_STRIDE
# processed frames and reuse the last raw class masks in between. The cheap
# per-frame persistence (EMA + collapse guard + feet-connected trim) still runs
# every frame, so the mask stays smooth and stays locked to your feet while the
# expensive inference happens 1/SEG_STRIDE as often. Obstacle detection (YOLO)
# is left running every frame because obstacles are time-critical. A slow walker
# barely moves in SEG_STRIDE frames, so the reused mask is still accurate.
SEG_STRIDE = 2                   # run SegFormer every Nth processed frame (1 = every frame)
                                 #   (~3 "off" frames of grace at 0.45 alpha)

# --- Obstacle model (YOLOv8n, COCO) ---
YOLO_PATH = "yolov8n.onnx"
YOLO_INPUT = 640                 # network input is YOLO_INPUT x YOLO_INPUT
YOLO_CONF = 0.35                 # min confidence to keep a detection
YOLO_IOU = 0.45                  # NMS IoU threshold
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
# Only classes that matter for walking navigation (trip/collision hazards).
# NOTE: this is a POST-processing filter -- the network still computes all 80
# class scores, so it does not make the model itself lighter (only trims noise
# and a little decode work). To truly shrink/speed it up, quantize or re-export.
OBSTACLE_NAMES = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "dog", "cat", "backpack", "suitcase", "chair", "potted plant", "couch",
}

# --- Guidance ---
# |offset| under this => "straight". Lowered from 8.0 now that the planner reports
# the path's tangent heading: a gentle but real bend comes out around 8 deg, which
# the old threshold sat exactly on top of and reported as straight. 6 deg is about
# a 24 m radius curve -- shallow enough to ignore, anything sharper is worth saying.
STRAIGHT_DEG = 6.0
STEER_EMA_ALPHA = 0.4            # steering angle smoothing (0=frozen, 1=no smooth)
# Fork = the walkable corridor visibly splits into two branches with a real
# non-walkable wedge between them. Real branches, measured on the actual clips,
# are only ~20-35px wide (0.06-0.11 of width) once they're a bit ahead of you --
# the old 0.12 (38px) branch floor silently threw every real fork away, which is
# why forks stopped being detected at all. We keep the branch/gap floors small
# but demand the split show up across several rows of a *dense* vertical scan: a
# genuine fork's wedge is tall (5-7 rows here), while a patchy-mask nick is 1-2.
FORK_MIN_SEG_FRAC = 0.06         # a branch must be >= this wide (frac of width, ~19px)
FORK_GAP_FRAC = 0.05             # min non-walkable gap to count as a real split (~16px)
FORK_MIN_ROWS = 3                # split must appear in >= this many dense scan rows
FORK_SCAN_N = 22                 # dense vertical scan rows for the fork test
FORK_SCAN_BOT = 0.95             # near edge of the fork scan (frac of height)
FORK_SCAN_TOP = 0.25             # far edge of the fork scan (frac of height)

# --- Metric path planning (planner.py) ---
# The guide line now comes from planning through a bird's-eye COSTMAP in metres
# rather than from averaging walkable pixels per image row, so obstacles and
# their distances actually steer it. The mask-only centre-line follower below is
# kept as the fallback for when the planner finds nothing usable (no ranged
# ground, or a costmap with no viable arc).
USE_BEV_PLANNER = True           # False = old image-space centre-line only
# Fork detection now runs on the metric bird's-eye grid (planner._fork_from_grid),
# where a "branch" is a real walkable width and the dividing wedge has to be known
# non-walkable ground. The old image-space row scan below measured gaps as a
# fraction of image WIDTH, which means centimetres at your feet and metres near the
# horizon -- so mask speckle read as a fork on 80% of IMG_7198 and ~56% of
# IMG_7200/7203, none of which contain a fork. Set False to fall back to it.
FORK_FROM_PLANNER = True
SHOW_BEV = False                 # append the costmap debug panel (--bev)

# --- Centre-line tracking (local-window centre-of-mass follower) ---
# The path direction is found by walking up the frame from your feet and, at each
# row, taking the centre-of-mass of the walkable pixels *within a window around
# the current heading*. The window keeps the line locked onto the corridor you're
# on, so a grassy patch or a side area off to one side can't yank the line the
# wrong way -- and because the window slides with the path, the line actually
# curves the way the path curves (the old segment-midpoint + straight-line-fit
# approach collapsed every path to a vertical line).
CENTERLINE_N_SCAN = 44           # rows scanned bottom->top (dense = smooth curve)
CENTERLINE_TOP_FRAC = 0.25       # follow the path up to this height (was 0.45)
CENTERLINE_BOT_FRAC = 0.97       # nearest row (at your feet)
CENTERLINE_HALF_FRAC = 0.22      # half-width of the tracking window (frac of width)
CENTERLINE_JUMP_FRAC = 0.13      # stop if the centre jumps more than this in one row
CENTERLINE_MIN_ROW_PIX = 4       # a row needs this many walkable px in-window to count
CENTERLINE_MAX_MISS = 3          # tolerate this many empty rows before stopping
CENTERLINE_SMOOTH_K = 7          # moving-average window over the tracked centres
CURVE_SAMPLE_N = 16              # fixed number of drawn samples (for cross-frame EMA)
CURVE_EMA_ALPHA = 0.35           # line smoothing across frames (lower = steadier)
LINE_HOLD_FRAMES = 8             # keep drawing the last line this long if path blips out
# Warnings like "fork" and "crossing ahead" describe a place you are walking toward,
# not an instant. Symmetric per-frame persistence made them strobe: the detector
# only has to disagree for a moment for the banner to drop and come straight back.
# So confirm over a few frames before raising one, then hold it for a fixed spell
# whatever the detector does next, and only drop it once the evidence has stayed
# away. The hold is in seconds rather than frames so it survives the frame rate
# changing between the Pi and a rendered clip.
# The rise threshold has to be a duration too, and a longer one than is obvious. A
# hold is an amplifier: raising on 3 frames turned a raw fork rate of 8% on IMG_7198
# -- a dozen scattered single-frame hits -- into a warning covering 64% of the clip.
# What separates a real junction from mask noise is not how often the detector fires
# but whether it fires CONTINUOUSLY, so the rise window is the discriminator. Measured
# over the windows in check_warnings.py, 0.6 s splits them completely: false forks and
# false openings in IMG_7198 disappear, while the real crossing in IMG_7200 and the
# real fork in IMG_7203 each still raise once and stay up. The delay costs nothing
# because both are detected 5-12 m out, several seconds before they matter.
WARN_RISE_S = 0.60               # unbroken evidence needed before a warning is raised
WARN_RISE_MIN_FRAMES = 2         # ...but never trust a single frame
WARN_HOLD_S = 3.0                # minimum time a raised warning stays up (readable)
WARN_CLEAR_S = 1.0               # evidence must be absent this long before it drops

# --- Decision cadence (don't re-announce a new bearing every single frame) ---
# A walking human can't act on a heading that jitters several times a second, so
# we keep blending the mask/line every frame (smooth visuals) but only *commit* a
# new spoken/printed directive on a slow cadence, then hold it. The committed
# bearing is also quantized so it reads as "11 deg right", not "10.6 -> 12.3 ...".
# 2.0 s was too long: at walking pace a turn came and went inside one hold window,
# so across whole clips the directive never changed (IMG_7201 committed the same cue
# for all 688 frames, IMG_7199 changed once). 1.0 s still keeps the announcement
# from chattering but tracks a real bend.
DECISION_PERIOD_S = 1.0          # re-evaluate the committed directive this often
# ...and never let the cadence hide a genuinely new situation. If the smoothed
# steering has moved this far from what we last announced, re-commit immediately
# rather than waiting out the window. Without this, an obstacle appearing mid-window
# is silently held for up to a second.
RECOMMIT_DELTA_DEG = 12.0
# ...but a smaller move is worth announcing at once while something is closing on you.
# Waiting out the cadence there means the arrow reads "bear left 10" for up to a second
# after the plan has committed to a 24 deg curve around a pedestrian, which understates
# how far you actually have to go and looks like the arrow ignoring the line.
HAZARD_RECOMMIT_DEG = 6.0
DIRECTIVE_QUANT_DEG = 5          # round the announced bearing to this step
DEFAULT_EFFECTIVE_FPS = 12.0     # fallback frames/sec when the caller can't say

# --- Crossing ("road ahead" while standing on a walkway) ---
CROSS_NEAR_BAND = 0.75           # rows below this frac = "near me"
CROSS_AHEAD_BAND = (0.35, 0.65)  # (top, bottom) frac band = "ahead of me"
CROSS_NEAR_WALKWAY_MIN = 0.12    # near band must be this walkway-covered
# A real crossing means you stand on a *walkway* and a road appears AHEAD. On a
# paved trail the whole corridor is road-classified, so the road runs right
# under your feet too -- that is NOT a crossing. Require the ground immediately
# in front of you to be mostly non-road, else treat road-ahead as just the trail
# continuing (no false "CROSSING AHEAD", no false blue band).
CROSS_NEAR_ROAD_MAX = 0.40       # near band road coverage above this => trail, not crossing
# The band test alone can't tell a paved *trail* (which perspective makes read as
# walkway near / road far) from a real street crossing -- both look identical to
# SegFormer. A real street has traffic; a trail does not. So we only call it a
# crossing (and only then paint blue) when a vehicle is actually seen ahead. This
# is what keeps trails 100% green: no cars => no crossing => no blue.
VEHICLE_NAMES = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}
CROSS_VEHICLE_BAND = (0.20, 0.75)  # a vehicle's base in this row range = "ahead, on the street"
# Fix E1: a real street crossing spans the width of the view; a driveway or a
# stray road patch only covers part of it. Require road across a wide fraction
# of the "ahead" band's *columns* (total coverage, not a single unbroken run --
# a crosswalk/path cutting through a real crossing would break an unbroken run).
CROSS_AHEAD_COVER_FRAC = 0.70    # frac of ahead-band columns with road => crossing

# --- Obstacle proximity ---
NEAR_BASE_FRAC = 0.60            # box base below this frac of height = "close"
NEAR_BOX_H_FRAC = 0.45           # or box taller than this frac = "close"

# --- Phase A: static/dynamic + coarse distance bins + collision direction ---
# Distance is reported as a coarse ZONE for now (not metres) -- accurate metric
# distance (+/- 1 m) needs calibration + stereo (Phase A+). The bins come from
# where the box base sits in the frame (prox) and how tall the box is: a nearer
# object sits lower and looks bigger. Tuned for the 320x240 processing frame.
IMMEDIATE_PROX = 0.85            # box base this low in frame  => ~arm's length
NEAR_PROX = 0.62                 # ~1.5-3 m zone
MID_PROX = 0.38                  # ~3-6 m zone (else "far")
IMMEDIATE_BOX_H = 0.60           # or box taller than this frac => immediate
NEAR_BOX_H = 0.42                # or box taller than this frac => near
# Approx horizontal field of view, used to turn a left/right pixel offset into a
# rough bearing when ground-plane ranging isn't available for an object.
NOMINAL_HFOV_DEG = 70.0

# --- Ground-plane monocular metric ranging ---
# The camera is fixed at a known height + downward tilt, so an object's ground
# contact point (box bottom-centre) maps to a real distance in metres. Set these
# to match your mounting. Distance bins below are now in METRES (not image cues).
CAM_HEIGHT_M = 55.5 * 0.0254     # camera height above ground (55.5 in -> 1.410 m)
CAM_TILT_DEG = 18.43             # nominal downward tilt; measured per clip, see below
# --- Measuring the tilt instead of assuming it -------------------------------
# Every metric thing we do -- ranging obstacles, the bird's-eye warp, where the sky
# veto applies -- is built on the camera's downward tilt, and a single hardcoded
# figure turned out to be wrong by up to 5 deg on this footage: the true horizon sits
# at processing row 77 in IMG_7198, 61 in IMG_7201 and 46 in IMG_7203, against 63
# assumed. That is not a small error. Too high and the walkable mask is allowed to
# climb into the sky; too low and sky gets warped into the grid as if it were ground
# tens of metres out, which is where the phantom forks and the collapsing masks came
# from. So we measure the horizon from the image each frame and back out the tilt.
#
# On the wearable this should come from the IMU instead: a body-mounted camera has a
# tilt that gravity knows exactly, for free, and without needing sky in view.
ADAPT_HORIZON = True
HORIZON_PCT = 80                 # percentile of per-column sky depth (see estimator)
HORIZON_MIN_SKY_COLS = 0.10      # need this fraction of columns starting in sky
HORIZON_WINDOW = 15              # frames of history the median is taken over
HORIZON_TILT_RANGE = (6.0, 32.0) # plausible mounting angles; clamp to these
# Rebuilding the warp is cheap but not free, and the estimate jitters by a fraction
# of a degree frame to frame, so only re-derive the geometry on a real move.
HORIZON_RETILT_DEG = 0.75
# Field of view across the camera's LONG sensor side (iPhone 16 Pro main 4K ~70 deg).
# Orientation-robust: works whether the clip is portrait or landscape (the frame
# may be auto-rotated to portrait 2160x3840). Tune using the drawn horizon line.
CAM_FOV_LONG_DEG = 70.0
DIST_IMMEDIATE_M = 1.5           # <= this => "immediate"
DIST_NEAR_M = 3.0                # <= this => "near"
DIST_MID_M = 6.0                 # <= this => "mid"; beyond => "far"
# Ground-plane distance explodes toward infinity near the horizon (a tiny pixel
# error becomes huge metres), so beyond this we don't trust the number: report
# "far" with no metric value rather than a nonsense "1244 m".
MAX_RANGE_M = 20.0
# |heading_deg| above this means the object is moving downward in the frame,
# i.e. toward you (see tracking.py heading convention).
TOWARD_USER_MIN_DEG = 100.0
# --- Reacting to something closing on you -----------------------------------
# Distance alone is the wrong trigger for a moving obstacle. "Close" above means
# within 3 m, but two people walking toward each other close a gap at roughly 2.8
# m/s, so 3 m is about one second's warning -- useless to a blind walker, and the
# reason the guidance in IMG_7204 stayed STRAIGHT while a pedestrian came straight
# on. What matters is how long until you meet, so we range the closing speed and act
# on time-to-contact. The speed is deliberately NOT ego-motion compensated: the gap
# shrinking is what threatens you, regardless of whose legs are closing it.
# 5 s, not the 3.5 s that seems sufficient. A pedestrian 7 m off closing at 1.5-2.5
# m/s sits right on a 3.5 s threshold, so the risk flag toggled frame to frame as the
# estimate wobbled either side of it -- and a threshold that marginal is worse than a
# generous one, because it decides whether we are avoiding at all. At 5 s the person
# is flagged once, well before they are close, and stays flagged.
HAZARD_TTC_S = 5.0               # act when contact is this near in time
# A hazard's warning stays on screen while the hazard is still in view, plus this
# much grace for frames where the detector misses it (see ObstacleAnnouncer).
OBST_TEXT_GRACE_S = 1.0
# The printed distance is held until it has moved this far. Somebody approaching at
# walking pace closes 0.05 m per frame, so a live "7.0m / 6.9m / 7.0m" redraws the
# warning several times a second -- the number is technically fresher and the line is
# materially harder to read. Half a metre is finer than the ranging is accurate anyway.
OBST_TEXT_STEP_M = 0.5
OBST_TEXT_SETTLE_S = 0.5         # wording waits this long on a change (see Settled)
# How far ahead (px, along the motion vector) to probe the walkable mask when
# deciding if a moving object's path will cross yours.
TOWARD_PATH_LOOKAHEAD = (8, 16, 24)
# Class priors: things that cannot move on their own vs things that can. A car
# is "dynamic" even when parked (it *may* move / a door may open); a bench never
# walks. Ambiguous carry-able items default to static unless seen clearly moving.
STATIC_PRIOR = {"bench", "fire hydrant", "stop sign", "parking meter",
                "potted plant", "traffic light"}
DYNAMIC_PRIOR = {"person", "bicycle", "car", "motorcycle", "bus", "train",
                 "truck", "dog", "cat"}

SAVE_PATH = "annotated_path.jpg"
VIDEO_OUT = "annotated_video.mp4"
SAVE_EVERY = 15


def open_camera(source):
    is_device = isinstance(source, int)
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2) if is_device else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera source {source!r}. Is the ESP32 cam connected?")
    if is_device:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always grab the freshest frame
    return cap


class WalkableSegmenter:
    """SegFormer-B0 (ADE20K) ONNX segmenter -> (walkable mask, road mask)."""

    def __init__(self, model_path=MODEL_PATH, input_size=INPUT_SIZE):
        if not os.path.exists(model_path):
            raise SystemExit(
                f"Model not found: {model_path}\n"
                "Export it once with:  python3 export_segformer.py"
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        self.sess = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.input_size = input_size
        self.walkway_ids = self._resolve_ids(model_path, WALKWAY_NAMES)
        self.road_ids = self._resolve_ids(model_path, ROAD_NAMES)
        self.walkable_ids = self.walkway_ids | self.road_ids
        self.structure_ids = self._resolve_ids(model_path, STRUCTURE_NAMES)
        self._road_ema = None            # running per-pixel road vote (Fix B)
        self._walk_ema = None            # running per-pixel walkable vote (persistence)
        self.horizon_row = None          # set via set_horizon(); see SKY_HORIZON_MARGIN_PX
        self._last_walkable = None       # last good mask, reused during a collapse
        self._area_ema = None            # running "expected" walkable area level
        self._collapse_hold = 0          # frames we've been holding through a collapse
        self._seg_t = 0                  # frames since SegFormer last actually ran
        self._sf_walk = None             # cached SegFormer walkable between runs
        self._sf_veto = None             # cached SegFormer structure veto between runs
        self._raw_road = None            # cached SegFormer road between runs

    def reset(self):
        """Clear all temporal state (call between independent clips/images)."""
        self._road_ema = None
        self._walk_ema = None
        self._last_walkable = None
        self._area_ema = None
        self._collapse_hold = 0
        self._seg_t = 0
        self._sf_walk = None
        self._sf_veto = None
        self._raw_road = None
        if not self.walkable_ids:
            raise SystemExit("Could not map any walkable names to model class ids; check config.json.")

    def set_horizon(self, horizon_ny, height=PROC_H):
        """Tell the segmenter which image row the ground plane vanishes at, so the
        sky test can be confined to where sky is geometrically possible."""
        self.horizon_row = int(round((horizon_ny + 0.5) * height))

    @staticmethod
    def _resolve_ids(model_path, names):
        """Read id2label from the exported config.json and select matching ids."""
        cfg_path = os.path.join(os.path.dirname(model_path), "config.json")
        ids = set()
        if not os.path.exists(cfg_path):
            return ids
        with open(cfg_path) as f:
            cfg = json.load(f)
        for idx, label in cfg.get("id2label", {}).items():
            tokens = {t.strip().lower() for t in label.replace(",", ";").split(";")}
            if tokens & names:
                ids.add(int(idx))
        return ids

    def _preprocess(self, proc_bgr):
        rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        x = rgb.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))[None]      # HWC -> 1CHW
        return np.ascontiguousarray(x)

    def __call__(self, proc_bgr):
        """Return (walkable, road, road_full), all uint8 0/255 at proc size.

        walkable/road are restricted to the region connected to your feet (what
        you actually follow). road_full is the temporally-smoothed road mask
        *before* that feet-connected trim, used for crossing detection so a
        cross-street separated from your feet by a median isn't discarded (Fix 2).
        """
        h, w = proc_bgr.shape[:2]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # SegFormer frame-striding: only run the heavy net every SEG_STRIDE frames.
        # From it we cache the semantic walkable, the structure veto, and the road
        # split; the cheap appearance mask (below) is rebuilt every frame, so the
        # path stays responsive while inference runs 1/SEG_STRIDE as often.
        if self._sf_walk is None or self._seg_t >= SEG_STRIDE:
            self._seg_t = 0
            x = self._preprocess(proc_bgr)
            logits = self.sess.run([self.output_name], {self.input_name: x})[0]  # [1,C,h',w']
            class_map = np.argmax(logits[0], axis=0).astype(np.int32)            # [h', w']

            def _mask(ids):
                m = np.isin(class_map, list(ids)).astype(np.uint8) * 255
                return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            self._sf_walk = _mask(self.walkable_ids)
            self._sf_veto = _mask(self.structure_ids)
            self._raw_road = _mask(self.road_ids)
        else:
            self._seg_t += 1
        sf_walk, veto, road = self._sf_walk, self._sf_veto, self._raw_road

        # Appearance-based path (every frame): grey surface, minus vivid grass,
        # sky, and SegFormer structures; unioned with any path SegFormer did find.
        walkable = self._appearance_walkable(proc_bgr, sf_walk, veto, kernel)
        # Nothing above the horizon rests on the ground, so nothing above it can be
        # walkable -- a grey building on the skyline passes the "grey surface" test
        # otherwise, which is how the mask ended up climbing into the sky. Ground
        # within a row or two of the horizon is tens of metres out and useless to
        # plan through, so cutting exactly at the estimate costs nothing.
        if self.horizon_row is not None:
            walkable[:max(0, self.horizon_row)] = 0

        walkable = self._smooth_walkable(walkable)  # temporal persistence (anti-flicker)
        walkable = cv2.morphologyEx(walkable, cv2.MORPH_CLOSE, kernel, iterations=2)
        walkable = self._fill_small_holes(walkable)  # shadow/seam speckle
        walkable = self._guard_collapse(walkable)   # reject sudden whole-path dropouts
        road_full = self._smooth_road(road)      # temporal stickiness (Fix B), untrimmed

        walkable = self._feet_component(walkable)     # keep the corridor at your feet
        road = cv2.bitwise_and(road_full, walkable)   # road within the followed region
        return walkable, road, road_full

    def _appearance_walkable(self, proc_bgr, sf_walk, veto, kernel):
        """Build the path mask from colour (see the module HSV notes).

        A pedestrian path is a low-saturation grey surface; grass is vivid green
        and sky is bright/desaturated. So: (grey OR SegFormer-walkable), then drop
        grass, sky, and anything SegFormer confidently calls a structure. This
        recovers gravel that SegFormer misses and rejects the grass field that it
        wrongly accepts -- the two failure modes behind the flaky green mask.
        """
        hsv = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        grass = (H >= GRASS_HUE_LO) & (H <= GRASS_HUE_HI) & (S > GRASS_SAT_MIN)
        sky = sky_mask(hsv)
        if self.horizon_row is not None:                # see SKY_HORIZON_MARGIN_PX
            sky[min(proc_bgr.shape[0], self.horizon_row + SKY_HORIZON_MARGIN_PX):, :] = False
        grayish = S < GRAY_SAT_MAX
        cand = (grayish | (sf_walk > 0)) & (~grass) & (~sky) & (veto == 0)
        cand = (cand.astype(np.uint8)) * 255
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel, iterations=3)
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kernel, iterations=1)
        return cand

    @staticmethod
    def _fill_small_holes(mask):
        """Fill holes fully enclosed by walkable pixels and smaller than
        HOLE_FILL_MAX_FRAC of the frame (see that constant for the size trade-off).

        Morphological closing only bridges gaps up to the kernel size; a shadow
        across the path is far bigger than a 5x5 ellipse but still nothing like an
        obstacle. This closes those properly by looking at connected components of
        the NON-walkable region and keeping only the ones that are either large or
        touching the frame edge.
        """
        inv = cv2.bitwise_not(mask)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
        if n <= 1:
            return mask
        h, w = mask.shape
        limit = HOLE_FILL_MAX_FRAC * h * w
        fill = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            touches_edge = x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h
            if not touches_edge and area <= limit:
                fill.append(i)
        if not fill:
            return mask
        out = mask.copy()
        out[np.isin(labels, fill)] = 255
        return out

    def _feet_component(self, mask):
        """Keep only the walkable blob connected to your feet (bottom-centre).

        Scans up the centre column for a seed, and if the exact centre is off the
        path (framing where grass is underfoot) falls back to the largest blob
        touching the bottom edge -- so we still lock onto the path you're on
        instead of dropping everything.
        """
        h, w = mask.shape
        num, labels = cv2.connectedComponents(mask)
        seed = 0
        for dy in range(0, max(1, h // 6)):
            s = int(labels[h - 1 - dy, w // 2])
            if s:
                seed = s
                break
        if seed == 0:
            bottom = set(np.unique(labels[h - 3:h, :])) - {0}
            if bottom:
                seed = max(bottom, key=lambda L: int((labels == L).sum()))
        if seed:
            return np.where(labels == seed, 255, 0).astype(np.uint8)
        return mask

    def _smooth_road(self, road):
        """Temporal 'stickiness' for the road mask (Fix B).

        A patch must read as road for a couple of frames before it turns blue,
        and stay clear a couple of frames before it turns back -- so a single
        shadow-driven argmax flip no longer flickers the walkway blue/green.
        This votes on the per-pixel *decision* over time; it does not blur the
        pixels themselves, so a moving camera is not smeared.
        """
        cur = (road > 0).astype(np.float32)
        if self._road_ema is None or self._road_ema.shape != cur.shape:
            self._road_ema = cur
        else:
            self._road_ema = ROAD_EMA_ALPHA * cur + (1.0 - ROAD_EMA_ALPHA) * self._road_ema
        return (self._road_ema >= ROAD_STICK_THRESH).astype(np.uint8) * 255

    def _guard_collapse(self, walkable):
        """Reject a sudden whole-mask dropout by reusing the last good mask.

        The per-pixel vote handles edge flicker; this handles the failure where
        the *entire* path blinks out for a few frames (bad segmentation), which
        made the guide line collapse and snap back. We track the running walkable
        area and, if a frame craters far below that level, hold the last good mask
        for up to COLLAPSE_HOLD_FRAMES. A path that truly ends stays gone past the
        window, so NO PATH still fires when the path genuinely runs out.
        """
        area = float((walkable > 0).mean())
        if self._area_ema is None:
            self._area_ema = area
            self._last_walkable = walkable.copy()
            return walkable
        collapsed = (self._last_walkable is not None
                     and self._area_ema >= COLLAPSE_MIN_AREA
                     and area < COLLAPSE_FRAC * self._area_ema
                     and self._collapse_hold < COLLAPSE_HOLD_FRAMES)
        if collapsed:
            self._collapse_hold += 1
            return self._last_walkable            # hold last good geometry
        self._collapse_hold = 0
        self._area_ema = AREA_EMA_ALPHA * area + (1.0 - AREA_EMA_ALPHA) * self._area_ema
        self._last_walkable = walkable.copy()
        return walkable

    def _smooth_walkable(self, walkable):
        """Temporal persistence for the walkable mask (anti-flicker).

        Same per-pixel vote idea as _smooth_road, but tuned to be *sticky-on*:
        a pixel goes green quickly and only fades after several frames of real
        absence. That keeps the entire path covered frame-to-frame -- no more
        chunks dropping out, no more whole-path "NO PATH" dropouts, and the
        path width stops pulsing (which is what was throwing off the centre-line).
        """
        cur = (walkable > 0).astype(np.float32)
        if self._walk_ema is None or self._walk_ema.shape != cur.shape:
            self._walk_ema = cur
        else:
            self._walk_ema = WALK_EMA_ALPHA * cur + (1.0 - WALK_EMA_ALPHA) * self._walk_ema
        return (self._walk_ema >= WALK_STICK_THRESH).astype(np.uint8) * 255


def letterbox(img, size):
    """Resize keeping aspect ratio and pad to a square `size`.

    Returns (padded_img, ratio, dw, dh) so detections can be mapped back with
    x_orig = (x_model - dw) / ratio.
    """
    h, w = img.shape[:2]
    ratio = min(size / w, size / h)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2.0, (size - nh) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, ratio, left, top


class ObstacleDetector:
    """YOLOv8n (COCO) ONNX detector returning relevant-class boxes in image coords."""

    def __init__(self, model_path=YOLO_PATH, input_size=YOLO_INPUT,
                 conf=YOLO_CONF, iou=YOLO_IOU):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        self.sess = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = input_size
        self.conf = conf
        self.iou = iou
        self.keep_ids = {i for i, n in enumerate(COCO_NAMES) if n in OBSTACLE_NAMES}

    def __call__(self, img_bgr):
        """Return a list of detections: dicts with box (x1,y1,x2,y2), conf, name."""
        h, w = img_bgr.shape[:2]
        padded, ratio, dw, dh = letterbox(img_bgr, self.input_size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None])
        out = self.sess.run(None, {self.input_name: x})[0]   # [1, 84, 8400]

        preds = np.squeeze(out, 0)
        if preds.shape[0] < preds.shape[1]:                  # [84, 8400] -> [8400, 84]
            preds = preds.T
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        keep = (confidences >= self.conf) & np.isin(class_ids, list(self.keep_ids))
        boxes_xywh, class_ids, confidences = boxes_xywh[keep], class_ids[keep], confidences[keep]
        if len(boxes_xywh) == 0:
            return []

        # cx,cy,w,h (model space) -> x,y,w,h top-left (model space) for NMS.
        xs = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        ys = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        rects = np.stack([xs, ys, boxes_xywh[:, 2], boxes_xywh[:, 3]], axis=1)
        idxs = cv2.dnn.NMSBoxes(rects.tolist(), confidences.tolist(), self.conf, self.iou)
        if len(idxs) == 0:
            return []
        idxs = np.array(idxs).flatten()

        dets = []
        for i in idxs:
            x, y, bw, bh = rects[i]
            x1 = int(round((x - dw) / ratio))
            y1 = int(round((y - dh) / ratio))
            x2 = int(round((x + bw - dw) / ratio))
            y2 = int(round((y + bh - dh) / ratio))
            x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
            y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
            cls = int(class_ids[i])
            dets.append({
                "box": (x1, y1, x2, y2),
                "conf": float(confidences[i]),
                "cls": cls,
                "name": COCO_NAMES[cls] if cls < len(COCO_NAMES) else str(cls),
            })
        return dets


def make_ground(width, height, tilt_deg=None):
    """Build the GroundPlane model for a camera frame of the given ACTUAL size
    (after any auto-rotation). Orientation-robust via the long-side FOV.

    tilt_deg overrides CAM_TILT_DEG, which is only a nominal mounting figure -- see
    HorizonTracker for why the real tilt has to be measured per clip.
    """
    return GroundPlane.from_fov(CAM_HEIGHT_M,
                                CAM_TILT_DEG if tilt_deg is None else tilt_deg,
                                CAM_FOV_LONG_DEG, width, height)


def sky_mask(hsv):
    """Boolean sky mask from an HSV image. See SKY_HUE_LO for why hue is needed."""
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (V > SKY_VAL_MIN) & ((S < SKY_SAT_MAX)
                                | ((H >= SKY_HUE_LO) & (H <= SKY_HUE_HI)))


def estimate_horizon_row(proc_bgr):
    """Image row where the sky meets the ground, or None if no sky is in view.

    Per column, find where the unbroken run of sky from the top of the frame ends;
    the horizon is a high percentile of those depths. Requiring the run to start at
    row 0 is what makes this safe: bright ground can look like sky in isolation but
    is never contiguous with the top of the frame. Anything standing above eye level
    -- trees, poles, houses -- ends its column's run early, which is why we take a
    high percentile rather than an average: the columns looking out over open ground
    are the ones that see the true horizon, and they are the deepest.
    """
    sky = sky_mask(cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2HSV))
    h, w = sky.shape
    ground = ~sky
    first = np.argmax(ground, axis=0)              # 0 where the top row is ground
    first[~ground.any(axis=0)] = h                 # column is sky all the way down
    from_top = sky[0]
    if from_top.sum() < HORIZON_MIN_SKY_COLS * w:
        return None
    return float(np.percentile(first[from_top], HORIZON_PCT))


def _distance_bin(prox, box_h_frac):
    """Coarse distance zone from image cues (fallback when metric range is N/A)."""
    if prox >= IMMEDIATE_PROX or box_h_frac >= IMMEDIATE_BOX_H:
        return "immediate"
    if prox >= NEAR_PROX or box_h_frac >= NEAR_BOX_H:
        return "near"
    if prox >= MID_PROX:
        return "mid"
    return "far"


def _closing_speed_ms(o, ground, w, h):
    """How fast the gap to a tracked object is shrinking, in m/s, or None.

    Projects the two ends of the track's history onto the ground and differences the
    range. See HAZARD_TTC_S on why this is left uncompensated for your own motion.
    """
    h0, h1 = o.get("hist0"), o.get("hist1")
    if not h0 or not h1 or ground is None:
        return None
    dt = h1[2] - h0[2]
    if dt <= 1e-3:
        return None
    a = ground.point(h0[0] / float(w) - 0.5, h0[1] / float(h) - 0.5)
    b = ground.point(h1[0] / float(w) - 0.5, h1[1] / float(h) - 0.5)
    if a is None or b is None:
        return None
    return (a["distance_m"] - b["distance_m"]) / dt


def _distance_bin_m(distance_m):
    """Distance zone from a metric distance (ground-plane ranging)."""
    if distance_m <= DIST_IMMEDIATE_M:
        return "immediate"
    if distance_m <= DIST_NEAR_M:
        return "near"
    if distance_m <= DIST_MID_M:
        return "mid"
    return "far"


def _classify_kind(name, moving):
    """static (won't move) vs dynamic (can move), from class prior + motion."""
    if name in STATIC_PRIOR:
        return "static"
    if name in DYNAMIC_PRIOR:
        return "dynamic"
    return "dynamic" if moving else "static"        # ambiguous carry-ables


def _side_word(bearing_deg):
    if bearing_deg <= -10.0:
        return "left"
    if bearing_deg >= 10.0:
        return "right"
    return "ahead"


def _heading_toward_path(o, mask):
    """Probe a few points ahead along the motion vector; True if any lands on the
    walkable mask (the object is drifting onto your path)."""
    if not o.get("moving"):
        return False
    vx, vy = o.get("vx", 0.0), o.get("vy", 0.0)
    norm = (vx * vx + vy * vy) ** 0.5
    if norm < 1e-3:
        return False
    ux, uy = vx / norm, vy / norm
    h, w = mask.shape
    for k in TOWARD_PATH_LOOKAHEAD:
        px, py = int(o["cx"] + ux * k), int(o["cy"] + uy * k)
        if 0 <= px < w and 0 <= py < h and mask[py, px] > 0:
            return True
    return False


def enrich_objects(objects, mask, ground=None):
    """Turn tracked objects into navigation-ready state (Phase A).

    Adds to each object dict:
      kind          static | dynamic
      prox          0 (far/top) .. 1 (underfoot)
      distance_m    metric ground distance (ground-plane) or None if above horizon
      size_m        (width_m, height_m) estimate or None
      distance_bin  immediate | near | mid | far  (from metres if available)
      bearing_deg   left(-) / right(+) of centre
      side          left | ahead | right
      on_path       box base sits on a walkable pixel
      toward_user   moving downward in frame (i.e. toward you)
      toward_path   motion vector will cross onto the walkable path
      collision_risk dynamic + close + heading onto your path/at you
      urgent        collision_risk OR anything close blocking the path

    ground: a GroundPlane for metric ranging (None -> fall back to image-cue bins).
    Returns (on_path_count, urgent_count).
    """
    h, w = mask.shape
    on_path_count, urgent_count = 0, 0
    for o in objects:
        x1, y1, x2, y2 = o["box"]
        cx = min(max(int(o["cx"]), 0), w - 1)
        cy = min(max(int(o["cy"]), 0), h - 1)
        box_h_frac = (y2 - y1) / float(h)

        o["prox"] = cy / float(h - 1)                  # 0 far .. 1 underfoot
        # Metric ranging from the ground-contact point (box bottom-centre).
        o["distance_m"] = None
        o["size_m"] = None
        gp = None
        if ground is not None:
            nx = o["cx"] / float(w) - 0.5              # fraction preserved across resize
            ny = o["cy"] / float(h) - 0.5
            gp = ground.point(nx, ny)
        if gp is not None and gp["distance_m"] <= MAX_RANGE_M:
            o["distance_m"] = gp["distance_m"]
            o["depth_m"] = gp["depth_m"]
            o["bearing_deg"] = gp["bearing_deg"]
            o["size_m"] = ground.size_m((x2 - x1) / float(w),
                                        (y2 - y1) / float(h), gp["depth_m"])
            o["distance_bin"] = _distance_bin_m(gp["distance_m"])
        elif gp is not None:
            # beyond reliable range (near horizon): trust bearing, not the metres
            o["bearing_deg"] = gp["bearing_deg"]
            o["distance_bin"] = "far"
        else:
            o["distance_bin"] = _distance_bin(o["prox"], box_h_frac)
            o["bearing_deg"] = ((cx - w / 2.0) / (w / 2.0)) * (NOMINAL_HFOV_DEG / 2.0)
        o["side"] = _side_word(o["bearing_deg"])
        o["kind"] = _classify_kind(o["name"], o.get("moving", False))
        o["on_path"] = bool(mask[cy, cx] > 0)

        heading = o.get("heading_deg")
        o["toward_user"] = bool(o.get("moving") and heading is not None
                                and abs(heading) >= TOWARD_USER_MIN_DEG)
        o["toward_path"] = _heading_toward_path(o, mask)

        # How soon we meet, if it is coming toward us at all (see HAZARD_TTC_S).
        o["closing_ms"] = _closing_speed_ms(o, ground, w, h)
        o["ttc_s"] = None
        if o["closing_ms"] and o["closing_ms"] > 0.15 and o["distance_m"]:
            o["ttc_s"] = o["distance_m"] / o["closing_ms"]

        close = o["distance_bin"] in ("immediate", "near")
        # Something close sitting on your path is a blocker regardless of type.
        blocking = o["on_path"] and close
        # A moving thing on (or heading onto) your path that is either already close
        # or about to reach you. Time-to-contact is what catches the pedestrian
        # walking straight at you from 7 m, which no distance threshold does in time.
        soon = close or (o["ttc_s"] is not None and o["ttc_s"] <= HAZARD_TTC_S)
        o["collision_risk"] = bool(
            o["kind"] == "dynamic" and soon
            and (o["on_path"] or o["toward_path"])
        )
        o["urgent"] = bool(blocking or o["collision_risk"])
        # keep "near" for backward compatibility with any older callers/labels
        o["near"] = close

        on_path_count += int(o["on_path"])
        urgent_count += int(o["urgent"])
    return on_path_count, urgent_count


def _row_segments(row, w):
    """Contiguous walkable runs in a boolean row, wider than FORK_MIN_SEG_FRAC."""
    segs, in_seg, start = [], False, 0
    for x in range(w):
        if row[x] and not in_seg:
            in_seg, start = True, x
        elif not row[x] and in_seg:
            in_seg = False
            segs.append((start, x - 1))
    if in_seg:
        segs.append((start, w - 1))
    return [s for s in segs if (s[1] - s[0]) >= FORK_MIN_SEG_FRAC * w]


def _track_centerline(mask):
    """Walk up the frame from the feet, tracking the centre-of-mass of walkable
    pixels within a sliding window around the current heading. Returns the list
    of (cx, y) centres, near -> far. The window is what makes the line follow the
    path's curve instead of collapsing to a vertical line."""
    h, w = mask.shape
    ys = np.linspace(CENTERLINE_BOT_FRAC, CENTERLINE_TOP_FRAC, CENTERLINE_N_SCAN) * (h - 1)
    half = int(CENTERLINE_HALF_FRAC * w)
    jump = CENTERLINE_JUMP_FRAC * w
    centers, prev, miss = [], w / 2.0, 0
    for yf in ys:
        y = int(round(yf))
        x0, x1 = max(0, int(prev - half)), min(w, int(prev + half) + 1)
        xs = np.where(mask[y, x0:x1] > 0)[0]
        if len(xs) < CENTERLINE_MIN_ROW_PIX:
            miss += 1
            if miss > CENTERLINE_MAX_MISS:
                break
            continue
        cx = x0 + float(xs.mean())
        if centers and abs(cx - prev) > jump:    # path lost / ambiguous jump -> stop
            break
        miss = 0
        centers.append((cx, y))
        prev = cx
    return centers


def _smooth1d(a, k):
    if len(a) < 3 or k < 3:
        return a
    k = min(k, len(a) if len(a) % 2 else len(a) - 1)
    if k < 3:
        return a
    ap = np.pad(a, k // 2, mode="edge")
    return np.convolve(ap, np.ones(k) / k, mode="valid")


def detect_fork(mask):
    """True if the walkable area splits into >=2 wide-enough branches separated
    by a real non-walkable gap, and that split holds across several rows of a
    dense vertical scan. The multi-row requirement is what tells a true branch (a
    tall wedge of grass/ground dividing two paths) apart from a one-row hole in a
    patchy mask -- so we announce a fork on a genuine split without the phantom
    forks a single noisy row used to trigger.
    """
    h, w = mask.shape
    fork_rows = 0
    for yf in np.linspace(FORK_SCAN_BOT, FORK_SCAN_TOP, FORK_SCAN_N):
        segs = _row_segments(mask[int(yf * (h - 1))] > 0, w)
        if len(segs) >= 2:
            gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
            if any(g >= FORK_GAP_FRAC * w for g in gaps):
                fork_rows += 1
    return fork_rows >= FORK_MIN_ROWS


def analyze(mask):
    """Follow the path centre-line (mask-only fallback when the metric planner
    has nothing usable -- see planner.py for the primary route).

    Returns (offset_deg, valid, curve_x, curve_ys, fork, frac):
      offset_deg: steering angle toward a look-ahead point (negative = left).
      curve_x/curve_ys: centre-line samples on a *fixed* y-grid so the drawn
                        line can be smoothed across frames (curve_x is None if
                        no path was found).
    """
    h, w = mask.shape
    centers = _track_centerline(mask)

    walkable_frac = float((mask > 0).mean())
    valid = walkable_frac >= MIN_WALKABLE_FRAC and len(centers) >= 1

    offset_deg, curve_x, curve_ys = 0.0, None, None

    if valid and len(centers) >= 2:
        cys = np.array([c[1] for c in centers], dtype=np.float32)
        cxs = _smooth1d(np.array([c[0] for c in centers], dtype=np.float32),
                        CENTERLINE_SMOOTH_K)
        order = np.argsort(cys)                  # ascending y (far -> near) for interp
        ys_s, xs_s = cys[order], cxs[order]
        # Sample the drawn line ONLY over the range we actually tracked (bottom of
        # frame -> last real centre), instead of a fixed grid that runs to the top
        # of the frame. Extrapolating past the last tracked point is what made the
        # line shoot straight up into the sky beyond the final dot.
        near_y, far_y = float(ys_s.max()), float(ys_s.min())
        curve_ys = np.linspace(near_y, far_y, CURVE_SAMPLE_N).astype(np.float32)
        # Follow the tracked centres directly (smoothed + interpolated) instead of
        # a global polynomial, so the line can't invent an S-bend the path lacks.
        curve_x = np.clip(np.interp(curve_ys, ys_s, xs_s), 0, w - 1).astype(np.float32)
        # Cue = the *heading* of the drawn line (near foot -> far end), so it
        # matches the way the line bends. Using where the path is going (not just
        # its lateral offset at one row) is what makes "bear left/right" agree
        # with a path that visibly curves.
        near_x, far_x = float(curve_x[0]), float(curve_x[-1])
        offset_deg = float(np.degrees(np.arctan2(far_x - near_x, max(1.0, near_y - far_y))))
    elif valid:
        cx = float(np.clip(centers[0][0], 0, w - 1))
        y0 = float(centers[0][1])
        curve_ys = np.full(CURVE_SAMPLE_N, y0, dtype=np.float32)
        curve_x = np.full(CURVE_SAMPLE_N, cx, dtype=np.float32)
        dx = centers[0][0] - w / 2.0
        offset_deg = float(np.degrees(np.arctan2(dx, h * 0.6)))

    return offset_deg, valid, curve_x, curve_ys, detect_fork(mask), walkable_frac


def detect_crossing(walkable, road):
    """Standing on a walkway with a *wide road band* spanning the way ahead =>
    a street crossing (Fix E1).

    Approximate (ADE20K has no dedicated crosswalk class): a painted crosswalk
    reads as "road". A real street crosses the full width of your view, whereas
    a driveway or a stray road patch only covers part of it. We measure the
    fraction of the "ahead" band's *columns* that contain road (total width
    coverage) rather than a single unbroken run, because a crosswalk or path
    cutting through a real crossing would otherwise break the run.

    `road` here is the untrimmed road mask (see WalkableSegmenter: road_full), so
    a cross-street separated from your feet by a median still counts (Fix 2).
    """
    h, w = walkable.shape
    walkway = cv2.bitwise_and(walkable, cv2.bitwise_not(road)) > 0
    road_b = road > 0
    near_walkway = walkway[int(CROSS_NEAR_BAND * h):, :]
    near_road = road_b[int(CROSS_NEAR_BAND * h):, :]
    a0, a1 = int(CROSS_AHEAD_BAND[0] * h), int(CROSS_AHEAD_BAND[1] * h)
    ahead = road_b[a0:a1, :]
    if near_walkway.size == 0 or ahead.size == 0:
        return False
    if near_walkway.mean() < CROSS_NEAR_WALKWAY_MIN:
        return False
    if near_road.mean() > CROSS_NEAR_ROAD_MAX:    # you're already on road => trail, not a crossing
        return False
    col_has_road = ahead.any(axis=0)             # a column counts if road anywhere in the band
    return bool(col_has_road.mean() >= CROSS_AHEAD_COVER_FRAC)


def _vehicle_ahead(dets, shape):
    """True if a vehicle's base sits in the 'ahead street' row band.

    Corroborates a geometric crossing: a paved trail reads like a crossing to the
    segmenter, but only a real street has traffic, so this gates out trails.
    """
    if not dets:
        return False
    h = shape[0]
    y0, y1 = CROSS_VEHICLE_BAND[0] * h, CROSS_VEHICLE_BAND[1] * h
    for d in dets:
        if d.get("name") in VEHICLE_NAMES:
            base = d["box"][3]                   # y2 (bottom of the box)
            if y0 <= base <= y1:
                return True
    return False


WARN_RANK = {"watch": 0, "ahead": 1, "stop": 2}


def _warn_kind(o):
    """How serious this object is, or None if it is not worth mentioning."""
    if o.get("collision_risk"):
        return "stop"
    if o.get("urgent"):
        return "ahead"
    if o.get("moving") and (o.get("on_path") or o.get("toward_path")):
        return "watch"
    return None


def _warn_why(o):
    """The clause explaining why an object matters."""
    if o.get("toward_user"):
        return "coming toward you"
    return "on your path" if o.get("on_path") else "crossing into your path"


class Settled:
    """A value that only changes once the replacement has held for a few frames.

    Wording derived from a threshold chatters when the measurement sits near it:
    "toward you" turns on at a heading of 100 deg, so a pedestrian walking almost
    straight at you crosses it repeatedly and the sentence rewrites itself twice a
    second. Real changes -- somebody actually crossing from your left to your right --
    persist, so a short confirmation keeps them and drops the chatter.
    """

    def __init__(self, need):
        self._need = max(1, need)
        self.value = None
        self._cand, self._n = None, 0

    def update(self, v):
        if self.value is None:
            self.value = v
        elif v == self.value:
            self._cand, self._n = None, 0
        elif v == self._cand:
            self._n += 1
            if self._n >= self._need:
                self.value, self._cand, self._n = v, None, 0
        else:
            self._cand, self._n = v, 1
        return self.value


class ObstacleAnnouncer:
    """Keeps an obstacle's warning on screen for as long as the obstacle is there.

    The flags a warning is built from -- collision_risk, urgent -- come from estimates
    that wobble frame to frame: time-to-contact for a pedestrian 7 m out swung between
    2.6 s and 8 s on consecutive frames of IMG_7198, so reading those flags directly
    blinked the text on and off several times a second. Unreadable, and worse than
    silence, because a warning that flashes reads as one that has been withdrawn.

    So the decision to warn is latched per tracked object rather than per frame: once
    something has been worth warning about it keeps its warning while that same track
    is still in view, and only its distance keeps updating. A warning can be upgraded
    (a pedestrian we were merely watching starts closing on us) but never quietly
    downgraded, and it clears only once the object itself is gone.
    """

    def __init__(self, fps):
        self._grace = max(1, int(round(OBST_TEXT_GRACE_S * max(1.0, fps))))
        self._settle = max(1, int(round(OBST_TEXT_SETTLE_S * max(1.0, fps))))
        self._held = {}                      # track id -> {kind, age, det, shown_m, ...}

    def update(self, objects):
        """Fold in this frame's objects. Returns [(kind, object), ...], worst first.
        Each returned object carries a `warn_dist_m`: the distance to print, which is
        held until it has really moved (see OBST_TEXT_STEP_M)."""
        seen = {o["id"]: o for o in objects if o.get("id") is not None}
        for tid, o in seen.items():
            kind = _warn_kind(o)
            if kind is None:
                continue
            prev = self._held.get(tid)
            if prev is not None and WARN_RANK[prev["kind"]] > WARN_RANK[kind]:
                kind = prev["kind"]          # never downgrade; see class docstring
            self._held[tid] = {
                "kind": kind, "age": 0, "det": o,
                "shown_m": (prev or {}).get("shown_m"),
                "side": (prev or {}).get("side") or Settled(self._settle),
                "why": (prev or {}).get("why") or Settled(self._settle)}
        for tid, h in list(self._held.items()):
            if tid in seen:
                h["age"], h["det"] = 0, seen[tid]
                dm = seen[tid].get("distance_m")
                if dm is not None and (h["shown_m"] is None
                                       or abs(dm - h["shown_m"]) >= OBST_TEXT_STEP_M):
                    h["shown_m"] = dm
                h["side"].update(seen[tid].get("side", ""))
                h["why"].update(_warn_why(seen[tid]))
            else:
                h["age"] += 1
                if h["age"] > self._grace:
                    del self._held[tid]
        out = [(h["kind"], dict(h["det"], warn_dist_m=h["shown_m"],
                                warn_side=h["side"].value, warn_why=h["why"].value))
               for h in self._held.values()]
        out.sort(key=lambda kd: (-WARN_RANK[kd[0]],
                                 kd[1].get("distance_m") if kd[1].get("distance_m")
                                 is not None else 1e9))
        return out


class HorizonTracker:
    """Tracks the horizon across frames and reports the camera's real tilt.

    See ADAPT_HORIZON. Single-frame estimates are noisy and go missing whenever the
    sky is not in view, so we keep a short history and take its median, then only
    hand back a new tilt once it has moved by more than HORIZON_RETILT_DEG -- the
    caller rebuilds its warp on that signal, so it should fire on real changes in how
    the camera is being carried, not on estimator jitter.
    """

    def __init__(self, fy_n, proc_h, tilt_deg=None):
        self.fy_n = float(fy_n)
        self.proc_h = int(proc_h)
        self._rows = deque(maxlen=HORIZON_WINDOW)
        self.tilt_deg = float(CAM_TILT_DEG if tilt_deg is None else tilt_deg)
        self.row = None

    def _tilt_from_row(self, row):
        ny = row / float(self.proc_h) - 0.5
        lo, hi = HORIZON_TILT_RANGE
        return min(hi, max(lo, math.degrees(math.atan(-ny / self.fy_n))))

    def update(self, proc_bgr):
        """Fold in one frame. True if the tilt moved enough to rebuild geometry."""
        row = estimate_horizon_row(proc_bgr)
        if row is not None:
            self._rows.append(row)
        if not self._rows:
            return False
        self.row = float(np.median(self._rows))
        tilt = self._tilt_from_row(self.row)
        if abs(tilt - self.tilt_deg) < HORIZON_RETILT_DEG:
            return False
        self.tilt_deg = tilt
        return True


class StickySignal:
    """A warning flag that is slow to raise, then stays up long enough to be read.

    Used for "fork" and "crossing ahead". Both describe a piece of ground you are
    approaching, so the useful behaviour is not to track the detector frame by frame
    but to latch: take a few frames of agreement before raising (one bad mask should
    not announce a junction), hold for WARN_HOLD_S once raised however the detector
    wobbles, and only drop after the evidence has been continuously absent for
    WARN_CLEAR_S. Timings are seconds internally so the same object behaves the same
    at 1.8 FPS on the Pi and at 30 FPS in a rendered clip.
    """

    def __init__(self, fps, hold_s=WARN_HOLD_S, clear_s=WARN_CLEAR_S,
                 rise_s=WARN_RISE_S):
        fps = max(1.0, fps)
        self._hold = max(1, int(round(hold_s * fps)))
        self._clear = max(1, int(round(clear_s * fps)))
        self._rise = max(1 if rise_s <= 0 else WARN_RISE_MIN_FRAMES,
                         int(round(rise_s * fps)))
        self.on = False
        self._evidence = 0        # consecutive frames the detector has agreed
        self._absent = 0          # consecutive frames without evidence
        self._held = 0            # frames since we raised it

    def update(self, raw):
        if self.on:
            self._held += 1
            self._absent = 0 if raw else self._absent + 1
            if self._held >= self._hold and self._absent >= self._clear:
                self.on, self._evidence = False, 0
        else:
            self._evidence = self._evidence + 1 if raw else 0
            if self._evidence >= self._rise:
                self.on, self._held, self._absent = True, 0, 0
        return self.on


class Guidance:
    """Temporal smoothing: EMA on the steering angle + hysteresis on the cue.

    Keeps the spoken/printed guidance from jittering frame-to-frame. The first
    update commits immediately (so single images report a real cue right away).
    """

    def __init__(self, fps=DEFAULT_EFFECTIVE_FPS):
        self.offset = None                       # EMA'd steering angle (per frame)
        self.curve_x = None                      # EMA'd centre-line x samples
        self.curve_y = None                      # EMA'd centre-line y samples
        self._fork_sig = StickySignal(fps)       # see StickySignal / WARN_HOLD_S
        self._cross_sig = StickySignal(fps)
        self.obstacles = ObstacleAnnouncer(fps)  # see ObstacleAnnouncer
        self._line_hold = 0                      # frames we've held the line through a blip
        # decision cadence: commit a directive every `period` frames, then hold it
        self.period = max(1, int(round(DECISION_PERIOD_S * max(1.0, fps))))
        self._t = self.period                    # force a commit on the first frame
        self.committed_offset = 0.0              # the announced (held) bearing
        self.committed_cue = None                # the announced (held) cue

    @staticmethod
    def _cue_for(offset, valid):
        if not valid:
            return "NO PATH"
        if abs(offset) < STRAIGHT_DEG:
            return "STRAIGHT"
        return "BEAR LEFT" if offset < 0 else "BEAR RIGHT"

    def update(self, raw_offset, valid, curve_x, curve_ys, raw_fork,
               raw_crossing=False, hazard=False):
        # Per-frame smoothing of the raw steering angle (kept continuous so the
        # committed value we snapshot on the cadence is already de-noised).
        if valid:
            self.offset = raw_offset if self.offset is None \
                else STEER_EMA_ALPHA * raw_offset + (1 - STEER_EMA_ALPHA) * self.offset

        # Ease the drawn centre-line (both x and y, since the line now ends at the
        # last tracked point and that endpoint can move) toward each new fit
        # instead of snapping to it, so it stops twitching frame-to-frame.
        if valid and curve_x is not None:
            if (self.curve_x is None or self.curve_x.shape != curve_x.shape):
                self.curve_x = np.asarray(curve_x, dtype=np.float32)
                self.curve_y = np.asarray(curve_ys, dtype=np.float32)
            else:
                self.curve_x = CURVE_EMA_ALPHA * curve_x + (1 - CURVE_EMA_ALPHA) * self.curve_x
                self.curve_y = CURVE_EMA_ALPHA * curve_ys + (1 - CURVE_EMA_ALPHA) * self.curve_y
            self._line_hold = 0
            curve_pts = [(int(x), int(y)) for x, y in zip(self.curve_x, self.curve_y)]
        elif self.curve_x is not None and self._line_hold < LINE_HOLD_FRAMES:
            self._line_hold += 1                 # brief path blip: keep the last line
            curve_pts = [(int(x), int(y)) for x, y in zip(self.curve_x, self.curve_y)]
        else:
            self.curve_x = self.curve_y = None
            curve_pts = []

        fork = self._fork_sig.update(raw_fork)
        crossing = self._cross_sig.update(raw_crossing)

        # Decision cadence: re-commit the announced bearing/cue every `period`
        # frames (~DECISION_PERIOD_S), holding it in between so a slow-walking
        # human gets a steady directive instead of a number that changes several
        # times a second. Four things override the hold, because waiting out the
        # window would mean withholding something the user needs now: the steering
        # having moved materially, the path appearing or disappearing, the very
        # first frame, and any real change of plan while a hazard is closing on us.
        self._t += 1
        raw = self.offset or 0.0
        # The announced bearing is the planned line's own heading, always. It used to
        # be overridden while something was closing on us -- pick the side the hazard
        # is not on, hold it -- and that was wrong in a way the pictures made obvious:
        # in IMG_7201 the plan curved hard left around a pedestrian while the override
        # announced "bear right" purely because their bearing read -6.8 deg, and in
        # IMG_7204 it clamped a 24 deg avoiding curve down to 10 deg. A directive that
        # contradicts the line cannot be followed, and between the two the line is the
        # one that knows where the ground is: it is chosen against a metric costmap
        # that has the obstacle inflated into it, not from a single bearing. So the
        # hazard no longer steers. What it still does is speak (see ObstacleAnnouncer)
        # and cut short the decision cadence, so a change of plan caused by something
        # closing on you is announced now rather than up to a second later.
        new_cue = self._cue_for(raw, valid)
        moved = abs(raw - self.committed_offset) >= RECOMMIT_DELTA_DEG
        appeared = (self.committed_cue == "NO PATH") != (new_cue == "NO PATH")
        urgent_change = bool(hazard) and (
            new_cue != self.committed_cue
            or abs(raw - self.committed_offset) >= HAZARD_RECOMMIT_DEG)
        if self._t >= self.period or moved or appeared or urgent_change:
            self._t = 0
            if abs(raw) < STRAIGHT_DEG:
                self.committed_offset = 0.0
            else:
                self.committed_offset = float(round(raw / DIRECTIVE_QUANT_DEG)
                                              * DIRECTIVE_QUANT_DEG)
            self.committed_cue = new_cue

        return self.committed_offset, self.committed_cue, curve_pts, fork, crossing


def _warnings_text(fork, crossing, warned):
    """Compose the warning line. `warned` is ObstacleAnnouncer.update()'s output,
    already latched and sorted worst-first, so this is pure formatting -- nothing
    here decides whether a warning appears or vanishes."""
    parts = []
    if crossing:
        parts.append("CROSSING AHEAD")
    if fork:
        parts.append("FORK")

    stop = [d for kind, d in warned if kind == "stop"]
    ahead = [d for kind, d in warned if kind == "ahead"]
    watch = [d for kind, d in warned if kind == "watch"]

    if stop:
        d = stop[0]
        parts.append(f"STOP: {d['name']} {_dist_str(d)} "
                     f"{d.get('warn_side') or ''}, "
                     f"{d.get('warn_why') or _warn_why(d)}".strip())
    elif ahead:
        names = sorted({f"{d['name']} {_dist_str(d)}".strip() for d in ahead})
        parts.append("ahead: " + ", ".join(names))

    if watch:
        w = watch[0]
        parts.append(f"watch: {w['name']} {w.get('warn_side') or ''}".strip())
    return " | ".join(parts)


def _put_text(vis, text, org, scale, color, thickness=1):
    """Draw text with a black contour so bright labels stay readable over any
    background (sky, pavement, foliage). The outline is a thicker black stroke
    drawn underneath the coloured glyphs."""
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


# Neon label palette (BGR) -- picked to pop against outdoor scenes.
NEON_CUE = (0, 255, 255)      # neon yellow  : main steering cue
NEON_WARN = (0, 128, 255)     # neon orange  : warnings banner
NEON_STOP = (0, 0, 255)       # neon red     : urgent / STOP obstacle
NEON_AHEAD = (0, 200, 255)    # neon amber   : on-path but distant obstacle
NEON_OFF = (200, 255, 0)      # neon cyan-green: off-path / not relevant


def _heading_arrow(heading_deg):
    """ASCII arrow for a heading (see tracking.py convention). Font-safe."""
    if heading_deg is None:
        return ""
    a = heading_deg
    if -45 <= a <= 45:
        return "^"                               # away from you
    if a >= 135 or a <= -135:
        return "v"                               # toward you
    return ">" if a > 0 else "<"


def _dist_str(d):
    """Metric distance like '2.4m' when available, else the coarse zone word.

    Prefers the held value the announcer stamped on, so a warning that stays up does
    not redraw itself every other frame just because the range wobbled by 10 cm.
    """
    dm = d.get("warn_dist_m") if d.get("warn_dist_m") is not None else d.get("distance_m")
    return f"{dm:.1f}m" if dm is not None else d.get("distance_bin", "")


def _object_label(d):
    """Compact overlay label, e.g. 'person#3 D 2.4m v' (id, kind, distance, heading)."""
    kind = "D" if d.get("kind") == "dynamic" else "S"
    oid = d.get("id")
    tag = f"{d['name']}#{oid}" if oid is not None else d["name"]
    parts = [tag, kind, _dist_str(d)]
    arrow = _heading_arrow(d.get("heading_deg")) if d.get("moving") else ""
    if arrow:
        parts.append(arrow)
    return " ".join(p for p in parts if p)


LABEL_SCALE = 0.4                # object label text scale
BANNER_RESERVE_PX = 46           # top band used by the cue + warning banners


def _label_org(x1, y1, y2, tw, th, w, h, reserve_top):
    """Pick a label origin (cv2 baseline point) that stays on-frame and clears
    the top banner band. Prefer just above the box; if that would collide with
    the banners, drop it just below the box instead."""
    lx = max(2, min(x1, w - tw - 2))             # keep the text inside the frame
    above = y1 - 4                               # baseline sits above the box top
    if above - th < reserve_top:                 # would hit the top banners
        below = y2 + th + 4                       # place under the box instead
        ly = below if below <= h - 2 else max(reserve_top + th, min(above, h - 2))
    else:
        ly = above
    return lx, ly


def draw(proc_bgr, walkable, road, cue, offset_deg, curve_pts, fork, crossing, dets,
         aim_frac=None, warned=()):
    vis = proc_bgr.copy()
    h, w = vis.shape[:2]
    overlay = vis.copy()
    overlay[walkable > 0] = (0, 255, 0)          # walkway = green
    overlay[road > 0] = (255, 160, 0)            # road = blue (crossing surface)
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

    # The line is drawn to the full planning depth, but the spoken directive only
    # describes the path as far as the lookahead. Perspective makes the far half of
    # the line bend much harder than the near half, so draw the announced stretch
    # solid and the preview beyond it thin: a "STRAIGHT" cue under a line whose tip
    # curves away then reads as what it is, rather than as a contradiction.
    aim_i = (len(curve_pts) - 1 if aim_frac is None
             else int(round(aim_frac * (len(curve_pts) - 1))))
    for i in range(len(curve_pts) - 1):
        near = i < aim_i
        cv2.line(vis, curve_pts[i], curve_pts[i + 1], (255, 0, 0),
                 2 if near else 1)
    for c in curve_pts[:aim_i:4]:
        cv2.circle(vis, c, 2, (255, 255, 0), -1)
    if 0 < aim_i < len(curve_pts):
        cv2.circle(vis, curve_pts[aim_i], 4, (255, 255, 0), 1)

    # The instruction, drawn as its own arrow from underfoot. It is the heading of the
    # planned line at the lookahead, so it points along the solid stretch of that line
    # by construction -- the arrow is there to make the committed ANGLE legible (a 24
    # deg avoiding curve versus a 4 deg drift is hard to read off a line in
    # perspective), not to say anything the line does not.
    a = math.radians(float(offset_deg))
    base, ln = (w // 2, h - 4), h * 0.22
    tip = (int(round(base[0] + math.sin(a) * ln)), int(round(base[1] - math.cos(a) * ln)))
    cv2.arrowedLine(vis, base, tip, (0, 0, 0), 5, tipLength=0.3)
    cv2.arrowedLine(vis, base, tip, (0, 255, 255), 2, tipLength=0.3)

    warn = _warnings_text(fork, crossing, warned)
    reserve_top = BANNER_RESERVE_PX if warn else 26   # only cue banner if no warning

    for d in dets or []:
        x1, y1, x2, y2 = d["box"]
        if d.get("urgent"):
            color = NEON_STOP                    # neon red: on path AND close -> STOP
        elif d.get("on_path"):
            color = NEON_AHEAD                   # neon amber: on path but distant
        else:
            color = NEON_OFF                     # neon cyan-green: off path / not relevant now
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, ((x1 + x2) // 2, y2), 3, color, -1)   # bottom-centre test point
        label = _object_label(d)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, 1)
        org = _label_org(x1, y1, y2, tw, th, w, h, reserve_top)
        _put_text(vis, label, org, LABEL_SCALE, color, 1)

    _put_text(vis, f"{cue} ({offset_deg:+.0f}deg)", (8, 20), 0.6, NEON_CUE, 2)
    if warn:
        _put_text(vis, warn, (8, 40), 0.45, NEON_WARN, 1)
    return vis


def plan_path(planner, walkable, objects):
    """Plan the guide line, preferring the metric costmap planner.

    Returns the same tuple shape as analyze() plus the PlanResult (None when the
    planner was unavailable or came up empty and we fell back to the mask-only
    centre-line). Obstacles must already be enriched with distance/size, which is
    why this runs *after* enrich_objects -- their metres are what the costmap is
    built from.
    """
    if planner is not None and USE_BEV_PLANNER:
        plan = planner.plan(walkable, objects)
        if plan.valid:
            frac = float((walkable > 0).mean())
            fork = plan.fork if FORK_FROM_PLANNER else detect_fork(walkable)
            return (plan.offset_deg, True, plan.curve_x, plan.curve_ys,
                    fork, frac, plan)
        return analyze(walkable) + (plan,)
    return analyze(walkable) + (None,)


def process_frame(segmenter, detector, guidance, proc, tracker=None, t=0.0,
                  ground=None, planner=None, show_bev=None, horizon=None):
    """Run both models + guidance on one BGR frame. Return (vis, info).

    tracker: an ObjectTracker (persists across frames) that links detections into
    tracks so we can classify static/dynamic and derive heading. t is the frame
    timestamp (seconds) used for that motion history. ground: a GroundPlane for
    metric distance/size (None -> image-cue distance bins only). planner: a
    BevPlanner that turns the mask + ranged obstacles into a planned line
    (None -> mask-only centre-line).
    """
    # Re-aim the camera model at the horizon we can actually see (see ADAPT_HORIZON)
    # before anything that depends on the geometry runs.
    if horizon is not None and ground is not None and horizon.update(proc):
        ground.set_tilt(horizon.tilt_deg)
        if planner is not None:
            planner.refresh_geometry()
    if ground is not None:
        segmenter.set_horizon(ground.horizon_ny(), proc.shape[0])
    t0 = time.perf_counter()
    walkable, road, road_full = segmenter(proc)
    dets = detector(proc) if detector else []
    ms = (time.perf_counter() - t0) * 1000.0

    objects = tracker.update(dets, t) if tracker is not None else dets
    # Range/size the obstacles FIRST: the planner's costmap is built from their
    # metres, so this ordering is what lets obstacles steer the line rather than
    # only trigger warnings.
    on_path, urgent = enrich_objects(objects, walkable, ground)
    dets = objects
    t1 = time.perf_counter()
    offset_raw, valid, curve_x, curve_ys, fork_raw, frac, plan = \
        plan_path(planner, walkable, objects)
    plan_ms = (time.perf_counter() - t1) * 1000.0
    # Crossing/junction from the corridor geometry (planner.CROSS_FAR_Z), falling
    # back to the old road-class test only when there is no plan to read it from.
    if plan is not None:
        crossing_raw = plan.opening
    else:
        crossing_raw = (detect_crossing(walkable, road_full)
                        and _vehicle_ahead(dets, walkable.shape))
    # Latch the obstacle warnings before drawing, so the text is decided by whether
    # the obstacle is still there rather than by this frame's flags.
    warned = guidance.obstacles.update(objects)
    offset, cue, curve_pts, fork, crossing = guidance.update(
        offset_raw, valid, curve_x, curve_ys, fork_raw, crossing_raw,
        any(kind == "stop" for kind, _ in warned))

    # Blue marks the open ground when a crossing/junction is called. It is drawn from
    # the walkable mask, not SegFormer's "road" class: the decision no longer uses
    # that class (see planner.CROSS_FAR_Z), and shading a region that had no part in
    # the decision is how the overlay came to disagree with the warning -- the road
    # class flickering on was read as a street appearing out of nowhere.
    crossing_overlay = np.zeros_like(walkable)
    if crossing:
        mh = walkable.shape[0]
        a0, a1 = int(CROSS_AHEAD_BAND[0] * mh), int(CROSS_AHEAD_BAND[1] * mh)
        crossing_overlay[a0:a1, :] = walkable[a0:a1, :]
    vis = draw(proc, walkable, crossing_overlay, cue, offset, curve_pts, fork,
               crossing, dets, aim_frac=(plan.aim_frac if plan is not None else None),
               warned=warned)
    if ground is not None:                           # faint horizon line (sanity check)
        hy = int(round((ground.horizon_ny() + 0.5) * proc.shape[0]))
        if 0 <= hy < proc.shape[0]:
            cv2.line(vis, (0, hy), (proc.shape[1] - 1, hy), (120, 120, 120), 1)

    planned = plan is not None and plan.valid
    if planned:
        _put_text(vis, f"plan conf {plan.confidence:.2f}  {plan.reach_z_m:.1f}m",
                  (8, proc.shape[0] - 8), 0.35, NEON_CUE, 1)
    if (SHOW_BEV if show_bev is None else show_bev) and plan is not None:
        vis = np.hstack([vis, planner.render(plan, proc.shape[0])])

    info = {"ms": ms, "plan_ms": plan_ms, "cue": cue, "offset": offset,
            "frac": frac, "fork": fork, "crossing": crossing, "dets": dets,
            "on_path": on_path, "urgent": urgent, "planned": planned,
            "confidence": (plan.confidence if planned else 0.0),
            "reach_m": (plan.reach_z_m if planned else 0.0),
            "kappa": (plan.kappa if planned else 0.0),
            "warn": _warnings_text(fork, crossing, warned)}
    return vis, info


def make_horizon(ground):
    """HorizonTracker for the processing frame, or None if adaptation is off."""
    if not ADAPT_HORIZON:
        return None
    return HorizonTracker(ground.fy_n, PROC_H)


def make_planner(ground):
    """BevPlanner for the processing frame, or None if the geometry is unusable."""
    if not USE_BEV_PLANNER:
        return None
    try:
        return BevPlanner(ground, PROC_W, PROC_H)
    except Exception as e:                            # bad tilt/FOV -> fall back
        print(f"(note) metric planner unavailable ({e}); using mask centre-line.")
        return None


def run_image(segmenter, detector, path, show_bev=False):
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Could not read image {path!r}")
    proc = cv2.resize(frame, (PROC_W, PROC_H))
    segmenter.reset()
    ground = make_ground(frame.shape[1], frame.shape[0])
    vis, info = process_frame(segmenter, detector, Guidance(), proc,
                              tracker=ObjectTracker(), t=0.0, ground=ground,
                              planner=make_planner(ground), show_bev=show_bev,
                              horizon=make_horizon(ground))
    cv2.imwrite(SAVE_PATH, vis)
    warn = f"  | {info['warn']}" if info["warn"] else ""
    src = "plan" if info["planned"] else "mask"
    print(f"[{info['ms']:5.0f} ms] {info['cue']:11s} offset={info['offset']:+5.0f}deg  "
          f"walkable={info['frac']*100:4.1f}%  {src} conf={info['confidence']:.2f} "
          f"reach={info['reach_m']:.1f}m{warn}  -> {SAVE_PATH}")


def run_camera(segmenter, detector, source, show_bev=False):
    cap = open_camera(source)
    mode = "SegFormer + YOLOv8n" if detector else "SegFormer only"
    print(f"Path-nav ({mode}) running (Ctrl+C to quit). Writing {SAVE_PATH}")
    guidance = Guidance()
    tracker = ObjectTracker()
    ground = make_ground(CAP_W, CAP_H)
    planner = make_planner(ground)
    horizon = make_horizon(ground)
    frame_idx, t_last = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            proc = cv2.resize(frame, (PROC_W, PROC_H))
            vis, info = process_frame(segmenter, detector, guidance, proc,
                                      tracker=tracker, t=time.time(), ground=ground,
                                      planner=planner, show_bev=show_bev,
                                      horizon=horizon)

            frame_idx += 1
            if frame_idx % SAVE_EVERY == 0:
                now = time.time()
                fps = SAVE_EVERY / (now - t_last)
                t_last = now
                cv2.imwrite(SAVE_PATH, vis)
                warn = f"  | {info['warn']}" if info["warn"] else ""
                print(f"[{fps:4.1f} FPS] {info['cue']:11s} "
                      f"offset={info['offset']:+5.0f}deg  walkable={info['frac']*100:4.1f}%{warn}")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        cap.release()


def _open_writer(fps, size, out_base=VIDEO_OUT):
    """Open a VideoWriter, falling back from mp4 to avi if the codec is missing."""
    stem = os.path.splitext(out_base)[0]
    for fourcc, path in ((cv2.VideoWriter_fourcc(*"mp4v"), stem + ".mp4"),
                         (cv2.VideoWriter_fourcc(*"XVID"), stem + ".avi")):
        writer = cv2.VideoWriter(path, fourcc, fps, size)
        if writer.isOpened():
            return writer, path
        writer.release()
    return None, None


def _round_or_none(v, places):
    return round(float(v), places) if v is not None else None


def _object_log(o):
    """Serializable per-object record for the JSONL export (distance + direction
    focused; no speed number, per the Phase A spec)."""
    heading = o.get("heading_deg")
    dm = o.get("distance_m")
    size = o.get("size_m")
    return {
        "id": o.get("id"),
        "name": o.get("name"),
        "kind": o.get("kind"),
        "box": [int(v) for v in o.get("box", (0, 0, 0, 0))],
        "distance_m": (round(float(dm), 2) if dm is not None else None),
        "size_m": ([round(float(size[0]), 2), round(float(size[1]), 2)]
                   if size is not None else None),
        "distance_bin": o.get("distance_bin"),
        "bearing_deg": round(float(o.get("bearing_deg", 0.0)), 1),
        "side": o.get("side"),
        "heading_deg": (round(float(heading), 1) if heading is not None else None),
        "moving": bool(o.get("moving", False)),
        "on_path": bool(o.get("on_path", False)),
        "toward_user": bool(o.get("toward_user", False)),
        "toward_path": bool(o.get("toward_path", False)),
        "collision_risk": bool(o.get("collision_risk", False)),
        "urgent": bool(o.get("urgent", False)),
        # How fast the gap is closing and how long until it shuts (HAZARD_TTC_S).
        # Without these in the log there is no way to tell a hazard that was never
        # seen from one that was seen and judged not close enough to mention.
        "closing_ms": _round_or_none(o.get("closing_ms"), 2),
        "ttc_s": _round_or_none(o.get("ttc_s"), 1),
    }


def _frame_log(idx, t_sec, info):
    return {
        "frame": idx,
        "t_sec": round(t_sec, 3),
        "cue": info["cue"],
        "offset_deg": round(float(info["offset"]), 1),
        "fork": bool(info["fork"]),
        "crossing": bool(info["crossing"]),
        # Where the line came from and how sure the planner was -- the fields to
        # look at when a cue disagrees with what the clip shows.
        "planned": bool(info.get("planned", False)),
        "confidence": round(float(info.get("confidence", 0.0)), 2),
        "reach_m": round(float(info.get("reach_m", 0.0)), 2),
        "kappa": round(float(info.get("kappa", 0.0)), 3),
        "objects": [_object_log(o) for o in info["dets"]],
    }


STILL_MIN_GAP = 12               # processed frames between saved stills
STILL_MAX = 40                   # cap on stills per clip
STILL_NEAR_M = 4.0               # save when an on-path obstacle is closer than this


def _still_worth_saving(info):
    """True on frames that show the planner reacting to something -- an urgent
    obstacle, or anything on the path within STILL_NEAR_M. These are the frames
    worth looking at; the rest of a walking clip is empty path."""
    if info.get("urgent"):
        return True
    for d in info.get("dets") or []:
        dm = d.get("distance_m")
        if d.get("on_path") and dm is not None and dm <= STILL_NEAR_M:
            return True
    return False


def run_video(segmenter, detector, path, stride=1, export_path=None, show_bev=False,
              stills_dir=None):
    """Replay a recorded walk: annotate every stride-th frame and log cue changes.

    export_path: optional JSONL file; one line per processed frame with the full
    tracked-object state (for offline review and future Phase D training data).
    stills_dir: optional directory to drop annotated PNGs of the frames where an
    obstacle is actually on the path, so they can be inspected without scrubbing
    the whole clip.
    """
    if not os.path.exists(path):
        raise SystemExit(f"Video not found: {path!r}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video {path!r}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 1e-3:
        src_fps = 15.0
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or PROC_W
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or PROC_H
    ground = make_ground(ow, oh)                 # metric ranging uses true frame aspect
    planner = make_planner(ground)
    horizon = make_horizon(ground)               # measures the real tilt per clip
    out_fps = max(1.0, src_fps / max(1, stride))
    # Derive the output name from the input so replaying several clips in a row
    # keeps a separate annotated_<name>.mp4 for each instead of clobbering one file.
    out_base = "annotated_" + os.path.basename(path)
    # Opened on the first annotated frame, because the debug panel changes the size.
    writer, out_path = None, None

    segmenter.reset()                            # fresh temporal state per clip
    if planner is not None:
        planner.reset()                          # don't carry a curvature between clips
    export_f = open(export_path, "w") if export_path else None
    print(f"Replaying {path} (src ~{src_fps:.0f} FPS, stride {stride})"
          + (f"  (+ {export_path})" if export_path else ""))
    guidance = Guidance(fps=out_fps)             # cadence in real (wall-clock) time
    tracker = ObjectTracker()
    idx, done, ms_sum, plan_ms_sum, last_key = 0, 0, 0.0, 0.0, None
    stem = os.path.splitext(os.path.basename(path))[0]
    if stills_dir:
        os.makedirs(stills_dir, exist_ok=True)
    n_stills, last_still = 0, -STILL_MIN_GAP
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride != 0:
                idx += 1
                continue
            proc = cv2.resize(frame, (PROC_W, PROC_H))
            t = idx / src_fps                    # timestamp drives motion history
            vis, info = process_frame(segmenter, detector, guidance, proc,
                                      tracker=tracker, t=t, ground=ground,
                                      planner=planner, show_bev=show_bev,
                                      horizon=horizon)
            if writer is None:
                writer, out_path = _open_writer(out_fps, (vis.shape[1], vis.shape[0]),
                                                out_base)
                if writer is None:
                    raise SystemExit("Could not open a VideoWriter "
                                     "(no mp4v/XVID codec available).")
                print(f"  -> {out_path}")
            writer.write(vis)
            done += 1
            ms_sum += info["ms"]
            plan_ms_sum += info["plan_ms"]

            if export_f is not None:
                export_f.write(json.dumps(_frame_log(idx, t, info)) + "\n")

            if (stills_dir and n_stills < STILL_MAX
                    and done - last_still >= STILL_MIN_GAP
                    and _still_worth_saving(info)):
                cv2.imwrite(os.path.join(stills_dir, f"{stem}_f{idx:05d}.png"), vis)
                n_stills, last_still = n_stills + 1, done

            # event log: print only when the cue or warning text changes (TTS-style)
            key = (info["cue"], info["warn"])
            if key != last_key:
                warn = f"  | {info['warn']}" if info["warn"] else ""
                print(f"[t={t:6.1f}s] {info['cue']:11s} offset={info['offset']:+5.0f}deg{warn}")
                last_key = key
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if export_f is not None:
            export_f.close()
    avg = ms_sum / done if done else 0.0
    plan_avg = plan_ms_sum / done if done else 0.0
    fps_txt = f"~{1000.0 / avg:.1f} FPS" if avg else "n/a"
    stills_txt = f", {n_stills} obstacle stills" if stills_dir else ""
    print(f"\nDone: {done} frames processed (avg {avg:.0f} ms/frame models "
          f"+ {plan_avg:.1f} ms planning, {fps_txt}). Wrote {out_path}{stills_txt}")


def load_detector():
    """Return an ObstacleDetector if yolov8n.onnx exists, else None (path-only)."""
    if not os.path.exists(YOLO_PATH):
        print(f"(note) {YOLO_PATH} not found -- running path segmentation only. "
              "Export it with: yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12")
        return None
    return ObstacleDetector()


def main(argv):
    args = argv[1:]
    show_bev = "--bev" in args
    if show_bev:
        args = [a for a in args if a != "--bev"]
    segmenter = WalkableSegmenter()
    detector = load_detector()

    if args and args[0] == "--image":
        if len(args) < 2:
            raise SystemExit("Usage: python3 path_nav.py --image PATH [--bev]")
        run_image(segmenter, detector, args[1], show_bev)
        return
    if args and args[0] == "--video":
        if len(args) < 2:
            raise SystemExit("Usage: python3 path_nav.py --video PATH "
                             "[--stride N] [--export tracks.jsonl] [--bev]")
        stride = 1
        if "--stride" in args:
            stride = max(1, int(args[args.index("--stride") + 1]))
        export_path = None
        if "--export" in args:
            export_path = args[args.index("--export") + 1]
        run_video(segmenter, detector, args[1], stride, export_path, show_bev)
        return

    source = DEVICE
    if args:
        source = int(args[0]) if args[0].isdigit() else args[0]
    run_camera(segmenter, detector, source, show_bev)


if __name__ == "__main__":
    main(sys.argv)
