#!/usr/bin/env python3
"""Metric path planning: decide where to walk, don't just average the mask.

Why this exists
---------------
`analyze()` in path_nav.py builds the guide line by taking, row by row in the
IMAGE, the centre-of-mass of walkable pixels. Two problems with that:

  1. It never sees obstacles. A person standing in the middle of the path does
     not move the line at all -- they only become a spoken warning. The line
     runs straight through them.
  2. It reasons in pixels. Two obstacles the same number of pixels left/right of
     centre are at completely different real distances depending on how high up
     the frame they sit, so avoidance logic bolted onto an image-space scan is
     geometrically wrong.

So we do it the way a robot does. `GroundPlane` already gives the real (X, Z)
metres of any ground pixel, so:

    image  ->  bird's-eye metric grid   (one fixed homography, precomputed)
           ->  cost per cell            (non-walkable, obstacle footprints,
                                         clearance, predicted motion)
           ->  score a fan of walkable arcs, keep the cheapest
           ->  unproject the winner back to image pixels to draw

Costs rather than booleans is the important part: a cell 20 cm from a wall is
legal but bad, so the winning line naturally runs down the middle of the
corridor and swings wide around a pedestrian instead of grazing them.

Why arcs instead of a grid search
---------------------------------
Every candidate is a constant-curvature path starting at your feet pointing
forward, so the winner is smooth and walkable by construction -- grid A* gives
staircase paths that need smoothing afterwards and can propose turns no walking
human would make. Three things also fall out of the search for free:

  * the winner's curvature IS the steering command,
  * the gap between best and second-best score is a confidence,
  * two well-separated good minima mean the corridor really does branch.

Cost: the arc-sample -> grid-index mapping is fixed, so per frame this is one
warpPerspective, one distanceTransform and a fancy-index -- a few ms at the
default 60x80 grid, against ~300 ms for SegFormer. Free in practice.

Conventions match ground_plane.py: metres, degrees, X right, Z forward, and a
negative steering angle means left.
"""
import math

import cv2
import numpy as np

# --- Bird's-eye grid -------------------------------------------------------
# 10 m wide x 14 m deep at 10 cm cells = 100 x 140 cells.
#
# We still only PLAN through the first ~6 m -- ground-plane range error grows fast
# with depth and a walking human needs only a few seconds of lookahead -- but the
# grid has to extend well past that to see a junction coming. At the camera's 18.4
# deg downward tilt a path fork sits 10-15 m out while it is still comfortably in
# frame, so an 8 m grid could not see a single one of the real forks in these clips
# (IMG_7203 has an obvious grass island splitting the gravel that fell entirely
# outside it). The width has to grow with the depth for the same reason: the 21.5
# deg half-angle view spans +/- 4.7 m at 12 m, so a +/- 3 m grid would clip the
# branches off. The extra cells cost the warp and the distance transform only, both
# of which are still around a millisecond.
BEV_HALF_W_M = 5.0
BEV_DEPTH_M = 14.0
BEV_CELL_M = 0.10
GRID_W = int(round(2.0 * BEV_HALF_W_M / BEV_CELL_M))
GRID_D = int(round(BEV_DEPTH_M / BEV_CELL_M))
# Everything at or just below the horizon row is unusable (a pixel there maps to
# a near-infinite distance), so we exclude that band from the warp entirely.
HORIZON_MARGIN_PX = 2

# --- Cell costs ------------------------------------------------------------
COST_BLOCKED = 1000.0        # observed non-walkable, or an obstacle footprint
# Cells no image pixel maps to: outside the field of view, which on a portrait
# clip is most of the grid's width close in (the narrow sensor axis is the
# horizontal one, so at 2 m we see barely +/- 0.8 m of a 6 m-wide grid).
# Deliberately a moderate preference, not a near-prohibition. Any real sideways
# manoeuvre leaves the visible cone, so pricing unknown ground too high made
# walking straight into an obstacle score better than going around it.
COST_UNKNOWN = 9.0
BODY_HALF_W_M = 0.30         # half shoulder width: less clearance than this = you don't fit
# Clearance beyond this earns no penalty, i.e. how much room counts as "plenty".
# There is a real trade-off here. Too small (0.90 m) and the term goes flat across
# any normal path, leaving nothing to separate the candidate arcs, so the line
# wanders (confidence averaged 0.27 on IMG_7204). Too large (2.00 m) and the term
# keeps pulling toward the widest available ground, which on an open field means the
# plan tracks the shape of the mask -- including whatever grass the mask wrongly
# swallowed -- and bends the same way for a whole clip (IMG_7199 read BEAR LEFT on
# 96% of frames). 1.2 m keeps you off the edges of a genuine 2-3 m path without
# chasing width beyond what a person needs; TIE_SLACK below handles the flat case
# properly instead.
CLEAR_PREF_M = 1.20
COST_CLEARANCE = 14.0        # weight of the "keep away from edges" term

# --- Obstacle footprints ---------------------------------------------------
OBST_MIN_HALF_W_M = 0.20     # floor on the stamped half-width (thin poles still block)
OBST_MAX_HALF_W_M = 1.60     # ceiling, so a bad size_m estimate can't wall off the grid
OBST_DEPTH_M = 0.45          # assumed front-to-back extent of a footprint
# Ground-plane range error grows with distance, so a far obstacle's footprint is
# padded more than a near one's: it might not be exactly where we think it is.
RANGE_PAD_FRAC = 0.10

# --- Predicted motion (dynamic layer) --------------------------------------
# tracking.py's headings are NOT ego-motion compensated, so a predicted position
# is a rough guess, not a trajectory. We therefore stamp a soft, widening cost
# rather than a hard block, and only look a couple of seconds ahead.
PRED_HORIZON_S = 1.6
PRED_STEPS = 3
PRED_PAD_STEP_M = 0.22       # footprint grows this much per prediction step
COST_PREDICTED = 55.0        # soft cost of "something may be here shortly"
PRED_MAX_SPEED_MS = 3.5      # reject absurd velocities from a bad projection
PRED_MIN_SPEED_MS = 0.25     # below this it isn't really going anywhere

# --- Candidate arcs --------------------------------------------------------
N_ARCS = 31                  # odd, so one candidate is exactly straight ahead
KAPPA_MAX = 0.30             # 1/m; ~3.3 m turn radius at full lock
ARC_LEN_M = 6.0              # arclength each candidate is scored over
ARC_SAMPLES = 30
PLAN_TARGET_Z_M = 5.0        # forward depth we'd like to reach; shortfall is penalised
# Getting somewhere is the primary objective, so this deliberately dominates the
# per-cell terms: a route that dead-ends two metres early must score worse than
# any detour that actually gets through, however scruffy the detour's cells are.
COST_PROGRESS = 25.0         # penalty per metre of target depth not reached
COST_CURVATURE = 5.0         # per 1/m; mild preference for going straight
# Among arcs scoring within this of the best, take the straightest one. On open
# ground many arcs are near-tied and the winner is then decided by mask noise, which
# is how a clip of a wide flat field ended up being told to bear left continuously.
# Making the tie-break explicit says the intended thing directly -- do not instruct
# someone to turn unless turning is clearly better -- rather than trying to buy the
# same behaviour by tuning weights against each other. It is deliberately small
# relative to the progress penalty (25 per metre short), so a genuine obstruction or
# a bend that runs an arc out of walkable ground still wins easily.
#
# Swept against real frames (debug_frame.py --sweep) rather than guessed, because
# the useful range is narrow: at 2.0 the real left-hand curve in IMG_7202 was flattened
# from 11.5 deg to 5.7 and stopped being announced at all, while at 0.5 it survives
# at 8.5 deg and the genuinely straight stretch of IMG_7205 still resolves to straight.
TIE_SLACK = 0.5
# Continuity with last frame's plan. Raised because on a wide open path many arcs
# score within noise of each other, and a weak continuity term let the winner hop
# between them frame to frame -- the "messy line" symptom. This is the principled
# place to damp that: inside the objective, rather than by smoothing the drawn line
# afterwards, which would only hide the jitter behind lag.
COST_PREV_KAPPA = 8.0        # per 1/m of change from last frame's plan
# A plan must reach this much farther than the nearest ground the camera can
# actually see (see _build_visibility) to count as a way forward. Measuring
# progress from the visible edge rather than from an absolute depth is what stops
# an arc that is walled off immediately from looking like it went somewhere.
MIN_PROGRESS_M = 1.0
# Steering is read off a point this far along the chosen arc, not off its far end.
# The far end of a 6 m arc exaggerates the turn badly -- an arc that only needs to
# clear a bin at 3 m ends up pointing 40 deg off, and a human told "bear right 40"
# will over-turn.
#
# 2.5 m proved too short in practice, though: at walking pace it describes under two
# seconds of path, so a bend starting three metres out was still being announced as
# STRAIGHT while the drawn line visibly curved, which reads as the directive
# contradicting the picture. 3.5 m covers about two and a half seconds -- far enough
# to name the bend you are walking into, near enough to still be about the next few
# steps rather than the far end of the plan.
LOOKAHEAD_S_M = 3.5
# ...but report the arc's TANGENT heading there (kappa * s), not the straight-line
# bearing to it. For a circular arc the chord bearing is almost exactly half the
# turn, so reporting the chord silently halved every directive: across the eight
# test clips the mean chosen curvature was 0.070 /m (a real, visible bend) while
# the announced angle averaged 2.2 deg, inside the STRAIGHT deadband. 79% of frames
# were announced STRAIGHT on paths that were plainly curving. The tangent is also
# the more natural instruction -- it is the direction the path is heading, and
# because every arc starts at your feet a lateral offset from the corridor centre
# still shows up as curvature, so corrections are not lost either.
CONF_SCALE = 25.0            # score gap that counts as full confidence
FORK_MIN_SEP_ARCS = 6        # score minima this many candidates apart are distinct

# --- Corridor opening (a crossing or junction ahead) ------------------------
# A street crossing, a forecourt and a T-junction all look the same from a walking
# camera: the corridor you are following stops being a corridor and opens out to the
# sides. That is a measurement the metric grid can make directly, which is what lets
# us drop the old test built on SegFormer's "road" class -- that class flickers on and
# off between frames, and every flicker read as a street appearing across the path.
#
# The measure is the fraction of the GROUND WE CAN SEE at a given depth that is
# walkable, not a width in metres, because the camera's 43 deg view spans only 1.6 m
# at 2 m but 5.5 m at 7 m. Raw widths therefore grow with depth no matter what the
# ground does, whereas the fraction does not: an ordinary 2 m path fills most of the
# view up close and about a third of it at 7 m, while ground that is open right across
# the view at 7 m really is open.
CROSS_FAR_Z = (5.0, 9.0)     # depth band that must be open (m)
CROSS_OPEN_FRAC = 0.90       # this much of the visible ground walkable => open
CROSS_MIN_OPEN_W_M = 5.0     # and at least this much walkable width, absolutely
CROSS_DEPTH_M = 1.5          # sustained over this much depth, contiguously

# --- Fork detection, in metres ---------------------------------------------
# The old detector scanned image rows for two walkable runs separated by a gap,
# with thresholds as a fraction of image WIDTH. In an image a fixed pixel gap means
# centimetres near your feet and metres up by the horizon, so a shadow notch high
# in the frame looked exactly like a real branch -- forks fired on 80% of IMG_7198
# and 55-57% of IMG_7200/7203, where there are none. In the bird's-eye grid the
# thresholds are real widths, so a branch has to be something a person could
# actually walk down, and the dividing wedge has to be genuinely non-walkable
# ground rather than merely unobserved.
# Depth band to search, bounded at both ends by what the geometry can actually
# support. It starts well out because a fork cannot even fit in view nearer than
# that: the 21.5 deg half-angle cone is only 1.6 m wide at 2 m, against the 2.0 m
# that two branches plus a dividing wedge need at minimum.
FORK_Z_RANGE = (4.0, 12.0)   # look for branches in this depth band (m)
# ...but the split has to reach back to at least this depth to count. Past ~9 m a
# single image row spans more than a metre of ground, the mask fragments into
# speckle, and any two surviving fragments read as branches -- searching freely out
# to 12 m had IMG_7198 claiming a fork on 82% of frames. Simply capping the search
# at 9 m does not work either, because a real junction is often first resolvable at
# 8.5-11 m (IMG_7205). What separates them is anchoring: a true fork starts near you
# and recedes, so its split is continuous from the reliable band outward, whereas
# horizon speckle appears only in the far rows and has nothing holding it down.
FORK_ANCHOR_Z_M = 9.0
FORK_BRANCH_W_M = 0.70       # each branch must be at least this wide to count
FORK_WEDGE_W_M = 0.70        # dividing non-walkable wedge must be at least this wide
FORK_DEPTH_M = 1.50          # the split must hold over this much depth, contiguously
FORK_WEDGE_SOLID = 0.6       # fraction of the wedge that must be known non-walkable

# Number of points in the returned image-space curve. Matches the drawn/EMA'd
# sample count in path_nav so the cross-frame smoothing lines up.
CURVE_N = 16


def _arc_xz(kappa, s):
    """(X, Z) at arclength s along a constant-curvature arc starting at the
    origin heading along +Z. Positive kappa curves right (+X)."""
    if abs(kappa) < 1e-6:
        return 0.0, s
    r = 1.0 / kappa
    a = kappa * s
    return r * (1.0 - math.cos(a)), r * math.sin(a)


class PlanResult:
    """Outcome of one planning pass."""

    def __init__(self, valid, offset_deg=0.0, curve_x=None, curve_ys=None,
                 kappa=0.0, reach_z_m=0.0, confidence=0.0, fork=False,
                 cost=None, path_uv=None, aim_frac=1.0, opening=False):
        self.valid = valid
        self.offset_deg = offset_deg
        self.curve_x = curve_x           # image-space x samples, near -> far
        self.curve_ys = curve_ys         # image-space y samples, near -> far
        self.kappa = kappa               # chosen curvature (1/m)
        self.reach_z_m = reach_z_m       # forward depth the plan actually reaches
        self.confidence = confidence     # 0..1, from the margin over the runner-up
        self.fork = fork                 # two distinct viable branches
        self.opening = opening           # ground opens out ahead: crossing/junction
        self.cost = cost                 # the cost grid, for the debug view
        self.path_uv = path_uv           # winner's grid cells, for the debug view
        # Where along curve_x/curve_ys (0..1) the announced angle was measured. The
        # drawn line runs to 6 m, but the directive describes the path at the
        # lookahead, and in an image the far end of the line bends far more
        # dramatically than the near part -- so without marking this, a plan that is
        # genuinely near-straight for the next few steps looks like it disagrees
        # with a STRAIGHT cue.
        self.aim_frac = aim_frac


class BevPlanner:
    """Bird's-eye costmap + arc planner for one fixed camera geometry.

    Build once per run (the homography and the arc->cell mapping depend only on
    the camera and the processing size, both fixed), then call plan() per frame.
    """

    def __init__(self, ground, proc_w, proc_h):
        self.ground = ground
        self.w = int(proc_w)
        self.h = int(proc_h)
        self.H = self._build_homography()
        self.H_inv = np.linalg.inv(self.H)
        self._build_visibility()
        self._precompute_arcs()
        self._prev_kappa = 0.0

    def reset(self):
        """Forget the previous plan. Call between independent clips/images so the
        continuity term can't carry a curvature over from unrelated footage."""
        self._prev_kappa = 0.0

    def refresh_geometry(self):
        """Rebuild everything that depends on the camera's aim.

        Call after GroundPlane.set_tilt(). The candidate arcs are defined in metres
        on the ground, so they survive a re-aim untouched; only the image<->grid warp
        and the map of what the camera can see have to be redone.
        """
        self.H = self._build_homography()
        self.H_inv = np.linalg.inv(self.H)
        self._build_visibility()

    # --- geometry ---------------------------------------------------------

    def _to_px(self, nx, ny):
        """Normalised frame offsets (-0.5..0.5) -> processing-frame pixels.

        Normalising by each dimension separately is what makes this survive the
        non-aspect-preserving resize from the camera frame down to PROC_W x
        PROC_H: a point's fraction across the frame is the same either way, and
        GroundPlane's focal lengths are per-fraction for exactly this reason.
        """
        return (nx + 0.5) * self.w, (ny + 0.5) * self.h

    @staticmethod
    def _to_cell(X, Z):
        """Metric ground point -> (col, row) in the bird's-eye grid. Row 0 is the
        far edge, so the grid reads like a map with 'ahead' at the top."""
        return ((X + BEV_HALF_W_M) / BEV_CELL_M,
                (BEV_DEPTH_M - Z) / BEV_CELL_M)

    def _build_homography(self):
        """Four ground points at known metres, projected to pixels, define the
        image->grid warp exactly (both are planes)."""
        pts = [(-1.0, 1.2), (1.0, 1.2), (-1.0, 5.0), (1.0, 5.0)]
        src, dst = [], []
        for X, Z in pts:
            p = self.ground.project(X, Z)
            if p is None:
                raise ValueError("camera geometry cannot see the calibration points")
            src.append(self._to_px(*p))
            dst.append(self._to_cell(X, Z))
        return cv2.getPerspectiveTransform(np.array(src, dtype=np.float32),
                                           np.array(dst, dtype=np.float32))

    def _build_visibility(self):
        """Work out which grid cells the camera can see, and how to fill in the
        blind strip right in front of your feet. All constant for a fixed camera.

        Warping a filled frame (minus the unusable band at and above the horizon)
        tells us which cells are observed. Two regions are not:

          * the wide near-field corners, outside the horizontal field of view;
          * a strip closer than ~1 m, *below* the bottom of the frame, because a
            camera at chest height tilted only slightly down cannot see its own
            feet.

        That near strip matters. Every arc starts at your feet, so if we called it
        "unknown" then every candidate would pay the same large unknown cost over
        its first few samples -- which drowns out the real differences between
        arcs. But that ground is not really unknown: you are standing on it, you
        saw it a moment ago, and you cannot avoid crossing it. So we extend the
        nearest visible row towards the camera, per column.

        Only for that strip, though. It has to be told apart from the sideways
        blind wedges, which on a portrait clip are huge -- the horizontal field of
        view is the NARROW sensor axis, so at 2 m the camera sees barely +/- 0.8 m
        of a grid that is 6 m wide. Extending the nearest visible row into those
        would invent metres of ground we have never seen. The two cases are
        distinguishable exactly: project each cell back and check whether it falls
        below the bottom of the frame (the strip at your feet) or outside it
        sideways (a wedge we must leave unknown).
        """
        src = np.full((self.h, self.w), 255, dtype=np.uint8)
        hy = int(round((self.ground.horizon_ny() + 0.5) * self.h)) + HORIZON_MARGIN_PX
        if hy > 0:
            src[:min(hy, self.h), :] = 0
        seen = cv2.warpPerspective(src, self.H, (GRID_W, GRID_D),
                                   flags=cv2.INTER_NEAREST) > 127

        rows = np.arange(GRID_D)[:, None] + np.zeros((1, GRID_W), dtype=int)
        cols = np.arange(GRID_W)[None, :] + np.zeros((GRID_D, 1), dtype=int)
        # Where each cell centre projects to in the frame (normalised offsets).
        Xc = (cols + 0.5) * BEV_CELL_M - BEV_HALF_W_M
        Zc = BEV_DEPTH_M - (rows + 0.5) * BEV_CELL_M
        nx, ny, in_front = self.ground.project_many(Xc, Zc)
        # Below the bottom edge of the frame, but still within it side to side:
        # the blind strip between your feet and the nearest ground you can see.
        near_blind = (ny > 0.5) & (np.abs(nx) <= 0.5) & in_front

        has_any = seen.any(axis=0)
        # Nearest visible row per column (row index grows toward the camera).
        nearest = np.where(has_any, (GRID_D - 1) - np.argmax(seen[::-1], axis=0), 0)
        self._near_blind = near_blind & has_any[None, :]
        self._fill_r = np.where(self._near_blind, nearest[None, :], rows)
        self._fill_c = cols
        self.valid = seen[self._fill_r, self._fill_c] | self._near_blind

        # Nearest ground we can genuinely see, straight ahead. A plan must get
        # meaningfully past this to count as having found a way forward -- measured
        # from the visible edge, not from an absolute depth, so it adapts to the
        # camera's tilt and height instead of being a magic number.
        centre = GRID_W // 2
        z_near = BEV_DEPTH_M
        if has_any[centre]:
            z_near = BEV_DEPTH_M - float(nearest[centre]) * BEV_CELL_M
        self.z_near_visible = z_near
        self.min_reach_z_m = z_near + MIN_PROGRESS_M

    def _precompute_arcs(self):
        """Map every (arc, sample) to a grid cell once. Per frame this turns the
        whole search into a single fancy-index into the cost grid."""
        kappas = np.linspace(-KAPPA_MAX, KAPPA_MAX, N_ARCS)
        s = np.linspace(ARC_LEN_M / ARC_SAMPLES, ARC_LEN_M, ARC_SAMPLES)
        X = np.zeros((N_ARCS, ARC_SAMPLES), dtype=np.float32)
        Z = np.zeros((N_ARCS, ARC_SAMPLES), dtype=np.float32)
        for i, k in enumerate(kappas):
            for j, sj in enumerate(s):
                X[i, j], Z[i, j] = _arc_xz(k, sj)
        u, v = self._to_cell(X, Z)
        self.kappas = kappas
        self.arc_s = s
        self._lookahead_i = int(np.argmin(np.abs(s - LOOKAHEAD_S_M)))
        self.arc_X, self.arc_Z = X, Z
        self.arc_ok = (u >= 0) & (u < GRID_W) & (v >= 0) & (v < GRID_D)
        self.arc_col = np.clip(u, 0, GRID_W - 1).astype(np.int32)
        self.arc_row = np.clip(v, 0, GRID_D - 1).astype(np.int32)

    # --- costmap ----------------------------------------------------------

    def _obstacle_mask(self, objects):
        """Hard-blocked cells for obstacles we can currently range."""
        blocked = np.zeros((GRID_D, GRID_W), dtype=np.uint8)
        for o in objects or []:
            d = o.get("distance_m")
            if d is None:
                continue                      # above horizon / past reliable range
            b = math.radians(o.get("bearing_deg", 0.0))
            X, Z = d * math.sin(b), d * math.cos(b)
            size = o.get("size_m")
            half = (size[0] / 2.0) if size else OBST_MIN_HALF_W_M
            half = min(max(half, OBST_MIN_HALF_W_M), OBST_MAX_HALF_W_M)
            pad = RANGE_PAD_FRAC * d          # farther = less sure where it is
            self._stamp(blocked, X, Z, half + pad, OBST_DEPTH_M / 2.0 + pad, 1)
        return blocked > 0

    def _predicted_cost(self, objects):
        """Soft cost where movers are likely to be over the next second or two.

        This is what makes the line route *behind* a crossing pedestrian rather
        than into the spot they are about to occupy. Velocity comes from
        projecting the two ends of the track's history onto the ground plane, so
        it is in real m/s -- but it is not ego-motion compensated, hence soft
        costs and a footprint that widens with every step forward in time.
        """
        layer = np.zeros((GRID_D, GRID_W), dtype=np.float32)
        for o in objects or []:
            if not o.get("moving") or o.get("kind") != "dynamic":
                continue
            vel = self._ground_velocity(o)
            if vel is None:
                continue
            (X, Z), (vx, vz) = vel
            size = o.get("size_m")
            half = (size[0] / 2.0) if size else OBST_MIN_HALF_W_M
            half = min(max(half, OBST_MIN_HALF_W_M), OBST_MAX_HALF_W_M)
            for step in range(1, PRED_STEPS + 1):
                dt = PRED_HORIZON_S * step / PRED_STEPS
                pad = PRED_PAD_STEP_M * step
                self._stamp(layer, X + vx * dt, Z + vz * dt,
                            half + pad, OBST_DEPTH_M / 2.0 + pad, COST_PREDICTED)
        return layer

    def _ground_velocity(self, o):
        """((X, Z), (vx, vz)) in metres and m/s from the track's history ends."""
        h0, h1 = o.get("hist0"), o.get("hist1")
        if not h0 or not h1:
            return None
        dt = h1[2] - h0[2]
        if dt <= 1e-3:
            return None
        a = self.ground.xz(h0[0] / self.w - 0.5, h0[1] / self.h - 0.5)
        b = self.ground.xz(h1[0] / self.w - 0.5, h1[1] / self.h - 0.5)
        if a is None or b is None:
            return None
        vx, vz = (b[0] - a[0]) / dt, (b[1] - a[1]) / dt
        speed = math.hypot(vx, vz)
        if speed < PRED_MIN_SPEED_MS or speed > PRED_MAX_SPEED_MS:
            return None
        return b, (vx, vz)

    @staticmethod
    def _stamp(grid, X, Z, half_w_m, half_d_m, value):
        """Paint a metric axis-aligned footprint into a grid (max-combined, so
        overlapping stamps don't accumulate into an artificially huge cost)."""
        c0, r1 = BevPlanner._to_cell(X - half_w_m, Z - half_d_m)
        c1, r0 = BevPlanner._to_cell(X + half_w_m, Z + half_d_m)
        c0, c1 = int(math.floor(c0)), int(math.ceil(c1))
        r0, r1 = int(math.floor(r0)), int(math.ceil(r1))
        c0, c1 = max(0, c0), min(GRID_W, c1)
        r0, r1 = max(0, r0), min(GRID_D, r1)
        if c0 >= c1 or r0 >= r1:
            return
        region = grid[r0:r1, c0:c1]
        np.maximum(region, value, out=region)

    def build_costmap(self, walkable, objects, strict=True):
        """Warp the mask into the grid and turn it into a cost per cell.

        strict=False drops the body-width inflation, used as a retry when the
        strict map leaves nowhere to go (standing in a doorway, say).

        Note on warping a mask that contains vertical things: an obstacle's
        pixels sit ABOVE its feet, and the homography maps higher rows to farther
        Z, so anything standing up smears into a wedge of blocked cells
        stretching away from you. For walls that is exactly right. For a person
        it over-blocks the ground behind them, which is conservative rather than
        unsafe -- and it is why we still stamp explicit metric footprints below,
        since those put a correctly *sized* obstacle at the right distance.
        """
        warped = cv2.warpPerspective(walkable, self.H, (GRID_W, GRID_D),
                                     flags=cv2.INTER_LINEAR)
        # Same near-strip extension as the visibility mask, so the blind ground at
        # your feet inherits whether the nearest visible ground is walkable.
        warped = warped[self._fill_r, self._fill_c]
        walk = (warped > 127) & self.valid
        obst = self._obstacle_mask(objects)

        unknown = (~self.valid) & (~obst)      # unseen, but nothing known to be there
        passable = (walk | unknown) & (~obst)
        known = walk & (~obst)                 # ground we have actually seen and like
        # The blind strip at your feet must never hard-block a plan. Every arc
        # starts at X=0, so one non-walkable cell underfoot -- a shadow, a moment
        # of mask flicker -- would otherwise block all 31 candidates at their first
        # sample and report "no path" for ground you are already standing on. An
        # actual detected obstacle there still blocks; a doubtful mask does not.
        underfoot = self._near_blind & (~obst)
        passable |= underfoot
        known |= underfoot
        # How far each cell is from the edge of the KNOWN-walkable corridor, in
        # metres. Measuring from the known region rather than from everything
        # traversable is essential: the unobserved side wedges are traversable, and
        # they are enormous (the horizontal field of view is only +/- 21.5 deg), so
        # including them put the "most open space" out in territory we have never
        # seen and pinned the plan at a constant sideways curvature. Measured this
        # way, the term does what it is meant to -- centre the line in the path we
        # can actually see -- and unknown ground gets zero clearance, so it is
        # costly without being forbidden.
        clear_m = cv2.distanceTransform(known.astype(np.uint8),
                                        cv2.DIST_L2, 3) * BEV_CELL_M
        if strict:
            # You physically do not fit within a body half-width of an obstruction.
            # Applied only to known ground: unknown cells have zero clearance by
            # construction, and blocking them here would forbid unseen ground
            # outright rather than merely pricing it.
            passable &= (clear_m >= BODY_HALF_W_M) | (~known)

        deficit = np.clip(1.0 - clear_m / CLEAR_PREF_M, 0.0, 1.0)
        cost = COST_CLEARANCE * (deficit ** 2)
        cost += np.where(unknown, COST_UNKNOWN, 0.0)
        # Passable-but-doubtful ground underfoot is not free: if the mask says the
        # strip you're standing on isn't walkable, prefer arcs that leave it sooner.
        cost += np.where(underfoot & (~walk), COST_UNKNOWN, 0.0)
        cost += self._predicted_cost(objects)
        cost[~passable] = COST_BLOCKED
        # Kept for fork detection, which needs to distinguish "known non-walkable"
        # from "merely unobserved" -- a distinction the scalar cost has lost.
        self._walk = walk
        self._solid = (~walk) & (~unknown)
        return cost.astype(np.float32)

    def _reachable(self, walk):
        """The walkable region actually connected to the ground under your feet.

        A second walkable surface is not a branch of your path unless you can walk
        to it. IMG_7198 runs alongside a gravel verge behind a fence: in the grid
        that is two walkable strips split by a solid wedge, which is the exact
        signature of a fork, and no threshold on widths or depths can tell the two
        apart. Connectivity can -- the verge never joins the path, whereas the
        branches of a real fork merge into the trunk you are standing on.
        """
        n, lab = cv2.connectedComponents(walk.astype(np.uint8), connectivity=8)
        if n <= 1:
            return walk
        c0 = GRID_W // 2
        band = lab[GRID_D - 6:, max(0, c0 - 4):c0 + 5]
        seeds = [int(v) for v in np.unique(band) if v > 0]
        return np.isin(lab, seeds) if seeds else walk

    def _opening_from_grid(self, walk):
        """True if the ground ahead opens right across the view -- see CROSS_FAR_Z.

        Takes the reachable walkable region, so an open field on the far side of a
        hedge does not count as the path opening out.
        """
        r0 = int((BEV_DEPTH_M - CROSS_FAR_Z[1]) / BEV_CELL_M)
        r1 = int((BEV_DEPTH_M - CROSS_FAR_Z[0]) / BEV_CELL_M)
        need = max(2, int(round(CROSS_DEPTH_M / BEV_CELL_M)))
        min_cells = CROSS_MIN_OPEN_W_M / BEV_CELL_M
        streak = 0
        for r in range(max(0, r0), min(GRID_D, r1)):
            seen = self.valid[r]
            n_seen = int(seen.sum())
            n_walk = int((walk[r] & seen).sum())
            open_row = (n_seen > 0 and n_walk >= min_cells
                        and n_walk >= CROSS_OPEN_FRAC * n_seen)
            streak = streak + 1 if open_row else 0
            if streak >= need:
                return True
        return False

    def _fork_from_grid(self):
        """True if the corridor really splits: two branches wide enough to walk,
        divided by a solid wedge, holding over a contiguous stretch of depth.

        Runs on the metric grid, so every threshold is a real width (see the
        FORK_* constants for why the old image-space version false-fired).
        """
        walk, solid = self._reachable(self._walk), self._solid
        r0 = int((BEV_DEPTH_M - FORK_Z_RANGE[1]) / BEV_CELL_M)
        r1 = int((BEV_DEPTH_M - FORK_Z_RANGE[0]) / BEV_CELL_M)
        r0, r1 = max(0, r0), min(GRID_D, r1)
        branch = max(1, int(round(FORK_BRANCH_W_M / BEV_CELL_M)))
        wedge = max(1, int(round(FORK_WEDGE_W_M / BEV_CELL_M)))
        need = max(2, int(round(FORK_DEPTH_M / BEV_CELL_M)))

        # Rows run far -> near, so a streak that is long enough AND has reached in
        # to FORK_ANCHOR_Z_M is a split that runs continuously from the trustworthy
        # band outward -- see FORK_ANCHOR_Z_M.
        anchor_row = int((BEV_DEPTH_M - FORK_ANCHOR_Z_M) / BEV_CELL_M)
        streak = 0
        for r in range(r0, r1):
            row = walk[r]
            runs, start = [], None
            for x in range(GRID_W + 1):
                on = x < GRID_W and row[x]
                if on and start is None:
                    start = x
                elif not on and start is not None:
                    if x - start >= branch:
                        runs.append((start, x))
                    start = None
            split = False
            for a, b in zip(runs, runs[1:]):
                gap0, gap1 = a[1], b[0]
                if gap1 - gap0 < wedge:
                    continue
                # The gap has to be actual non-walkable ground, not a patch we
                # simply could not see -- otherwise the unobserved side wedges
                # would read as a fork on every single frame.
                if solid[r, gap0:gap1].mean() >= FORK_WEDGE_SOLID:
                    split = True
                    break
            streak = streak + 1 if split else 0
            if streak >= need and r >= anchor_row:
                return True
        return False

    # --- search -----------------------------------------------------------

    def _score_arcs(self, cost):
        """Score every candidate arc. Returns (scores, reach index per arc)."""
        c = cost[self.arc_row, self.arc_col]
        c = np.where(self.arc_ok, c, COST_BLOCKED)     # off-grid counts as blocked

        blocked = c >= COST_BLOCKED
        any_blocked = blocked.any(axis=1)
        first = np.argmax(blocked, axis=1)             # 0 when nothing is blocked
        n_ok = np.where(any_blocked, first, ARC_SAMPLES)   # samples before blockage

        # Mean soft cost over the traversable prefix of each arc.
        csum = np.cumsum(np.where(blocked, 0.0, c), axis=1)
        idx = np.clip(n_ok - 1, 0, ARC_SAMPLES - 1)
        rows = np.arange(N_ARCS)
        mean_cost = np.where(n_ok > 0, csum[rows, idx] / np.maximum(n_ok, 1), 0.0)

        z_end = np.where(n_ok > 0, self.arc_Z[rows, idx], 0.0)
        shortfall = np.clip(PLAN_TARGET_Z_M - z_end, 0.0, None)

        scores = (mean_cost
                  + COST_PROGRESS * shortfall
                  + COST_CURVATURE * np.abs(self.kappas)
                  + COST_PREV_KAPPA * np.abs(self.kappas - self._prev_kappa))
        # An arc that does not get meaningfully past the nearest visible ground has
        # not found a way anywhere. Rule those out here, inside the search, rather
        # than vetoing the winner afterwards -- otherwise one dead-end candidate
        # scoring well would mask every viable alternative.
        viable = (n_ok > 0) & (z_end >= self.min_reach_z_m)
        self._terms = {"scores": np.where(viable, scores, np.inf),
                       "mean_cost": mean_cost,
                       "progress": COST_PROGRESS * shortfall,
                       "curvature": COST_CURVATURE * np.abs(self.kappas),
                       "continuity": COST_PREV_KAPPA * np.abs(self.kappas
                                                              - self._prev_kappa),
                       "z_end": z_end, "viable": viable}
        return np.where(viable, scores, np.inf), n_ok, z_end

    def explain(self, walkable, objects):
        """Per-arc breakdown of the last-computed score, for tuning the weights.

        Returns (result, terms). Which term dominates tells you whether the plan is
        responding to the scene or to fixed geometry.
        """
        r = self.plan(walkable, objects)
        return r, dict(self._terms, kappa=self.kappas.copy())

    def plan(self, walkable, objects):
        """Plan one frame. Returns a PlanResult (valid=False if nothing works)."""
        cost = self.build_costmap(walkable, objects, strict=True)
        scores, n_ok, z_end = self._score_arcs(cost)
        if not np.isfinite(scores).any():
            # Nowhere fits a full body width -- retry allowing tight squeezes
            # rather than reporting no path at all.
            cost = self.build_costmap(walkable, objects, strict=False)
            scores, n_ok, z_end = self._score_arcs(cost)
        if not np.isfinite(scores).any():
            # No arc works, but a split corridor is still worth announcing -- the
            # fork lives in the costmap, not in the chosen arc, and the branches may
            # simply diverge faster than the steering fan can follow.
            reach = self._reachable(self._walk)
            return PlanResult(False, cost=cost, fork=self._fork_from_grid(),
                              opening=self._opening_from_grid(reach))

        best = int(np.argmin(scores))
        # Break near-ties toward going straight (see TIE_SLACK).
        near = np.flatnonzero(scores <= scores[best] + TIE_SLACK)
        best = int(near[np.argmin(np.abs(self.kappas[near]))])
        # Confidence from the margin over the best *distinct* alternative, so an
        # arc's immediate neighbours (always near-identical) don't hide it.
        mask = np.abs(np.arange(N_ARCS) - best) >= FORK_MIN_SEP_ARCS
        rival = float(np.min(scores[mask])) if np.isfinite(scores[mask]).any() else np.inf
        margin = rival - float(scores[best])
        confidence = float(np.clip(margin / CONF_SCALE, 0.0, 1.0))

        fork = self._fork_from_grid()
        opening = self._opening_from_grid(self._reachable(self._walk))
        self._prev_kappa = float(self.kappas[best])

        k = int(n_ok[best])
        X = self.arc_X[best, :k]
        Z = self.arc_Z[best, :k]
        curve_x, curve_ys = self._to_image_curve(X, Z)
        # Heading of the path at the carrot LOOKAHEAD_S_M along the arc (or at the
        # far end, if the plan is shorter than that). See LOOKAHEAD_S_M on why this
        # is the tangent and not the chord.
        aim = min(k - 1, self._lookahead_i)
        offset_deg = float(math.degrees(self.kappas[best] * self.arc_s[aim]))

        u, v = self._to_cell(X, Z)
        return PlanResult(True, offset_deg=offset_deg, curve_x=curve_x,
                          curve_ys=curve_ys, kappa=self._prev_kappa,
                          reach_z_m=float(Z[-1]), confidence=confidence,
                          fork=fork, opening=opening, cost=cost,
                          path_uv=np.stack([u, v], axis=1),
                          aim_frac=(aim / (k - 1) if k > 1 else 1.0))

    def _to_image_curve(self, X, Z):
        """Resample the winning arc to CURVE_N points and unproject to pixels."""
        n = len(X)
        if n >= 2:
            t = np.linspace(0.0, n - 1.0, CURVE_N)
            X = np.interp(t, np.arange(n), X)
            Z = np.interp(t, np.arange(n), Z)
        else:
            X = np.full(CURVE_N, float(X[0]))
            Z = np.full(CURVE_N, float(Z[0]))
        u, v = self._to_cell(X, Z)
        cells = np.stack([u, v], axis=1).astype(np.float32).reshape(-1, 1, 2)
        px = cv2.perspectiveTransform(cells, self.H_inv).reshape(-1, 2)
        cx = np.clip(px[:, 0], 0, self.w - 1).astype(np.float32)
        cy = np.clip(px[:, 1], 0, self.h - 1).astype(np.float32)
        return cx, cy

    # --- debug view -------------------------------------------------------

    def render(self, result, out_h):
        """Colour picture of the costmap with the chosen path, for tuning.

        Green = cheap and clear, yellow/orange = tight or unseen, red = blocked,
        blue = the plan. Bottom of the image is at your feet.
        """
        # Scale to exactly the requested height so the panel always sits flush
        # beside the camera view, whatever depth the grid is configured for.
        s = out_h / float(GRID_D)
        out_w = max(1, int(round(GRID_W * s)))
        cost = result.cost
        if cost is None:
            return np.zeros((out_h, out_w, 3), dtype=np.uint8)
        blocked = cost >= COST_BLOCKED
        soft = np.clip(cost / (COST_CLEARANCE + COST_UNKNOWN), 0.0, 1.0)
        img = np.zeros((GRID_D, GRID_W, 3), dtype=np.uint8)
        img[..., 1] = ((1.0 - soft) * 220 + 35).astype(np.uint8)      # G
        img[..., 2] = (soft * 235).astype(np.uint8)                   # R
        img[blocked] = (40, 40, 210)
        if result.path_uv is not None:
            for u, v in result.path_uv:
                iu, iv = int(round(u)), int(round(v))
                if 0 <= iu < GRID_W and 0 <= iv < GRID_D:
                    img[iv, iu] = (255, 150, 0)
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        # metre rings, so distances are readable at a glance
        for z in range(2, int(BEV_DEPTH_M) + 1, 2):
            _, v = self._to_cell(0.0, z)
            y = int(round(v * s))
            if 0 <= y < img.shape[0]:
                cv2.line(img, (0, y), (img.shape[1] - 1, y), (90, 90, 90), 1)
                cv2.putText(img, f"{z}m", (2, max(9, y - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        return img
