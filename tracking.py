"""Lightweight multi-object tracker for Phase A (SORT-style, IoU association).

Purpose: YOLO reports fresh boxes every frame with no memory, so we cannot tell
"the same person as last frame" from "a brand-new person", and we cannot say
which way anything is moving. This module links detections across frames into
persistent tracks (stable ids) and derives a per-track *heading* from the recent
position history. That heading is what the collision logic in path_nav.py needs
to answer "is this thing coming toward my path?".

Design notes:
  - Pure NumPy/stdlib -- no Kalman filter, no scipy, no extra model. Association
    is greedy IoU matching within the same class, which is plenty for
    walking-pace scenes and keeps the Pi cost negligible.
  - We track the box *bottom-centre* (the ground-contact point), because that is
    also what distance/most-obstacle reasoning keys off.
  - Heading is measured in IMAGE space and is NOT ego-motion compensated in
    Phase A: while you walk, static objects also drift in the frame. So heading
    is a useful approximation ("drifting toward path / toward me") but absolute
    world direction waits for the depth work in Phase A+. Speed is deliberately
    NOT exposed as a number (only a moving/not-moving decision is made).

Coordinate/heading convention (image axes: x right, y down):
    heading_deg = degrees(atan2(dx, -dy))
      0    = moving up   (away from you, toward top of frame)
    +/-180 = moving down (toward you, toward bottom of frame)
     +90   = moving right
     -90   = moving left
"""
import math


# --- Association / lifecycle ---
IOU_MATCH = 0.30          # min IoU (same class) to link a detection to a track
MAX_COAST_FRAMES = 2      # keep reporting a briefly-missed track this many frames
DROP_AGE_FRAMES = 12      # delete a track after this many frames with no match

# --- Motion / heading ---
HISTORY_LEN = 15          # bottom-centre points kept per track
HEADING_MIN_POINTS = 3    # need at least this many points to trust a heading
MOTION_EPS_PX = 5.0       # net displacement (px) below this over the window = "still"


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    """One tracked object: stable id, latest box, and recent motion history."""

    def __init__(self, track_id, det, t):
        self.id = track_id
        self.name = det["name"]
        self.cls = det["cls"]
        self.box = det["box"]
        self.conf = det["conf"]
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.history = [(*self._base(det["box"]), t)]   # [(cx, cy, t), ...]

    @staticmethod
    def _base(box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2.0, float(y2)               # bottom-centre

    def update(self, det, t):
        self.name = det["name"]
        self.cls = det["cls"]
        self.box = det["box"]
        self.conf = det["conf"]
        self.hits += 1
        self.time_since_update = 0
        self.history.append((*self._base(det["box"]), t))
        if len(self.history) > HISTORY_LEN:
            self.history = self.history[-HISTORY_LEN:]

    def mark_missed(self):
        self.age += 1
        self.time_since_update += 1

    def predicted_box(self):
        """Last box shifted by the most recent inter-frame displacement.

        A constant-velocity guess of where the object is *now*. Matching against
        this (instead of the stale last box) lets fast movers -- a crossing
        cyclist -- still associate frame-to-frame, where raw IoU would break.
        """
        if len(self.history) < 2:
            return self.box
        (x0, y0, _), (x1, y1, _) = self.history[-2], self.history[-1]
        dx, dy = x1 - x0, y1 - y0
        bx1, by1, bx2, by2 = self.box
        return (bx1 + dx, by1 + dy, bx2 + dx, by2 + dy)

    def motion(self):
        """Return (moving, heading_deg, vx, vy) from the position history.

        vx, vy is the net image displacement (px) across the tracked window and
        is used only to decide direction (not to report a speed). heading_deg is
        None when the track hasn't moved enough to trust a direction.
        """
        if len(self.history) < HEADING_MIN_POINTS:
            return False, None, 0.0, 0.0
        x0, y0, _ = self.history[0]
        x1, y1, _ = self.history[-1]
        vx, vy = x1 - x0, y1 - y0
        if math.hypot(vx, vy) < MOTION_EPS_PX:
            return False, None, vx, vy
        heading = math.degrees(math.atan2(vx, -vy))
        return True, heading, vx, vy

    def as_dict(self, coasted):
        moving, heading, vx, vy = self.motion()
        cx, cy = self._base(self.box)
        return {
            "id": self.id,
            "name": self.name,
            "cls": self.cls,
            "conf": self.conf,
            "box": self.box,
            "cx": cx,
            "cy": cy,
            "moving": moving,
            "heading_deg": heading,
            "vx": vx,
            "vy": vy,
            "coasted": coasted,
            "hits": self.hits,
            "age": self.age,
            # Raw endpoints of the tracked window, (x, y, t). vx/vy above are in
            # pixels, which cannot be turned into metres per second; with the two
            # timestamped ground-contact points the planner can project both onto
            # the ground plane and get a real velocity for motion prediction.
            "hist0": self.history[0],
            "hist1": self.history[-1],
        }


class ObjectTracker:
    """Greedy IoU tracker. Call update(dets, t) once per processed frame."""

    def __init__(self, iou_match=IOU_MATCH, max_coast=MAX_COAST_FRAMES,
                 drop_age=DROP_AGE_FRAMES):
        self.iou_match = iou_match
        self.max_coast = max_coast
        self.drop_age = drop_age
        self.tracks = []
        self._next_id = 1

    def update(self, dets, t):
        """Associate detections to tracks; return the list of active object dicts.

        Active = matched this frame, or missed for <= max_coast frames (so a
        one-frame YOLO dropout doesn't make an obstacle vanish). Coasted tracks
        keep their last box and are flagged coasted=True.
        """
        for tr in self.tracks:
            tr.mark_missed()

        unmatched = set(range(len(dets)))
        # Build all candidate (iou, track_idx, det_idx) pairs for same-class boxes.
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di in unmatched:
                d = dets[di]
                if d["cls"] != tr.cls:
                    continue
                iou = _iou(tr.predicted_box(), d["box"])
                if iou >= self.iou_match:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)                    # greediest (highest IoU) first

        used_tracks, used_dets = set(), set()
        for iou, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            self.tracks[ti].update(dets[di], t)
            used_tracks.add(ti)
            used_dets.add(di)
            unmatched.discard(di)

        for di in unmatched:                        # spawn tracks for new boxes
            self.tracks.append(Track(self._next_id, dets[di], t))
            self._next_id += 1

        self.tracks = [tr for tr in self.tracks if tr.time_since_update <= self.drop_age]

        active = []
        for tr in self.tracks:
            if tr.time_since_update == 0:
                active.append(tr.as_dict(coasted=False))
            elif tr.time_since_update <= self.max_coast:
                active.append(tr.as_dict(coasted=True))
        return active
