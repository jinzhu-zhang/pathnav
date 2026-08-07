#!/usr/bin/env python3
"""Ground-plane monocular ranging: turn an image pixel into a metric distance.

Idea (one camera, no stereo): the camera is fixed at a known HEIGHT above the
ground and a known downward TILT. The ground is assumed flat. A ray cast through
a pixel therefore hits the ground at a computable distance -- so the point where
an object touches the ground (the bottom-centre of its box) tells us how far away
it is, in metres. Objects lower in the frame are nearer; higher = farther.

Only valid for things resting ON the ground (people, obstacles on the path). A
pixel at/above the horizon never meets the ground -> distance is undefined.

Coordinates are NORMALISED (fraction of frame, -0.5..0.5) so this is independent
of the processing resolution and of any aspect-ratio resize: a point's fraction
across the frame is the same whether measured on the full 4K frame or the small
processing frame. The camera's true aspect ratio (width/height) is supplied
separately so the vertical focal length is correct.

Camera frame convention: x right, y DOWN, z forward. World: same, gravity = +y.
The optical axis is pitched DOWN by `tilt` from horizontal.
"""
import math

import numpy as np


class GroundPlane:
    def __init__(self, height_m, tilt_deg, fx_n, fy_n):
        """height_m : camera height above ground (m)
        tilt_deg   : downward tilt of the optical axis from horizontal (deg)
        fx_n, fy_n : normalised focal lengths (f_px / width, f_px / height)
        Prefer building via from_fov(), which handles orientation correctly.
        """
        self.h = float(height_m)
        self.theta = math.radians(tilt_deg)
        self.fx_n = float(fx_n)
        self.fy_n = float(fy_n)
        self._ct, self._st = math.cos(self.theta), math.sin(self.theta)

    @classmethod
    def from_fov(cls, height_m, tilt_deg, fov_long_deg, width, height):
        """Build from the camera's field of view across its LONG sensor side.

        This is orientation-robust: iPhone (and most) videos are stored/rotated so
        the frame may be portrait or landscape, but the lens is the same. We take
        the single pixel focal length from the long dimension's FOV, then derive
        both normalised focals from the ACTUAL frame width/height (square pixels).
        """
        long_side = max(width, height)
        f_px = (long_side / 2.0) / math.tan(math.radians(fov_long_deg) / 2.0)
        return cls(height_m, tilt_deg, f_px / float(width), f_px / float(height))

    def set_tilt(self, tilt_deg):
        """Re-aim the model at a new measured downward tilt, in place.

        The tilt is the one camera parameter we cannot take on trust: height and FOV
        are fixed by the hardware, but how steeply the camera looks down changes with
        how it is carried. Callers that cache anything derived from the geometry (a
        warp, a visibility map) must rebuild it after calling this.
        """
        self.theta = math.radians(float(tilt_deg))
        self._ct, self._st = math.cos(self.theta), math.sin(self.theta)

    def _world_ray(self, nx, ny):
        """Camera-ray (nx,ny normalised offsets from centre) rotated into world by
        the downward tilt. Returns (wx, wy, wz); wy>0 points toward the ground."""
        rx, ry, rz = nx / self.fx_n, ny / self.fy_n, 1.0
        wx = rx
        wy = self._ct * ry + self._st * rz
        wz = -self._st * ry + self._ct * rz
        return wx, wy, wz

    def point(self, nx, ny):
        """Return dict(distance_m, depth_m, bearing_deg) for a ground pixel, or
        None if the ray is at/above the horizon (never meets the ground).

        distance_m : horizontal ground distance from the camera to the point.
        depth_m    : forward (optical-axis-ish) component, used for size scaling.
        bearing_deg: left(-)/right(+) angle of the point.
        """
        wx, wy, wz = self._world_ray(nx, ny)
        if wy <= 1e-6:                       # at/above horizon -> no ground hit
            return None
        t = self.h / wy
        X, Z = t * wx, t * wz
        return {
            "distance_m": math.hypot(X, Z),
            "depth_m": Z,
            "bearing_deg": math.degrees(math.atan2(X, Z)),
        }

    def xz(self, nx, ny):
        """Ground position (X right, Z forward) in metres, or None above horizon."""
        p = self.point(nx, ny)
        if p is None:
            return None
        b = math.radians(p["bearing_deg"])
        return p["distance_m"] * math.sin(b), p["distance_m"] * math.cos(b)

    def project(self, X, Z):
        """Inverse of point(): the normalised pixel (nx, ny) that sees the ground
        point (X right, Z forward) in metres. None if it falls behind the camera.

        Used to build the image->bird's-eye homography: pick a few ground points
        at known metres, project them to pixels, and the two sets of four
        correspondences define the warp exactly (the ground is a plane).
        """
        nx, ny, ok = self.project_many(X, Z)
        return (nx, ny) if ok else None

    def project_many(self, X, Z):
        """Array-friendly project(): returns (nx, ny, ok) with ok False where the
        point is behind the camera. Works on scalars or NumPy arrays, so a caller
        can project a whole bird's-eye grid in one go without reaching into the
        camera model's internals."""
        rz = self._st * self.h + self._ct * Z
        ok = rz > 1e-6
        safe = np.where(ok, rz, 1.0) if hasattr(rz, "shape") else (rz if ok else 1.0)
        ry = self._ct * self.h - self._st * Z
        return self.fx_n * X / safe, self.fy_n * ry / safe, ok

    def size_m(self, box_w_frac, box_h_frac, depth_m):
        """Approximate physical (width, height) in metres of a box at `depth_m`,
        given its width/height as fractions of the frame."""
        return (box_w_frac * depth_m / self.fx_n,
                box_h_frac * depth_m / self.fy_n)

    def horizon_ny(self):
        """Normalised vertical position (-0.5..0.5) of the horizon line, for a
        visual sanity-check overlay. Above this row nothing touches the ground."""
        return -math.tan(self.theta) * self.fy_n
