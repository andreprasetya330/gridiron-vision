"""Field registration: pixels to yards.

This is the hardest part of the vision pipeline and the one that decides whether
anything downstream is worth reading. A homography error of two yards moves a
safety from twelve deep to ten deep, which is the difference between Cover 4 and
Cover 3 in the model's eyes.

Three registration paths, in descending order of how much you should trust them:

1. `KeypointRegistrar` - a trained keypoint model that finds yard-line and hash
   intersections. Best, and needs labeled frames to train.
2. `manual_homography` - you click four or more known field points once per camera
   setup. For fixed All-22 cameras this is genuinely enough, and it is exact.
3. `LineRegistrar` - Hough-transform yard line detection with a vanishing-point
   constraint. Fully automatic, works on clean wide-angle film, and reports its own
   confidence so bad fits get flagged rather than silently used.

Every path returns a homography plus a `reprojection_error_yd`, and the pipeline
refuses to emit coordinates it cannot vouch for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gridiron.fields import FIELD_LENGTH_YD, FIELD_WIDTH_YD, FieldSpec, get_field
from gridiron.vision import require


@dataclass
class Registration:
    """A pixel-to-field mapping for one frame."""

    homography: np.ndarray | None
    reprojection_error_yd: float = float("inf")
    n_correspondences: int = 0
    method: str = "none"
    notes: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.homography is not None and self.reprojection_error_yd < 3.0

    def to_field(self, points: np.ndarray) -> np.ndarray:
        """Image points (N,2) -> field coordinates in yards (N,2)."""
        if self.homography is None:
            return np.full((len(points), 2), np.nan)
        cv2 = require("cv2")
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.homography.astype(np.float64))
        return out.reshape(-1, 2)

    def to_image(self, points: np.ndarray) -> np.ndarray:
        if self.homography is None:
            return np.full((len(points), 2), np.nan)
        cv2 = require("cv2")
        inverse = np.linalg.inv(self.homography)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, inverse).reshape(-1, 2)


def field_template(league: str = "ncaa", yard_step: int = 5) -> dict[str, np.ndarray]:
    """Canonical field landmarks in yards.

    Origin at the back of the left end zone, x downfield 0-120, y across 0-53.3.
    Hash positions come from the league spec, which is the only thing that differs
    between college and the NFL and the thing that most affects coverage reads.
    """
    spec: FieldSpec = get_field(league)
    yard_lines = np.arange(10, FIELD_LENGTH_YD - 10 + 1, yard_step, dtype=np.float64)

    rows = {
        "bottom_sideline": 0.0,
        "left_hash": spec.left_hash_y,
        "right_hash": spec.right_hash_y,
        "top_sideline": FIELD_WIDTH_YD,
    }

    points: dict[str, np.ndarray] = {}
    for name, y in rows.items():
        points[name] = np.array([[x, y] for x in yard_lines], dtype=np.float64)
    points["all"] = np.concatenate(list(points.values()), axis=0)
    return points


def manual_homography(
    image_points: np.ndarray, field_points: np.ndarray, league: str = "ncaa"
) -> Registration:
    """Fit from hand-clicked correspondences. Exact and boring, which is ideal."""
    cv2 = require("cv2")
    image_points = np.asarray(image_points, dtype=np.float64)
    field_points = np.asarray(field_points, dtype=np.float64)

    if len(image_points) < 4 or len(image_points) != len(field_points):
        return Registration(None, notes=["need at least 4 matched point pairs"])

    H, inliers = cv2.findHomography(image_points, field_points, cv2.RANSAC, 5.0)
    if H is None:
        return Registration(None, notes=["homography fit failed"])

    projected = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2).astype(np.float32), H
    ).reshape(-1, 2)
    error = float(np.mean(np.linalg.norm(projected - field_points, axis=1)))

    return Registration(
        homography=H,
        reprojection_error_yd=error,
        n_correspondences=int(inliers.sum()) if inliers is not None else len(image_points),
        method="manual",
    )


class LineRegistrar:
    """Automatic registration from painted field lines.

    Finds the white lines, separates the near-vertical yard lines from the
    near-horizontal sidelines and hashes, then matches intersections against the
    field template. Works well on wide, static All-22 angles and poorly on
    broadcast footage with heavy camera motion - which it will tell you, via the
    reprojection error, rather than quietly returning nonsense.
    """

    def __init__(self, league: str = "ncaa", white_threshold: int = 190) -> None:
        self.league = league
        self.spec = get_field(league)
        self.white_threshold = white_threshold

    def line_mask(self, frame: np.ndarray) -> np.ndarray:
        cv2 = require("cv2")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Field paint is high-value and low-saturation regardless of turf colour.
        mask = cv2.inRange(hsv, (0, 0, self.white_threshold), (180, 60, 255))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    def detect_lines(self, frame: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        cv2 = require("cv2")
        mask = self.line_mask(frame)
        edges = cv2.Canny(mask, 50, 150, apertureSize=3)
        segments = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360,
            threshold=90,
            minLineLength=max(40, frame.shape[0] // 12),
            maxLineGap=25,
        )
        if segments is None:
            return [], []

        # HoughLinesP returns (N,1,4) on most OpenCV builds and (N,4) on some;
        # reshaping covers both rather than betting on the installed version.
        segments = np.asarray(segments).reshape(-1, 4)

        vertical, horizontal = [], []
        for seg in segments:
            x1, y1, x2, y2 = (float(v) for v in seg)
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle > 90:
                angle = 180 - angle
            if angle > 55:
                vertical.append(np.array([x1, y1, x2, y2]))
            elif angle < 25:
                horizontal.append(np.array([x1, y1, x2, y2]))
        return vertical, horizontal

    def register(self, frame: np.ndarray) -> Registration:
        vertical, horizontal = self.detect_lines(frame)
        if len(vertical) < 3 or len(horizontal) < 2:
            return Registration(
                None,
                method="lines",
                notes=[
                    f"found {len(vertical)} yard lines and {len(horizontal)} horizontal "
                    "lines, which is not enough to solve a homography"
                ],
            )

        vertical = _merge_collinear(vertical)
        horizontal = _merge_collinear(horizontal)

        mask = self.line_mask(frame)
        rows = self.label_rows(horizontal, mask)
        if rows is None:
            return Registration(
                None,
                method="lines",
                notes=[
                    "could not tell sidelines from hash rows, so the horizontal "
                    "lines cannot be assigned to field positions"
                ],
            )
        sidelines, dashed = rows

        # Column identity belongs to the yard line, not to the intersection.
        # Grouping intersections by image x looks equivalent and is not: under
        # perspective one yard line meets the near sideline and the far sideline
        # at image columns hundreds of pixels apart, so the near end of one line
        # groups with the far end of its neighbour and the fitted field comes out
        # sheared. Ordering the lines themselves left to right sidesteps it.
        ordered = _order_vertical(vertical, frame.shape)

        def fit(labeled: list[tuple[np.ndarray, float]]) -> Registration:
            # Absolute yardage needs the painted numbers, so anchor the middle
            # line at midfield and report the ambiguity rather than hiding it.
            start = 60.0 - 5.0 * (len(ordered) - 1) / 2
            intersections: list[tuple[float, float, float, float]] = []
            for i, v in enumerate(ordered):
                for h, field_y in labeled:
                    point = _intersect(v, h)
                    if point is not None and _inside(point, frame.shape):
                        intersections.append(
                            (point[0], point[1], start + 5.0 * i, field_y)
                        )
            if len(intersections) < 4:
                return Registration(
                    None, method="lines", notes=["too few line intersections to fit"]
                )
            return _fit_from_grid(intersections)

        # The two sidelines span the field, so they alone pin the homography.
        # Hash rows are then placed by measuring them through that fit rather than
        # by where they sit in the frame: perspective compresses the far half, so
        # a hash 3/8 of the way across the field lands past the halfway row in
        # image coordinates and reads as the wrong hash.
        base = fit(sidelines)
        if base.homography is None or not dashed:
            return base

        labeled = list(sidelines)
        hashes = [self.spec.left_hash_y, self.spec.right_hash_y]
        for h in dashed:
            midpoint = np.array([[(h[0] + h[2]) / 2, (h[1] + h[3]) / 2]])
            field_y = float(base.to_field(midpoint)[0][1])
            nearest = min(hashes, key=lambda y: abs(y - field_y))
            if abs(nearest - field_y) <= 3.0:
                labeled.append((h, nearest))

        if len(labeled) == len(sidelines):
            return base

        refined = fit(labeled)
        return refined if refined.homography is not None else base

    def label_rows(
        self, horizontal: list[np.ndarray], mask: np.ndarray, solid_ratio: float = 0.6
    ) -> tuple[list[tuple[np.ndarray, float]], list[np.ndarray]] | None:
        """Separate sidelines from hash rows, and label the sidelines.

        Geometry alone cannot do this. A homography has enough freedom to carry
        any three parallel lines onto any other three, so fitting every candidate
        assignment and keeping the lowest reprojection error picks an arbitrary
        one - they all fit about equally well, and the wrong ones put the defense
        in the stands behind a confident-looking half-yard error.

        What separates them is what the lines are made of. A sideline is painted
        continuously; a hash row is a series of one-yard ticks that the Hough
        transform bridges into a line that is mostly gap. Measuring how much of
        each line is actually painted tells the two apart, and it holds on real
        film for the same reason it holds here.

        Returns the labeled sidelines and the leftover dashed rows, which the
        caller places once it has a homography to measure them with.
        """
        if len(horizontal) < 2:
            return None

        scored = [(h, _line_fill_ratio(mask, h)) for h in horizontal]
        solid = [h for h, ratio in scored if ratio >= solid_ratio]
        dashed = [h for h, ratio in scored if ratio < solid_ratio]
        if len(solid) < 2:
            return None

        # The camera looks across the field from one sideline, so the nearer
        # sideline sits lower in the frame and carries field y = 0.
        solid.sort(key=lambda h: -(h[1] + h[3]) / 2)
        near, far = solid[0], solid[-1]
        if abs((near[1] + near[3]) / 2 - (far[1] + far[3]) / 2) < 1e-6:
            return None

        return [(near, 0.0), (far, FIELD_WIDTH_YD)], dashed


class FixedRegistrar:
    """One homography, reused for every frame.

    For a locked-off All-22 camera this is not a shortcut, it is the correct
    answer: the mapping genuinely does not change, so solving it once from
    clicked points beats re-estimating it 150 times and averaging the noise. It
    is also the control case for validation - with registration held exact, any
    remaining coordinate error belongs to some other stage.
    """

    def __init__(self, registration: Registration) -> None:
        self.registration = registration

    def register(self, frame: np.ndarray) -> Registration:
        return self.registration

    @classmethod
    def from_homography(
        cls, image_to_field: np.ndarray, error_yd: float = 0.0, method: str = "fixed"
    ) -> FixedRegistrar:
        return cls(
            Registration(
                homography=np.asarray(image_to_field, dtype=np.float64),
                reprojection_error_yd=error_yd,
                method=method,
            )
        )


class KeypointRegistrar:
    """Registration from a trained field-keypoint model.

    Expects a YOLO-pose style model whose keypoints correspond, in order, to the
    template points returned by `field_template`. Training one needs a few hundred
    labeled frames; until then, use `manual_homography` for fixed cameras.
    """

    def __init__(self, weights: str, league: str = "ncaa", min_confidence: float = 0.5) -> None:
        self.weights = weights
        self.league = league
        self.min_confidence = min_confidence
        self.template = field_template(league)["all"]
        self._model = None

    @property
    def model(self):
        if self._model is None:
            ultralytics = require("ultralytics")
            self._model = ultralytics.YOLO(self.weights)
        return self._model

    def register(self, frame: np.ndarray) -> Registration:
        results = self.model.predict(frame, verbose=False)
        if not results or results[0].keypoints is None:
            return Registration(None, method="keypoints", notes=["no keypoints detected"])

        keypoints = results[0].keypoints
        xy = keypoints.xy[0].cpu().numpy()
        conf = (
            keypoints.conf[0].cpu().numpy()
            if keypoints.conf is not None
            else np.ones(len(xy))
        )

        keep = conf >= self.min_confidence
        if keep.sum() < 4:
            return Registration(
                None,
                method="keypoints",
                notes=[f"only {int(keep.sum())} keypoints above confidence threshold"],
            )

        n = min(len(xy), len(self.template))
        mask = keep[:n]
        registration = manual_homography(xy[:n][mask], self.template[:n][mask], self.league)
        registration.method = "keypoints"
        return registration


class SmoothedRegistration:
    """Temporal smoothing across frames.

    A per-frame homography jitters, and jitter in the homography becomes jitter in
    every player's coordinates, which the velocity features then amplify. Holding a
    rolling median of recent valid fits is a cheap, robust fix for the fixed or
    slowly panning cameras that All-22 uses.
    """

    def __init__(self, window: int = 9, max_error_yd: float = 2.5) -> None:
        self.window = window
        self.max_error_yd = max_error_yd
        self._history: list[np.ndarray] = []
        self._last_good: Registration | None = None

    def update(self, registration: Registration) -> Registration:
        if registration.homography is not None and registration.reprojection_error_yd <= self.max_error_yd:
            normalized = registration.homography / registration.homography[2, 2]
            self._history.append(normalized)
            if len(self._history) > self.window:
                self._history.pop(0)
            self._last_good = registration

        if not self._history:
            return registration

        smoothed = np.median(np.stack(self._history, axis=0), axis=0)
        error = (
            registration.reprojection_error_yd
            if registration.homography is not None
            else (self._last_good.reprojection_error_yd if self._last_good else float("inf"))
        )
        notes = list(registration.notes)
        if registration.homography is None:
            notes.append("frame fit failed; using smoothed homography from recent frames")

        return Registration(
            homography=smoothed,
            reprojection_error_yd=error,
            n_correspondences=registration.n_correspondences,
            method=f"{registration.method}+smoothed",
            notes=notes,
        )


def _line_params(segment: np.ndarray) -> tuple[float, float]:
    """A segment as (angle in degrees mod 180, signed perpendicular offset)."""
    x1, y1, x2, y2 = (float(v) for v in segment)
    theta = np.arctan2(y2 - y1, x2 - x1)
    degrees = np.degrees(theta) % 180.0
    radians = np.radians(degrees)
    # Perpendicular offset of the line from the origin, taken in a canonical
    # direction so that a segment and its reverse describe the same line.
    rho = -np.sin(radians) * x1 + np.cos(radians) * y1
    return float(degrees), float(rho)


def _merge_collinear(
    segments: list[np.ndarray], angle_tol: float = 4.0, offset_tol: float = 12.0
):
    """Collapse Hough fragments of the same painted line into one segment.

    Grouping is by angle and perpendicular offset - the Hough parameters - rather
    than by how close the fragments' midpoints are. Midpoint distance sounds
    equivalent and is not: a yard line runs most of the frame's height, Canny
    finds both of its painted edges, and Hough breaks each edge into pieces
    covering different stretches of it. Two pieces of one line can sit hundreds of
    pixels apart by midpoint while lying a pixel from each other, so they survive
    as separate lines and the detected spacing comes out alternating 108 pixels
    and 2. Everything downstream then reads a field whose yard lines are two feet
    apart.
    """
    if not segments:
        return []

    groups: list[dict] = []
    for seg in segments:
        angle, rho = _line_params(seg)
        placed = False
        for group in groups:
            delta = abs(angle - group["angle"])
            delta = min(delta, 180.0 - delta)  # 179 degrees is close to 1 degree
            if delta < angle_tol and abs(rho - group["rho"]) < offset_tol:
                group["points"].extend([seg[0:2], seg[2:4]])
                placed = True
                break
        if not placed:
            groups.append({"angle": angle, "rho": rho, "points": [seg[0:2], seg[2:4]]})

    merged: list[np.ndarray] = []
    for group in groups:
        points = np.array(group["points"], dtype=np.float64)
        radians = np.radians(group["angle"])
        direction = np.array([np.cos(radians), np.sin(radians)])
        projection = points @ direction
        merged.append(
            np.array([*points[np.argmin(projection)], *points[np.argmax(projection)]])
        )
    return merged


def _intersect(a: np.ndarray, b: np.ndarray) -> tuple[float, float] | None:
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return float(px), float(py)


def _inside(point: tuple[float, float], shape: tuple[int, ...], margin: float = 5.0) -> bool:
    h, w = shape[:2]
    x, y = point
    return -margin <= x <= w + margin and -margin <= y <= h + margin


def _line_fill_ratio(mask: np.ndarray, segment: np.ndarray, samples: int = 200) -> float:
    """What fraction of a detected line is actually painted.

    Near 1.0 for a sideline, far below it for a row of hash ticks that the Hough
    transform has bridged into a single line.
    """
    x1, y1, x2, y2 = (float(v) for v in segment)
    height, width = mask.shape[:2]
    us = np.linspace(x1, x2, samples)
    vs = np.linspace(y1, y2, samples)

    hits = 0
    counted = 0
    for u, v in zip(us, vs):
        col, row = int(round(u)), int(round(v))
        if not (0 <= col < width and 0 <= row < height):
            continue
        counted += 1
        # A tolerance of a pixel or two, since the fitted line and the painted
        # one need not agree exactly.
        lo_r, hi_r = max(0, row - 2), min(height, row + 3)
        lo_c, hi_c = max(0, col - 2), min(width, col + 3)
        if mask[lo_r:hi_r, lo_c:hi_c].any():
            hits += 1
    return hits / counted if counted else 0.0


def _order_vertical(
    vertical: list[np.ndarray], shape: tuple[int, ...]
) -> list[np.ndarray]:
    """Sort yard lines left to right, measured at a single image row.

    Comparing them anywhere else is not well defined: they converge toward the
    vanishing point, so two lines can swap order between the top and bottom of
    the frame.
    """
    row = shape[0] * 0.9

    def x_at(segment: np.ndarray) -> float:
        x1, y1, x2, y2 = (float(v) for v in segment)
        if abs(y2 - y1) < 1e-6:
            return (x1 + x2) / 2
        return x1 + (x2 - x1) * (row - y1) / (y2 - y1)

    return sorted(vertical, key=x_at)


def _fit_from_grid(
    intersections: list[tuple[float, float, float, float]],
) -> Registration:
    """Fit a homography from intersections whose field position is known."""
    points = np.array(intersections, dtype=np.float64)
    registration = manual_homography(points[:, :2], points[:, 2:4])
    if registration.homography is None:
        return registration

    registration.method = "lines"
    registration.notes.append(
        "Yard-line identity was inferred from spacing, so downfield position is "
        "anchored at midfield and may be offset by a multiple of 5 yards. Depths "
        "relative to the line of scrimmage, which is what coverage depends on, are "
        "unaffected."
    )
    return registration
