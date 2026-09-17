"""Team assignment and offense/defense resolution.

Two questions, in order:

1. **Which team is each player on?** Cluster torso colours. Jerseys are designed to
   be maximally separable at a glance, which makes this the one easy problem in the
   whole vision pipeline.
2. **Which cluster is the defense?** Colour cannot answer that. Geometry can: at
   the snap, the offense has five linemen packed shoulder to shoulder on the line
   of scrimmage, and no defense ever looks like that. Finding the tightest cluster
   of five and calling that side the offense is far more reliable than anything
   based on which direction play is moving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gridiron.vision import require
from gridiron.vision.detect import Detection


@dataclass
class TeamAssignment:
    labels: dict[int, int] = field(default_factory=dict)  # track_id -> cluster 0/1
    offense_cluster: int | None = None
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)
    centers: np.ndarray | None = None

    def side_of(self, track_id: int) -> str | None:
        cluster = self.labels.get(track_id)
        if cluster is None or self.offense_cluster is None:
            return None
        return "offense" if cluster == self.offense_cluster else "defense"

    def nearest_side(self, feature: np.ndarray) -> str | None:
        """Assign a player who was not at the snap by Lab distance to the two jerseys."""
        if self.centers is None or self.offense_cluster is None:
            return None
        dist = np.linalg.norm(self.centers - feature.reshape(1, -1), axis=1)
        cluster = int(np.argmin(dist))
        return "offense" if cluster == self.offense_cluster else "defense"


def torso_crop(frame: np.ndarray, detection: Detection) -> np.ndarray:
    """The middle of the jersey, avoiding helmet, pants, and grass."""
    h, w = frame.shape[:2]
    x1 = int(np.clip(detection.x1 + 0.25 * (detection.x2 - detection.x1), 0, w - 1))
    x2 = int(np.clip(detection.x2 - 0.25 * (detection.x2 - detection.x1), 1, w))
    y1 = int(np.clip(detection.y1 + 0.20 * (detection.y2 - detection.y1), 0, h - 1))
    y2 = int(np.clip(detection.y1 + 0.55 * (detection.y2 - detection.y1), 1, h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]


def jersey_feature(frame: np.ndarray, detection: Detection) -> np.ndarray:
    """Colour descriptor for one player.

    Lab rather than RGB because Lab distance tracks how different two colours look
    to a human, and jersey design is optimized for exactly that. The green mask
    removes turf bleeding in around the shoulders.
    """
    cv2 = require("cv2")
    crop = torso_crop(frame, detection)
    if crop.size == 0:
        return np.zeros(3, dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    not_grass = ~cv2.inRange(hsv, (30, 40, 40), (90, 255, 255)).astype(bool)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)

    pixels = lab[not_grass] if not_grass.sum() > 20 else lab.reshape(-1, 3)
    sat = hsv[not_grass][:, 1] if not_grass.sum() > 20 else hsv.reshape(-1, 3)[:, 1]
    # White numbers and glare pull a red jersey into the white cluster. Keep the
    # more saturated half of the crop when the jersey actually has colour.
    if (sat > 50).sum() >= 15:
        pixels = pixels[sat >= np.median(sat)]
    return np.median(pixels, axis=0).astype(np.float32)


def assign_teams(
    frame: np.ndarray,
    tracked: list[Any],
    field_positions: dict[int, tuple[float, float]] | None = None,
) -> TeamAssignment:
    """Cluster players into two teams and identify the offense.

    Single-frame clustering, kept for callers that only have one frame. The film
    pipeline prefers `assign_teams_from_roster`, which clusters the median of a
    whole track and is far steadier.
    """
    from sklearn.cluster import KMeans

    players = [t for t in tracked if getattr(t.detection, "class_name", "player") == "player"]
    if len(players) < 6:
        return TeamAssignment(notes=["too few players detected to cluster teams"])

    features = np.stack([jersey_feature(frame, t.detection) for t in players])
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(features)
    labels = {t.track_id: int(label) for t, label in zip(players, kmeans.labels_)}

    separation = float(np.linalg.norm(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]))
    # Lab distances above roughly 25 are visually obvious; below 12 the two teams
    # are wearing similar colours and every downstream number is suspect.
    confidence = float(np.clip((separation - 12.0) / 30.0, 0.0, 1.0))

    assignment = TeamAssignment(
        labels=labels, confidence=confidence, centers=kmeans.cluster_centers_
    )
    if confidence < 0.35:
        assignment.notes.append(
            f"jersey colours are only {separation:.0f} Lab units apart; team assignment "
            "is unreliable on this film"
        )

    if field_positions:
        assignment.offense_cluster = _find_offense(labels, field_positions)
        if assignment.offense_cluster is None:
            assignment.notes.append(
                "could not find an offensive line formation, so offense/defense could "
                "not be resolved"
            )
    return assignment


def assign_teams_from_roster(roster: Any, field_positions: dict[int, tuple[float, float]]):
    """Turn track-level colour labels into an offense/defense assignment.

    Colour says which two groups exist; only geometry says which one is the
    offense, because no defense ever lines up five abreast on the ball.
    """
    assignment = TeamAssignment(
        labels=dict(roster.team_of),
        confidence=roster.confidence,
        centers=roster.centers,
        notes=list(roster.notes),
    )
    if not assignment.labels:
        return assignment
    if field_positions:
        assignment.offense_cluster = _find_offense(assignment.labels, field_positions)
        if assignment.offense_cluster is None:
            assignment.notes.append(
                "could not find an offensive line formation, so offense/defense could "
                "not be resolved"
            )
    return assignment


def _find_offense(
    labels: dict[int, int], field_positions: dict[int, tuple[float, float]]
) -> int | None:
    """The offense is the side whose five most depth-packed players form a wide line.

    Searching for five consecutive players along the sideline (y) looks right and
    is not: the quarterback and the backs stand in the gaps of the offensive line,
    so that window is the line plus the backfield and its depth span is six yards.
    An absolute cutoff of four yards then rejects the true offense on most snaps.

    Searching along depth (x) instead finds the five players who already share an
    x, and the one that is also wide is the line. A stacked backfield is tight in
    both directions; a defensive front is wider but not as shallow. The relative
    score is what decides, not a magic number of yards.
    """
    scored: list[tuple[float, int]] = []
    for cluster in (0, 1):
        points = [
            field_positions[tid]
            for tid, label in labels.items()
            if label == cluster and tid in field_positions
        ]
        window = _line_window(points)
        if window is None:
            continue
        scored.append((_line_score(window), cluster))
    if not scored:
        return None
    scored.sort()
    return scored[0][1]


def _line_window(points: list[tuple[float, float]]) -> np.ndarray | None:
    """The five players most likely to be a line: packed in depth, spread in width."""
    if len(points) < 5:
        return None
    arr = np.array(points)
    order = np.argsort(arr[:, 0])
    best, best_score = None, np.inf
    for start in range(0, len(order) - 4):
        window = arr[order[start : start + 5]]
        score = _line_score(window)
        if score < best_score:
            best_score, best = score, window
    return best


def _line_score(window: np.ndarray) -> float:
    """Lower is more line-like: small depth, then large width."""
    depth = float(window[:, 0].max() - window[:, 0].min())
    width = float(window[:, 1].max() - window[:, 1].min())
    return depth - 0.15 * min(width, 12.0)


def estimate_line_of_scrimmage(
    field_positions: dict[int, tuple[float, float]],
    labels: dict[int, int],
    offense_cluster: int,
) -> float | None:
    """Downfield coordinate of the line of scrimmage, from the offensive line.

    The same five players `_find_offense` used, so the origin and the side
    assignment cannot disagree about where the line is.
    """
    points = [
        field_positions[tid]
        for tid, label in labels.items()
        if label == offense_cluster and tid in field_positions
    ]
    window = _line_window(points)
    if window is None:
        return None
    return float(np.median(window[:, 0]))
