"""Detection and tracking metrics, scored against corpus ground truth.

The generic metrics are here because you cannot tune a detector without them.
The reason this module is not just a wrapper around a standard mAP function is
the tracking half.

Standard multi-object tracking treats every identity switch as one error. That is
the wrong loss for this project. Two linebackers swapping IDs changes nothing:
the coverage model reads a set of defenders, and the set is identical. A safety
swapping with a corner is a disaster, because the cue vocabulary asks questions
like "how deep is the deepest defender" and "did a corner press", and after that
swap it answers both wrong for the rest of the play. So switches are counted, but
they are also split by whether they crossed a role boundary, and it is the
crossing ones that gate whether a play is usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from gridiron.vision import require
from gridiron.vision.detect import Detection

Box = tuple[float, float, float, float]


def iou_matrix(pred: Sequence[Box], truth: Sequence[Box]) -> np.ndarray:
    if not pred or not truth:
        return np.zeros((len(pred), len(truth)), dtype=np.float64)

    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    ix1 = np.maximum(p[:, None, 0], t[None, :, 0])
    iy1 = np.maximum(p[:, None, 1], t[None, :, 1])
    ix2 = np.minimum(p[:, None, 2], t[None, :, 2])
    iy2 = np.minimum(p[:, None, 3], t[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)

    area_p = np.clip(p[:, 2] - p[:, 0], 0, None) * np.clip(p[:, 3] - p[:, 1], 0, None)
    area_t = np.clip(t[:, 2] - t[:, 0], 0, None) * np.clip(t[:, 3] - t[:, 1], 0, None)
    union = area_p[:, None] + area_t[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def greedy_match(
    pred: Sequence[Box],
    truth: Sequence[Box],
    iou_threshold: float = 0.5,
    scores: Sequence[float] | None = None,
) -> list[tuple[int, int, float]]:
    """Match predictions to truth, best IoU first. Returns (pred, truth, iou)."""
    ious = iou_matrix(pred, truth)
    if ious.size == 0:
        return []

    order = (
        np.argsort(-np.asarray(scores, dtype=np.float64))
        if scores is not None
        else np.arange(len(pred))
    )
    taken_truth: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for pi in order:
        row = ious[pi].copy()
        for ti in taken_truth:
            row[ti] = -1.0
        ti = int(np.argmax(row))
        if row[ti] >= iou_threshold:
            taken_truth.add(ti)
            matches.append((int(pi), ti, float(row[ti])))
    return matches


@dataclass
class DetectionMetrics:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    frames: int = 0
    matched_ious: list[float] = field(default_factory=list)
    # Kept rather than reduced to a number, because average precision is a
    # ranking statistic: pooling it across clips means pooling the scores, and
    # averaging per-clip values gives a different and wrong answer.
    scored: list[tuple[float, bool]] = field(default_factory=list)

    @property
    def average_precision(self) -> float:
        return average_precision(
            self.scored, self.true_positives + self.false_negatives
        )

    def merge(self, other: "DetectionMetrics") -> "DetectionMetrics":
        self.true_positives += other.true_positives
        self.false_positives += other.false_positives
        self.false_negatives += other.false_negatives
        self.frames += other.frames
        self.matched_ious.extend(other.matched_ious)
        self.scored.extend(other.scored)
        return self

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mean_iou(self) -> float:
        return float(np.mean(self.matched_ious)) if self.matched_ious else 0.0

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "average_precision": round(self.average_precision, 4),
            "mean_iou": round(self.mean_iou, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "frames": self.frames,
        }


def average_precision(
    scored: list[tuple[float, bool]], n_truth: int
) -> float:
    """Area under the precision-recall curve, by the all-points interpolation."""
    if not scored or n_truth <= 0:
        return 0.0
    scored = sorted(scored, key=lambda s: -s[0])
    tp = np.cumsum([1.0 if hit else 0.0 for _, hit in scored])
    fp = np.cumsum([0.0 if hit else 1.0 for _, hit in scored])
    recall = tp / n_truth
    precision = tp / np.maximum(tp + fp, 1e-9)

    # Make precision monotonically decreasing, then integrate over recall.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def evaluate_detections(
    per_frame: Iterable[tuple[list[Detection], list[Box]]],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Score detections against truth boxes, frame by frame."""
    metrics = DetectionMetrics()

    for detections, truth_boxes in per_frame:
        metrics.frames += 1
        pred_boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
        scores = [d.confidence for d in detections]
        matches = greedy_match(pred_boxes, truth_boxes, iou_threshold, scores)

        matched_preds = {pi for pi, _, _ in matches}
        for _, _, iou in matches:
            metrics.matched_ious.append(iou)
        metrics.true_positives += len(matches)
        metrics.false_positives += len(pred_boxes) - len(matches)
        metrics.false_negatives += len(truth_boxes) - len(matches)

        for pi, score in enumerate(scores):
            metrics.scored.append((score, pi in matched_preds))

    return metrics


@dataclass
class TrackingMetrics:
    id_switches: int = 0
    role_crossing_switches: int = 0
    side_crossing_switches: int = 0
    truth_tracks: int = 0
    mostly_tracked: int = 0
    fragments: int = 0
    frames: int = 0

    @property
    def mostly_tracked_fraction(self) -> float:
        return self.mostly_tracked / self.truth_tracks if self.truth_tracks else 0.0

    @property
    def switches_per_track(self) -> float:
        return self.id_switches / self.truth_tracks if self.truth_tracks else 0.0

    @property
    def usable(self) -> bool:
        """Whether cue extraction can trust this play.

        Keyed on role-crossing switches rather than the raw count, for the reason
        in the module docstring.
        """
        if not self.truth_tracks:
            return False
        return (
            self.role_crossing_switches <= 1
            and self.mostly_tracked_fraction >= 0.8
        )

    def as_dict(self) -> dict:
        return {
            "id_switches": self.id_switches,
            "role_crossing_switches": self.role_crossing_switches,
            "side_crossing_switches": self.side_crossing_switches,
            "switches_per_track": round(self.switches_per_track, 4),
            "mostly_tracked_fraction": round(self.mostly_tracked_fraction, 4),
            "truth_tracks": self.truth_tracks,
            "fragments": self.fragments,
            "frames": self.frames,
            "usable": self.usable,
        }


def evaluate_tracking(
    per_frame: Iterable[tuple[list[Any], list[dict]]],
    iou_threshold: float = 0.4,
) -> TrackingMetrics:
    """Score tracker identity against truth identity.

    `per_frame` yields (tracked_detections, truth_entries), where a truth entry
    carries `track_id`, `box`, and ideally `role` and `side`.
    """
    metrics = TrackingMetrics()
    # Which truth id each tracker id was last seen on, and vice versa.
    tracker_to_truth: dict[int, int] = {}
    truth_meta: dict[int, dict] = {}
    truth_seen: dict[int, int] = {}
    truth_matched: dict[int, int] = {}
    last_frame_matched: dict[int, bool] = {}

    for tracked, truth_entries in per_frame:
        metrics.frames += 1
        truth_boxes = [tuple(e["box"]) for e in truth_entries]
        for entry in truth_entries:
            truth_meta.setdefault(entry["track_id"], entry)
            truth_seen[entry["track_id"]] = truth_seen.get(entry["track_id"], 0) + 1

        pred_boxes = [
            (t.detection.x1, t.detection.y1, t.detection.x2, t.detection.y2)
            for t in tracked
        ]
        matches = greedy_match(pred_boxes, truth_boxes, iou_threshold)

        matched_truth_this_frame: set[int] = set()
        for pi, ti, _ in matches:
            tracker_id = tracked[pi].track_id
            truth_id = truth_entries[ti]["track_id"]
            matched_truth_this_frame.add(truth_id)
            truth_matched[truth_id] = truth_matched.get(truth_id, 0) + 1

            previous = tracker_to_truth.get(tracker_id)
            if previous is not None and previous != truth_id:
                metrics.id_switches += 1
                before, after = truth_meta.get(previous, {}), truth_meta.get(truth_id, {})
                if before.get("side") != after.get("side"):
                    metrics.side_crossing_switches += 1
                if _role_group(before.get("role")) != _role_group(after.get("role")):
                    metrics.role_crossing_switches += 1
            tracker_to_truth[tracker_id] = truth_id

        for truth_id in truth_seen:
            was = last_frame_matched.get(truth_id, False)
            now = truth_id in matched_truth_this_frame
            if was and not now:
                metrics.fragments += 1
            last_frame_matched[truth_id] = now

    metrics.truth_tracks = len(truth_seen)
    metrics.mostly_tracked = sum(
        1
        for tid, seen in truth_seen.items()
        if seen and truth_matched.get(tid, 0) / seen >= 0.8
    )
    return metrics


# Roles collapse to depth responsibility, because that is the axis the cue
# vocabulary reads. Swapping two underneath zone defenders leaves every cue
# unchanged - they are at similar depth doing similar things, and the cues are
# geometric. Swapping a deep zone defender with an underneath one moves a
# trajectory ten yards vertically, which changes the safety count and the
# deepest-defender depth, and those are the cues that separate the shells.
_ROLE_GROUPS = {
    "deep_zone": "deep",
    "underneath_zone": "shallow",
    "man": "shallow",
    "blitz": "rush",
    # Position labels, for whenever real film brings them instead.
    "s": "deep",
    "fs": "deep",
    "ss": "deep",
    "cb": "shallow",
    "lb": "shallow",
    "dl": "rush",
    "de": "rush",
    "dt": "rush",
    "edge": "rush",
}


def _role_group(role: str | None) -> str:
    if not role:
        return "unknown"
    return _ROLE_GROUPS.get(role.lower(), role.lower())


def evaluate_clip(
    video_path: Path,
    truth: Any,
    detector: Any,
    tracker: Any | None = None,
    max_frames: int | None = None,
    iou_threshold: float = 0.5,
) -> tuple[DetectionMetrics, TrackingMetrics]:
    """Run a detector (and optionally a tracker) over a clip and score both."""
    from gridiron.vision.track import build_tracker

    cv2 = require("cv2")
    tracker = tracker if tracker is not None else build_tracker(frame_rate=int(truth.fps))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    detection_pairs: list[tuple[list[Detection], list[Box]]] = []
    tracking_pairs: list[tuple[list[Any], list[dict]]] = []
    try:
        index = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            if max_frames is not None and index >= max_frames:
                break

            entries = truth.frames.get(str(index), [])
            detections = detector.detect(frame)
            detection_pairs.append((detections, [tuple(e["box"]) for e in entries]))
            tracking_pairs.append((tracker.update(detections, index), entries))
    finally:
        cap.release()

    return (
        evaluate_detections(detection_pairs, iou_threshold=iou_threshold),
        evaluate_tracking(tracking_pairs),
    )


@dataclass
class RegistrationError:
    """How far off a homography is, measured where the players are.

    Reported in yards on the field rather than pixels, because a pixel is worth
    very different amounts of field at the near and far sidelines, and because
    yards are the units the coverage model was trained in.
    """

    lateral_rmse_yd: float = float("inf")
    downfield_rmse_yd: float = float("inf")
    downfield_offset_yd: float = 0.0
    raw_rmse_yd: float = float("inf")
    method: str = "none"

    @property
    def usable(self) -> bool:
        """Whether coverage reads survive this much error.

        The downfield offset is excluded on purpose. Line-based registration
        cannot tell the 30 from the 35 without reading the painted numbers, so it
        anchors at midfield and can sit a whole multiple of five yards off. That
        constant shift is harmless: every coverage cue is a depth relative to the
        line of scrimmage, and the pipeline locates the line of scrimmage from the
        players themselves, so a uniform slide cancels. Error that varies across
        the field does not cancel, and that is what these two terms measure.
        """
        return self.lateral_rmse_yd < 1.5 and self.downfield_rmse_yd < 1.5

    def as_dict(self) -> dict:
        return {
            "lateral_rmse_yd": round(self.lateral_rmse_yd, 3),
            "downfield_rmse_yd": round(self.downfield_rmse_yd, 3),
            "downfield_offset_yd": round(self.downfield_offset_yd, 3),
            "raw_rmse_yd": round(self.raw_rmse_yd, 3),
            "method": self.method,
            "usable": self.usable,
        }


def registration_error(
    estimated: Any,
    truth_image_to_field: np.ndarray,
    frame_shape: tuple[int, ...],
    samples: int = 12,
    margin: float = 0.1,
) -> RegistrationError:
    """Compare an estimated homography to a known one, in field yards.

    Sampling happens over the image rather than over the field, so the error is
    weighted the way the frame is: the far sideline occupies few pixels and gets
    few samples, which is correct, because few players are found there.
    """
    if getattr(estimated, "homography", None) is None:
        return RegistrationError(method=getattr(estimated, "method", "none"))

    cv2 = require("cv2")
    height, width = frame_shape[:2]
    us = np.linspace(width * margin, width * (1 - margin), samples)
    # Only the lower part of the frame is field; the top is stands and sky.
    vs = np.linspace(height * 0.45, height * (1 - margin), samples)
    grid = np.array([[u, v] for v in vs for u in us], dtype=np.float32).reshape(-1, 1, 2)

    truth_xy = cv2.perspectiveTransform(
        grid, np.asarray(truth_image_to_field, dtype=np.float64)
    ).reshape(-1, 2)
    got_xy = cv2.perspectiveTransform(
        grid, np.asarray(estimated.homography, dtype=np.float64)
    ).reshape(-1, 2)

    ok = np.all(np.isfinite(truth_xy), axis=1) & np.all(np.isfinite(got_xy), axis=1)
    if not ok.any():
        return RegistrationError(method=getattr(estimated, "method", "none"))

    dx = got_xy[ok, 0] - truth_xy[ok, 0]
    dy = got_xy[ok, 1] - truth_xy[ok, 1]
    offset = float(np.median(dx))

    return RegistrationError(
        lateral_rmse_yd=float(np.sqrt(np.mean(dy**2))),
        downfield_rmse_yd=float(np.sqrt(np.mean((dx - offset) ** 2))),
        downfield_offset_yd=offset,
        raw_rmse_yd=float(np.sqrt(np.mean(dx**2 + dy**2))),
        method=getattr(estimated, "method", "unknown"),
    )


def evaluate_registration_on_clip(
    video_path: Path,
    truth: Any,
    registrar: Any,
    frames_to_probe: int = 5,
) -> list[RegistrationError]:
    """Register several frames of a clip and score each against the truth."""
    cv2 = require("cv2")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        wanted = (
            set(np.linspace(0, total - 1, frames_to_probe).astype(int).tolist())
            if total
            else set(range(frames_to_probe))
        )
        errors: list[RegistrationError] = []
        index = -1
        while wanted:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            if index not in wanted:
                continue
            wanted.discard(index)
            errors.append(
                registration_error(
                    registrar.register(frame), truth.image_to_field, frame.shape
                )
            )
        return errors
    finally:
        cap.release()


def evaluate_corpus(
    corpus: Any,
    detector: Any,
    max_frames_per_clip: int | None = 60,
    iou_threshold: float = 0.5,
    split: str | None = None,
) -> dict:
    """Score a detector across clips that have ground truth.

    Pass `split="val"` after training on this corpus. The splits are per clip, so
    this is the only number that means anything once the detector has been fit.
    """
    from gridiron.vision.render import RenderedTruth

    detection = DetectionMetrics()
    tracking = TrackingMetrics()
    clips = 0
    usable_clips = 0

    for clip in corpus:
        if split is not None and clip.split != split:
            continue
        if not clip.truth_path or not Path(clip.truth_path).exists():
            continue
        if not clip.video.exists():
            continue
        truth = RenderedTruth.load(Path(clip.truth_path))
        det, trk = evaluate_clip(
            clip.video,
            truth,
            detector,
            max_frames=max_frames_per_clip,
            iou_threshold=iou_threshold,
        )
        clips += 1
        usable_clips += int(trk.usable)

        detection.merge(det)

        tracking.id_switches += trk.id_switches
        tracking.role_crossing_switches += trk.role_crossing_switches
        tracking.side_crossing_switches += trk.side_crossing_switches
        tracking.truth_tracks += trk.truth_tracks
        tracking.mostly_tracked += trk.mostly_tracked
        tracking.fragments += trk.fragments
        tracking.frames += trk.frames

    pooled = tracking.as_dict()
    # Usability is a per-play gate - a play is kept or thrown out whole - so the
    # pooled flag would answer a question nobody asks. Count the plays instead.
    pooled.pop("usable", None)
    pooled["usable_clips"] = usable_clips
    pooled["usable_fraction"] = round(usable_clips / clips, 4) if clips else 0.0

    return {
        "clips": clips,
        "detection": detection.as_dict(),
        "tracking": pooled,
    }
