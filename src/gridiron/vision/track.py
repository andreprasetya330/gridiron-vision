"""Multi-object tracking.

ByteTrack when supervision is installed, and a self-contained IoU tracker when it
is not, so the pipeline degrades rather than dying. Football tracking is easier
than the general case - the camera is wide and stable, players do not leave the
scene - and harder in one specific way: twenty-two people in near-identical
uniforms cross each other constantly, so identity swaps are the failure mode that
matters.

A swap between two defenders playing the same role costs nothing. A swap between a
safety and a corner ruins the cue extraction for that play, which is why the
tracker reports per-track continuity and the pipeline propagates it into track
quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gridiron.vision.detect import Detection


@dataclass
class TrackedDetection:
    detection: Detection
    track_id: int
    frame_index: int


@dataclass
class TrackerConfig:
    max_age: int = 15  # frames a track survives unmatched
    min_hits: int = 3
    iou_threshold: float = 0.3
    max_distance_px: float = 90.0


class SimpleTracker:
    """IoU plus centroid-distance tracker.

    Deliberately simple. On stable wide-angle film with 30 fps and players moving
    under 10 yards/second, greedy IoU matching with a distance gate holds identity
    about as well as a Kalman filter, and it has no dependencies and no tuning.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._next_id = 0
        self._tracks: dict[int, dict[str, Any]] = {}

    def update(self, detections: list[Detection], frame_index: int) -> list[TrackedDetection]:
        cfg = self.config
        active = {
            tid: t for tid, t in self._tracks.items() if frame_index - t["last_frame"] <= cfg.max_age
        }
        self._tracks = active

        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()

        pairs: list[tuple[float, int, int]] = []
        for di, det in enumerate(detections):
            for tid, track in active.items():
                iou = _iou(det, track["box"])
                distance = float(np.linalg.norm(np.array(det.center) - np.array(track["center"])))
                if iou < cfg.iou_threshold and distance > cfg.max_distance_px:
                    continue
                cost = (1.0 - iou) + distance / max(cfg.max_distance_px, 1.0)
                pairs.append((cost, di, tid))

        for _, di, tid in sorted(pairs):
            if di in assignments or tid in used_tracks:
                continue
            assignments[di] = tid
            used_tracks.add(tid)

        output: list[TrackedDetection] = []
        for di, det in enumerate(detections):
            tid = assignments.get(di)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"hits": 0, "created": frame_index}
            track = self._tracks.setdefault(tid, {"hits": 0, "created": frame_index})
            track.update(
                {
                    "box": (det.x1, det.y1, det.x2, det.y2),
                    "center": det.center,
                    "last_frame": frame_index,
                    "hits": track.get("hits", 0) + 1,
                }
            )
            output.append(TrackedDetection(det, tid, frame_index))
        return output

    def track_stats(self) -> dict[int, dict[str, Any]]:
        return self._tracks


class ByteTrackAdapter:
    """supervision's ByteTrack, when available."""

    def __init__(self, frame_rate: int = 30) -> None:
        from gridiron.vision import require

        sv = require("supervision")
        self.sv = sv
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The `ByteTrack` was deprecated",
                category=FutureWarning,
            )
            self.tracker = sv.ByteTrack(frame_rate=frame_rate)

    def update(self, detections: list[Detection], frame_index: int) -> list[TrackedDetection]:
        if not detections:
            return []
        sv = self.sv
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_ids = np.array([d.class_id for d in detections], dtype=int)

        sv_detections = sv.Detections(xyxy=boxes, confidence=confidence, class_id=class_ids)
        tracked = self.tracker.update_with_detections(sv_detections)

        output: list[TrackedDetection] = []
        for i in range(len(tracked)):
            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
            tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else i
            output.append(
                TrackedDetection(
                    Detection(
                        x1,
                        y1,
                        x2,
                        y2,
                        float(tracked.confidence[i]) if tracked.confidence is not None else 1.0,
                        int(tracked.class_id[i]) if tracked.class_id is not None else 0,
                    ),
                    tid,
                    frame_index,
                )
            )
        return output


def build_tracker(frame_rate: int = 30, prefer_bytetrack: bool = True):
    if prefer_bytetrack:
        try:
            return ByteTrackAdapter(frame_rate=frame_rate)
        except Exception:
            pass
    return SimpleTracker()


def continuity_score(tracks: dict[int, list[int]], n_frames: int) -> float:
    """Fraction of frames the average track was actually observed.

    Low continuity means the tracker is fragmenting identities, which shows up
    downstream as defenders teleporting and velocities exploding.
    """
    if not tracks or n_frames <= 0:
        return 0.0
    spans = [len(frames) / n_frames for frames in tracks.values()]
    return float(np.mean(spans))


def _iou(det: Detection, box: tuple[float, float, float, float] | None) -> float:
    if box is None:
        return 0.0
    ax1, ay1, ax2, ay2 = det.x1, det.y1, det.x2, det.y2
    bx1, by1, bx2, by2 = box
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0
