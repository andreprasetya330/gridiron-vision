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

from dataclasses import dataclass
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
    """IoU plus predicted-centroid tracker.

    Players on All-22 move a few dozen pixels per frame. Matching the last box
    alone swaps identities every time two similar jerseys cross. Predicting the
    next center from recent velocity, then smoothing the accepted box, is the
    cheapest Kalman that still holds a rusher through the pile.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._next_id = 0
        self._tracks: dict[int, dict[str, Any]] = {}

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        frame: np.ndarray | None = None,
    ) -> list[TrackedDetection]:
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
                dt = max(1, frame_index - int(track["last_frame"]))
                vx, vy = track.get("velocity", (0.0, 0.0))
                predicted = (
                    track["center"][0] + vx * dt,
                    track["center"][1] + vy * dt,
                )
                iou = _iou(det, track["box"])
                distance = float(np.linalg.norm(np.array(det.center) - np.array(predicted)))
                gate = cfg.max_distance_px * (1.0 + 0.12 * (dt - 1))
                if iou < cfg.iou_threshold and distance > gate:
                    continue
                cost = (1.0 - iou) + distance / max(gate, 1.0)
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
                self._tracks[tid] = {
                    "hits": 0,
                    "created": frame_index,
                    "velocity": (0.0, 0.0),
                }
            track = self._tracks.setdefault(
                tid, {"hits": 0, "created": frame_index, "velocity": (0.0, 0.0)}
            )
            prev_center = track.get("center")
            prev_frame = track.get("last_frame", frame_index)
            prev_box = track.get("box")
            alpha = 0.55 if prev_box is not None else 1.0
            box = (
                alpha * det.x1 + (1.0 - alpha) * prev_box[0],
                alpha * det.y1 + (1.0 - alpha) * prev_box[1],
                alpha * det.x2 + (1.0 - alpha) * prev_box[2],
                alpha * det.y2 + (1.0 - alpha) * prev_box[3],
            ) if prev_box is not None else (det.x1, det.y1, det.x2, det.y2)
            center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            dt = max(1, frame_index - int(prev_frame)) if prev_center is not None else 1
            if prev_center is not None:
                inst = (
                    (center[0] - prev_center[0]) / dt,
                    (center[1] - prev_center[1]) / dt,
                )
                old_v = track.get("velocity", (0.0, 0.0))
                velocity = (0.65 * old_v[0] + 0.35 * inst[0], 0.65 * old_v[1] + 0.35 * inst[1])
            else:
                velocity = (0.0, 0.0)
            track.update(
                {
                    "box": box,
                    "center": center,
                    "last_frame": frame_index,
                    "hits": track.get("hits", 0) + 1,
                    "velocity": velocity,
                }
            )
            smoothed = Detection(
                box[0], box[1], box[2], box[3], det.confidence, det.class_id, det.class_name
            )
            output.append(TrackedDetection(smoothed, tid, frame_index))
        return output

    def track_stats(self) -> dict[int, dict[str, Any]]:
        return self._tracks


class _ResultsView:
    """The minimal `Results`-like object ultralytics' trackers consume.

    They touch `xywh`, `xyxy`, `conf`, `cls`, `len()`, and boolean indexing, so
    a view over our own detections is enough. Going through this shim rather
    than `model.track()` keeps the detector injectable, which is what lets the
    pipeline be tested against known-truth boxes.

    `xyxy` is not optional: camera-motion compensation uses it to mask players
    out before estimating the camera transform, and without it GMC silently
    falls back to an identity transform - the tracker still runs, it just stops
    compensating for the pan, which is the whole reason it is here.
    """

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    @property
    def xyxy(self) -> np.ndarray:
        if self.xywh.size == 0:
            return self.xywh.reshape(0, 4)
        cx, cy, w, h = (self.xywh[:, i] for i in range(4))
        return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, mask) -> _ResultsView:
        return _ResultsView(self.xywh[mask], self.conf[mask], self.cls[mask])


class BotSortTracker:
    """BoT-SORT: Kalman motion model plus global camera-motion compensation.

    All-22 pans and zooms on almost every snap. A tracker that assumes a fixed
    camera spends its whole distance budget explaining the pan, and then has
    nothing left to tell two crossing defenders apart - which is the identity
    churn that makes an overlay look like it is guessing. GMC estimates the
    frame-to-frame camera transform with sparse optical flow and subtracts it,
    so the motion the tracker sees is the motion of the players.
    """

    def __init__(self, frame_rate: int = 30, track_buffer: int = 60) -> None:
        from types import SimpleNamespace

        from ultralytics.trackers import BOTSORT

        self.args = SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=0.25,
            track_low_thresh=0.08,
            # Only a confident box may *start* an identity, while weak boxes are
            # still used to extend one. That asymmetry is what stops a flickering
            # low-score detection from spawning a fresh track every few frames.
            new_track_thresh=0.55,
            # Players disappear behind the pile for most of a second. Holding a
            # lost track that long is what lets it be re-found as itself.
            track_buffer=track_buffer,
            match_thresh=0.85,
            fuse_score=True,
            gmc_method="sparseOptFlow",
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            with_reid=False,
            model="auto",
        )
        self.tracker = BOTSORT(self.args)
        self.frame_rate = frame_rate

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        frame: np.ndarray | None = None,
    ) -> list[TrackedDetection]:
        if not detections:
            return []
        xywh = np.array(
            [
                [
                    (d.x1 + d.x2) / 2.0,
                    (d.y1 + d.y2) / 2.0,
                    d.x2 - d.x1,
                    d.y2 - d.y1,
                ]
                for d in detections
            ],
            dtype=np.float32,
        )
        conf = np.array([d.confidence for d in detections], dtype=np.float32)
        cls = np.array([d.class_id for d in detections], dtype=np.float32)

        tracks = self.tracker.update(_ResultsView(xywh, conf, cls), frame)

        output: list[TrackedDetection] = []
        for row in np.asarray(tracks):
            if row.size < 7:
                continue
            x1, y1, x2, y2 = (float(v) for v in row[:4])
            track_id = int(row[4])
            score = float(row[5])
            class_id = int(row[6])
            source = detections[int(row[7])] if row.size > 7 else None
            output.append(
                TrackedDetection(
                    Detection(
                        x1,
                        y1,
                        x2,
                        y2,
                        score,
                        class_id,
                        source.class_name if source else "player",
                    ),
                    track_id,
                    frame_index,
                )
            )
        return output


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
    """Best available tracker, degrading rather than failing.

    BoT-SORT first because camera-motion compensation is the single biggest win
    on panning film; the hand-rolled tracker last so the pipeline still runs
    with only numpy installed.
    """
    try:
        return BotSortTracker(frame_rate=frame_rate)
    except Exception:
        pass
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
