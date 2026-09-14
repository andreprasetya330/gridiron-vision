"""Video in, `PlayTracks` out.

This is the seam in the whole architecture. Everything upstream of here is
computer vision; everything downstream consumes one JSON contract and neither
knows nor cares that a camera was ever involved. A future Hudl, Catapult, or PFF
importer produces the same object and skips this module entirely.

The pipeline is deliberately loud about failure. A play processed from film with
two safeties missing and a shaky homography still produces output, but it produces
output stamped with a quality score that makes it downweighted in training, sorted
into the review queue in the UI, and excluded from tell mining.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from gridiron.fields import FIELD_CENTER_Y, get_field
from gridiron.tracking.schema import (
    N_FRAMES,
    POST_SNAP_SECONDS,
    PRE_SNAP_SECONDS,
    PlayerTrack,
    PlayTracks,
    Situation,
    TrackQuality,
    resample_to_grid,
)
from gridiron.vision import require
from gridiron.vision.detect import Detection, DetectorConfig, PlayerDetector, filter_sideline_detections
from gridiron.vision.registration import (
    LineRegistrar,
    Registration,
    SmoothedRegistration,
)
from gridiron.vision.snap import find_snap, motion_energy, refine_with_ball, segment_plays
from gridiron.vision.teams import assign_teams, estimate_line_of_scrimmage
from gridiron.vision.track import build_tracker


@dataclass
class PipelineConfig:
    league: str = "ncaa"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    registration_weights: str | None = None
    manual_correspondences: tuple[np.ndarray, np.ndarray] | None = None
    stride: int = 1
    max_frames: int | None = None
    single_play: bool = True  # a Hudl-style cutup with one snap
    play_direction: str = "right"


def read_frames(
    video_path: Path, stride: int = 1, max_frames: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    cv2 = require("cv2")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    index = 0
    emitted = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                yield index, frame
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            index += 1
    finally:
        capture.release()


def video_fps(video_path: Path) -> float:
    cv2 = require("cv2")
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    return float(fps)


class FilmPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        detector: Any | None = None,
        registrar: Any | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        # Both stages are injectable so the pipeline can be validated against
        # known ground truth with one stage swapped out at a time. Isolating a
        # failure to detection, registration, or the geometry between them is
        # otherwise guesswork.
        self.detector = detector if detector is not None else PlayerDetector(self.config.detector)
        self._injected_registrar = registrar
        self.spec = get_field(self.config.league)

    def _registrar(self):
        if self._injected_registrar is not None:
            return self._injected_registrar
        if self.config.registration_weights:
            from gridiron.vision.registration import KeypointRegistrar

            return KeypointRegistrar(self.config.registration_weights, self.config.league)
        return LineRegistrar(self.config.league)

    def process(self, video_path: Path, play_id: str | None = None) -> list[PlayTracks]:
        video_path = Path(video_path)
        fps = video_fps(video_path)
        registrar = self._registrar()
        smoother = SmoothedRegistration()
        tracker = build_tracker(frame_rate=int(round(fps)))

        frames_meta: list[dict[str, Any]] = []
        raw_frames: list[np.ndarray] = []

        for frame_index, frame in read_frames(
            video_path, self.config.stride, self.config.max_frames
        ):
            detections = filter_sideline_detections(self.detector.detect(frame), frame.shape[0])
            players = [d for d in detections if d.class_name == "player"]
            balls = [d for d in detections if d.class_name == "ball"]

            tracked = tracker.update(players, frame_index)

            if self.config.manual_correspondences is not None:
                from gridiron.vision.registration import manual_homography

                registration = manual_homography(
                    *self.config.manual_correspondences, self.config.league
                )
            else:
                registration = registrar.register(frame)
            registration = smoother.update(registration)

            field_positions: dict[int, tuple[float, float]] = {}
            if registration.homography is not None and tracked:
                foot_points = np.array([t.detection.foot_point for t in tracked])
                mapped = registration.to_field(foot_points)
                for t, point in zip(tracked, mapped):
                    if np.all(np.isfinite(point)):
                        field_positions[t.track_id] = (float(point[0]), float(point[1]))

            ball_point = None
            if balls and registration.homography is not None:
                mapped = registration.to_field(np.array([balls[0].foot_point]))
                if np.all(np.isfinite(mapped[0])):
                    ball_point = (float(mapped[0][0]), float(mapped[0][1]))

            frames_meta.append(
                {
                    "frame_index": frame_index,
                    "tracked": tracked,
                    "field_positions": field_positions,
                    "registration": registration,
                    "ball": ball_point,
                    "n_detections": len(players),
                }
            )
            raw_frames.append(frame)

        if not frames_meta:
            return []

        positions_by_frame = [m["field_positions"] for m in frames_meta]
        energy = motion_energy(positions_by_frame)

        if self.config.single_play:
            snap = find_snap(energy, fps=fps / max(self.config.stride, 1))
            if snap is None:
                snap_indices = [len(frames_meta) // 3]
                snap_confidence = 0.0
            else:
                snap_indices = [snap.frame_index]
                snap_confidence = snap.confidence
            segments = [(max(0, i - int(fps * 3)), i, min(len(frames_meta) - 1, i + int(fps * 5))) for i in snap_indices]
            confidences = [snap_confidence]
        else:
            found = segment_plays(energy, fps=fps / max(self.config.stride, 1))
            segments = [(s.start_frame, s.snap_frame, s.end_frame) for s in found]
            confidences = [s.confidence for s in found]

        plays: list[PlayTracks] = []
        for i, ((start, snap_idx, end), confidence) in enumerate(zip(segments, confidences)):
            ball_positions = {
                j: m["ball"] for j, m in enumerate(frames_meta) if m["ball"] is not None
            }
            snap_idx = refine_with_ball(snap_idx, ball_positions, fps)
            play = self._build_play(
                video_path=video_path,
                frames_meta=frames_meta,
                raw_frames=raw_frames,
                start=start,
                snap_idx=snap_idx,
                end=end,
                fps=fps / max(self.config.stride, 1),
                snap_confidence=confidence,
                play_id=play_id or f"{video_path.stem}-{i:03d}",
            )
            if play is not None:
                plays.append(play)
        return plays

    def _build_play(
        self,
        video_path: Path,
        frames_meta: list[dict[str, Any]],
        raw_frames: list[np.ndarray],
        start: int,
        snap_idx: int,
        end: int,
        fps: float,
        snap_confidence: float,
        play_id: str,
    ) -> PlayTracks | None:
        snap_meta = frames_meta[min(snap_idx, len(frames_meta) - 1)]
        snap_positions = snap_meta["field_positions"]
        if len(snap_positions) < 10:
            return None

        assignment = assign_teams(
            raw_frames[min(snap_idx, len(raw_frames) - 1)],
            snap_meta["tracked"],
            snap_positions,
        )
        notes = list(assignment.notes)

        if assignment.offense_cluster is None:
            # Colour clustering produced two groups, but neither looks like a
            # line. Guessing is still better than dropping the play: the overlay
            # can show it for review, and a missing play cannot.
            from collections import Counter

            counts = Counter(assignment.labels.values())
            if counts:
                assignment.offense_cluster = counts.most_common(1)[0][0]
                notes.append(
                    "offense/defense could not be resolved from formation; "
                    "sides are a guess and this play should be reviewed"
                )
            else:
                notes.append("offense/defense unresolved; play emitted for manual review")
                return None

        los_x = estimate_line_of_scrimmage(
            snap_positions, assignment.labels, assignment.offense_cluster
        )
        if los_x is None:
            notes.append("line of scrimmage could not be estimated")
            return None

        # The median of everyone on the field looks like it should skew toward
        # trips, and it does, but it still beat every landmark-based alternative
        # tried in scripts/ball_spot_study.py - the alignment-based ones are less
        # biased and much noisier, and the tail is what breaks a play.
        ball_y = snap_meta["ball"][1] if snap_meta["ball"] else float(
            np.median([p[1] for p in snap_positions.values()])
        )

        direction = self.config.play_direction
        sign = 1.0 if direction == "right" else -1.0

        # Collect each track's observations in normalized coordinates.
        series: dict[int, dict[str, list[float]]] = {}
        for j in range(start, min(end + 1, len(frames_meta))):
            meta = frames_meta[j]
            t = (j - snap_idx) / fps
            if t < -PRE_SNAP_SECONDS - 0.5 or t > POST_SNAP_SECONDS + 0.5:
                continue
            for track_id, (fx, fy) in meta["field_positions"].items():
                entry = series.setdefault(track_id, {"t": [], "x": [], "y": []})
                entry["t"].append(t)
                entry["x"].append(sign * (fx - los_x))
                entry["y"].append(sign * (fy - ball_y))

        players: list[PlayerTrack] = []
        for track_id, entry in series.items():
            side = assignment.side_of(track_id)
            if side is None:
                continue
            x = resample_to_grid(np.array(entry["t"]), np.array(entry["x"]))
            y = resample_to_grid(np.array(entry["t"]), np.array(entry["y"]))
            if not np.isfinite(x).any():
                continue
            players.append(
                PlayerTrack(track_id=f"{'O' if side == 'offense' else 'D'}_{track_id}", side=side, x=x, y=y)
            )

        if not players:
            return None

        defenders = [p for p in players if p.side == "defense"]
        offense = [p for p in players if p.side == "offense"]

        registrations = [
            frames_meta[j]["registration"]
            for j in range(start, min(end + 1, len(frames_meta)))
            if frames_meta[j]["registration"].homography is not None
        ]
        registration_error = (
            float(np.median([r.reprojection_error_yd for r in registrations]))
            if registrations
            else float("inf")
        )

        full_defense_frames = sum(
            1
            for p_frame in range(N_FRAMES)
            if sum(1 for d in defenders if np.isfinite(d.x[p_frame])) >= 11
        )

        confidences = [
            t.detection.confidence
            for j in range(start, min(end + 1, len(frames_meta)))
            for t in frames_meta[j]["tracked"]
        ]

        if snap_confidence < 0.3:
            notes.append("snap frame is uncertain, so pre-snap cues may be misaligned")

        quality = TrackQuality(
            defenders_detected=len(defenders),
            offense_detected=len(offense),
            mean_detection_conf=float(np.mean(confidences)) if confidences else 0.0,
            registration_error_yd=registration_error if np.isfinite(registration_error) else 99.0,
            frames_with_full_defense=full_defense_frames / N_FRAMES,
            notes=notes,
        )

        situation = Situation(
            ball_y_from_center=round(float(sign * (ball_y - FIELD_CENTER_Y)), 2),
            hash_side=self.spec.hash_of(ball_y),
            league=self.config.league,
        )

        homography = None
        if snap_meta["registration"].homography is not None:
            homography = snap_meta["registration"].homography.tolist()

        frame_h, frame_w = raw_frames[min(snap_idx, len(raw_frames) - 1)].shape[:2]

        return PlayTracks(
            play_id=play_id,
            source="film",
            players=players,
            situation=situation,
            quality=quality,
            video_path=str(video_path),
            snap_frame_in_video=snap_idx,
            video_fps=fps,
            homography=homography,
            origin_x=float(los_x),
            origin_y=float(ball_y),
            play_direction=direction,
            video_width=int(frame_w),
            video_height=int(frame_h),
        )


def process_video(
    video_path: Path,
    config: PipelineConfig | None = None,
    play_id: str | None = None,
    detector: Any | None = None,
    registrar: Any | None = None,
) -> list[PlayTracks]:
    return FilmPipeline(config, detector=detector, registrar=registrar).process(
        video_path, play_id=play_id
    )
