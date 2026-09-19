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

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gridiron.fields import FIELD_CENTER_Y, FIELD_LENGTH_YD, FIELD_WIDTH_YD, get_field
from gridiron.tracking.schema import (
    N_FRAMES,
    POST_SNAP_SECONDS,
    PRE_SNAP_SECONDS,
    PlayerTrack,
    PlayTracks,
    Situation,
    TrackQuality,
    make_time_grid,
    resample_to_grid,
    smooth_track,
)
from gridiron.vision import require
from gridiron.vision.detect import DetectorConfig, PlayerDetector, filter_sideline_detections
from gridiron.vision.identity import (
    TrackRoster,
    apply_merges,
    build_track_embeddings,
    build_track_profiles,
    classify_tracks,
    finalize_film_players,
    merge_fragments,
)
from gridiron.vision.registration import (
    LineRegistrar,
    SmoothedRegistration,
)
from gridiron.vision.snap import find_snap, motion_energy, refine_with_ball, segment_plays
from gridiron.vision.teams import assign_teams_from_roster, estimate_line_of_scrimmage
from gridiron.vision.track import build_tracker

# Overlay follows the rest of a Hudl cutup; coverage scoring still uses ±3s.
FILM_POST_SNAP_SECONDS = 8.0


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
    # auto: Roboflow when an API key is present and no detector/registrar is injected.
    backend: str = "auto"


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


class _DetRef:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence


class _TrackedRef:
    def __init__(self, track_id: int, confidence: float) -> None:
        self.track_id = track_id
        self.detection = _DetRef(confidence)


class FilmPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        detector: Any | None = None,
        registrar: Any | None = None,
        workflow: Any | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        # Both stages are injectable so the pipeline can be validated against
        # known ground truth with one stage swapped out at a time. Isolating a
        # failure to detection, registration, or the geometry between them is
        # otherwise guesswork.
        self._injected_detector = detector
        self._injected_registrar = registrar
        self._workflow = workflow
        self.predictions: list[dict[str, Any]] = []
        self.spec = get_field(self.config.league)
        if detector is not None:
            self.detector = detector
        elif self._uses_local_backend():
            self.detector = PlayerDetector(self.config.detector)
        else:
            self.detector = None

    def _uses_local_backend(self) -> bool:
        if self._injected_detector is not None or self._injected_registrar is not None:
            return True
        if self.config.backend == "local":
            return True
        if self.config.backend == "roboflow":
            return False
        from gridiron.config import RoboflowSettings

        return not RoboflowSettings().enabled

    def _registrar(self):
        if self._injected_registrar is not None:
            return self._injected_registrar
        if self.config.registration_weights:
            from gridiron.vision.registration import KeypointRegistrar

            return KeypointRegistrar(self.config.registration_weights, self.config.league)
        return LineRegistrar(self.config.league)

    def process(self, video_path: Path, play_id: str | None = None) -> list[PlayTracks]:
        video_path = Path(video_path)
        self.predictions = []
        if video_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            return self._process_image(video_path, play_id=play_id)
        if not self._uses_local_backend():
            return self._process_roboflow(video_path, play_id=play_id)
        return self._process_local(video_path, play_id=play_id)

    def _process_local(self, video_path: Path, play_id: str | None = None) -> list[PlayTracks]:
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
            # Officials are not rejected here. At All-22 scale a per-frame
            # appearance test is a coin flip, and rejecting a real player on
            # even a few frames breaks his track in half. They are removed
            # after tracking, once a whole track's appearance can be read.
            players = [d for d in detections if d.class_name == "player"]
            balls = [d for d in detections if d.class_name == "ball"]

            tracked = tracker.update(players, frame_index, frame)

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
                for t, point in zip(tracked, mapped, strict=True):
                    if not np.all(np.isfinite(point)):
                        continue
                    fx, fy = float(point[0]), float(point[1])
                    if not _on_field(fx, fy):
                        continue
                    field_positions[t.track_id] = (fx, fy)
                field_positions = _drop_sideline_isolates(field_positions)

            ball_point = None
            if balls and registration.homography is not None:
                mapped = registration.to_field(np.array([balls[0].foot_point]))
                if np.all(np.isfinite(mapped[0])) and _on_field(float(mapped[0][0]), float(mapped[0][1])):
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

        segments, confidences = self._find_segments(frames_meta, fps)

        # Identity is decided on the snap window only. Broadcast film follows
        # the ball, so a few seconds after the snap the frame is half sideline
        # and the bench outnumbers the offense. Those frames say nothing useful
        # about who is playing, and they actively poison the appearance model.
        window: set[int] = set()
        for _start, snap_idx, _end in segments:
            lo = max(0, snap_idx - int(fps * PRE_SNAP_SECONDS))
            hi = min(len(frames_meta), snap_idx + int(fps * POST_SNAP_SECONDS) + 1)
            window.update(range(lo, hi))
        if not window:
            window = set(range(len(frames_meta)))

        on_field_ids: set[int] = set()
        for j in window:
            on_field_ids.update(frames_meta[j]["field_positions"])

        # One appearance decision per track, over its whole life in the window.
        profiles = build_track_profiles(
            frames_meta, raw_frames, keep_ids=on_field_ids, frames=window
        )
        embeddings = build_track_embeddings(
            frames_meta, raw_frames, keep_ids=on_field_ids, frames=window
        )
        roster = classify_tracks(profiles, embeddings)

        # Rejoining fragments needs the team labels, so it has to follow
        # classification rather than precede it - but the appearance of a track
        # does not change when it is given a longer life, so the roster stays
        # valid and only the ids it refers to move.
        merges = merge_fragments(frames_meta, embeddings, roster, fps)
        if merges:
            apply_merges(frames_meta, merges)
            roster.team_of = {
                merges.get(tid, tid): team for tid, team in roster.team_of.items()
            }
            roster.officials = {merges.get(tid, tid) for tid in roster.officials}
            roster.notes.append(
                f"rejoined {len(merges)} track fragment(s) to the player they belong to"
            )

        if roster.officials:
            for meta in frames_meta:
                meta["tracked"] = [
                    t for t in meta["tracked"] if t.track_id not in roster.officials
                ]
                meta["field_positions"] = {
                    tid: pos
                    for tid, pos in meta["field_positions"].items()
                    if tid not in roster.officials
                }
            # That first snap was found with the crowd and the officiating crew
            # still in the positions. Neither moves with the play, so they flatten
            # the motion energy the snap is read from - and the more of them the
            # homography let through, the further off the answer. Now that they
            # are gone the energy means something different, so ask again rather
            # than refine a number built on people who are not in the play.
            segments, confidences = self._find_segments(frames_meta, fps)

        plays: list[PlayTracks] = []
        for i, ((start, snap_idx, end), confidence) in enumerate(
            zip(segments, confidences, strict=True)
        ):
            ball_positions = {
                j: m["ball"] for j, m in enumerate(frames_meta) if m["ball"] is not None
            }
            snap_idx = refine_with_ball(snap_idx, ball_positions, fps)
            snap_idx = _refine_with_formation(frames_meta, snap_idx, fps)
            if len(frames_meta[snap_idx]["field_positions"]) >= 16:
                confidence = max(confidence, 0.55)
            play = self._build_play(
                video_path=video_path,
                frames_meta=frames_meta,
                raw_frames=raw_frames,
                roster=roster,
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

    def _workflow_client(self):
        if self._workflow is not None:
            return self._workflow
        from gridiron.vision.roboflow import WorkflowClient

        self._workflow = WorkflowClient()
        return self._workflow

    def _process_image(self, image_path: Path, play_id: str | None = None) -> list[PlayTracks]:
        cv2 = require("cv2")
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"could not open image: {image_path}")
        result = self._workflow_client().run_frame(frame)
        return self._plays_from_workflow_frames(
            source_path=image_path,
            frames=[(0, frame, result)],
            fps=10.0,
            play_id=play_id or image_path.stem,
        )

    def _process_roboflow(self, video_path: Path, play_id: str | None = None) -> list[PlayTracks]:
        fps = video_fps(video_path)
        client = self._workflow_client()
        frames: list[tuple[int, np.ndarray, Any]] = []
        for frame_index, frame in read_frames(
            video_path, self.config.stride, self.config.max_frames
        ):
            frames.append((frame_index, frame, client.run_frame(frame)))
        if not frames:
            return []
        return self._plays_from_workflow_frames(
            source_path=video_path,
            frames=frames,
            fps=fps / max(self.config.stride, 1),
            play_id=play_id,
        )

    def _plays_from_workflow_frames(
        self,
        source_path: Path,
        frames: list[tuple[int, np.ndarray, Any]],
        fps: float,
        play_id: str | None,
    ) -> list[PlayTracks]:
        from gridiron.vision.registration import Registration
        from gridiron.vision.roboflow import coverage_prediction_row

        frames_meta: list[dict[str, Any]] = []
        raw_frames: list[np.ndarray] = []
        prev_positions: dict[int, tuple[float, float]] = {}
        prev_sides: dict[int, str] = {}
        next_id = 1

        for frame_index, frame, parsed in frames:
            assigned, next_id = _associate_field_players(
                parsed.players, prev_positions, prev_sides, next_id
            )
            field_positions = {
                tid: (player.field_x, player.field_y) for tid, player in assigned.items()
            }
            sides = {tid: player.side for tid, player in assigned.items() if player.side}
            homography = np.asarray(parsed.homography, dtype=np.float64) if parsed.homography else None
            registration = Registration(
                homography=homography,
                method="roboflow-uga",
                reprojection_error_yd=1.5,
                notes=list(parsed.notes),
            )
            frames_meta.append(
                {
                    "frame_index": frame_index,
                    "tracked": [
                        _TrackedRef(tid, player.confidence) for tid, player in assigned.items()
                    ],
                    "field_positions": field_positions,
                    "sides": sides,
                    "registration": registration,
                    "ball": None,
                    "n_detections": len(assigned),
                    "notes": list(parsed.notes),
                    "coverage": parsed.coverage,
                    "parsed": parsed,
                }
            )
            raw_frames.append(frame)
            prev_positions = field_positions
            prev_sides = {tid: side for tid, side in sides.items() if side}

        if not frames_meta:
            return []

        segments, confidences = self._find_segments(frames_meta, fps * max(self.config.stride, 1))
        empty_roster = TrackRoster()
        plays: list[PlayTracks] = []
        for i, ((start, snap_idx, end), confidence) in enumerate(
            zip(segments, confidences, strict=True)
        ):
            this_id = play_id or f"{source_path.stem}-{i:03d}"
            play = self._build_play(
                video_path=source_path,
                frames_meta=frames_meta,
                raw_frames=raw_frames,
                roster=empty_roster,
                start=start,
                snap_idx=snap_idx,
                end=end,
                fps=fps,
                snap_confidence=max(confidence, 0.5),
                play_id=this_id,
            )
            if play is None:
                continue
            snap_parsed = frames_meta[min(snap_idx, len(frames_meta) - 1)].get("parsed")
            if snap_parsed is not None:
                _save_workflow_overlays(this_id, snap_parsed)
                if snap_parsed.coverage.coverage:
                    self.predictions.append(
                        coverage_prediction_row(
                            this_id,
                            snap_parsed.coverage,
                            quality_score=play.quality.score,
                            usable=play.quality.usable,
                        )
                    )
            plays.append(play)
        return plays

    def _find_segments(
        self, frames_meta: list[dict[str, Any]], fps: float
    ) -> tuple[list[tuple[int, int, int]], list[float]]:
        """Locate each play's snap from how much the field is moving."""
        energy = motion_energy([m["field_positions"] for m in frames_meta])
        rate = fps / max(self.config.stride, 1)

        if not self.config.single_play:
            found = segment_plays(energy, fps=rate)
            return (
                [(s.start_frame, s.snap_frame, s.end_frame) for s in found],
                [s.confidence for s in found],
            )

        snap = find_snap(energy, fps=rate)
        if snap is None:
            return [(0, len(frames_meta) // 3, len(frames_meta) - 1)], [0.0]
        return [(0, snap.frame_index, len(frames_meta) - 1)], [snap.confidence]

    def _build_play(
        self,
        video_path: Path,
        frames_meta: list[dict[str, Any]],
        raw_frames: list[np.ndarray],
        roster: TrackRoster,
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

        labeled_sides: dict[int, str] = {}
        for meta in frames_meta[start : min(end + 1, len(frames_meta))]:
            labeled_sides.update(meta.get("sides") or {})
        snap_sides = {
            tid: labeled_sides[tid] for tid in snap_positions if tid in labeled_sides
        }

        if snap_sides:
            from gridiron.vision.teams import TeamAssignment

            labels = {tid: 0 if side == "offense" else 1 for tid, side in snap_sides.items()}
            assignment = TeamAssignment(labels=labels, offense_cluster=0, notes=[])
            notes = list(snap_meta.get("notes") or [])
        else:
            assignment = assign_teams_from_roster(roster, snap_positions)
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

        remaining = max(0.0, (len(frames_meta) - 1 - snap_idx) / max(fps, 1e-6))
        post = float(min(FILM_POST_SNAP_SECONDS, max(POST_SNAP_SECONDS, remaining)))
        grid = make_time_grid(PRE_SNAP_SECONDS, post)
        if grid.shape[0] < N_FRAMES:
            grid = make_time_grid()
        snap_index = int(np.argmin(np.abs(grid)))

        # Collect each track's observations in normalized coordinates. Officials
        # are already gone; `process` stripped them before the snap was found.
        series: dict[int, dict[str, list[float]]] = {}
        for j in range(start, min(end + 1, len(frames_meta))):
            meta = frames_meta[j]
            t = (j - snap_idx) / max(fps, 1e-6)
            if t < -PRE_SNAP_SECONDS - 0.5 or t > post + 0.5:
                continue
            for track_id, (fx, fy) in meta["field_positions"].items():
                entry = series.setdefault(track_id, {"t": [], "x": [], "y": []})
                entry["t"].append(t)
                entry["x"].append(sign * (fx - los_x))
                entry["y"].append(sign * (fy - ball_y))

        still = len(frames_meta) == 1
        if still:
            notes.append("still frame: formation is held at the snap, so velocities are zero")
            ticks = [round(float(t), 2) for t in np.arange(-PRE_SNAP_SECONDS, 0.01, 0.1)]
            for entry in series.values():
                if entry["t"]:
                    x0, y0 = entry["x"][0], entry["y"][0]
                    entry["t"] = ticks
                    entry["x"] = [x0] * len(ticks)
                    entry["y"] = [y0] * len(ticks)

        players: list[PlayerTrack] = []
        min_obs = 1 if still else 8
        for track_id, entry in series.items():
            if len(entry["t"]) < min_obs:
                continue
            side = assignment.side_of(track_id)
            if side is None:
                continue
            x = smooth_track(resample_to_grid(np.array(entry["t"]), np.array(entry["x"]), grid=grid))
            y = smooth_track(resample_to_grid(np.array(entry["t"]), np.array(entry["y"]), grid=grid))
            if not np.isfinite(x).any():
                continue
            ys = y[np.isfinite(y)]
            if ys.size and float(np.median(np.abs(ys))) > 24.0:
                continue
            players.append(
                PlayerTrack(
                    track_id=f"{'O' if side == 'offense' else 'D'}_{track_id}",
                    side=side,
                    x=x,
                    y=y,
                )
            )

        players = finalize_film_players(players, snap_index=snap_index)

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
            snap_frame_in_video=int(snap_meta.get("frame_index", snap_idx)),
            video_fps=fps,
            homography=homography,
            origin_x=float(los_x),
            origin_y=float(ball_y),
            play_direction=direction,
            video_width=int(frame_w),
            video_height=int(frame_h),
            time_grid=grid,
        )


def _formation_spread(positions: dict[int, tuple[float, float]]) -> float:
    """How far apart the players are in depth. Small means they are still lined up."""
    if len(positions) < 8:
        return float("inf")
    return float(np.std([p[0] for p in positions.values()]))


def _refine_with_formation(
    frames_meta: list[dict[str, Any]],
    snap_idx: int,
    fps: float,
    lookback_s: float = 2.5,
    lookahead_s: float = 0.15,
    tolerance: float = 1.15,
) -> int:
    """Pull the snap back to the last frame the offense was still in formation.

    Motion energy finds the play, not the snap, and on broadcast film it lands
    late: the camera zooms as the play develops, so players get bigger, the
    detector finds more of them, and anything keyed on how *many* players are
    visible drifts toward the middle of the play. Counting was the old
    criterion and it put the snap a full second past the handoff.

    Depth spread does not have that bias. Twenty-two players straddling the line
    are compact in x no matter how far away the camera is, and the snap is the
    moment that stops being true - so take the last frame that is still within a
    little of the tightest alignment seen.

    `lookahead_s` is a short settle rather than a search: energy can trip on the
    last pre-snap shift, and one step of slack covers it. Both it and the
    tolerance were fitted in scripts/snap_tolerance_study.py against the
    synthetic clip, the only film where the true snap frame is known exactly.
    """
    lo = max(0, snap_idx - int(fps * lookback_s))
    # A little forward too: motion energy can trip on the last pre-snap shift,
    # and the snap is then a few frames the other way.
    hi = min(len(frames_meta) - 1, snap_idx + int(fps * lookahead_s))
    counts = [len(frames_meta[j]["field_positions"]) for j in range(lo, hi + 1)]
    if not counts:
        return snap_idx
    # Frames where the detector only found a handful look artificially compact.
    need = max(8, int(0.6 * max(counts)))
    candidates = [
        j for j in range(lo, hi + 1) if len(frames_meta[j]["field_positions"]) >= need
    ]
    if not candidates:
        return snap_idx

    spreads = {j: _formation_spread(frames_meta[j]["field_positions"]) for j in candidates}
    tightest = min(spreads.values())
    if not np.isfinite(tightest):
        return snap_idx
    still_formed = [j for j in candidates if spreads[j] <= tightest * tolerance]
    return max(still_formed) if still_formed else snap_idx


def _on_field(x: float, y: float, margin_x: float = 1.5, margin_y: float = 2.5) -> bool:
    """Drop crowd and graphics that the homography throws off the turf."""
    return (
        -margin_x <= x <= FIELD_LENGTH_YD + margin_x
        and margin_y <= y <= FIELD_WIDTH_YD - margin_y
    )


def _drop_sideline_isolates(
    positions: dict[int, tuple[float, float]], band: float = 2.8
) -> dict[int, tuple[float, float]]:
    """Officials and the chain gang stand on the paint; receivers do not live there alone."""
    if len(positions) < 8:
        return positions
    kept: dict[int, tuple[float, float]] = {}
    for tid, (fx, fy) in positions.items():
        on_paint = fy <= band or fy >= FIELD_WIDTH_YD - band
        if not on_paint:
            kept[tid] = (fx, fy)
            continue
        nearby = sum(
            1
            for ox, oy in positions.values()
            if abs(ox - fx) <= 8.0 and abs(oy - fy) <= 8.0
        )
        if nearby > 2:
            kept[tid] = (fx, fy)
    return kept


def _associate_field_players(
    players: list[Any],
    prev_positions: dict[int, tuple[float, float]],
    prev_sides: dict[int, str],
    next_id: int,
    max_yards: float = 6.0,
) -> tuple[dict[int, Any], int]:
    """Greedy nearest-neighbor IDs on field yards, staying on the same side."""
    unused = set(prev_positions)
    assigned: dict[int, Any] = {}
    for player in players:
        best_id = None
        best_dist = max_yards
        for tid in unused:
            if prev_sides.get(tid) and player.side and prev_sides[tid] != player.side:
                continue
            px, py = prev_positions[tid]
            dist = float(np.hypot(player.field_x - px, player.field_y - py))
            if dist < best_dist:
                best_id = tid
                best_dist = dist
        if best_id is None:
            best_id = next_id
            next_id += 1
        else:
            unused.remove(best_id)
        assigned[best_id] = player
    return assigned, next_id


def _save_workflow_overlays(play_id: str, parsed: Any) -> None:
    from gridiron.config import subdir

    folder = subdir("film", "overlays")
    if getattr(parsed, "minimap_png", None):
        (folder / f"{play_id}_minimap.png").write_bytes(parsed.minimap_png)
    if getattr(parsed, "output_png", None):
        (folder / f"{play_id}_output.png").write_bytes(parsed.output_png)


def process_video(
    video_path: Path,
    config: PipelineConfig | None = None,
    play_id: str | None = None,
    detector: Any | None = None,
    registrar: Any | None = None,
    workflow: Any | None = None,
) -> list[PlayTracks]:
    return FilmPipeline(
        config, detector=detector, registrar=registrar, workflow=workflow
    ).process(video_path, play_id=play_id)
