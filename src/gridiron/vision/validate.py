"""Measure how much the vision pipeline costs you.

Every stage between the camera and the coverage model degrades the signal, and
the only question that matters is whether what survives is still good enough to
call a coverage. This module answers that with numbers instead of vibes.

The protocol: take plays whose coordinates are known exactly, render them to
video, run the pipeline on the video, and compare what came back to what went in.
Then run the same coverage model over both and report the accuracy gap. That gap
is the domain gap, and it is the single number that decides whether film-derived
predictions can be trusted or need to be labeled "low confidence" in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gridiron.tracking.schema import N_FRAMES, PlayTracks
from gridiron.vision.pipeline import PipelineConfig, process_video
from gridiron.vision.render import (
    CameraSpec,
    RenderedPlayerDetector,
    RenderedTruth,
    TruthBoxDetector,
    render_play,
)


@dataclass
class PlayComparison:
    play_id: str
    recovered: bool
    n_truth_players: int = 0
    n_recovered_players: int = 0
    matched: int = 0
    position_rmse_yd: float = float("nan")
    position_p90_yd: float = float("nan")
    snap_frame_error: int | None = None
    side_accuracy: float = float("nan")
    registration_error_yd: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "play_id": self.play_id,
            "recovered": self.recovered,
            "n_truth_players": self.n_truth_players,
            "n_recovered_players": self.n_recovered_players,
            "matched": self.matched,
            "position_rmse_yd": _round(self.position_rmse_yd),
            "position_p90_yd": _round(self.position_p90_yd),
            "snap_frame_error": self.snap_frame_error,
            "side_accuracy": _round(self.side_accuracy),
            "registration_error_yd": _round(self.registration_error_yd),
            "notes": self.notes,
        }


def _round(value: float, digits: int = 3) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def compare_play(truth_play: PlayTracks, recovered: PlayTracks | None, truth: RenderedTruth) -> PlayComparison:
    """Match recovered tracks to true ones and measure the damage."""
    result = PlayComparison(
        play_id=truth_play.play_id,
        recovered=recovered is not None,
        n_truth_players=len(truth_play.players),
    )
    if recovered is None:
        result.notes.append("pipeline produced no play from this video")
        return result

    result.n_recovered_players = len(recovered.players)
    result.registration_error_yd = recovered.quality.registration_error_yd

    if recovered.snap_frame_in_video is not None:
        result.snap_frame_error = int(recovered.snap_frame_in_video - truth.snap_frame)

    # Match on mean position over the frames both tracks were observed. Greedy
    # nearest-neighbour is enough: the question is whether the geometry survived,
    # not whether an assignment algorithm can rescue it.
    truth_by_id = {p.track_id: p for p in truth_play.players}
    pairs: list[tuple[float, str, PlayTracks]] = []
    for rec in recovered.players:
        for tid, tru in truth_by_id.items():
            distance = _mean_distance(tru, rec)
            if np.isfinite(distance):
                pairs.append((distance, tid, rec))

    used_truth: set[str] = set()
    used_rec: set[str] = set()
    matches: list[tuple[Any, Any]] = []
    for distance, tid, rec in sorted(pairs, key=lambda item: item[0]):
        if tid in used_truth or rec.track_id in used_rec or distance > 5.0:
            continue
        used_truth.add(tid)
        used_rec.add(rec.track_id)
        matches.append((truth_by_id[tid], rec))

    result.matched = len(matches)
    if not matches:
        result.notes.append("no recovered track landed within 5 yards of a real one")
        return result

    errors: list[float] = []
    side_hits = 0
    for tru, rec in matches:
        ok = np.isfinite(tru.x) & np.isfinite(tru.y) & np.isfinite(rec.x) & np.isfinite(rec.y)
        if ok.any():
            errors.extend(np.hypot(tru.x[ok] - rec.x[ok], tru.y[ok] - rec.y[ok]).tolist())
        side_hits += int(tru.side == rec.side)

    if errors:
        arr = np.asarray(errors, dtype=np.float64)
        result.position_rmse_yd = float(np.sqrt(np.mean(arr**2)))
        result.position_p90_yd = float(np.percentile(arr, 90))
    result.side_accuracy = side_hits / len(matches)
    return result


def _mean_distance(a: PlayTracks, b: PlayTracks) -> float:
    ok = np.isfinite(a.x) & np.isfinite(a.y) & np.isfinite(b.x) & np.isfinite(b.y)
    if ok.sum() < max(3, N_FRAMES // 8):
        return float("inf")
    return float(np.mean(np.hypot(a.x[ok] - b.x[ok], a.y[ok] - b.y[ok])))


@dataclass
class ValidationReport:
    comparisons: list[PlayComparison] = field(default_factory=list)
    coverage_agreement: float | None = None
    coverage_accuracy_truth: float | None = None
    coverage_accuracy_film: float | None = None
    detector_recall: float | None = None

    @property
    def recovery_rate(self) -> float:
        if not self.comparisons:
            return float("nan")
        return sum(1 for c in self.comparisons if c.recovered) / len(self.comparisons)

    @property
    def median_position_error(self) -> float:
        values = [c.position_rmse_yd for c in self.comparisons if np.isfinite(c.position_rmse_yd)]
        return float(np.median(values)) if values else float("nan")

    @property
    def median_side_accuracy(self) -> float:
        values = [c.side_accuracy for c in self.comparisons if np.isfinite(c.side_accuracy)]
        return float(np.median(values)) if values else float("nan")

    @property
    def snap_frame_mae(self) -> float:
        values = [abs(c.snap_frame_error) for c in self.comparisons if c.snap_frame_error is not None]
        return float(np.mean(values)) if values else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_plays": len(self.comparisons),
            "recovery_rate": _round(self.recovery_rate),
            "median_position_rmse_yd": _round(self.median_position_error),
            "median_side_accuracy": _round(self.median_side_accuracy),
            "snap_frame_mae": _round(self.snap_frame_mae),
            "detector_recall": _round(self.detector_recall),
            "coverage_agreement": _round(self.coverage_agreement),
            "coverage_accuracy_truth": _round(self.coverage_accuracy_truth),
            "coverage_accuracy_film": _round(self.coverage_accuracy_film),
            "plays": [c.to_dict() for c in self.comparisons],
        }

    def summary(self) -> str:
        lines = [
            f"plays: {len(self.comparisons)}  recovered: {self.recovery_rate:.0%}",
            f"median position RMSE: {self.median_position_error:.2f} yd",
            f"median offense/defense accuracy: {self.median_side_accuracy:.0%}",
            f"snap frame mean absolute error: {self.snap_frame_mae:.1f} frames",
        ]
        if self.detector_recall is not None:
            lines.append(
                f"colour detector recall (separate probe): {self.detector_recall:.0%}"
            )
        if self.coverage_agreement is not None:
            lines.append(
                f"coverage: {self.coverage_accuracy_truth:.0%} on true tracks vs "
                f"{self.coverage_accuracy_film:.0%} on film-derived tracks "
                f"({self.coverage_agreement:.0%} of predictions agree)"
            )
        return "\n".join(lines)


def validate_pipeline(
    plays: list[PlayTracks],
    work_dir: Path,
    camera: CameraSpec | None = None,
    fps: float = 30.0,
    jitter_px: float = 0.0,
    box_jitter_px: float = 0.0,
    miss_rate: float = 0.0,
    model: Any | None = None,
    keep_video: bool = False,
    probe_detector: bool = True,
    los_x: float = 55.0,
) -> ValidationReport:
    """Render, process, and compare. `model` is any object with `.predict(frame)`.

    Detection is supplied from ground truth so that a failure points at the
    geometry rather than at the blob detector. `probe_detector` separately scores
    the colour detector's recall on the same frames, which keeps detection
    honest without letting it mask everything downstream.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report = ValidationReport()

    recovered_plays: list[PlayTracks | None] = []
    recalls: list[float] = []

    for i, play in enumerate(plays):
        video_path = work_dir / f"{_slug(play.play_id)}.mp4"
        truth = render_play(
            play,
            video_path,
            camera=camera or CameraSpec.around(los_x),
            fps=fps,
            los_x=los_x,
            jitter_px=jitter_px,
            seed=i,
        )
        config = PipelineConfig(league=play.situation.league, single_play=True)
        detector = TruthBoxDetector(
            truth, box_jitter_px=box_jitter_px, miss_rate=miss_rate, seed=i
        )
        try:
            produced = process_video(
                video_path, config=config, play_id=play.play_id, detector=detector
            )
        except Exception as exc:  # a crash is a result, and should be reported as one
            comparison = PlayComparison(play_id=play.play_id, recovered=False)
            comparison.notes.append(f"pipeline raised {type(exc).__name__}: {exc}")
            report.comparisons.append(comparison)
            recovered_plays.append(None)
            continue

        best = produced[0] if produced else None
        report.comparisons.append(compare_play(play, best, truth))
        recovered_plays.append(best)

        if probe_detector:
            recalls.append(detector_recall(video_path, truth))

        if not keep_video:
            video_path.unlink(missing_ok=True)
            video_path.with_suffix(".truth.json").unlink(missing_ok=True)

    if recalls:
        report.detector_recall = float(np.mean(recalls))
    if model is not None:
        _score_domain_gap(report, plays, recovered_plays, model)
    return report


def detector_recall(
    video_path: Path, truth: RenderedTruth, frames_to_probe: int = 5, iou_threshold: float = 0.3
) -> float:
    """Fraction of real players the colour detector finds on sampled frames."""
    from gridiron.vision.pipeline import read_frames

    detector = RenderedPlayerDetector()
    indices = np.linspace(0, max(len(truth.frames) - 1, 0), frames_to_probe).astype(int)
    wanted = set(int(i) for i in indices)

    hits = total = 0
    for frame_index, frame in read_frames(Path(video_path)):
        if frame_index not in wanted:
            continue
        entries = truth.frames.get(str(frame_index), [])
        if not entries:
            continue
        found = detector.detect(frame)
        for entry in entries:
            total += 1
            if any(_box_iou(entry["box"], (d.x1, d.y1, d.x2, d.y2)) >= iou_threshold for d in found):
                hits += 1
    return hits / total if total else float("nan")


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def _score_domain_gap(
    report: ValidationReport,
    truth_plays: list[PlayTracks],
    film_plays: list[PlayTracks | None],
    model: Any,
) -> None:
    """How much accuracy the camera costs, on the same plays either way."""
    from gridiron.coverage.bridge import measure_gap

    paired_clean: list[PlayTracks] = []
    paired_film: list[PlayTracks] = []
    for truth, recovered in zip(truth_plays, film_plays):
        if recovered is None or truth.coverage is None:
            continue
        paired_clean.append(truth)
        paired_film.append(recovered)
    if not paired_clean:
        return

    gap = measure_gap(paired_clean, paired_film, model, film_source="recovered")
    report.coverage_accuracy_truth = gap.clean_accuracy
    report.coverage_accuracy_film = gap.film_accuracy
    report.coverage_agreement = gap.agreement


def comparisons_to_frame(report: ValidationReport) -> pd.DataFrame:
    return pd.DataFrame([c.to_dict() for c in report.comparisons])


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text)
