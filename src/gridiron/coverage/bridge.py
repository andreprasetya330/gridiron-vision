"""The camera costs accuracy. This module measures how much.

A model trained on Big Data Bowl tracking - every player, every frame, to the
inch - will look better than it is the first time it sees film. Film loses
safeties behind other players, jitters coordinates through the homography, and
sometimes places the line of scrimmage a yard off. Training with dropout and
jitter is the mitigation; this module is the measurement, without which you
cannot tell whether the mitigation worked.

Two comparisons, both on the same snaps:

- **Clean vs recovered.** The synthetic (or BDB) tracks against the tracks the
  vision pipeline produced from film of those same snaps. That is the real
  domain gap.
- **Clean vs augmented.** The same tracks with film-like noise applied in
  software. Faster, and it is what you run in CI; it is not a substitute for
  the first comparison once real film exists.

Confidence is reported with expected calibration error. A coverage call printed
at 82% that is right 60% of the time is a lie, and a coach will remember the
lie longer than the 22 correct calls around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from gridiron.coverage.baseline import CoverageMetrics, evaluate_predictions
from gridiron.coverage.dataset import AugmentConfig
from gridiron.taxonomy import COVERAGES
from gridiron.tracking.schema import PlayTracks


@dataclass
class CalibrationBin:
    confidence: float
    accuracy: float
    n: int

    @property
    def gap(self) -> float:
        return abs(self.confidence - self.accuracy)


@dataclass
class DomainGap:
    n: int
    clean_accuracy: float
    film_accuracy: float
    agreement: float
    clean_ece: float
    film_ece: float
    clean_metrics: dict = field(default_factory=dict)
    film_metrics: dict = field(default_factory=dict)
    film_source: str = "recovered"

    @property
    def drop(self) -> float:
        return self.clean_accuracy - self.film_accuracy

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "clean_accuracy": round(self.clean_accuracy, 4),
            "film_accuracy": round(self.film_accuracy, 4),
            "drop": round(self.drop, 4),
            "agreement": round(self.agreement, 4),
            "clean_ece": round(self.clean_ece, 4),
            "film_ece": round(self.film_ece, 4),
            "film_source": self.film_source,
            "clean": self.clean_metrics,
            "film": self.film_metrics,
        }

    def summary(self) -> str:
        return (
            f"n={self.n}  clean {self.clean_accuracy:.3f} -> film {self.film_accuracy:.3f} "
            f"(drop {self.drop:.3f})  ECE {self.film_ece:.3f}  source={self.film_source}"
        )


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> tuple[float, list[CalibrationBin]]:
    """How far the printed probabilities are from actual hit rates.

    Bins by the predicted class's confidence, not by the true class, because
    that is the number a coach reads.
    """
    conf = probs.max(axis=1)
    pred = np.array(COVERAGES)[probs.argmax(axis=1)]
    correct = pred == np.asarray(y_true)

    bins: list[CalibrationBin] = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        # The last bin is closed on the right so a 1.0 confidence is counted.
        in_bin = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        accuracy = float(correct[in_bin].mean())
        confidence = float(conf[in_bin].mean())
        bins.append(CalibrationBin(confidence=confidence, accuracy=accuracy, n=n))
        ece += (n / len(y_true)) * abs(confidence - accuracy)
    return float(ece), bins


def _predict_matrix(model: Any, plays: list[PlayTracks]) -> np.ndarray:
    """(n, n_classes) probabilities, columns in COVERAGES order."""
    if hasattr(model, "predict_play"):
        rows = []
        for play in plays:
            result = model.predict_play(play)
            rows.append([result["probabilities"].get(c, 0.0) for c in COVERAGES])
        return np.array(rows, dtype=np.float64)

    from gridiron.coverage.features import build_feature_frame

    frame = build_feature_frame(plays, mode=getattr(model, "mode", "postsnap"))
    raw = np.asarray(model.predict_proba(frame), dtype=np.float64)
    classes = list(getattr(model, "classes", COVERAGES))
    aligned = np.zeros((len(raw), len(COVERAGES)), dtype=np.float64)
    for i, name in enumerate(classes):
        if name in COVERAGES:
            aligned[:, COVERAGES.index(name)] = raw[:, i]
    return aligned


def measure_gap(
    clean: list[PlayTracks],
    film: list[PlayTracks],
    model: Any,
    film_source: str = "recovered",
) -> DomainGap:
    """Score the same snaps twice: once as they are, once as the camera saw them."""
    if len(clean) != len(film):
        raise ValueError("clean and film lists must be paired, one-to-one")
    labels = [p.coverage for p in clean]
    if any(c is None for c in labels):
        raise ValueError("every clean play needs a coverage label")

    # Film plays inherit the label of their pair. They do not get to carry their
    # own, because the vision pipeline does not know the coverage.
    for film_play, label in zip(film, labels):
        film_play.coverage = label

    clean_probs = _predict_matrix(model, clean)
    film_probs = _predict_matrix(model, film)
    y = np.array(labels)

    clean_pred = np.array(COVERAGES)[clean_probs.argmax(axis=1)]
    film_pred = np.array(COVERAGES)[film_probs.argmax(axis=1)]
    clean_ece, _ = expected_calibration_error(y, clean_probs)
    film_ece, _ = expected_calibration_error(y, film_probs)

    return DomainGap(
        n=len(clean),
        clean_accuracy=float((clean_pred == y).mean()),
        film_accuracy=float((film_pred == y).mean()),
        agreement=float((clean_pred == film_pred).mean()),
        clean_ece=clean_ece,
        film_ece=film_ece,
        clean_metrics=evaluate_predictions(y, clean_probs, list(COVERAGES)).to_dict(),
        film_metrics=evaluate_predictions(y, film_probs, list(COVERAGES)).to_dict(),
        film_source=film_source,
    )


def apply_film_noise(
    plays: Iterable[PlayTracks],
    augment: AugmentConfig | None = None,
    seed: int = 0,
) -> list[PlayTracks]:
    """A cheap stand-in for the vision pipeline, used when there is no film yet.

    Drops deep defenders, jitters coordinates, and shifts the line of scrimmage
    - the three things the camera actually does. The resulting plays are still
    PlayTracks, so they go through the same scoring path as recovered film.
    """
    from copy import deepcopy

    aug = augment or AugmentConfig(dropout_prob=0.8, max_dropped=2, jitter_yd=0.6, los_shift_yd=0.5)
    rng = np.random.default_rng(seed)
    noisy: list[PlayTracks] = []

    for play in plays:
        clone = deepcopy(play)
        defenders = [p for p in clone.players if p.side == "defense"]
        if aug.enabled and rng.random() < aug.dropout_prob and len(defenders) > aug.max_dropped:
            deepest = sorted(
                defenders,
                key=lambda p: -np.nanmax(np.where(np.isfinite(p.x), p.x, -np.inf)),
            )
            drop = {p.track_id for p in deepest[: int(rng.integers(1, aug.max_dropped + 1))]}
            clone.players = [p for p in clone.players if p.track_id not in drop]

        los_shift = float(rng.normal(0, aug.los_shift_yd)) if aug.enabled else 0.0
        for player in clone.players:
            jitter_x = rng.normal(0, aug.jitter_yd, size=player.x.shape).astype(np.float32)
            jitter_y = rng.normal(0, aug.jitter_yd, size=player.y.shape).astype(np.float32)
            player.x = player.x + jitter_x + los_shift
            player.y = player.y + jitter_y
        noisy.append(clone)
    return noisy


def score_plays(plays: list[PlayTracks], model: Any) -> list[dict]:
    """Run a model over a list of plays and return the overlay payload."""
    if hasattr(model, "predict_play"):
        return [model.predict_play(play) for play in plays]

    aligned = _predict_matrix(model, plays)
    out = []
    for i, play in enumerate(plays):
        order = np.argsort(-aligned[i])
        out.append(
            {
                "play_id": play.play_id,
                "coverage": COVERAGES[int(order[0])],
                "confidence": float(aligned[i][order[0]]),
                "probabilities": {c: float(v) for c, v in zip(COVERAGES, aligned[i])},
                "runner_up": COVERAGES[int(order[1])],
                "roles": {},
                "quality_score": round(play.quality.score, 3),
                "usable": play.quality.usable,
            }
        )
    return out


def write_predictions(predictions: list[dict], path: Path) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    return path


def upsert_predictions(predictions: list[dict], path: Path) -> Path:
    """Replace rows that share a play_id; leave the rest of the overlay file alone."""
    import json

    path = Path(path)
    existing: list[dict] = []
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            existing = raw
    by_id = {row.get("play_id"): row for row in existing if row.get("play_id")}
    for row in predictions:
        play_id = row.get("play_id")
        if play_id:
            by_id[play_id] = row
    return write_predictions(list(by_id.values()), path)
