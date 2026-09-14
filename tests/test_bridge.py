"""Calibration and the domain gap, against answers known by construction."""

from __future__ import annotations

import numpy as np
import pytest

from gridiron.coverage.bridge import (
    apply_film_noise,
    expected_calibration_error,
    measure_gap,
    score_plays,
)
from gridiron.taxonomy import COVERAGES


def test_perfectly_calibrated_predictions_have_near_zero_ece():
    rng = np.random.default_rng(0)
    n = 2000
    # Each play is assigned a true class, then a probability vector whose top
    # entry equals p and is correct with frequency p.
    p = rng.uniform(0.4, 0.95, size=n)
    true_idx = rng.integers(0, len(COVERAGES), size=n)
    pred_idx = true_idx.copy()
    wrong = rng.random(n) > p
    pred_idx[wrong] = (true_idx[wrong] + 1) % len(COVERAGES)

    probs = np.full((n, len(COVERAGES)), (1 - p)[:, None] / (len(COVERAGES) - 1))
    probs[np.arange(n), pred_idx] = p
    y = np.array(COVERAGES)[true_idx]

    ece, bins = expected_calibration_error(y, probs, n_bins=8)
    assert ece < 0.05
    assert bins


def test_overconfident_predictions_have_large_ece():
    n = 400
    true_idx = np.zeros(n, dtype=int)
    probs = np.full((n, len(COVERAGES)), 0.01)
    probs[:, 1] = 0.93  # always 93% sure of the wrong class
    y = np.array(COVERAGES)[true_idx]
    ece, _ = expected_calibration_error(y, probs)
    assert ece > 0.8


def test_film_noise_lowers_accuracy_on_a_real_model(synthetic_plays):
    """The gap this project exists to measure has to actually exist.

    If applying film-like noise did not change the answer, either the noise is
    too gentle to be a stand-in for a camera or the model is not reading the
    coordinates it claims to read.
    """
    from gridiron.coverage.baseline import CoverageBaseline
    from gridiron.coverage.features import build_feature_frame

    labeled = [p for p in synthetic_plays if p.coverage][:400]
    features = build_feature_frame(labeled, mode="postsnap")
    split = int(0.7 * len(labeled))
    model = CoverageBaseline(mode="postsnap")
    model.fit(features.iloc[:split])

    holdout = labeled[split:]
    noisy = apply_film_noise(holdout, seed=3)
    gap = measure_gap(holdout, noisy, model, film_source="augmented")

    assert gap.n == len(holdout)
    assert gap.clean_accuracy > gap.film_accuracy
    assert gap.drop > 0.02


def test_score_plays_writes_overlay_payload(synthetic_plays):
    from gridiron.coverage.baseline import CoverageBaseline
    from gridiron.coverage.features import build_feature_frame

    labeled = [p for p in synthetic_plays if p.coverage][:80]
    features = build_feature_frame(labeled, mode="postsnap")
    model = CoverageBaseline(mode="postsnap").fit(features)
    payload = score_plays(labeled[:5], model)
    assert len(payload) == 5
    for row in payload:
        assert row["coverage"] in COVERAGES
        assert 0.0 <= row["confidence"] <= 1.0
        assert set(row["probabilities"]) == set(COVERAGES)
        assert "play_id" in row
