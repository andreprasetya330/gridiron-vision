"""The statistics have to be right, or the tool is worse than useless."""

import numpy as np
import pytest

from gridiron.tells.stats import (
    benjamini_hochberg,
    beta_binomial_posterior,
    estimate_prior_strength,
    exponential_recency_weights,
    two_proportion_pvalue,
    weighted_binomial_counts,
    wilson_interval,
)


def test_wilson_interval_stays_in_bounds_at_extremes():
    """The normal approximation produces impossible bounds here; Wilson must not."""
    interval = wilson_interval(3, 3)
    assert 0.0 <= interval.low <= interval.high <= 1.0
    assert interval.low < 1.0, "3-for-3 must not imply certainty"

    zero = wilson_interval(0, 8)
    assert zero.low == 0.0
    assert zero.high > 0.0


def test_wilson_interval_narrows_with_sample_size():
    small = wilson_interval(6, 10)
    large = wilson_interval(60, 100)
    assert large.width < small.width


def test_shrinkage_pulls_small_samples_toward_baseline():
    """3-for-3 should read as a lean, not as a 100% tendency."""
    baseline = 0.30
    shrunk = beta_binomial_posterior(3, 3, baseline, prior_strength=20.0)
    assert baseline < shrunk < 0.55

    # Real volume should move it much closer to the observed rate.
    heavy = beta_binomial_posterior(60, 60, baseline, prior_strength=20.0)
    assert heavy > 0.70


def test_shrinkage_is_monotonic_in_evidence():
    baseline = 0.25
    rates = [beta_binomial_posterior(k, 20, baseline) for k in range(0, 21, 5)]
    assert rates == sorted(rates)


def test_prior_strength_shrinks_hard_when_teams_do_not_differ():
    """If all variation is sampling noise, shrinkage should be aggressive."""
    rng = np.random.default_rng(0)
    counts = np.full(12, 40)
    rates = rng.binomial(40, 0.3, size=12) / 40
    strength = estimate_prior_strength(rates, counts)
    assert strength > 40

    # Genuinely different teams should produce gentle shrinkage.
    spread_rates = np.array([0.05, 0.1, 0.2, 0.45, 0.6, 0.8, 0.9, 0.15, 0.7, 0.35, 0.5, 0.25])
    spread_strength = estimate_prior_strength(spread_rates, counts)
    assert spread_strength < strength


def test_benjamini_hochberg_controls_false_discoveries_under_the_null():
    """With no real effects, BH should reject almost nothing."""
    rng = np.random.default_rng(11)
    null_pvalues = rng.uniform(0, 1, size=400)
    rejected, qvalues = benjamini_hochberg(null_pvalues, alpha=0.10)
    assert rejected.sum() <= 4, "BH is letting through far too many null findings"
    assert np.all(qvalues >= null_pvalues - 1e-9)


def test_benjamini_hochberg_still_finds_real_effects():
    pvalues = np.concatenate([np.full(20, 1e-6), np.random.default_rng(3).uniform(0.2, 1, 380)])
    rejected, _ = benjamini_hochberg(pvalues, alpha=0.10)
    assert rejected[:20].all()


def test_benjamini_hochberg_qvalues_are_monotone():
    pvalues = np.array([0.001, 0.01, 0.02, 0.3, 0.5])
    _, qvalues = benjamini_hochberg(pvalues)
    assert np.all(np.diff(qvalues) >= -1e-12)


def test_two_proportion_pvalue_detects_a_real_gap():
    assert two_proportion_pvalue(18, 20, 0.30, baseline_trials=5000) < 0.001
    assert two_proportion_pvalue(6, 20, 0.30, baseline_trials=5000) > 0.5


def test_two_proportion_pvalue_respects_a_thin_baseline():
    """A thin baseline should make us less certain, not more."""
    thin = two_proportion_pvalue(15, 20, 0.30, baseline_trials=25)
    thick = two_proportion_pvalue(15, 20, 0.30, baseline_trials=10000)
    assert thin > thick


def test_recency_weights_decay_with_a_three_game_half_life():
    weights = exponential_recency_weights(np.array([10, 7, 4, 1]), current_week=10, half_life=3.0)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.5, abs=0.01)
    assert weights[2] == pytest.approx(0.25, abs=0.01)


def test_weighted_counts_do_not_inflate_effective_sample_size():
    """Downweighted history must not smuggle in false confidence."""
    successes_mask = np.array([True] * 10 + [False] * 10)
    equal = np.ones(20)
    _, n_equal = weighted_binomial_counts(successes_mask, equal)
    assert n_equal == pytest.approx(20.0)

    skewed = np.concatenate([np.ones(2), np.full(18, 0.05)])
    _, n_skewed = weighted_binomial_counts(successes_mask, skewed)
    assert n_skewed < 10, "effective n must collapse when weight is concentrated"
