"""The statistics that keep tell mining honest.

A defense plays roughly 65 snaps a game. Across ~20 cues with a few values each,
times 8 coverages, mining an opponent means running several hundred hypothesis
tests against a sample that small. Chance alone will produce a fistful of
gorgeous, completely fictional tells every single week.

Four defenses against that, all applied:

1. a minimum sample floor, because nothing below it is worth testing
2. beta-binomial shrinkage toward the national rate, so 3-for-3 reads as a lean
   rather than a certainty
3. Wilson intervals, which behave near 0 and 1 where the normal approximation
   falls apart and where interesting tells live
4. Benjamini-Hochberg FDR control across the entire cue-by-coverage grid
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    @property
    def width(self) -> float:
        return self.high - self.low

    def excludes(self, value: float) -> bool:
        return value < self.low or value > self.high


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because tells cluster near 0 and 1,
    exactly where the normal interval produces impossible bounds like 1.08.
    """
    if trials <= 0:
        return Interval(0.0, 1.0)
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    p = successes / trials
    denom = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    return Interval(float(max(0.0, center - half)), float(min(1.0, center + half)))


def beta_binomial_posterior(
    successes: int, trials: int, prior_rate: float, prior_strength: float = 20.0
) -> float:
    """Posterior mean rate, shrunk toward the national baseline.

    `prior_strength` is in pseudo-observations: 20 means "treat the league rate as
    though we had already seen this team run 20 snaps in this situation". Small
    samples therefore land near the league rate and only real volume moves them.
    """
    prior_rate = float(np.clip(prior_rate, 1e-6, 1 - 1e-6))
    alpha = prior_rate * prior_strength
    beta = (1 - prior_rate) * prior_strength
    return float((successes + alpha) / (trials + alpha + beta))


# A prior worth more pseudo-observations than a cue ever gets real ones would
# make shrinkage, not evidence, decide every answer. A defense plays ~65 snaps a
# game, and a given pre-snap look shows up on a fraction of those, so a season
# yields roughly 100-200 observations of a common cue. Capping the prior near 100
# keeps it influential on thin samples and overrideable on thick ones.
MAX_PRIOR_STRENGTH = 100.0
NO_VARIATION_STRENGTH = 50.0
MIN_PEERS_FOR_ESTIMATE = 4


def estimate_prior_strength(
    rates: np.ndarray, counts: np.ndarray, default: float = 20.0
) -> float:
    """Empirical-Bayes estimate of shrinkage strength from across-team spread.

    If every team behaves identically given a cue, observed variation is pure
    sampling noise and shrinkage should be aggressive. If teams genuinely differ,
    shrinkage should be gentle. A method-of-moments estimate separates the two.

    The estimate is capped, and deliberately so. With only a handful of peer
    teams the moment estimator has enormous variance and routinely collapses to
    "no between-team variation at all", which would shrink a genuine 8x tendency
    down into invisibility. Capping trades a little statistical purity for not
    silently deleting the findings the tool exists to produce.
    """
    rates = np.asarray(rates, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    keep = counts > 0
    rates, counts = rates[keep], counts[keep]
    if rates.size < MIN_PEERS_FOR_ESTIMATE:
        return default

    weights = counts / counts.sum()
    mean = float(np.sum(weights * rates))
    if mean <= 0 or mean >= 1:
        return default

    observed_var = float(np.sum(weights * (rates - mean) ** 2))
    expected_sampling_var = float(np.sum(weights * mean * (1 - mean) / counts))
    between_var = observed_var - expected_sampling_var

    if between_var <= 1e-9:
        return NO_VARIATION_STRENGTH
    strength = mean * (1 - mean) / between_var - 1
    return float(np.clip(strength, 2.0, MAX_PRIOR_STRENGTH))


def two_proportion_pvalue(
    successes: int,
    trials: int,
    baseline_rate: float,
    baseline_trials: int | None = None,
    alternative: str = "two-sided",
) -> float:
    """P-value for "this team's rate differs from the national rate".

    When the baseline comes from a large corpus its own uncertainty is negligible
    and an exact binomial test against a fixed rate is right. When the baseline is
    itself thin - which is the situation early on, before the film corpus grows -
    a two-proportion test that accounts for both samples is the honest choice.
    """
    if trials <= 0:
        return 1.0
    baseline_rate = float(np.clip(baseline_rate, 1e-9, 1 - 1e-9))

    if baseline_trials is None or baseline_trials > 20 * trials:
        return float(
            stats.binomtest(successes, trials, baseline_rate, alternative=alternative).pvalue
        )

    baseline_successes = int(round(baseline_rate * baseline_trials))
    table = [
        [successes, trials - successes],
        [baseline_successes, baseline_trials - baseline_successes],
    ]
    try:
        return float(stats.fisher_exact(table, alternative=alternative)[1])
    except ValueError:
        return 1.0


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR control.

    Returns (rejected, qvalues). Alpha defaults to 0.10 rather than 0.05 because
    the cost asymmetry runs the other way here: showing a coach one shaky tell
    that he can check on film is cheaper than hiding a real one, as long as the
    hit rate is displayed alongside it.
    """
    pvalues = np.asarray(pvalues, dtype=np.float64)
    n = pvalues.size
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float64)

    order = np.argsort(pvalues)
    ranked = pvalues[order]
    ranks = np.arange(1, n + 1)

    qvalues_sorted = np.minimum.accumulate((ranked * n / ranks)[::-1])[::-1]
    qvalues_sorted = np.clip(qvalues_sorted, 0.0, 1.0)

    qvalues = np.empty(n, dtype=np.float64)
    qvalues[order] = qvalues_sorted
    return qvalues <= alpha, qvalues


def exponential_recency_weights(
    weeks: np.ndarray, current_week: float, half_life: float = 3.0
) -> np.ndarray:
    """Weight plays by how recent they are.

    A defense in week 11 is not the defense it was in week 2. A three-game
    half-life keeps the last month dominant without throwing away early volume
    entirely.
    """
    weeks = np.asarray(weeks, dtype=np.float64)
    age = np.clip(current_week - weeks, 0, None)
    return np.power(0.5, age / max(half_life, 1e-6))


def weighted_binomial_counts(
    successes_mask: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Effective (successes, trials) under recency weighting.

    Rescaled to preserve effective sample size rather than raw weight mass, so
    downweighted history cannot smuggle in false confidence.
    """
    weights = np.asarray(weights, dtype=np.float64)
    successes_mask = np.asarray(successes_mask, dtype=bool)
    if weights.sum() <= 0:
        return 0.0, 0.0
    effective_n = float(weights.sum() ** 2 / np.sum(weights**2))
    rate = float(np.sum(weights[successes_mask]) / np.sum(weights))
    return rate * effective_n, effective_n
