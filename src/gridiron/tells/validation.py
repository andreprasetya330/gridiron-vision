"""Forward validation: does a tell hold up on film it was not mined from?

This is the feature that decides whether anyone uses the tool. A coach will not
act on "we found a correlation." He will act on "we found this in weeks 1 through
7, and since then it has held on 11 of 14 snaps."

The protocol is deliberately strict. Mine on weeks 1..k, score on week k+1 only,
then roll forward. A tell never gets evaluated on data that produced it, and the
reported hit rate is the pooled out-of-sample rate across every fold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gridiron.tells.mining import MiningConfig, Tell, TellMiner
from gridiron.tells.stats import wilson_interval


@dataclass
class TellValidation:
    cue_key: str
    cue_value: str
    coverage: str
    mined_rate: float
    # The rate the tell was mined against: this team's own base rate carried
    # through the league's effect for this cue. Beating the raw league average is
    # not evidence of a tell if the team simply plays that coverage a lot.
    expected_rate: float
    holdout_hits: int
    holdout_trials: int
    first_seen_week: int | None = None
    folds: int = 0

    @property
    def direction(self) -> int:
        """+1 for "they run this", -1 for "they avoid this".

        Both are real findings, but they are opposite predictions and must never
        be pooled. An avoidance tell is *supposed* to have a low hit rate; mixing
        it into an overall hit rate makes a working engine look broken.
        """
        return 1 if self.mined_rate >= self.expected_rate else -1

    @property
    def holdout_rate(self) -> float:
        return self.holdout_hits / self.holdout_trials if self.holdout_trials else float("nan")

    @property
    def held_up(self) -> bool:
        """Did the edge survive out of sample, in the direction it predicted?

        The bar is a Wilson bound clearing the expected rate. Beating it on a
        point estimate proves nothing at these sample sizes.
        """
        if self.holdout_trials < 5:
            return False
        interval = wilson_interval(self.holdout_hits, self.holdout_trials)
        if self.direction > 0:
            return interval.low > self.expected_rate
        return interval.high < self.expected_rate

    @property
    def shrinkage(self) -> float:
        """How much of the in-sample edge evaporated. Large values mean overfitting."""
        if not np.isfinite(self.holdout_rate):
            return float("nan")
        return (self.mined_rate - self.holdout_rate) * self.direction

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "direction": self.direction,
                "holdout_rate": None if not np.isfinite(self.holdout_rate) else round(self.holdout_rate, 4),
                "held_up": self.held_up,
                "shrinkage": None if not np.isfinite(self.shrinkage) else round(self.shrinkage, 4),
            }
        )
        return payload


@dataclass
class ValidationResult:
    team: str
    validations: list[TellValidation] = field(default_factory=list)
    folds: int = 0
    n_mined_total: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def predictive(self) -> list[TellValidation]:
        """Tells that say "they will run this", which is what a hit rate means."""
        return [v for v in self.validations if v.direction > 0]

    @property
    def avoidance(self) -> list[TellValidation]:
        return [v for v in self.validations if v.direction < 0]

    @property
    def survival_rate(self) -> float:
        scored = [v for v in self.validations if v.holdout_trials >= 5]
        if not scored:
            return float("nan")
        return sum(1 for v in scored if v.held_up) / len(scored)

    @property
    def pooled_hit_rate(self) -> float:
        """Out-of-sample hit rate across predictive tells only.

        Avoidance tells are scored separately by `pooled_avoidance_rate`. Pooling
        the two would average a prediction against its own opposite.
        """
        return _pooled_rate(self.predictive)

    @property
    def pooled_expected(self) -> float:
        """Sample-weighted expected rate for the same cue/coverage pairs."""
        return _pooled_expected(self.predictive)

    @property
    def pooled_avoidance_rate(self) -> float:
        """For avoidance tells, a rate *below* `pooled_avoidance_expected` is success."""
        return _pooled_rate(self.avoidance)

    @property
    def pooled_avoidance_expected(self) -> float:
        return _pooled_expected(self.avoidance)

    def to_dict(self) -> dict[str, Any]:
        def maybe(value: float) -> float | None:
            return None if not np.isfinite(value) else round(value, 4)

        return {
            "team": self.team,
            "folds": self.folds,
            "n_mined_total": self.n_mined_total,
            "n_predictive": len(self.predictive),
            "n_avoidance": len(self.avoidance),
            "survival_rate": maybe(self.survival_rate),
            "pooled_hit_rate": maybe(self.pooled_hit_rate),
            "pooled_expected": maybe(self.pooled_expected),
            "pooled_avoidance_rate": maybe(self.pooled_avoidance_rate),
            "pooled_avoidance_expected": maybe(self.pooled_avoidance_expected),
            "notes": self.notes,
            "tells": [v.to_dict() for v in self.validations],
        }


def _pooled_rate(entries: list[TellValidation]) -> float:
    hits = sum(v.holdout_hits for v in entries)
    trials = sum(v.holdout_trials for v in entries)
    return hits / trials if trials else float("nan")


def _pooled_expected(entries: list[TellValidation]) -> float:
    trials = sum(v.holdout_trials for v in entries)
    if not trials:
        return float("nan")
    return sum(v.expected_rate * v.holdout_trials for v in entries) / trials


def forward_validate(
    cues: pd.DataFrame,
    team: str,
    config: MiningConfig | None = None,
    min_train_weeks: int = 4,
) -> ValidationResult:
    """Roll through the season mining on the past and scoring on the next week."""
    config = config or MiningConfig()
    result = ValidationResult(team=team)

    if "week" not in cues.columns or cues["week"].isna().all():
        result.notes.append("no week column, so forward validation is not possible")
        return result

    frame = cues[cues["coverage"].notna()].copy()
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame = frame.dropna(subset=["week"])
    weeks = sorted(frame["week"].unique())

    if len(weeks) <= min_train_weeks:
        result.notes.append(
            f"only {len(weeks)} weeks available; need more than {min_train_weeks} to validate forward"
        )
        return result

    accumulator: dict[tuple[str, str, str], TellValidation] = {}

    for i in range(min_train_weeks, len(weeks)):
        train_weeks = set(weeks[:i])
        test_week = weeks[i]

        train = frame[frame["week"].isin(train_weeks)]
        test = frame[(frame["week"] == test_week) & (frame["defense_team"] == team)]
        if test.empty:
            continue

        mined = TellMiner(train, config).mine(team, current_week=float(weeks[i - 1]))
        result.n_mined_total += len(mined.tells)
        if not mined.tells:
            continue
        result.folds += 1

        for tell in mined.tells:
            key = (tell.cue_key, tell.cue_value, tell.coverage)
            entry = accumulator.get(key)
            if entry is None:
                entry = TellValidation(
                    cue_key=tell.cue_key,
                    cue_value=tell.cue_value,
                    coverage=tell.coverage,
                    mined_rate=tell.shrunk_rate,
                    expected_rate=tell.expected_rate,
                    holdout_hits=0,
                    holdout_trials=0,
                    first_seen_week=int(test_week),
                )
                accumulator[key] = entry

            matched = test[test[tell.cue_key].astype(str) == tell.cue_value]
            if matched.empty:
                continue
            entry.holdout_trials += len(matched)
            entry.holdout_hits += int((matched["coverage"] == tell.coverage).sum())
            entry.folds += 1
            # Keep the most recent mined estimate; it reflects the most film.
            entry.mined_rate = tell.shrunk_rate
            entry.expected_rate = tell.expected_rate

    result.validations = sorted(
        accumulator.values(), key=lambda v: (-v.holdout_trials, -v.mined_rate)
    )
    if not result.validations:
        result.notes.append(
            "no tell was mined early enough to be tested on later film; the season is "
            "too short or the defense is too consistent to tip"
        )
    return result


def attach_validation(tells: list[Tell], validation: ValidationResult) -> list[dict[str, Any]]:
    """Join mined tells to their out-of-sample record for reporting."""
    lookup = {(v.cue_key, v.cue_value, v.coverage): v for v in validation.validations}
    rows: list[dict[str, Any]] = []
    for tell in tells:
        payload = tell.to_dict()
        record = lookup.get((tell.cue_key, tell.cue_value, tell.coverage))
        if record is not None and record.holdout_trials > 0:
            payload["validation"] = {
                "holdout_hits": record.holdout_hits,
                "holdout_trials": record.holdout_trials,
                "holdout_rate": round(record.holdout_rate, 4),
                "held_up": record.held_up,
            }
            payload["evidence_line"] = (
                f"held on {record.holdout_hits} of {record.holdout_trials} snaps "
                f"since it was found"
            )
        else:
            payload["validation"] = None
            payload["evidence_line"] = "not yet tested out of sample"
        rows.append(payload)
    return rows
