"""Tell mining: find where a defense deviates from the national baseline.

The unit of discovery is a (cue value, coverage) pair. For each one we ask: when
this team shows this look, how often do they play this coverage, and is that
different from what everyone else does after showing the same look?

Three details that matter more than they look:

- The national baseline **excludes the team being scouted**. Otherwise a team with
  a strong habit contributes to the very average it is measured against, which
  shrinks its own apparent tell. With six teams in a conference that bias is large.
- The comparison controls for the team's **own** coverage distribution. A team that
  plays Cover 3 on 45% of snaps against a league average of 30% will beat the
  league conditional rate after almost every cue, and a naive comparison reports
  each of those as a separate tell. None of them are: they are one fact about the
  team, counted forty times. The null here is the team's own base rate carried
  through the league's cue effect, so a tell has to mean "given this look they
  play it more than *they* usually do", which is the only version a coach can use.
- Ranking uses absolute rate as well as lift. A cue that moves Cover 3 from 30% to
  55% is statistically interesting; a cue that means Cover 0 is coming 90% of the
  time is what actually gets a play called. The score rewards both.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from gridiron.cues.vocabulary import CUE_KEYS
from gridiron.taxonomy import COVERAGES
from gridiron.tells.stats import (
    benjamini_hochberg,
    beta_binomial_posterior,
    estimate_prior_strength,
    exponential_recency_weights,
    two_proportion_pvalue,
    weighted_binomial_counts,
    wilson_interval,
)

# Situation cues describe the down and distance, not what the defense showed.
# They are still mineable - "3rd and long means quarters" is a real tendency - but
# they are tagged separately because a tell you can see is worth more than a
# tendency you have to remember.
SITUATION_CUES = {"down", "distance_band", "field_zone", "hash_side", "personnel", "score_state"}

CUE_PHRASES: dict[str, str] = {
    "shell": "the pre-snap shell is {value}",
    "deep_defender_count": "{value} defender(s) align deep",
    "deepest_depth_band": "the deepest defender is {value} yards off",
    "safety_depth_band": "the safeties average {value} yards deep",
    "safety_split_band": "the safety split is {value}",
    "late_rotation": "there is late rotation: {value}",
    "rotation_direction": "the late rotation goes {value}",
    "press_corners": "{value} corner(s) press",
    "corner_leverage": "the corners play {value} leverage",
    "corner_depth_band": "the corners align {value}",
    "box_count_band": "the box count is {value}",
    "los_defenders_band": "{value} defenders are on the line",
    "slot_defender_depth_band": "the slot defender is {value}",
    "defense_spread_band": "the defensive front is {value}",
    "linebacker_depth_band": "the linebackers are {value}",
    "motion_response": "the response to motion is {value}",
    "field_safety_bias": "the deep safety shades {value}",
    "down": "it is {value} down",
    "distance_band": "the distance is {value}",
    "field_zone": "the ball is in {value}",
    "hash_side": "the ball is on the {value} hash",
    "personnel": "the offense is in {value} personnel",
    "score_state": "the defense is {value}",
}


def phrase_cue(cue_key: str, cue_value: str) -> str:
    template = CUE_PHRASES.get(cue_key, cue_key + " is {value}")
    return template.format(value=cue_value)


@dataclass
class Tell:
    team: str
    cue_key: str
    cue_value: str
    coverage: str
    n_cue: int
    n_cue_coverage: int
    raw_rate: float
    shrunk_rate: float
    baseline_rate: float
    baseline_n: int
    # What this team plays overall, and what that implies for this cue once the
    # league-wide effect of the cue is applied. `lift` is measured against the
    # latter, so it isolates the cue rather than the team's general preference.
    team_base_rate: float
    expected_rate: float
    lift: float
    ci_low: float
    ci_high: float
    p_value: float
    q_value: float
    significant: bool
    support: float
    score: float
    baseline_source: str = "observed"
    kind: str = "alignment"
    prior_strength: float = 20.0

    @property
    def description(self) -> str:
        direction = "run" if self.lift > 0 else "avoid"
        return (
            f"When {phrase_cue(self.cue_key, self.cue_value)}, {self.team} {direction} "
            f"{self.coverage} on {self.shrunk_rate:.0%} of snaps, against "
            f"{self.expected_rate:.0%} expected from their own mix "
            f"({self.team_base_rate:.0%} overall, league {self.baseline_rate:.0%} "
            f"after this look), n={self.n_cue}"
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["description"] = self.description
        return payload


def passes_effect_floor(tell: Tell, min_lift: float, min_ratio: float, min_rate: float) -> bool:
    """Is this difference big enough to bother a coach with?

    Absolute lift alone is the wrong test. A cue that takes Cover 2 Zone from 2%
    of snaps to 13% only moves the rate by eleven points, so an absolute floor
    throws it away - but going from "they never do this" to "one snap in eight"
    is one of the most actionable things you can tell a defense's opponent. So a
    tell qualifies on either a large absolute move or a large relative one, with
    the relative path gated on the resulting rate being worth acting on at all.
    """
    if abs(tell.lift) >= min_lift:
        return True
    if tell.lift > 0:
        return tell.shrunk_rate >= max(min_ratio * tell.expected_rate, min_rate)
    if tell.shrunk_rate > 0:
        return tell.expected_rate >= max(min_ratio * tell.shrunk_rate, min_rate)
    return False


def transported_rate(team_base: float, league_cue: float, league_base: float) -> float:
    """The team's own rate, moved by however much the cue moves the league.

    Working in log-odds rather than raw percentage points because a cue that adds
    ten points to a 30% coverage cannot add ten points to one the team plays 3% of
    the time - the arithmetic version predicts rates above 1 for common coverages
    and negative rates for rare ones. Odds ratios compose cleanly and stay in range.
    """
    eps = 1e-3
    tb = float(np.clip(team_base, eps, 1 - eps))
    lc = float(np.clip(league_cue, eps, 1 - eps))
    lb = float(np.clip(league_base, eps, 1 - eps))

    def logit(p: float) -> float:
        return float(np.log(p / (1 - p)))

    z = logit(tb) + logit(lc) - logit(lb)
    return float(1.0 / (1.0 + np.exp(-z)))


@dataclass
class MiningConfig:
    min_support: int = 10
    min_lift: float = 0.10
    # Relative-change escape hatch for rare-but-decisive coverages.
    min_ratio: float = 2.5
    min_ratio_rate: float = 0.08
    min_absolute_rate: float = 0.0
    fdr_alpha: float = 0.10
    prior_strength: float | None = None  # None means estimate it empirically
    use_recency: bool = True
    half_life_weeks: float = 3.0
    max_tells: int = 25
    include_situation_cues: bool = True
    baseline_source: str = "observed"
    min_quality: float = 0.0
    # Unconditional coverage mix, usually from Big Data Bowl. When set, this
    # replaces the mix computed from other teams in the current table - that is
    # how a college corpus borrows an NFL national baseline.
    league_rates: dict[str, float] | None = None


def config_for_corpus(source: str, **kwargs: Any) -> MiningConfig:
    """Attach a borrowed NFL baseline when mining college film.

    Synthetic and BDB corpora are self-contained: their own table *is* the
    league. Film is not, and comparing a Husky defense to two other filmed
    opponents is not a national baseline.
    """
    config = MiningConfig(**kwargs)
    if source in {"film", "pff", "hudl"}:
        from gridiron.tracking.bdb import load_league_rates

        rates = load_league_rates()
        if rates:
            config.league_rates = rates
            config.baseline_source = "nfl-big-data-bowl"
    return config


@dataclass
class MiningResult:
    team: str
    tells: list[Tell] = field(default_factory=list)
    n_plays: int = 0
    n_tests: int = 0
    n_significant: int = 0
    coverage_distribution: dict[str, float] = field(default_factory=dict)
    baseline_distribution: dict[str, float] = field(default_factory=dict)
    baseline_source: str = "observed"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "n_plays": self.n_plays,
            "n_tests": self.n_tests,
            "n_significant": self.n_significant,
            "coverage_distribution": self.coverage_distribution,
            "baseline_distribution": self.baseline_distribution,
            "baseline_source": self.baseline_source,
            "notes": self.notes,
            "tells": [t.to_dict() for t in self.tells],
        }


class TellMiner:
    """Mines one cue table for team-specific coverage tendencies."""

    def __init__(self, cues: pd.DataFrame, config: MiningConfig | None = None) -> None:
        self.config = config or MiningConfig()
        required = {"defense_team", "coverage"}
        missing = required - set(cues.columns)
        if missing:
            raise ValueError(f"cue table missing columns: {sorted(missing)}")
        self.cues = cues[cues["coverage"].notna()].copy()
        if "quality_score" in self.cues.columns and self.config.min_quality > 0:
            self.cues = self.cues[self.cues["quality_score"] >= self.config.min_quality]

    @property
    def cue_keys(self) -> list[str]:
        keys = [k for k in CUE_KEYS if k in self.cues.columns]
        if not self.config.include_situation_cues:
            keys = [k for k in keys if k not in SITUATION_CUES]
        return keys

    def _weights(self, frame: pd.DataFrame, current_week: float | None) -> np.ndarray:
        if not self.config.use_recency or "week" not in frame.columns:
            return np.ones(len(frame), dtype=np.float64)
        weeks = pd.to_numeric(frame["week"], errors="coerce").to_numpy(dtype=np.float64)
        if np.isnan(weeks).all():
            return np.ones(len(frame), dtype=np.float64)
        weeks = np.nan_to_num(weeks, nan=np.nanmedian(weeks))
        ref = current_week if current_week is not None else float(np.nanmax(weeks))
        return exponential_recency_weights(weeks, ref, self.config.half_life_weeks)

    def mine(self, team: str, current_week: float | None = None) -> MiningResult:
        cfg = self.config
        team_plays = self.cues[self.cues["defense_team"] == team]
        others = self.cues[self.cues["defense_team"] != team]

        result = MiningResult(
            team=team, n_plays=len(team_plays), baseline_source=cfg.baseline_source
        )
        if team_plays.empty:
            result.notes.append(f"no labeled plays found for {team}")
            return result
        if others.empty:
            result.notes.append(
                "no other teams in the corpus, so there is no national baseline to "
                "compare against; tells cannot be mined"
            )
            return result

        weights = self._weights(team_plays, current_week)
        team_coverage = team_plays["coverage"].to_numpy()

        result.coverage_distribution = {
            c: round(float(np.sum(weights[team_coverage == c]) / weights.sum()), 4)
            for c in COVERAGES
        }
        other_coverage = others["coverage"].to_numpy()
        if cfg.league_rates:
            league_base = {c: float(cfg.league_rates.get(c, 0.0)) for c in COVERAGES}
            total = sum(league_base.values()) or 1.0
            league_base = {c: v / total for c, v in league_base.items()}
        else:
            league_base = {c: float(np.mean(other_coverage == c)) for c in COVERAGES}
        result.baseline_distribution = {c: round(league_base[c], 4) for c in COVERAGES}
        team_base = {c: float(np.sum(weights[team_coverage == c]) / weights.sum()) for c in COVERAGES}

        candidates: list[Tell] = []
        pvalues: list[float] = []

        for cue_key in self.cue_keys:
            team_values = team_plays[cue_key].astype(str).to_numpy()
            other_values = others[cue_key].astype(str).to_numpy()

            for cue_value in pd.unique(team_values):
                if cue_value in ("unknown", "nan"):
                    continue
                cue_mask = team_values == cue_value
                n_cue_raw = int(cue_mask.sum())
                if n_cue_raw < cfg.min_support:
                    continue

                baseline_mask = other_values == cue_value
                baseline_n = int(baseline_mask.sum())
                if baseline_n < cfg.min_support:
                    continue

                baseline_covs = other_coverage[baseline_mask]
                cue_weights = weights[cue_mask]
                cue_covs = team_coverage[cue_mask]

                # Shrinkage strength is estimated per (cue value, coverage) from
                # how much teams actually differ after showing this look.
                peer_groups = list(others[baseline_mask].groupby("defense_team"))

                for coverage in COVERAGES:
                    baseline_rate = float((baseline_covs == coverage).mean())
                    if baseline_rate <= 0.0 and not (cue_covs == coverage).any():
                        continue

                    successes, trials = weighted_binomial_counts(cue_covs == coverage, cue_weights)
                    if trials < cfg.min_support:
                        continue

                    k = int(round(successes))
                    n = int(round(trials))
                    raw_rate = successes / trials if trials else 0.0

                    expected = transported_rate(
                        team_base[coverage], baseline_rate, league_base[coverage]
                    )

                    if cfg.prior_strength is not None:
                        strength = cfg.prior_strength
                    else:
                        rates = [float((g["coverage"] == coverage).mean()) for _, g in peer_groups]
                        counts = [len(g) for _, g in peer_groups]
                        strength = estimate_prior_strength(np.array(rates), np.array(counts))

                    shrunk = beta_binomial_posterior(k, n, expected, strength)
                    lift = shrunk - expected

                    interval = wilson_interval(k, n)
                    # The expected rate is itself estimated, from the team's own
                    # season and from the peers who showed this look. Using the
                    # smaller of the two keeps the test from treating it as known.
                    p = two_proportion_pvalue(
                        k, n, expected, baseline_trials=min(baseline_n, len(team_plays))
                    )

                    candidates.append(
                        Tell(
                            team=team,
                            cue_key=cue_key,
                            cue_value=str(cue_value),
                            coverage=coverage,
                            n_cue=n,
                            n_cue_coverage=k,
                            raw_rate=round(raw_rate, 4),
                            shrunk_rate=round(shrunk, 4),
                            baseline_rate=round(baseline_rate, 4),
                            baseline_n=baseline_n,
                            team_base_rate=round(team_base[coverage], 4),
                            expected_rate=round(expected, 4),
                            lift=round(lift, 4),
                            ci_low=round(interval.low, 4),
                            ci_high=round(interval.high, 4),
                            p_value=p,
                            q_value=1.0,
                            significant=False,
                            support=round(n_cue_raw / len(team_plays), 4),
                            score=0.0,
                            baseline_source=cfg.baseline_source,
                            kind="situation" if cue_key in SITUATION_CUES else "alignment",
                            prior_strength=round(float(strength), 2),
                        )
                    )
                    pvalues.append(p)

        result.n_tests = len(candidates)
        if not candidates:
            result.notes.append(
                "no cue reached the minimum sample threshold; more film is needed"
            )
            return result

        rejected, qvalues = benjamini_hochberg(np.array(pvalues), alpha=cfg.fdr_alpha)
        for tell, q, ok in zip(candidates, qvalues, rejected):
            tell.q_value = round(float(q), 5)
            tell.significant = bool(ok)
            tell.score = round(_score(tell), 4)

        result.n_significant = int(sum(1 for t in candidates if t.significant))

        keep = [
            t
            for t in candidates
            if t.significant
            and passes_effect_floor(t, cfg.min_lift, cfg.min_ratio, cfg.min_ratio_rate)
            and t.shrunk_rate >= cfg.min_absolute_rate
        ]
        keep.sort(key=lambda t: -t.score)
        result.tells = _deduplicate(keep)[: cfg.max_tells]

        if not result.tells:
            result.notes.append(
                "nothing survived FDR correction and the effect-size floor. That is a "
                "real finding, not a failure: this defense does not tip its coverage "
                "in any way this corpus can detect."
            )
        return result

    def mine_all(self, current_week: float | None = None) -> dict[str, MiningResult]:
        teams = sorted(self.cues["defense_team"].dropna().unique())
        return {team: self.mine(team, current_week) for team in teams}


def _score(tell: Tell) -> float:
    """Rank tells by how much a coach can actually do with them.

    Four factors: how far it moves the odds, how surprising that move is relative
    to the league rate, how confident we are, and how often the look appears. A
    40-point tell that shows up twice a game beats a 15-point tell on a look they
    run once a season.
    """
    effect = abs(tell.lift)
    if tell.lift > 0 and tell.expected_rate > 1e-6:
        ratio = tell.shrunk_rate / tell.expected_rate
        if ratio > 1.0:
            effect = max(effect, tell.shrunk_rate * min(float(np.log2(ratio)) / 3.0, 1.0))
    certainty = 1.0 - min(tell.q_value, 1.0)
    volume = np.log1p(tell.n_cue) / np.log1p(60)
    absolute = tell.shrunk_rate if tell.lift > 0 else (1.0 - tell.shrunk_rate)
    frequency = min(tell.support * 4, 1.0)
    return float(effect * (0.5 + 0.5 * certainty) * (0.4 + 0.6 * volume) * absolute * (0.5 + 0.5 * frequency))


def _deduplicate(tells: Iterable[Tell]) -> list[Tell]:
    """Collapse near-duplicate findings.

    `shell=2-high` and `deep_defender_count=2` are the same observation wearing
    two hats. Reporting both inflates the apparent number of tells and wastes a
    coach's attention, so only the highest-scoring member of each
    (cue value, coverage) family survives.
    """
    families: dict[tuple[str, str, str], Tell] = {}
    aliases = {
        "deep_defender_count": "shell",
        "safety_depth_band": "shell",
        "deepest_depth_band": "shell",
        "corner_depth_band": "press_corners",
        "rotation_direction": "late_rotation",
    }
    for tell in tells:
        family = aliases.get(tell.cue_key, tell.cue_key)
        key = (family, tell.cue_value, tell.coverage)
        existing = families.get(key)
        if existing is None or tell.score > existing.score:
            families[key] = tell
    return sorted(families.values(), key=lambda t: -t.score)


def tells_to_frame(result: MiningResult) -> pd.DataFrame:
    if not result.tells:
        return pd.DataFrame(
            columns=[
                "team",
                "cue_key",
                "cue_value",
                "coverage",
                "shrunk_rate",
                "expected_rate",
                "baseline_rate",
                "lift",
                "n_cue",
                "q_value",
                "score",
            ]
        )
    return pd.DataFrame([t.to_dict() for t in result.tells])
