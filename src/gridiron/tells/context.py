"""Season context: recency, opponent adjustment, and scheme changes.

Three ways a season's worth of film lies to you:

1. **Staleness.** A defense in week 11 is not the defense it was in week 2.
   Handled by exponential recency weighting rather than by throwing early film
   away, so volume still counts for something.
2. **Schedule.** A defense that played three option teams looks run-heavy in a
   way that says nothing about how they will play you. Handled by adjusting for
   who they faced.
3. **Scheme change.** A new coordinator, or a mid-season philosophical shift,
   makes prior film actively misleading rather than merely stale. Handled by
   detecting the changepoint and refusing to mine across it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from gridiron.taxonomy import COVERAGES
from gridiron.tells.stats import exponential_recency_weights


@dataclass
class CoordinatorChange:
    team: str
    season: int
    effective_week: int
    note: str = ""


@dataclass
class SeasonContext:
    half_life_weeks: float = 3.0
    coordinator_changes: list[CoordinatorChange] = field(default_factory=list)
    changepoint_alpha: float = 0.01
    min_weeks_per_side: int = 3

    @classmethod
    def load(cls, path: Path) -> SeasonContext:
        if not Path(path).exists():
            return cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            half_life_weeks=payload.get("half_life_weeks", 3.0),
            coordinator_changes=[
                CoordinatorChange(**c) for c in payload.get("coordinator_changes", [])
            ],
        )

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {
                    "half_life_weeks": self.half_life_weeks,
                    "coordinator_changes": [c.__dict__ for c in self.coordinator_changes],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def change_week_for(self, team: str, season: int | None) -> int | None:
        for change in self.coordinator_changes:
            if change.team == team and (season is None or change.season == season):
                return change.effective_week
        return None

    def usable_window(
        self, cues: pd.DataFrame, team: str, season: int | None = None
    ) -> tuple[pd.DataFrame, list[str]]:
        """Drop film from before a scheme change, recorded or detected."""
        notes: list[str] = []
        frame = cues[cues["defense_team"] == team].copy()
        if frame.empty or "week" not in frame.columns:
            return cues, notes

        frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
        recorded = self.change_week_for(team, season)
        detected = detect_scheme_change(
            frame, alpha=self.changepoint_alpha, min_weeks_per_side=self.min_weeks_per_side
        )

        cut = recorded if recorded is not None else detected
        if cut is None:
            return cues, notes

        if recorded is not None:
            notes.append(
                f"{team} changed defensive coordinators in week {recorded}; film before "
                f"that week is excluded from tell mining."
            )
        else:
            notes.append(
                f"Their coverage mix shifts sharply at week {detected}, which usually means "
                f"a scheme change or a personnel loss. Film before that week is excluded."
            )

        keep_team = frame[frame["week"] >= cut]
        others = cues[cues["defense_team"] != team]
        return pd.concat([others, keep_team], ignore_index=True), notes

    def weights_for(self, cues: pd.DataFrame, current_week: float | None = None) -> np.ndarray:
        weeks = pd.to_numeric(cues.get("week"), errors="coerce").to_numpy(dtype=np.float64)
        if weeks.size == 0 or np.isnan(weeks).all():
            return np.ones(len(cues), dtype=np.float64)
        weeks = np.nan_to_num(weeks, nan=float(np.nanmedian(weeks)))
        ref = current_week if current_week is not None else float(np.nanmax(weeks))
        return exponential_recency_weights(weeks, ref, self.half_life_weeks)


def detect_scheme_change(
    team_cues: pd.DataFrame, alpha: float = 0.01, min_weeks_per_side: int = 3
) -> int | None:
    """Find the week where a defense's coverage mix changes most abruptly.

    Sweeps every candidate split, runs a chi-square test on the coverage
    distribution before versus after, and returns the best split only if it
    survives a Bonferroni correction over the number of splits tried. Without that
    correction this would "detect" a change in basically every season.
    """
    frame = team_cues.dropna(subset=["coverage", "week"])
    weeks = sorted(frame["week"].unique())
    if len(weeks) < 2 * min_weeks_per_side:
        return None

    candidates = weeks[min_weeks_per_side : len(weeks) - min_weeks_per_side + 1]
    if not candidates:
        return None

    best_week, best_p = None, 1.0
    for cut in candidates:
        before = frame[frame["week"] < cut]["coverage"]
        after = frame[frame["week"] >= cut]["coverage"]
        if len(before) < 30 or len(after) < 30:
            continue
        table = np.array(
            [
                [(before == c).sum() for c in COVERAGES],
                [(after == c).sum() for c in COVERAGES],
            ],
            dtype=np.float64,
        )
        keep = table.sum(axis=0) > 0
        table = table[:, keep]
        if table.shape[1] < 2:
            continue
        try:
            _, p, _, _ = stats.chi2_contingency(table)
        except ValueError:
            continue
        if p < best_p:
            best_p, best_week = p, int(cut)

    threshold = alpha / max(len(candidates), 1)
    return best_week if best_p < threshold else None


def opponent_adjusted_distribution(
    cues: pd.DataFrame, team: str, weights: np.ndarray | None = None
) -> dict[str, dict[str, float]]:
    """Separate what a defense chooses to do from what its schedule forced.

    For each play, compute what the rest of the league does against that same
    offense. The difference between a team's raw rate and that expectation is the
    part that is actually about the defense, which is the part that transfers to
    the game you are preparing for.
    """
    frame = cues[cues["coverage"].notna()].copy()
    team_plays = frame[frame["defense_team"] == team]
    if team_plays.empty:
        return {}

    if weights is None:
        weights = np.ones(len(team_plays), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size != len(team_plays):
        weights = np.ones(len(team_plays), dtype=np.float64)

    league_rate = {c: float((frame["coverage"] == c).mean()) for c in COVERAGES}

    expected = {c: [] for c in COVERAGES}
    for offense, group in team_plays.groupby("offense_team"):
        peers = frame[(frame["offense_team"] == offense) & (frame["defense_team"] != team)]
        if len(peers) < 20:
            rates = league_rate
        else:
            rates = {c: float((peers["coverage"] == c).mean()) for c in COVERAGES}
        for c in COVERAGES:
            expected[c].extend([rates[c]] * len(group))

    covers = team_plays["coverage"].to_numpy()
    total_weight = weights.sum()

    out: dict[str, dict[str, float]] = {}
    for c in COVERAGES:
        raw = float(np.sum(weights[covers == c]) / total_weight) if total_weight else 0.0
        exp = float(np.mean(expected[c])) if expected[c] else league_rate[c]
        out[c] = {
            "raw": round(raw, 4),
            "expected_from_schedule": round(exp, 4),
            "adjusted": round(raw - exp + league_rate[c], 4),
            "league": round(league_rate[c], 4),
            "schedule_effect": round(exp - league_rate[c], 4),
        }
    return out


def season_narrative(
    cues: pd.DataFrame, team: str, current_week: float | None = None
) -> list[str]:
    """Plain statements about how this defense has trended, for the report."""
    frame = cues[(cues["defense_team"] == team) & cues["coverage"].notna()].copy()
    if frame.empty or "week" not in frame.columns:
        return []

    frame["week"] = pd.to_numeric(frame["week"], errors="coerce")
    frame = frame.dropna(subset=["week"])
    weeks = sorted(frame["week"].unique())
    if len(weeks) < 4:
        return []

    split = weeks[len(weeks) // 2]
    early = frame[frame["week"] < split]["coverage"]
    late = frame[frame["week"] >= split]["coverage"]

    lines: list[str] = []
    for coverage in COVERAGES:
        early_rate = float((early == coverage).mean()) if len(early) else 0.0
        late_rate = float((late == coverage).mean()) if len(late) else 0.0
        delta = late_rate - early_rate
        if abs(delta) < 0.08 or max(early_rate, late_rate) < 0.08:
            continue
        direction = "up" if delta > 0 else "down"
        lines.append(
            f"{coverage} is {direction} from {early_rate:.0%} in the first half of the "
            f"season to {late_rate:.0%} since week {int(split)}."
        )
    return lines[:4]
