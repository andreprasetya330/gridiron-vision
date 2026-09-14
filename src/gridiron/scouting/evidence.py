"""The evidence bundle.

Everything a scouting report can possibly claim is computed here, deterministically,
with its sample size, its baseline, its confidence, and the ids of the exact plays
behind it. The prose layer downstream is allowed to rearrange these facts and
nothing else.

That separation is the whole trust model. A coach who clicks a claim lands on the
nine snaps that produced it. A claim with no plays behind it cannot be written,
because the writer never sees anything but this structure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gridiron.db.store import tendency_table
from gridiron.taxonomy import COVERAGES, MAN_COVERAGES
from gridiron.tells.context import (
    SeasonContext,
    opponent_adjusted_distribution,
    season_narrative,
)
from gridiron.tells.mining import MiningConfig, MiningResult, TellMiner
from gridiron.tells.validation import ValidationResult, attach_validation, forward_validate


@dataclass
class EvidenceBundle:
    team: str
    season: int | None = None
    through_week: int | None = None
    self_scout: bool = False

    n_plays: int = 0
    n_labeled_plays: int = 0
    n_usable_plays: int = 0
    baseline_source: str = "observed"
    baseline_is_borrowed: bool = False

    coverage_distribution: dict[str, float] = field(default_factory=dict)
    baseline_distribution: dict[str, float] = field(default_factory=dict)
    opponent_adjusted: dict[str, dict[str, float]] = field(default_factory=dict)
    man_zone_split: dict[str, float] = field(default_factory=dict)

    tells: list[dict[str, Any]] = field(default_factory=list)
    tendencies: list[dict[str, Any]] = field(default_factory=list)
    season_trend: list[str] = field(default_factory=list)
    what_beat_them: list[dict[str, Any]] = field(default_factory=list)
    personnel: list[dict[str, Any]] = field(default_factory=list)

    validation_summary: dict[str, Any] = field(default_factory=dict)
    mining_summary: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def headline(self) -> str:
        if not self.coverage_distribution:
            return f"No coverage data for {self.team}."
        top = max(self.coverage_distribution.items(), key=lambda kv: kv[1])
        return f"{self.team} plays {top[0]} on {top[1]:.0%} of snaps."


def _clip_ids(cues: pd.DataFrame, cue_key: str, cue_value: str, coverage: str, limit: int = 12) -> list[str]:
    matched = cues[(cues[cue_key].astype(str) == cue_value) & (cues["coverage"] == coverage)]
    if "play_id" not in matched.columns:
        return []
    return [str(v) for v in matched["play_id"].head(limit).tolist()]


def build_evidence(
    cues: pd.DataFrame,
    team: str,
    season: int | None = None,
    through_week: int | None = None,
    mining_config: MiningConfig | None = None,
    context: SeasonContext | None = None,
    play_context: pd.DataFrame | None = None,
    roster: pd.DataFrame | None = None,
    self_scout: bool = False,
    validate: bool = True,
) -> EvidenceBundle:
    context = context or SeasonContext()
    mining_config = mining_config or MiningConfig()

    bundle = EvidenceBundle(
        team=team,
        season=season,
        through_week=through_week,
        self_scout=self_scout,
        baseline_source=mining_config.baseline_source,
        baseline_is_borrowed=mining_config.baseline_source != "observed",
    )

    frame = cues.copy()
    if through_week is not None and "week" in frame.columns:
        frame = frame[pd.to_numeric(frame["week"], errors="coerce") <= through_week]

    team_frame = frame[frame["defense_team"] == team]
    bundle.n_plays = len(team_frame)
    bundle.n_labeled_plays = int(team_frame["coverage"].notna().sum()) if not team_frame.empty else 0
    if "usable" in team_frame.columns:
        bundle.n_usable_plays = int(team_frame["usable"].fillna(False).astype(bool).sum())

    if team_frame.empty:
        bundle.caveats.append(f"No plays found for {team}. Ingest film or data first.")
        return bundle

    # --- Season context: drop film from before a scheme change ---------------
    windowed, context_notes = context.usable_window(frame, team, season)
    bundle.caveats.extend(context_notes)

    windowed_team = windowed[windowed["defense_team"] == team]
    weights = context.weights_for(windowed_team, through_week)

    # --- Distributions -------------------------------------------------------
    labeled = windowed_team[windowed_team["coverage"].notna()]
    if labeled.empty:
        bundle.caveats.append(
            f"{team} has plays but no coverage labels. Run the film pipeline or import "
            "per-snap PFF data before scouting."
        )
        return bundle

    label_weights = context.weights_for(labeled, through_week)
    total_weight = label_weights.sum()
    covers = labeled["coverage"].to_numpy()
    bundle.coverage_distribution = {
        c: round(float(np.sum(label_weights[covers == c]) / total_weight), 4) for c in COVERAGES
    }

    others = windowed[(windowed["defense_team"] != team) & windowed["coverage"].notna()]
    if not others.empty:
        other_covers = others["coverage"].to_numpy()
        bundle.baseline_distribution = {
            c: round(float((other_covers == c).mean()), 4) for c in COVERAGES
        }

    man_rate = sum(bundle.coverage_distribution.get(c, 0.0) for c in MAN_COVERAGES)
    league_man = sum(bundle.baseline_distribution.get(c, 0.0) for c in MAN_COVERAGES)
    bundle.man_zone_split = {
        "man": round(man_rate, 4),
        "zone": round(1 - man_rate, 4),
        "league_man": round(league_man, 4),
        "delta": round(man_rate - league_man, 4),
    }

    bundle.opponent_adjusted = opponent_adjusted_distribution(windowed, team, label_weights)
    bundle.season_trend = season_narrative(windowed, team, through_week)

    # --- Tells ---------------------------------------------------------------
    mining: MiningResult = TellMiner(windowed, mining_config).mine(team, current_week=through_week)
    bundle.mining_summary = {
        "n_tests": mining.n_tests,
        "n_significant": mining.n_significant,
        "fdr_alpha": mining_config.fdr_alpha,
        "min_support": mining_config.min_support,
        "min_lift": mining_config.min_lift,
    }
    bundle.caveats.extend(mining.notes)

    validation = ValidationResult(team=team)
    if validate:
        validation = forward_validate(windowed, team, mining_config)
        bundle.validation_summary = validation.to_dict()
        bundle.caveats.extend(validation.notes)

    enriched = attach_validation(mining.tells, validation)
    for row, tell in zip(enriched, mining.tells):
        row["clip_play_ids"] = _clip_ids(
            windowed_team, tell.cue_key, tell.cue_value, tell.coverage
        )
    bundle.tells = enriched

    # --- Situational tendencies ---------------------------------------------
    try:
        tendencies = tendency_table(windowed, team, by=["down", "distance_band"], min_plays=10)
        if not tendencies.empty:
            tendencies = tendencies.reindex(
                tendencies["delta"].abs().sort_values(ascending=False).index
            )
            bundle.tendencies = tendencies.head(12).to_dict("records")
    except KeyError:
        pass

    # --- What beat them ------------------------------------------------------
    if play_context is not None and not play_context.empty:
        bundle.what_beat_them = _what_beat_them(play_context, team)

    if roster is not None and not roster.empty:
        bundle.personnel = _personnel_notes(roster)

    _add_caveats(bundle, mining_config)
    return bundle


def _what_beat_them(play_context: pd.DataFrame, team: str) -> list[dict[str, Any]]:
    frame = play_context[play_context["defense_team"] == team].copy()
    if frame.empty or "epa" not in frame.columns:
        return []

    frame["epa"] = pd.to_numeric(frame["epa"], errors="coerce")
    frame = frame.dropna(subset=["epa"])
    if frame.empty:
        return []

    frame["distance"] = pd.to_numeric(frame["distance"], errors="coerce")
    frame["distance_band"] = pd.cut(
        frame["distance"], bins=[-0.1, 3, 7, 11, 100], labels=["short", "medium", "long", "very-long"]
    )

    grouped = (
        frame.groupby(["down", "distance_band", "play_type"], observed=True)
        .agg(plays=("play_id", "count"), epa_per_play=("epa", "mean"), success_rate=("success", "mean"))
        .reset_index()
    )
    grouped = grouped[grouped["plays"] >= 8].sort_values("epa_per_play", ascending=False)
    return grouped.head(8).to_dict("records")


def _personnel_notes(roster: pd.DataFrame) -> list[dict[str, Any]]:
    columns = {c.lower(): c for c in roster.columns}
    position = columns.get("position")
    name = columns.get("first_name") or columns.get("name")
    if not position:
        return []
    secondary = roster[roster[position].astype(str).str.upper().isin(["CB", "S", "DB", "LB"])]
    if secondary.empty:
        return []
    keep = [c for c in (name, columns.get("last_name"), position, columns.get("jersey"), columns.get("year")) if c]
    return secondary[keep].head(20).to_dict("records")


def _add_caveats(bundle: EvidenceBundle, config: MiningConfig) -> None:
    if bundle.baseline_is_borrowed:
        bundle.caveats.append(
            "The national baseline is borrowed from NFL data because the college film "
            "corpus is not yet large enough to build its own. Treat every league-rate "
            "comparison as directional until that changes."
        )
    if bundle.n_labeled_plays < 150:
        bundle.caveats.append(
            f"Only {bundle.n_labeled_plays} labeled snaps for {bundle.team}. At this sample "
            "size a tell needs a very large effect to clear the significance bar, so the "
            "absence of tells here means 'not enough film', not 'no tells'."
        )
    if bundle.n_plays and bundle.n_usable_plays / max(bundle.n_plays, 1) < 0.7:
        bundle.caveats.append(
            f"Only {bundle.n_usable_plays} of {bundle.n_plays} plays passed the track-quality "
            "check. Coverage on the rest is a guess made with defenders missing."
        )
    validated = [t for t in bundle.tells if t.get("validation")]
    held = [t for t in validated if t["validation"].get("held_up")]
    if validated and not held:
        bundle.caveats.append(
            "None of the mined tells have held up on later film yet. Treat them as "
            "hypotheses to check on tape, not as calls to make."
        )
