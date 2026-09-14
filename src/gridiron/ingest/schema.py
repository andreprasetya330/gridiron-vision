"""The one normalized play-context table.

A play from film, a row from CollegeFootballData, and a line in a PFF export all
describe the same event. They only become joinable if they agree on a key, so
everything is forced into `(season, week, defense_team, offense_team, play_id)`
with a normalized team name.

Team naming is the boring part that breaks everything. CFBD says "Washington",
PFF says "WASH", nflverse says "WAS", and a Hudl export says whatever the video
coordinator typed. One alias table, applied at every boundary.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

PLAY_CONTEXT_COLUMNS: list[str] = [
    "play_id",
    "season",
    "week",
    "defense_team",
    "offense_team",
    "down",
    "distance",
    "yardline",
    "quarter",
    "clock_seconds",
    "score_margin",
    "offense_personnel",
    "play_type",
    "yards_gained",
    "epa",
    "success",
    "coverage",
    "coverage_source",
    "source",
]

# Extend freely. The right long-term answer is a lookup keyed on the CFBD team id,
# but that only helps for CFBD; hand-maintained aliases are what cover the rest.
TEAM_ALIASES: dict[str, str] = {
    "washington huskies": "Washington",
    "uw": "Washington",
    "wash": "Washington",
    "u of w": "Washington",
    "washington st": "Washington State",
    "wsu": "Washington State",
    "oregon ducks": "Oregon",
    "ore": "Oregon",
    "oregon st": "Oregon State",
    "osu-or": "Oregon State",
    "cal": "California",
    "california golden bears": "California",
    "stanford cardinal": "Stanford",
    "asu": "Arizona State",
    "arizona st": "Arizona State",
    "usc trojans": "USC",
    "southern california": "USC",
    "ucla bruins": "UCLA",
}

_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


def normalize_team(name: Any) -> str | None:
    """Canonicalize a team name so joins across sources actually land."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    text = unicodedata.normalize("NFKD", str(name)).strip()
    if not text:
        return None
    key = _SPACES.sub(" ", _PUNCT.sub("", text.lower())).strip()
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    # Title-case as a fallback, preserving all-caps acronyms like USC and UCLA.
    if text.isupper() and len(text) <= 5:
        return text
    return text.strip()


def success_from_play(down: Any, distance: Any, yards_gained: Any) -> bool | None:
    """Standard success rate definition: 50/70/100% of needed yards by down."""
    try:
        down = int(down)
        distance = float(distance)
        gained = float(yards_gained)
    except (TypeError, ValueError):
        return None
    if distance <= 0:
        return None
    thresholds = {1: 0.5, 2: 0.7, 3: 1.0, 4: 1.0}
    return gained >= thresholds.get(down, 1.0) * distance


def to_play_context(rows: list[dict[str, Any]], source: str) -> pd.DataFrame:
    """Coerce arbitrary source rows into the shared schema."""
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=PLAY_CONTEXT_COLUMNS)

    for column in PLAY_CONTEXT_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df["source"] = df["source"].fillna(source)
    for column in ("defense_team", "offense_team"):
        df[column] = df[column].map(normalize_team)

    for column in ("season", "week", "down", "quarter"):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    for column in ("distance", "yardline", "clock_seconds", "yards_gained", "epa", "score_margin"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    missing_success = df["success"].isna()
    if missing_success.any():
        df.loc[missing_success, "success"] = df.loc[missing_success].apply(
            lambda r: success_from_play(r["down"], r["distance"], r["yards_gained"]), axis=1
        )

    return df[PLAY_CONTEXT_COLUMNS]


def merge_context(*frames: pd.DataFrame) -> pd.DataFrame:
    """Stack context from several sources, preferring rows that carry a coverage label."""
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        return pd.DataFrame(columns=PLAY_CONTEXT_COLUMNS)
    combined = pd.concat(usable, ignore_index=True)
    combined["_has_coverage"] = combined["coverage"].notna().astype(int)
    combined = combined.sort_values("_has_coverage", ascending=False)
    combined = combined.drop_duplicates(subset=["play_id"], keep="first")
    return combined.drop(columns="_has_coverage").reset_index(drop=True)
