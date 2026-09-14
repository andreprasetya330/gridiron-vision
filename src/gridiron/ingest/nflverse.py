"""nflverse loaders.

nflfastR publishes play-by-play as Parquet on GitHub releases with no key and no
rate limit. It carries EPA, success, personnel, and win probability for every NFL
play back to 1999, which makes it the reference corpus for the situational half of
the analysis - and, alongside Big Data Bowl coverage labels, the source of the
borrowed national baseline the college side leans on until its own film corpus
grows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from gridiron.config import subdir
from gridiron.ingest.schema import normalize_team, to_play_context

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"


def download_pbp(season: int, force: bool = False) -> Path:
    cache = subdir("cache", "nflverse")
    path = cache / f"play_by_play_{season}.parquet"
    if path.exists() and not force:
        return path
    url = PBP_URL.format(season=season)
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def load_pbp(season: int, columns: list[str] | None = None) -> pd.DataFrame:
    path = download_pbp(season)
    return pd.read_parquet(path, columns=columns)


PBP_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "week",
    "posteam",
    "defteam",
    "down",
    "ydstogo",
    "yardline_100",
    "qtr",
    "quarter_seconds_remaining",
    "score_differential",
    "offense_personnel",
    "play_type",
    "yards_gained",
    "epa",
    "success",
]


def pbp_to_context(season: int, team: str | None = None) -> pd.DataFrame:
    """nflfastR play-by-play -> the shared play-context schema."""
    available = pd.read_parquet(download_pbp(season)).columns
    columns = [c for c in PBP_COLUMNS if c in available]
    pbp = load_pbp(season, columns=columns)

    if team:
        canonical = normalize_team(team)
        pbp = pbp[pbp["defteam"].map(normalize_team) == canonical]

    rows: list[dict[str, Any]] = []
    for _, row in pbp.iterrows():
        rows.append(
            {
                "play_id": f"nflverse-{row.get('game_id')}-{row.get('play_id')}",
                "season": row.get("season"),
                "week": row.get("week"),
                "defense_team": row.get("defteam"),
                "offense_team": row.get("posteam"),
                "down": row.get("down"),
                "distance": row.get("ydstogo"),
                "yardline": row.get("yardline_100"),
                "quarter": row.get("qtr"),
                "clock_seconds": row.get("quarter_seconds_remaining"),
                "score_margin": -row.get("score_differential")
                if pd.notna(row.get("score_differential"))
                else None,
                "offense_personnel": row.get("offense_personnel"),
                "play_type": row.get("play_type"),
                "yards_gained": row.get("yards_gained"),
                "epa": row.get("epa"),
                "success": row.get("success"),
                "source": "nflverse",
            }
        )
    return to_play_context(rows, "nflverse")


def situational_epa(context: pd.DataFrame, team: str) -> pd.DataFrame:
    """What actually beat this defense, by situation.

    Feeds the "what beat them" section of the scouting report. A tendency is only
    worth exploiting if exploiting it has been working.
    """
    frame = context[context["defense_team"] == normalize_team(team)].copy()
    if frame.empty:
        return pd.DataFrame()

    frame["down"] = pd.to_numeric(frame["down"], errors="coerce")
    frame["distance"] = pd.to_numeric(frame["distance"], errors="coerce")
    frame["distance_band"] = pd.cut(
        frame["distance"],
        bins=[-0.1, 3, 7, 11, 100],
        labels=["short", "medium", "long", "very-long"],
    )

    grouped = (
        frame.groupby(["down", "distance_band", "play_type"], observed=True)
        .agg(
            plays=("play_id", "count"),
            epa_per_play=("epa", "mean"),
            success_rate=("success", "mean"),
            yards_per_play=("yards_gained", "mean"),
        )
        .reset_index()
    )
    return grouped[grouped["plays"] >= 8].sort_values("epa_per_play", ascending=False)
