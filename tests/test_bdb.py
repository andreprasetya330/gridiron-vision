"""Big Data Bowl helpers that do not need the 17 week CSVs."""

from __future__ import annotations

import numpy as np

from gridiron.taxonomy import COVERAGES
from gridiron.tracking.bdb import (
    league_coverage_rates,
    load_league_rates,
    normalize_coverage_label,
    save_league_rates,
)
from gridiron.tracking.schema import N_FRAMES, PlayerTrack, PlayTracks


def _labeled(coverage: str, n: int = 1) -> list[PlayTracks]:
    plays = []
    for i in range(n):
        plays.append(
            PlayTracks(
                play_id=f"bdb-2018090600-{i}-{coverage}",
                source="bdb",
                players=[
                    PlayerTrack(
                        track_id="D_1",
                        side="defense",
                        x=np.zeros(N_FRAMES, dtype=np.float32),
                        y=np.zeros(N_FRAMES, dtype=np.float32),
                    )
                ],
                coverage=coverage,
                week=1,
            )
        )
    return plays


def test_coverage_aliases_land_in_the_shared_vocabulary():
    assert normalize_coverage_label("Cover 3 Sky") == "Cover 3 Zone"
    assert normalize_coverage_label("Quarters") == "Cover 4 Zone"
    assert normalize_coverage_label("not a coverage") is None


def test_league_rates_are_the_observed_mix(tmp_path):
    plays = _labeled("Cover 3 Zone", 7) + _labeled("Cover 1 Man", 3)
    rates = league_coverage_rates(plays)
    assert rates["Cover 3 Zone"] == 0.7
    assert rates["Cover 1 Man"] == 0.3
    assert sum(rates.values()) == 1.0
    assert set(rates) == set(COVERAGES)

    path = save_league_rates(plays, tmp_path / "coverage_rates.json")
    loaded = load_league_rates(path)
    assert loaded is not None
    assert loaded["Cover 3 Zone"] == 0.7
    assert loaded["Prevent"] == 0.0


def test_copy_csvs_flattens_nested_kagglehub_layout(tmp_path):
    from gridiron.tracking.bdb import _copy_csvs

    nested = tmp_path / "cache" / "nfl-big-data-bowl-2021"
    nested.mkdir(parents=True)
    (nested / "games.csv").write_text("gameId\n1\n", encoding="utf-8")
    (nested / "inner").mkdir()
    (nested / "inner" / "week1.csv").write_text("x\n0\n", encoding="utf-8")
    dest = tmp_path / "raw" / "bdb"
    assert _copy_csvs(nested, dest) == 2
    assert (dest / "games.csv").exists()
    assert (dest / "week1.csv").exists()
