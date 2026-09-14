"""Ingestion: the boring joins that break everything when they are wrong."""

import pandas as pd
import pytest

from gridiron.ingest.pff import PffDropFolderSource, PffSessionSource, map_coverage, sniff
from gridiron.ingest.schema import merge_context, normalize_team, success_from_play, to_play_context


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Washington", "Washington"),
        ("washington huskies", "Washington"),
        ("UW", "Washington"),
        ("  WASH  ", "Washington"),
        ("Oregon St", "Oregon State"),
        ("USC", "USC"),
        (None, None),
    ],
)
def test_team_names_canonicalize(raw, expected):
    assert normalize_team(raw) == expected


def test_success_uses_the_standard_thresholds():
    assert success_from_play(1, 10, 5) is True
    assert success_from_play(1, 10, 4) is False
    assert success_from_play(2, 10, 7) is True
    assert success_from_play(3, 5, 4) is False
    assert success_from_play(3, 5, 5) is True
    assert success_from_play(None, 10, 5) is None


def test_play_context_fills_the_full_schema():
    frame = to_play_context([{"play_id": "x", "defense_team": "uw", "down": 3}], "test")
    assert frame.loc[0, "defense_team"] == "Washington"
    assert frame.loc[0, "source"] == "test"
    assert "coverage" in frame.columns


def test_merge_prefers_rows_that_carry_coverage():
    a = to_play_context([{"play_id": "p1", "defense_team": "UW"}], "cfbd")
    b = to_play_context([{"play_id": "p1", "defense_team": "UW", "coverage": "Cover 3 Zone"}], "pff")
    merged = merge_context(a, b)
    assert len(merged) == 1
    assert merged.loc[0, "coverage"] == "Cover 3 Zone"


def test_cfbd_maps_nested_clock_and_personnel_into_play_context():
    from gridiron.ingest.cfbd import defensive_plays_to_context

    plays = pd.DataFrame(
        [
            {
                "id": 99,
                "defense": "Washington",
                "offense": "Oregon",
                "down": 3,
                "distance": 8,
                "yardsToGoal": 42,
                "period": 2,
                "clock.minutes": 1,
                "clock.seconds": 15,
                "offenseScore": 14,
                "defenseScore": 10,
                "playType": "Pass Incompletion",
                "yardsGained": 0,
                "ppa": -0.4,
                "offensePersonnel": "1 RB, 1 TE, 3 WR",
            }
        ]
    )
    frame = defensive_plays_to_context(plays, season=2025, week=3)
    row = frame.iloc[0]
    assert row["play_id"] == "cfbd-99"
    assert row["defense_team"] == "Washington"
    assert row["clock_seconds"] == 75
    assert row["score_margin"] == -4
    assert row["offense_personnel"] == "11"
    assert row["success"] is False
    assert row["source"] == "cfbd"


def test_cfbd_rejects_a_missing_key(monkeypatch):
    from gridiron.ingest.cfbd import CFBDClient, CFBDConfig, CFBDError

    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    with pytest.raises(CFBDError, match="No CollegeFootballData API key"):
        CFBDClient(CFBDConfig(api_key=None)).config.resolve_key()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Cover 3", "Cover 3 Zone"),
        ("cover-3 seam", "Cover 3 Zone"),
        ("Quarters", "Cover 4 Zone"),
        ("2 Man", "Cover 2 Man"),
        ("C0", "Cover 0 Man"),
        ("Cover 3 Zone", "Cover 3 Zone"),
        ("nonsense", None),
    ],
)
def test_pff_coverage_vocabulary_maps_onto_ours(raw, expected):
    assert map_coverage(raw) == expected


def test_sniffs_a_per_snap_export(tmp_path):
    path = tmp_path / "pff_snaps.csv"
    pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "play_id": [10, 11, 12],
            "team": ["WASH", "WASH", "WASH"],
            "coverage_scheme": ["Cover 3", "Cover 1", "Quarters"],
            "week": [1, 1, 1],
        }
    ).to_csv(path, index=False)

    result = sniff(path)
    assert result.kind == "per_snap"
    assert result.has_play_level_coverage
    assert result.mapping["coverage"] == "coverage_scheme"


def test_sniffs_a_player_grade_export(tmp_path):
    path = tmp_path / "grades.csv"
    pd.DataFrame(
        {"player": ["A", "B"], "position": ["CB", "S"], "grades_coverage_defense": [78.1, 66.4]}
    ).to_csv(path, index=False)
    assert sniff(path).kind == "player_grades"


def test_unknown_export_is_flagged_not_guessed(tmp_path):
    path = tmp_path / "mystery.csv"
    pd.DataFrame({"alpha": [1], "beta": [2]}).to_csv(path, index=False)
    result = sniff(path)
    assert result.kind == "unknown"
    assert any("COLUMN_ALIASES" in note for note in result.notes)


def test_drop_folder_maps_per_snap_into_play_context(tmp_path):
    (tmp_path / "snaps.csv").write_text(
        "game_id,play_id,team,opponent,week,season,coverage\n"
        "1,10,WASH,Oregon,3,2025,Cover 3\n"
        "1,11,WASH,Oregon,3,2025,Cover 1\n",
        encoding="utf-8",
    )
    context = PffDropFolderSource(tmp_path).play_context()
    assert len(context) == 2
    assert context["defense_team"].tolist() == ["Washington", "Washington"]
    assert context["coverage"].tolist() == ["Cover 3 Zone", "Cover 1 Man"]
    assert set(context["coverage_source"]) == {"pff"}


def test_session_source_stays_unimplemented_on_purpose():
    source = PffSessionSource()
    assert source.available() is False
    with pytest.raises(NotImplementedError) as excinfo:
        source.load()
    assert "terms of service" in str(excinfo.value)
