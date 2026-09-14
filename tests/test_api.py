"""The API is read-only over files on disk. These tests pin the contract the UI consumes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gridiron.taxonomy import COVERAGES
from gridiron.tracking.schema import N_FRAMES, PlayerTrack, PlayTracks, save_play
import numpy as np


@pytest.fixture
def api_env(tmp_path, monkeypatch):
    plays = tmp_path / "plays"
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    plays.mkdir()
    reports.mkdir()
    models.mkdir()

    monkeypatch.setenv("GRIDIRON_DATA_DIR", str(tmp_path))
    # config caches the data dir; drop the cache so the env var is seen.
    from gridiron import config as cfg

    cfg.data_dir.cache_clear()

    play = PlayTracks(
        play_id="2025-W01-Washington-0000",
        source="synthetic",
        players=[
            PlayerTrack(
                track_id="D_1",
                side="defense",
                x=np.zeros(N_FRAMES, dtype=np.float32),
                y=np.zeros(N_FRAMES, dtype=np.float32),
            )
        ],
        season=2025,
        week=1,
        defense_team="Washington",
        offense_team="Oregon",
        coverage="Cover 3 Zone",
        video_path=None,
    )
    save_play(play, plays / f"{play.play_id}.json")

    (plays / "predictions.json").write_text(
        json.dumps(
            [
                {
                    "play_id": play.play_id,
                    "coverage": "Cover 3 Zone",
                    "confidence": 0.81,
                    "probabilities": {c: (0.81 if c == "Cover 3 Zone" else 0.03) for c in COVERAGES},
                    "runner_up": "Cover 1 Man",
                    "roles": {},
                    "quality_score": 0.9,
                    "usable": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    (reports / "washington.md").write_text("# Washington\nThey play Cover 3.", encoding="utf-8")

    from gridiron.api.main import _clear_api_caches, create_app

    _clear_api_caches()
    from fastapi.testclient import TestClient

    yield TestClient(create_app())
    _clear_api_caches()
    cfg.data_dir.cache_clear()


def test_health_reports_counts(api_env):
    r = api_env.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["plays"] == 1
    assert body["predictions"] == 1
    assert body["coverages"] == COVERAGES


def test_list_plays_joins_predictions(api_env):
    r = api_env.get("/api/plays")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    play = body["plays"][0]
    assert play["defense_team"] == "Washington"
    assert play["coverage_predicted"] == "Cover 3 Zone"
    assert play["coverage_truth"] == "Cover 3 Zone"


def test_get_play_includes_tracks_and_prediction(api_env):
    r = api_env.get("/api/plays/2025-W01-Washington-0000")
    assert r.status_code == 200
    body = r.json()
    assert body["players"][0]["track_id"] == "D_1"
    assert body["prediction"]["coverage"] == "Cover 3 Zone"
    assert "time_grid" in body


def test_unknown_play_is_404(api_env):
    assert api_env.get("/api/plays/nope").status_code == 404


def test_reports_are_listed_and_served(api_env):
    listed = api_env.get("/api/reports").json()
    assert listed["reports"][0]["slug"] == "washington"
    body = api_env.get("/api/reports/washington").json()
    assert "Cover 3" in body["markdown"]


def test_team_filter(api_env):
    assert api_env.get("/api/plays", params={"team": "Washington"}).json()["total"] == 1
    assert api_env.get("/api/plays", params={"team": "Oregon State"}).json()["total"] == 0
