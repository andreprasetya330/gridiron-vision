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
                    "presnap": {"coverage": "Cover 2 Zone", "confidence": 0.62, "probabilities": {}},
                    "disguise": {
                        "showed": "Cover 2 Zone",
                        "ran": "Cover 3 Zone",
                        "showed_shell": "2-high",
                        "ran_shell": "1-high",
                        "kind": "shell",
                        "disguised": True,
                        "family_mismatch": False,
                        "shell_mismatch": True,
                    },
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
    assert play["disguised"] is True
    assert play["coverage_presnap"] == "Cover 2 Zone"
    assert play["coverage_family"] == "Zone"
    assert play["coverage_shell"] == "3+-high"


def test_get_play_includes_tracks_and_prediction(api_env):
    r = api_env.get("/api/plays/2025-W01-Washington-0000")
    assert r.status_code == 200
    body = r.json()
    assert body["players"][0]["track_id"] == "D_1"
    assert body["prediction"]["coverage"] == "Cover 3 Zone"
    assert body["prediction"]["disguise"]["disguised"] is True
    assert body["coverage_family"] == "Zone"
    assert body["media_kind"] is None
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


def test_teams_respect_source(api_env):
    body = api_env.get("/api/teams").json()
    assert body["teams"][0]["team"] == "Washington"
    film = api_env.get("/api/teams", params={"source": "film"}).json()
    assert film["teams"] == []


def test_disguise_filter(api_env):
    assert api_env.get("/api/plays", params={"disguised_only": True}).json()["total"] == 1


def test_ingest_files_play_under_team(api_env, monkeypatch):
    from gridiron.taxonomy import COVERAGES
    from gridiron.tracking.schema import N_FRAMES, PlayerTrack, PlayTracks
    from gridiron.vision.ingest import FilmIngestResult
    import numpy as np

    def fake_ingest(request):
        from gridiron.config import plays_dir
        from gridiron.tracking.schema import save_play

        play = PlayTracks(
            play_id="georgia-clip",
            source="film",
            players=[
                PlayerTrack(
                    track_id="D_1",
                    side="defense",
                    x=np.zeros(N_FRAMES, dtype=np.float32),
                    y=np.zeros(N_FRAMES, dtype=np.float32),
                )
            ],
            defense_team=request.defense_team,
            offense_team=request.offense_team,
            video_path=str(request.path),
            vision_model="andre-4cotb/american-football-player-trackin-1-rfdetr-small-t1",
            vision_frames=[
                {
                    "boxes": [
                        {
                            "class_name": "defense_player",
                            "side": "defense",
                            "confidence": 0.9,
                            "box": [10.0, 20.0, 40.0, 80.0],
                        }
                    ],
                    "players": [
                        {
                            "track_id": "D_1",
                            "side": "defense",
                            "class_name": "defense_player",
                            "minimap_x": 560.0,
                            "minimap_y": 200.0,
                        }
                    ],
                }
            ],
            minimap_width=1200,
            minimap_height=533,
        )
        save_play(play, plays_dir("film") / f"{play.play_id}.json")
        prediction = {
            "play_id": play.play_id,
            "coverage": "Cover 2 Zone",
            "confidence": 0.84,
            "probabilities": {c: (0.84 if c == "Cover 2 Zone" else 0.02) for c in COVERAGES},
            "runner_up": "Cover 4 Zone",
            "roles": {},
            "quality_score": 0.7,
            "usable": True,
        }
        return FilmIngestResult(plays=[play], predictions=[prediction])

    monkeypatch.setattr("gridiron.api.main.ingest_film", fake_ingest)
    response = api_env.post(
        "/api/film/ingest",
        files={"file": ("uga.jpg", b"fake-bytes", "image/jpeg")},
        data={"defense_team": "Georgia", "offense_team": "Notre Dame"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    play = body["plays"][0]
    assert play["defense_team"] == "Georgia"
    assert play["coverage_family"] == "Zone"
    assert play["prediction"]["coverage"] == "Cover 2 Zone"
    assert play["media_kind"] == "image"
    assert play["vision_frames"][0]["boxes"][0]["box"] == [10.0, 20.0, 40.0, 80.0]
    assert play["vision_frames"][0]["players"][0]["minimap_x"] == 560.0
    listed = api_env.get("/api/plays", params={"source": "film", "team": "Georgia"}).json()
    assert listed["total"] == 1
