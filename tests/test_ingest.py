"""Film ingest files a clip under a team and scores coverage from tracks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gridiron.taxonomy import family_of
from gridiron.tracking.schema import N_FRAMES, PlayerTrack, PlayTracks, load_play
from gridiron.vision.ingest import FilmIngestError, FilmIngestRequest, ingest_film


def _play(play_id: str = "clip") -> PlayTracks:
    return PlayTracks(
        play_id=play_id,
        source="film",
        players=[
            PlayerTrack(
                track_id="D_1",
                side="defense",
                x=np.zeros(N_FRAMES, dtype=np.float32),
                y=np.zeros(N_FRAMES, dtype=np.float32),
            )
        ],
        video_path=None,
    )


def test_family_of_splits_man_and_zone():
    assert family_of("Cover 2 Zone") == "Zone"
    assert family_of("Cover 1 Man") == "Man"
    assert family_of("Prevent") == "Prevent"
    assert family_of(None) is None


def test_ingest_rejects_missing_team(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIDIRON_DATA_DIR", str(tmp_path))
    from gridiron import config as cfg

    cfg.data_dir.cache_clear()
    clip = tmp_path / "play.mp4"
    clip.write_bytes(b"not-a-real-video")
    with pytest.raises(FilmIngestError, match="defense team"):
        ingest_film(FilmIngestRequest(path=clip, original_name="play.mp4", defense_team="  "))
    cfg.data_dir.cache_clear()


def test_ingest_tags_team_and_writes_play(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIDIRON_DATA_DIR", str(tmp_path))
    from gridiron import config as cfg

    cfg.data_dir.cache_clear()
    clip = tmp_path / "uga-still.jpg"
    clip.write_bytes(b"fake-image")

    def fake_process(self, video_path, play_id=None):
        play = _play(play_id or "x")
        play.video_path = str(video_path)
        return [play]

    monkeypatch.setattr("gridiron.vision.pipeline.FilmPipeline.process", fake_process)
    monkeypatch.setattr("gridiron.coverage.bridge.load_default_models", lambda: (None, None))

    result = ingest_film(
        FilmIngestRequest(
            path=clip,
            original_name="uga-still.jpg",
            defense_team="Georgia",
            offense_team="Notre Dame",
        )
    )
    assert len(result.plays) == 1
    play = result.plays[0]
    assert play.defense_team == "Georgia"
    assert play.offense_team == "Notre Dame"
    assert play.play_id.startswith("georgia-uga-still")
    saved = load_play(Path(tmp_path) / "plays" / "film" / f"{play.play_id}.json")
    assert saved.defense_team == "Georgia"
    assert Path(saved.video_path).exists()
    cfg.data_dir.cache_clear()
