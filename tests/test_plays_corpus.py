"""Synthetic, BDB, and film plays must not share a directory."""

from __future__ import annotations

import numpy as np

from gridiron.tracking.schema import N_FRAMES, PlayerTrack, PlayTracks, load_corpus, save_play


def _play(play_id: str, source: str) -> PlayTracks:
    return PlayTracks(
        play_id=play_id,
        source=source,
        players=[
            PlayerTrack(
                track_id="D_1",
                side="defense",
                x=np.zeros(N_FRAMES, dtype=np.float32),
                y=np.zeros(N_FRAMES, dtype=np.float32),
            )
        ],
        week=1,
        defense_team="Washington",
        offense_team="Oregon",
    )


def test_auto_prefers_bdb_over_a_leftover_synthetic_season(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIDIRON_DATA_DIR", str(tmp_path))
    from gridiron import config as cfg

    cfg.data_dir.cache_clear()
    save_play(_play("synth-1", "synthetic"), cfg.plays_dir("synthetic") / "synth-1.json")
    save_play(_play("bdb-2018090600-1", "bdb"), cfg.plays_dir("bdb") / "bdb-2018090600-1.json")

    directory, recursive, resolved = cfg.resolve_play_directory("auto")
    assert resolved == "bdb"
    assert directory == cfg.plays_dir("bdb")
    assert recursive is False

    loaded = load_corpus("auto")
    assert [p.play_id for p in loaded] == ["bdb-2018090600-1"]

    mixed = load_corpus("all")
    assert {p.source for p in mixed} == {"bdb", "synthetic"}
    cfg.data_dir.cache_clear()


def test_auto_falls_through_to_synthetic_when_that_is_all_there_is(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIDIRON_DATA_DIR", str(tmp_path))
    from gridiron import config as cfg

    cfg.data_dir.cache_clear()
    save_play(_play("synth-1", "synthetic"), cfg.plays_dir("synthetic") / "synth-1.json")
    _directory, _rec, resolved = cfg.resolve_play_directory("auto")
    assert resolved == "synthetic"
    cfg.data_dir.cache_clear()
