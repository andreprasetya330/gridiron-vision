"""The play contract is what every other stage depends on."""

import json

import numpy as np
import pytest

from gridiron.tracking.schema import (
    N_FRAMES,
    SNAP_INDEX,
    TIME_GRID,
    PlayerTrack,
    PlayTracks,
    Situation,
    TrackQuality,
    clear_plays,
    frame_at,
    infer_game_id,
    load_plays,
    make_time_grid,
    resample_to_grid,
    save_play,
)


def _track(track_id="D_1", side="defense", depth=12.0):
    return PlayerTrack(
        track_id=track_id,
        side=side,
        x=np.full(N_FRAMES, depth, dtype=np.float32),
        y=np.zeros(N_FRAMES, dtype=np.float32),
    )


def test_time_grid_has_snap_at_zero():
    assert TIME_GRID[SNAP_INDEX] == pytest.approx(0.0)
    assert TIME_GRID[0] == pytest.approx(-2.0)
    assert TIME_GRID[-1] == pytest.approx(3.0)


def test_frame_at_rounds_to_nearest():
    assert frame_at(0.0) == SNAP_INDEX
    assert TIME_GRID[frame_at(2.5)] == pytest.approx(2.5)


def test_track_rejects_wrong_length():
    with pytest.raises(ValueError):
        PlayerTrack(track_id="x", side="defense", x=np.zeros(5), y=np.zeros(5))


def test_film_grid_roundtrip_is_longer_than_coverage_window():
    grid = make_time_grid(2.0, 6.0)
    track = PlayerTrack(
        track_id="D_1",
        side="defense",
        x=np.zeros(len(grid), dtype=np.float32),
        y=np.zeros(len(grid), dtype=np.float32),
    )
    play = PlayTracks(play_id="film-1", source="film", players=[track], time_grid=grid)
    restored = PlayTracks.from_dict(play.to_dict())
    assert len(restored.time_grid) == len(grid)
    assert restored.time_grid[-1] == pytest.approx(6.0)
    assert restored.array("defense").shape == (1, N_FRAMES, 2)


def test_json_roundtrip_preserves_gaps_as_nan():
    """A missing defender must survive serialization as missing, not as zero."""
    track = _track()
    track.x[10:15] = np.nan
    play = PlayTracks(
        play_id="p1",
        source="synthetic",
        players=[track],
        coverage="Cover 3 Zone",
        season=2025,
        week=1,
        defense_team="Washington",
        offense_team="Oregon",
    )

    payload = json.loads(json.dumps(play.to_dict()))
    restored = PlayTracks.from_dict(payload)

    assert np.isnan(restored.players[0].x[10:15]).all()
    assert restored.players[0].x[0] == pytest.approx(12.0)
    assert restored.coverage == "Cover 3 Zone"
    assert restored.game_id == play.game_id


def test_velocity_is_zero_for_a_stationary_player():
    vx, vy = _track().velocity(0.0)
    assert vx == pytest.approx(0.0)
    assert vy == pytest.approx(0.0)


def test_velocity_measures_downfield_movement():
    track = _track()
    track.x = TIME_GRID.astype(np.float32) * 5.0  # 5 yards per second
    vx, _ = track.velocity(1.0)
    assert vx == pytest.approx(5.0, abs=0.2)


def test_quality_flags_missing_defenders_as_unusable():
    assert TrackQuality(defenders_detected=11).usable
    assert not TrackQuality(defenders_detected=9).usable
    assert not TrackQuality(defenders_detected=11, registration_error_yd=4.0).usable


def test_quality_score_falls_when_tracking_degrades():
    clean = TrackQuality().score
    degraded = TrackQuality(
        defenders_detected=9, registration_error_yd=2.0, frames_with_full_defense=0.5
    ).score
    assert degraded < clean


def test_resample_refuses_to_bridge_long_gaps():
    """A defender missing for a second should stay missing, not be invented."""
    times = np.array([-2.0, -1.9, 1.5, 2.0])
    values = np.array([0.0, 1.0, 20.0, 21.0])
    out = resample_to_grid(times, values, max_gap=0.4)

    assert np.isfinite(out[frame_at(-2.0)])
    assert np.isnan(out[frame_at(0.0)]), "interpolated straight through a 3.4s gap"


def test_resample_bridges_short_gaps():
    times = np.arange(-2.0, 3.01, 0.2)
    values = times * 2.0
    out = resample_to_grid(times, values, max_gap=0.4)
    assert np.isfinite(out).all()
    assert out[frame_at(1.0)] == pytest.approx(2.0, abs=0.1)


def test_situation_field_side_follows_ball_placement():
    assert Situation(ball_y_from_center=6.0).field_side == "left"
    assert Situation(ball_y_from_center=-6.0).field_side == "right"
    assert Situation(ball_y_from_center=0.2).field_side == "none"


def test_array_is_deterministic_regardless_of_player_order():
    a = _track("D_1")
    b = _track("D_2", depth=8.0)
    forward = PlayTracks(play_id="p", source="s", players=[a, b]).array("defense")
    backward = PlayTracks(play_id="p", source="s", players=[b, a]).array("defense")
    assert np.allclose(forward, backward)


def test_game_id_is_inferred_from_bdb_play_id():
    assert infer_game_id("bdb-2018090600-75") == "2018090600"
    assert infer_game_id("bdb2025-2022090800-75") == "2022090800"
    play = PlayTracks(play_id="bdb-2018090600-75", source="bdb", players=[_track()])
    assert play.game_id == "2018090600"
    assert play.game_key == "2018090600"


def test_game_id_groups_synthetic_plays_by_matchup():
    play = PlayTracks(
        play_id="2025-W03-Washington-0012",
        source="synthetic",
        players=[_track()],
        season=2025,
        week=3,
        defense_team="Washington",
        offense_team="Oregon",
    )
    assert play.game_id == "2025-W03-Washington-Oregon"


def test_load_plays_skips_predictions_json(tmp_path):
    play = PlayTracks(play_id="p1", source="synthetic", players=[_track()])
    save_play(play, tmp_path / "p1.json")
    (tmp_path / "predictions.json").write_text("[]", encoding="utf-8")
    loaded = load_plays(tmp_path)
    assert [p.play_id for p in loaded] == ["p1"]


def test_clear_plays_leaves_predictions_alone(tmp_path):
    play = PlayTracks(play_id="p1", source="synthetic", players=[_track()])
    save_play(play, tmp_path / "p1.json")
    pred = tmp_path / "predictions.json"
    pred.write_text("[]", encoding="utf-8")
    assert clear_plays(tmp_path) == 1
    assert not (tmp_path / "p1.json").exists()
    assert pred.exists()
