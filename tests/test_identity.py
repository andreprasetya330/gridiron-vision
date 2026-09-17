"""Referee rejection and rusher vs coverage roles."""

from __future__ import annotations

import numpy as np
import pytest

from gridiron.tracking.schema import N_FRAMES, SNAP_INDEX, PlayerTrack, make_time_grid
from gridiron.vision.identity import (
    TrackProfile,
    assign_defensive_roles,
    classify_tracks,
    finalize_film_players,
)


def _defender(track_id: str, x: float, y: float, n: int = N_FRAMES) -> PlayerTrack:
    return PlayerTrack(
        track_id=track_id,
        side="defense",
        x=np.full(n, x, dtype=np.float32),
        y=np.full(n, y, dtype=np.float32),
    )


def test_line_player_is_a_rusher_not_coverage():
    rusher = _defender("D_1", 1.5, 2.0)
    safety = _defender("D_2", 13.0, 8.0)
    corner = _defender("D_3", 6.0, 14.0)
    assign_defensive_roles([rusher, safety, corner], snap_index=SNAP_INDEX)
    assert rusher.role == "blitz"
    assert safety.role == "deep_zone"
    assert corner.role == "man"


def test_linebacker_who_crashes_the_pocket_is_a_rusher():
    lb = _defender("D_4", 5.0, 0.0)
    lb.x[SNAP_INDEX + 15 :] = 1.2
    assign_defensive_roles([lb], snap_index=SNAP_INDEX)
    assert lb.role == "blitz"


def test_make_time_grid_keeps_the_coverage_window():
    grid = make_time_grid(2.0, 8.0)
    assert grid[0] == pytest.approx(-2.0)
    assert grid[SNAP_INDEX] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(8.0)
    assert len(grid) > N_FRAMES


def _profile(track_id: int, luma: float, a: float = 0.0, b: float = 0.0) -> TrackProfile:
    return TrackProfile(
        track_id=track_id,
        samples=40,
        lab=np.array([luma, 128.0 + a, 128.0 + b], dtype=np.float32),
        height_px=70.0,
    )


def _measured_roster() -> dict[int, TrackProfile]:
    """Track luminances measured off the Sugar Bowl clip: 2 officials, 2 teams."""
    profiles = {0: _profile(0, 25.0, -5, 5), 1: _profile(1, 36.0, -9, 9)}
    baylor = [56.0, 66.0, 71.0, 75.0, 78.0, 84.0, 90.0, 94.0, 102.0, 113.0, 116.0]
    for i, luma in enumerate(baylor, start=2):
        profiles[i] = _profile(i, luma, -17, 24)
    georgia = [125.0, 134.0, 143.0, 152.0, 166.0, 171.0, 175.0, 187.0, 190.0, 245.0]
    for i, luma in enumerate(georgia, start=2 + len(baylor)):
        profiles[i] = _profile(i, luma, -3, 8)
    return profiles


def test_officials_are_found_by_the_luminance_gap_below_the_players():
    roster = classify_tracks(_measured_roster())
    assert roster.officials == {0, 1}
    assert 0 not in roster.team_of and 1 not in roster.team_of
    assert len(set(roster.team_of.values())) == 2


def test_a_team_in_dark_jerseys_is_not_deleted_as_officials():
    """Eleven dark tracks are a team wearing black, not an officiating crew."""
    profiles = {i: _profile(i, 28.0 + i, -4, 6) for i in range(11)}
    profiles.update({i: _profile(i, 190.0 + i, -3, 8) for i in range(11, 22)})
    roster = classify_tracks(profiles)
    assert roster.officials == set()
    assert len(roster.team_of) == 22


def test_teams_survive_a_stadium_shadow_across_the_field():
    """Half the field in shade must not be read as the two teams.

    Same two uniforms, but luminance is scrambled across both rosters. Only hue
    is a real team signal, so the split has to follow hue.
    """
    profiles = {}
    for i in range(11):  # one uniform, lit from bright sun to deep shade
        profiles[i] = _profile(i, 70.0 + 10.0 * i, -17, 24)
    for i in range(11, 22):  # the other uniform, interleaved on luminance
        profiles[i] = _profile(i, 75.0 + 10.0 * (i - 11), -3, 8)
    roster = classify_tracks(profiles)
    assert roster.officials == set()
    first = {roster.team_of[i] for i in range(11)}
    second = {roster.team_of[i] for i in range(11, 22)}
    assert len(first) == 1 and len(second) == 1 and first != second


def _embedding(seed: int, centre: np.ndarray, jitter: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = centre + rng.normal(0.0, jitter, size=centre.shape)
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def test_embeddings_outrank_colour_when_both_are_present():
    """Colour says one thing, appearance says another; appearance wins."""
    dim = 32
    home = np.zeros(dim, dtype=np.float32)
    home[0] = 1.0
    away = np.zeros(dim, dtype=np.float32)
    away[1] = 1.0

    profiles, embeddings = {}, {}
    for i in range(11):  # colour is deliberately uninformative here
        profiles[i] = _profile(i, 100.0, -4, 6)
        embeddings[i] = _embedding(i, home)
    for i in range(11, 22):
        profiles[i] = _profile(i, 100.0, -4, 6)
        embeddings[i] = _embedding(i, away)

    roster = classify_tracks(profiles, embeddings)
    first = {roster.team_of[i] for i in range(11)}
    second = {roster.team_of[i] for i in range(11, 22)}
    assert len(first) == 1 and len(second) == 1 and first != second


def _squads(dim: int = 32, crew: int = 3, gap: float = 1.0):
    """Two teams that look somewhat alike, plus a crew that looks like neither.

    The teams share a component because two sets of padded players on grass are
    genuinely more similar to each other than either is to an official - that
    resemblance is what the outsider test measures against.
    """
    shared = np.zeros(dim, dtype=np.float32)
    shared[0] = 1.0
    home, away = shared.copy(), shared.copy()
    home[1], away[2] = 0.35, 0.35
    officials = np.zeros(dim, dtype=np.float32)
    officials[3] = gap

    profiles, embeddings = {}, {}
    for i in range(11):
        profiles[i], embeddings[i] = _profile(i, 100.0), _embedding(i, home)
    for i in range(11, 22):
        profiles[i], embeddings[i] = _profile(i, 100.0), _embedding(i, away)
    for i in range(22, 22 + crew):
        profiles[i], embeddings[i] = _profile(i, 30.0), _embedding(i, officials)
    return profiles, embeddings


def test_a_group_that_looks_like_neither_team_is_dropped():
    profiles, embeddings = _squads(crew=3)
    roster = classify_tracks(profiles, embeddings)

    assert roster.officials == {22, 23, 24}
    assert set(roster.team_of) == set(range(22))
    first = {roster.team_of[i] for i in range(11)}
    second = {roster.team_of[i] for i in range(11, 22)}
    assert len(first) == 1 and len(second) == 1 and first != second


def test_two_clean_teams_lose_nobody():
    """No crew on screen means no track should be discarded."""
    profiles, embeddings = _squads(crew=0)
    roster = classify_tracks(profiles, embeddings)

    assert roster.officials == set()
    assert len(roster.team_of) == 22
    first = {roster.team_of[i] for i in range(11)}
    second = {roster.team_of[i] for i in range(11, 22)}
    assert len(first) == 1 and len(second) == 1 and first != second


def test_a_crew_too_large_to_be_a_crew_is_not_discarded():
    """Rather than throw away a quarter of the film, defer to colour."""
    profiles, embeddings = _squads(crew=11)
    roster = classify_tracks(profiles, embeddings)

    assert roster.officials == set()


def test_no_gap_means_no_official_is_invented():
    """A clip with no crew on screen must not donate a dark player to the gap."""
    profiles = {i: _profile(i, 70.0 + 4.0 * i, -14, 20) for i in range(11)}
    profiles.update({i: _profile(i, 150.0 + 4.0 * i, -3, 8) for i in range(11, 22)})
    roster = classify_tracks(profiles)
    assert roster.officials == set()


def test_too_few_tracks_refuses_to_guess():
    roster = classify_tracks({0: _profile(0, 30.0), 1: _profile(1, 180.0)})
    assert roster.officials == set()
    assert roster.team_of == {}
    assert roster.notes


def test_geometry_moves_a_downfield_offense_player_to_defense():
    safety = PlayerTrack(
        track_id="O_9",
        side="offense",
        x=np.full(N_FRAMES, 12.0, dtype=np.float32),
        y=np.zeros(N_FRAMES, dtype=np.float32),
    )
    qb = PlayerTrack(
        track_id="O_1",
        side="offense",
        x=np.full(N_FRAMES, -5.0, dtype=np.float32),
        y=np.zeros(N_FRAMES, dtype=np.float32),
    )
    out = finalize_film_players([safety, qb], snap_index=SNAP_INDEX)
    sides = {p.track_id: p.side for p in out}
    assert sides["D_9"] == "defense"
    assert sides["O_1"] == "offense"
    assert next(p.role for p in out if p.track_id == "D_9") == "deep_zone"
