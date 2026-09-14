"""Snap detection and team assignment, tested against cases with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from gridiron.vision.snap import find_snap, motion_energy, refine_with_ball, segment_plays
from gridiron.vision.teams import TeamAssignment, _find_offense, estimate_line_of_scrimmage


def test_motion_energy_ignores_a_camera_pan():
    """A pan moves everyone the same way; that must not look like a snap."""
    frames = []
    for i in range(10):
        frames.append({tid: (float(i), float(tid)) for tid in range(11)})
    energy = motion_energy(frames)
    assert energy.max() < 0.01


def test_find_snap_on_a_stillness_then_burst():
    energy = np.concatenate(
        [np.full(30, 0.2), np.full(40, 4.0), np.full(20, 1.0)]
    )
    snap = find_snap(energy, fps=30.0)
    assert snap is not None
    # The burst starts at frame 30; smoothing may shift it by a few frames.
    assert 25 <= snap.frame_index <= 35


def test_segment_plays_respects_minimum_gap():
    energy = np.zeros(300)
    energy[40] = 10.0
    energy[50] = 10.0  # same snap, too close
    energy[200] = 10.0
    segments = segment_plays(energy, fps=30.0, min_gap_seconds=3.0)
    assert len(segments) == 2
    assert segments[1].snap_frame > segments[0].snap_frame + 80


def test_ball_refines_but_does_not_invent_a_snap():
    ball = {i: (float(i), 0.0) for i in range(10, 40)}
    # Biggest jump is between 24 and 25.
    ball[25] = (40.0, 0.0)
    assert refine_with_ball(20, ball, fps=30.0) == 24
    assert refine_with_ball(20, {}, fps=30.0) == 20


def test_offense_is_the_side_with_five_packed_linemen():
    # Offense: five linemen on x=0, plus skill players. Defense: a 4-3 look.
    positions = {}
    labels = {}
    for i, y in enumerate([-6, -3, 0, 3, 6]):
        positions[i] = (0.0, float(y))
        labels[i] = 0
    for i, y in enumerate([-20, -12, 12, 20]):
        positions[10 + i] = (-1.0, float(y))
        labels[10 + i] = 0
    for i, (x, y) in enumerate(
        [(1.5, -8), (1.5, -3), (1.5, 3), (1.5, 8), (12.0, 0), (14.0, -10), (14.0, 10)]
    ):
        positions[20 + i] = (x, y)
        labels[20 + i] = 1

    assert _find_offense(labels, positions) == 0
    los = estimate_line_of_scrimmage(positions, labels, 0)
    assert los == pytest.approx(0.0, abs=0.1)


def test_backs_in_the_line_gaps_do_not_hide_the_offense():
    """21 personnel puts the quarterback and a back in the y-gaps of the line.

    A search along the sideline then scores the 'line' as six yards deep and
    rejects it. The play still has an obvious offensive line; it just is not
    five consecutive people in y.
    """
    positions = {}
    labels = {}
    # OL at x=0, with the QB and FB sitting in the y-gaps.
    for i, (x, y) in enumerate(
        [(0.0, -6), (0.0, -3), (-2.5, -1.5), (0.0, 0), (-1.5, 1.5), (0.0, 3), (0.0, 6)]
    ):
        positions[i] = (x, y)
        labels[i] = 0
    for i, y in enumerate([-18, -12, 12, 18]):
        positions[10 + i] = (-1.0, float(y))
        labels[10 + i] = 0
    # A compact Cover-0 front, the shape that used to beat the old cutoff.
    for i, (x, y) in enumerate(
        [(1.2, -8), (1.2, -3), (1.3, 3), (1.2, 8), (3.0, 0), (12.0, -8), (12.0, 8)]
    ):
        positions[20 + i] = (x, y)
        labels[20 + i] = 1

    assert _find_offense(labels, positions) == 0
    assert estimate_line_of_scrimmage(positions, labels, 0) == pytest.approx(0.0, abs=0.1)


def test_side_of_follows_the_offense_cluster():
    assignment = TeamAssignment(labels={1: 0, 2: 1}, offense_cluster=0)
    assert assignment.side_of(1) == "offense"
    assert assignment.side_of(2) == "defense"
    assert assignment.side_of(99) is None
