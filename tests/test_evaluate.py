"""Metric code is the thing you cannot debug by looking at its output.

If the detector scores 0.7 you have no way to tell a mediocre detector from a
broken metric, so these tests pin the metrics against cases whose answers are
known by construction.
"""

from __future__ import annotations

import pytest

from gridiron.vision.detect import Detection
from gridiron.vision.evaluate import (
    _role_group,
    average_precision,
    evaluate_detections,
    evaluate_tracking,
    greedy_match,
    iou_matrix,
)


def box(x1, y1, x2, y2):
    return (float(x1), float(y1), float(x2), float(y2))


def det(x1, y1, x2, y2, confidence=0.9):
    return Detection(x1, y1, x2, y2, confidence, 0, "player")


class _Tracked:
    def __init__(self, detection, track_id):
        self.detection = detection
        self.track_id = track_id


def test_iou_of_identical_and_disjoint_boxes():
    m = iou_matrix([box(0, 0, 10, 10)], [box(0, 0, 10, 10), box(50, 50, 60, 60)])
    assert m[0, 0] == pytest.approx(1.0)
    assert m[0, 1] == pytest.approx(0.0)


def test_iou_of_half_overlap():
    m = iou_matrix([box(0, 0, 10, 10)], [box(5, 0, 15, 10)])
    assert m[0, 0] == pytest.approx(50 / 150)


def test_greedy_match_prefers_confident_predictions():
    """Two predictions on one truth box: the confident one should claim it."""
    truth = [box(0, 0, 10, 10)]
    pred = [box(0, 0, 9, 9), box(0, 0, 10, 10)]
    matches = greedy_match(pred, truth, iou_threshold=0.5, scores=[0.1, 0.99])
    assert len(matches) == 1
    assert matches[0][0] == 1


def test_one_truth_box_cannot_be_matched_twice():
    truth = [box(0, 0, 10, 10)]
    pred = [box(0, 0, 10, 10), box(1, 1, 11, 11)]
    assert len(greedy_match(pred, truth, iou_threshold=0.5)) == 1


def test_perfect_detections_score_perfectly():
    truth = [box(0, 0, 10, 10), box(30, 30, 40, 40)]
    frames = [([det(*t) for t in truth], truth)] * 3
    metrics = evaluate_detections(frames)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.average_precision == pytest.approx(1.0)
    assert metrics.false_positives == metrics.false_negatives == 0


def test_missed_and_spurious_detections_are_counted_separately():
    truth = [box(0, 0, 10, 10), box(30, 30, 40, 40)]
    predictions = [det(0, 0, 10, 10), det(80, 80, 90, 90)]
    metrics = evaluate_detections([(predictions, truth)])
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1  # the spurious one
    assert metrics.false_negatives == 1  # the missed one
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)


def test_average_precision_rewards_ranking():
    """Same detections, different confidence order, different AP."""
    good = average_precision([(0.9, True), (0.8, True), (0.2, False)], n_truth=2)
    bad = average_precision([(0.9, False), (0.8, True), (0.2, True)], n_truth=2)
    assert good == pytest.approx(1.0)
    assert bad < good


def test_tracking_is_clean_when_identity_holds():
    truth_entries = [
        {"track_id": 1, "box": box(0, 0, 10, 10), "side": "defense", "role": "deep_zone"},
        {"track_id": 2, "box": box(50, 0, 60, 10), "side": "defense", "role": "man"},
    ]
    frames = []
    for _ in range(5):
        tracked = [
            _Tracked(det(0, 0, 10, 10), 100),
            _Tracked(det(50, 0, 60, 10), 200),
        ]
        frames.append((tracked, truth_entries))

    metrics = evaluate_tracking(frames)
    assert metrics.id_switches == 0
    assert metrics.truth_tracks == 2
    assert metrics.mostly_tracked_fraction == 1.0
    assert metrics.usable


def test_swap_within_a_role_group_is_counted_but_forgiven():
    """Two underneath defenders trading IDs changes no cue, so the play survives."""
    a = {"track_id": 1, "box": box(0, 0, 10, 10), "side": "defense", "role": "man"}
    b = {"track_id": 2, "box": box(50, 0, 60, 10), "side": "defense", "role": "man"}

    frames = [([_Tracked(det(0, 0, 10, 10), 100), _Tracked(det(50, 0, 60, 10), 200)], [a, b])]
    # The tracker keeps its ids where they were, but the players have traded places.
    frames += [
        ([_Tracked(det(0, 0, 10, 10), 200), _Tracked(det(50, 0, 60, 10), 100)], [a, b])
    ] * 4

    metrics = evaluate_tracking(frames)
    assert metrics.id_switches > 0
    assert metrics.role_crossing_switches == 0
    assert metrics.usable


def test_swap_across_a_role_group_makes_the_play_unusable():
    """A safety trading identity with a man defender breaks the depth cues."""
    deep = {"track_id": 1, "box": box(0, 0, 10, 10), "side": "defense", "role": "deep_zone"}
    shallow = {"track_id": 2, "box": box(50, 0, 60, 10), "side": "defense", "role": "man"}

    frames = [
        ([_Tracked(det(0, 0, 10, 10), 100), _Tracked(det(50, 0, 60, 10), 200)], [deep, shallow])
    ]
    frames += [
        ([_Tracked(det(0, 0, 10, 10), 200), _Tracked(det(50, 0, 60, 10), 100)], [deep, shallow])
    ] * 4

    metrics = evaluate_tracking(frames)
    assert metrics.role_crossing_switches >= 2
    assert not metrics.usable


def test_offense_defense_swap_is_flagged():
    off = {"track_id": 1, "box": box(0, 0, 10, 10), "side": "offense", "role": None}
    dfn = {"track_id": 2, "box": box(50, 0, 60, 10), "side": "defense", "role": "man"}
    frames = [
        ([_Tracked(det(0, 0, 10, 10), 100), _Tracked(det(50, 0, 60, 10), 200)], [off, dfn]),
        ([_Tracked(det(0, 0, 10, 10), 200), _Tracked(det(50, 0, 60, 10), 100)], [off, dfn]),
    ]
    assert evaluate_tracking(frames).side_crossing_switches >= 2


def test_dropped_track_lowers_mostly_tracked():
    entries = [
        {"track_id": 1, "box": box(0, 0, 10, 10), "side": "defense", "role": "man"},
        {"track_id": 2, "box": box(50, 0, 60, 10), "side": "defense", "role": "man"},
    ]
    # The second player is detected in only one frame out of five.
    frames = [([_Tracked(det(0, 0, 10, 10), 100)], entries)] * 4
    frames += [
        ([_Tracked(det(0, 0, 10, 10), 100), _Tracked(det(50, 0, 60, 10), 200)], entries)
    ]

    metrics = evaluate_tracking(frames)
    assert metrics.truth_tracks == 2
    assert metrics.mostly_tracked == 1
    assert not metrics.usable


def test_role_groups_collapse_to_depth_responsibility():
    assert _role_group("deep_zone") == _role_group("S") == "deep"
    assert _role_group("man") == _role_group("underneath_zone") == "shallow"
    assert _role_group("blitz") == _role_group("DL") == "rush"
    assert _role_group(None) == "unknown"
