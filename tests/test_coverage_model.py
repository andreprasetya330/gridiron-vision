"""Feature engineering and the gradient boosting baseline."""

import numpy as np
import pytest

from gridiron.coverage.baseline import CoverageBaseline, evaluate_predictions, split_by_week
from gridiron.coverage.features import build_feature_frame, postsnap_features
from gridiron.taxonomy import COVERAGES, MAN_COVERAGES, is_man, shell_of
from gridiron.tracking.schema import N_FRAMES, PlayerTrack, PlayTracks


def test_shell_taxonomy_is_coherent():
    assert shell_of("Cover 0 Man") == "0-high"
    assert shell_of("Cover 1 Man") == "1-high"
    assert shell_of("Cover 2 Zone") == "2-high"
    assert shell_of("Cover 4 Zone") == "3+-high"
    assert is_man("Cover 2 Man") and not is_man("Cover 2 Zone")


def test_mirror_score_separates_man_from_zone(synthetic_plays):
    """The core post-snap discriminator has to actually discriminate."""
    man_scores, zone_scores = [], []
    for play in synthetic_plays[:400]:
        features = postsnap_features(play)
        score = features["mirror_score_mean"]
        if not np.isfinite(score):
            continue
        (man_scores if play.coverage in MAN_COVERAGES else zone_scores).append(score)

    assert man_scores and zone_scores
    assert np.mean(man_scores) > np.mean(zone_scores), (
        "man defenders should track receivers more closely than zone defenders"
    )


def test_deep_defender_counts_track_the_shell(synthetic_plays):
    by_coverage: dict[str, list[float]] = {}
    for play in synthetic_plays[:600]:
        value = postsnap_features(play)["deep_count_25s"]
        if np.isfinite(value):
            by_coverage.setdefault(play.coverage, []).append(value)

    cover3 = np.mean(by_coverage["Cover 3 Zone"])
    cover0 = np.mean(by_coverage["Cover 0 Man"])
    assert cover3 > cover0 + 1.0


def test_feature_frame_has_no_infinities(synthetic_plays):
    frame = build_feature_frame(synthetic_plays[:200], mode="postsnap")
    numeric = frame.select_dtypes("number")
    assert not np.isinf(numeric.to_numpy(dtype=float, na_value=0.0)).any()


def test_split_by_week_does_not_leak(synthetic_plays):
    frame = build_feature_frame(synthetic_plays[:800], mode="presnap")
    train, test = split_by_week(frame, holdout_weeks=2)
    assert not set(train["week"]) & set(test["week"])
    assert len(train) > 0 and len(test) > 0


def test_single_week_falls_back_to_holding_out_games():
    """The public BDB coverage labels are week 1. Do not 80/20 the play list."""
    import pandas as pd

    from gridiron.coverage.baseline import describe_split, split_plays

    rows = []
    for game in range(10):
        for play in range(8):
            rows.append(
                {
                    "play_id": f"bdb-20180906{game:02d}-{play}",
                    "week": 1,
                    "game_id": f"20180906{game:02d}",
                    "coverage": "Cover 3 Zone",
                }
            )
    frame = pd.DataFrame(rows)
    train, test = split_by_week(frame, holdout_weeks=3)
    assert not set(train["game_id"]) & set(test["game_id"])
    assert len(train) > 0 and len(test) > 0
    assert set(train["week"]) == {1}
    assert set(test["week"]) == {1}
    assert "game split" in describe_split(train, test)

    plays = [
        PlayTracks(
            play_id=row["play_id"],
            source="bdb",
            players=[
                PlayerTrack(
                    track_id="D_1",
                    side="defense",
                    x=np.zeros(N_FRAMES, dtype=np.float32),
                    y=np.zeros(N_FRAMES, dtype=np.float32),
                )
            ],
            week=1,
            game_id=row["game_id"],
        )
        for row in rows
    ]
    train_plays, val_plays = split_plays(plays, holdout_weeks=3)
    train_games = {p.game_id for p in train_plays}
    val_games = {p.game_id for p in val_plays}
    assert train_games and val_games
    assert not train_games & val_games


@pytest.mark.slow
def test_postsnap_baseline_beats_the_majority_class_decisively(synthetic_plays):
    frame = build_feature_frame(synthetic_plays, mode="postsnap")
    train, test = split_by_week(frame, holdout_weeks=3)
    metrics = CoverageBaseline(mode="postsnap").fit(train).evaluate(test)

    assert metrics.accuracy > metrics.majority_accuracy + 0.25
    assert metrics.man_zone_accuracy > 0.80
    assert metrics.shell_accuracy > 0.70


@pytest.mark.slow
def test_presnap_model_is_predictive_but_weaker_than_postsnap(synthetic_plays):
    """Pre-snap should beat chance and still lose to post-snap. Both matter."""
    pre = build_feature_frame(synthetic_plays, mode="presnap")
    post = build_feature_frame(synthetic_plays, mode="postsnap")

    pre_train, pre_test = split_by_week(pre, holdout_weeks=3)
    post_train, post_test = split_by_week(post, holdout_weeks=3)

    pre_metrics = CoverageBaseline(mode="presnap").fit(pre_train).evaluate(pre_test)
    post_metrics = CoverageBaseline(mode="postsnap").fit(post_train).evaluate(post_test)

    assert pre_metrics.accuracy > pre_metrics.majority_accuracy + 0.05
    assert post_metrics.accuracy > pre_metrics.accuracy


def test_evaluate_reports_a_perfect_model_as_perfect():
    classes = list(COVERAGES)
    y_true = np.array([classes[i % len(classes)] for i in range(40)])
    proba = np.zeros((40, len(classes)))
    for i, label in enumerate(y_true):
        proba[i, classes.index(label)] = 1.0

    metrics = evaluate_predictions(y_true, proba, classes)
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.man_zone_accuracy == pytest.approx(1.0)
    assert metrics.shell_accuracy == pytest.approx(1.0)
