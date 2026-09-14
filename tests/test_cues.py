"""Cue extraction has to recover the football, not just run without error."""

import numpy as np

from gridiron.cues.vocabulary import CUE_KEYS, NUMERIC_CUE_KEYS, extract_cues


def test_every_cue_key_is_populated(synthetic_plays):
    cues = extract_cues(synthetic_plays[0])
    for key in CUE_KEYS:
        assert key in cues.categorical, f"missing categorical cue {key}"
    for key in NUMERIC_CUE_KEYS:
        assert key in cues.numeric, f"missing numeric cue {key}"


def test_shell_matches_the_coverage_family(synthetic_cues):
    """Two-high shells should show up as two-high far more often in quarters than in Cover 3."""
    quarters = synthetic_cues[synthetic_cues["coverage"] == "Cover 4 Zone"]
    cover3 = synthetic_cues[synthetic_cues["coverage"] == "Cover 3 Zone"]

    quarters_two_high = (quarters["shell"] == "2-high").mean()
    cover3_two_high = (cover3["shell"] == "2-high").mean()

    assert quarters_two_high > 0.5
    assert quarters_two_high > cover3_two_high + 0.2


def test_zero_coverage_puts_nobody_deep(synthetic_cues):
    cover0 = synthetic_cues[synthetic_cues["coverage"] == "Cover 0 Man"]
    assert (cover0["deep_defender_count"] == "0").mean() > 0.75


def test_prevent_is_the_deepest_look(synthetic_cues):
    prevent = synthetic_cues[synthetic_cues["coverage"] == "Prevent"]["safety_depth_mean_n"].mean()
    cover1 = synthetic_cues[synthetic_cues["coverage"] == "Cover 1 Man"]["safety_depth_mean_n"].mean()
    assert prevent > cover1 + 4


def test_man_coverage_presses_more_than_zone(synthetic_cues):
    man = synthetic_cues[synthetic_cues["coverage"].isin(["Cover 0 Man", "Cover 1 Man"])]
    zone = synthetic_cues[synthetic_cues["coverage"].isin(["Cover 3 Zone", "Cover 4 Zone"])]
    assert man["press_corner_count_n"].mean() > zone["press_corner_count_n"].mean() + 0.4


def test_cues_survive_missing_defenders(synthetic_config):
    """Film loses players. Cue extraction must degrade, not crash."""
    from gridiron.tracking.synthetic import SyntheticConfig, generate_season

    degraded = list(
        generate_season(
            SyntheticConfig(
                weeks=1, plays_per_game=25, seed=3, degraded_fraction=1.0, teams=synthetic_config.teams
            )
        )
    )
    for play in degraded:
        cues = extract_cues(play)
        assert cues.numeric["defenders_observed_n"] <= 11
        assert isinstance(cues.categorical["shell"], str)


def test_situation_cues_come_through(synthetic_plays):
    cues = extract_cues(synthetic_plays[0])
    assert cues.categorical["down"] in {"1", "2", "3", "4", "unknown"}
    assert cues.categorical["distance_band"] in {"short", "medium", "long", "very-long", "unknown"}
    assert cues.categorical["hash_side"] in {"left", "right", "middle", "unknown"}


def test_numeric_cues_are_finite_or_deliberately_nan(synthetic_plays):
    for play in synthetic_plays[:50]:
        for key, value in extract_cues(play).numeric.items():
            assert not np.isinf(value), f"{key} produced an infinity"
