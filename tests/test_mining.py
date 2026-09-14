"""Does the miner find real tells, and does it stay quiet when there are none?

This is the test that matters most in the whole suite. On real film you can never
check the answer - you only ever see the tells you found, never the ones you
invented. The synthetic season plants tells of a known size in known teams and
leaves one team deliberately clean, so both failure modes are observable.
"""

import pytest

from gridiron.tells.mining import MiningConfig, TellMiner
from gridiron.tells.validation import forward_validate
from gridiron.tracking.synthetic import planted_tell_summary

# Each planted trait shows up through several cues, since the cue vocabulary is
# geometric and a trait like "safeties bail early" moves depth, shell, and split
# together.
TRAIT_CUES = {
    "late_rotation": {"late_rotation", "rotation_direction"},
    "press_corners": {"press_corners", "corner_depth_band", "corner_leverage"},
    "deep_safeties": {"safety_depth_band", "shell", "deepest_depth_band", "deep_defender_count"},
    "nickel_creep": {"slot_defender_depth_band"},
    "heavy_box": {"box_count_band", "linebacker_depth_band", "los_defenders_band"},
}


@pytest.fixture(scope="module")
def mined(synthetic_cues):
    config = MiningConfig(min_support=10, min_lift=0.10, fdr_alpha=0.10)
    return TellMiner(synthetic_cues, config).mine_all(current_week=10)


def test_recovers_the_strongest_planted_tells(mined, synthetic_config):
    """Every planted tell with a large effect should be found."""
    planted = planted_tell_summary(synthetic_config)
    strong = [p for p in planted if abs(p["effect_size"]) >= 0.35]
    assert strong, "the fixture should plant at least one large tell"

    missed = []
    for row in strong:
        result = mined.get(row["team"])
        expected_cues = TRAIT_CUES[row["trait"]]
        found = any(
            tell.cue_key in expected_cues and tell.coverage == row["coverage"]
            for tell in (result.tells if result else [])
        )
        if not found:
            missed.append(f"{row['team']} / {row['trait']} -> {row['coverage']}")

    assert not missed, f"failed to recover planted tells: {missed}"


def test_control_team_produces_almost_nothing(mined):
    """California has no planted tells. Anything found there is a false positive."""
    control = mined["California"]
    assert control.n_tests > 100, "the control team should still be tested extensively"
    assert len(control.tells) <= 3, (
        f"mined {len(control.tells)} tells from a team with none planted: "
        f"{[t.description for t in control.tells]}"
    )


def test_fdr_correction_actually_bites(synthetic_cues):
    """Without correction, far more findings survive. That gap is the point."""
    lenient = TellMiner(
        synthetic_cues, MiningConfig(fdr_alpha=1.0, min_lift=0.0, max_tells=10_000)
    ).mine("California")
    strict = TellMiner(
        synthetic_cues, MiningConfig(fdr_alpha=0.05, min_lift=0.10, max_tells=10_000)
    ).mine("California")
    assert len(strict.tells) < len(lenient.tells)


def test_shrinkage_keeps_reported_rates_off_the_extremes(mined):
    for result in mined.values():
        for tell in result.tells:
            assert 0.0 < tell.shrunk_rate < 1.0
            if tell.n_cue < 20:
                assert tell.shrunk_rate < 0.97, (
                    f"{tell.description} reports near-certainty from {tell.n_cue} snaps"
                )


def test_every_tell_carries_its_evidence(mined):
    for result in mined.values():
        for tell in result.tells:
            assert tell.n_cue >= 10
            assert tell.baseline_n >= 10
            assert tell.ci_low <= tell.shrunk_rate <= tell.ci_high or tell.ci_low <= tell.raw_rate <= tell.ci_high
            assert 0.0 <= tell.q_value <= 1.0
            assert tell.description


def test_minimum_support_is_enforced(synthetic_cues):
    result = TellMiner(synthetic_cues, MiningConfig(min_support=25)).mine("Washington")
    assert all(tell.n_cue >= 25 for tell in result.tells)


def test_baseline_excludes_the_team_being_scouted(synthetic_cues):
    """A team must not contribute to the average it is measured against."""
    team = "Oregon State"
    result = TellMiner(synthetic_cues, MiningConfig()).mine(team)
    everyone = synthetic_cues[synthetic_cues["coverage"].notna()]

    for tell in result.tells[:5]:
        matched = everyone[everyone[tell.cue_key].astype(str) == tell.cue_value]
        with_team = float((matched["coverage"] == tell.coverage).mean())
        assert abs(tell.baseline_rate - with_team) > 1e-9 or len(
            matched[matched["defense_team"] == team]
        ) == 0


def test_mining_reports_nothing_when_there_is_no_league_to_compare_to(synthetic_cues):
    solo = synthetic_cues[synthetic_cues["defense_team"] == "Washington"]
    result = TellMiner(solo, MiningConfig()).mine("Washington")
    assert result.tells == []
    assert any("baseline" in note for note in result.notes)


def test_forward_validation_beats_the_baseline_on_a_team_with_real_tells(synthetic_cues):
    result = forward_validate(synthetic_cues, "Oregon State", MiningConfig(), min_train_weeks=4)
    assert result.validations, "should have mined and tested at least one tell"
    assert result.predictive, "should have mined at least one 'they will run this' tell"
    assert result.pooled_hit_rate > result.pooled_expected, (
        "tells mined on early weeks did not outperform their expected rate on later weeks"
    )


def test_forward_validation_scores_avoidance_tells_in_their_own_direction(synthetic_cues):
    """An avoidance tell succeeds by coming in low, so it must not be pooled with the rest."""
    result = forward_validate(synthetic_cues, "Oregon State", MiningConfig(), min_train_weeks=4)
    for entry in result.avoidance:
        assert entry.mined_rate < entry.expected_rate
        if entry.held_up:
            assert entry.holdout_rate < entry.expected_rate


def test_forward_validation_never_scores_on_training_film(synthetic_cues):
    """The holdout counts must be smaller than the whole season, or weeks leaked."""
    result = forward_validate(synthetic_cues, "Washington", MiningConfig(), min_train_weeks=4)
    team_plays = (synthetic_cues["defense_team"] == "Washington").sum()
    for validation in result.validations:
        assert validation.holdout_trials < team_plays


def test_injected_league_rates_replace_the_table_mix(synthetic_cues):
    from gridiron.taxonomy import COVERAGES

    skewed = {c: 0.0 for c in COVERAGES}
    skewed["Prevent"] = 1.0
    observed = TellMiner(synthetic_cues, MiningConfig()).mine("Washington")
    borrowed = TellMiner(
        synthetic_cues,
        MiningConfig(league_rates=skewed, baseline_source="nfl-big-data-bowl"),
    ).mine("Washington")
    assert borrowed.baseline_source == "nfl-big-data-bowl"
    assert borrowed.baseline_distribution["Prevent"] == pytest.approx(1.0)
    assert observed.baseline_distribution["Prevent"] != pytest.approx(1.0)


def test_film_corpus_borrows_cached_bdb_rates(tmp_path, monkeypatch):
    from gridiron.tells.mining import config_for_corpus
    from gridiron.tracking.bdb import save_league_rates
    from gridiron.tracking.schema import N_FRAMES, PlayerTrack, PlayTracks
    import numpy as np

    monkeypatch.setenv("GRIDIRON_DATA_DIR", str(tmp_path))
    from gridiron import config as cfg

    cfg.data_dir.cache_clear()
    plays = [
        PlayTracks(
            play_id="bdb-1-1",
            source="bdb",
            players=[
                PlayerTrack(
                    track_id="D_1",
                    side="defense",
                    x=np.zeros(N_FRAMES, dtype=np.float32),
                    y=np.zeros(N_FRAMES, dtype=np.float32),
                )
            ],
            coverage="Cover 2 Zone",
        )
    ]
    save_league_rates(plays)
    film = config_for_corpus("film")
    assert film.baseline_source == "nfl-big-data-bowl"
    assert film.league_rates is not None
    assert film.league_rates["Cover 2 Zone"] == pytest.approx(1.0)
    synthetic = config_for_corpus("synthetic")
    assert synthetic.baseline_source == "observed"
    assert synthetic.league_rates is None
    cfg.data_dir.cache_clear()
