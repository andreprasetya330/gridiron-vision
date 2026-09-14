"""Evidence bundles and reports: every claim must be traceable."""

import re

import pytest

from gridiron.scouting.evidence import build_evidence
from gridiron.scouting.report import _numbers_are_grounded, render_markdown
from gridiron.tells.context import SeasonContext, detect_scheme_change, opponent_adjusted_distribution
from gridiron.tells.mining import MiningConfig


@pytest.fixture(scope="module")
def bundle(synthetic_cues):
    return build_evidence(synthetic_cues, "Washington", through_week=10, mining_config=MiningConfig())


def test_bundle_reports_a_real_distribution(bundle):
    total = sum(bundle.coverage_distribution.values())
    assert total == pytest.approx(1.0, abs=0.01)
    assert bundle.n_labeled_plays > 0


def test_every_tell_links_to_its_plays(bundle):
    assert bundle.tells, "Washington has planted tells and should produce some"
    for tell in bundle.tells:
        assert tell["clip_play_ids"], f"{tell['description']} has no supporting clips"
        assert len(tell["clip_play_ids"]) <= 12


def test_report_renders_without_an_llm(bundle):
    markdown = render_markdown(bundle)
    assert "# Scouting report: Washington" in markdown
    assert "## Coverage mix" in markdown
    assert "## Tells" in markdown
    assert "Read this with the caveats" in markdown


def test_report_states_sample_sizes_next_to_claims(bundle):
    markdown = render_markdown(bundle)
    for tell in bundle.tells[:3]:
        assert f"n={tell['n_cue']}" in markdown


def test_self_scout_changes_the_framing(synthetic_cues):
    bundle = build_evidence(
        synthetic_cues, "Oregon State", through_week=10, mining_config=MiningConfig(), self_scout=True
    )
    assert "Self-scout" in render_markdown(bundle)


def test_missing_team_produces_a_caveat_not_a_crash(synthetic_cues):
    bundle = build_evidence(synthetic_cues, "Nonexistent State", mining_config=MiningConfig())
    assert bundle.n_plays == 0
    assert any("No plays found" in c for c in bundle.caveats)


def test_borrowed_baseline_is_disclosed(synthetic_cues):
    bundle = build_evidence(
        synthetic_cues,
        "Washington",
        mining_config=MiningConfig(baseline_source="nfl-big-data-bowl"),
    )
    assert bundle.baseline_is_borrowed
    assert any("borrowed" in c for c in bundle.caveats)


def test_narrative_guard_rejects_invented_numbers():
    payload = {"coverage_distribution": {"Cover 3 Zone": 0.42}}
    assert _numbers_are_grounded("They play Cover 3 on 42% of snaps.", payload)
    assert not _numbers_are_grounded("They play Cover 3 on 71% of snaps.", payload)


def test_narrative_guard_allows_rounding_slack():
    payload = {"rate": 0.415}
    assert _numbers_are_grounded("about 42% of the time", payload)


def test_scheme_change_is_detected_where_it_was_planted(synthetic_cues):
    """Arizona State changes coordinators in week 8; the detector should notice."""
    team_cues = synthetic_cues[synthetic_cues["defense_team"] == "Arizona State"]
    detected = detect_scheme_change(team_cues, alpha=0.01, min_weeks_per_side=3)
    assert detected is not None
    assert abs(detected - 8) <= 1


def test_no_scheme_change_is_detected_for_a_stable_defense(synthetic_cues):
    team_cues = synthetic_cues[synthetic_cues["defense_team"] == "California"]
    assert detect_scheme_change(team_cues, alpha=0.01, min_weeks_per_side=3) is None


def test_context_excludes_film_from_before_a_coordinator_change(synthetic_cues):
    context = SeasonContext()
    windowed, notes = context.usable_window(synthetic_cues, "Arizona State", season=2025)
    assert notes, "a detected scheme change should be explained to the reader"
    before = (synthetic_cues["defense_team"] == "Arizona State").sum()
    after = (windowed["defense_team"] == "Arizona State").sum()
    assert after < before


def test_opponent_adjustment_separates_schedule_from_scheme(synthetic_cues):
    adjusted = opponent_adjusted_distribution(synthetic_cues, "Washington")
    assert adjusted
    for values in adjusted.values():
        assert set(values) == {"raw", "expected_from_schedule", "adjusted", "league", "schedule_effect"}
        assert 0.0 <= values["raw"] <= 1.0


def test_markdown_has_no_unrendered_placeholders(bundle):
    markdown = render_markdown(bundle)
    assert not re.search(r"\{[a-z_]+\}", markdown)
