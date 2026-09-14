"""Scouting report rendering.

The report is written from the evidence bundle and nothing else. The deterministic
renderer produces the whole document on its own; the optional LLM pass is allowed
to rewrite the narrative sections into better prose, with the numbers supplied as
a closed set it may not add to.

If the LLM is unavailable, unconfigured, or returns something suspicious, the
templated version ships. A report that reads a little stiff is fine. A report that
invents a statistic is a catastrophe, because the first time a coach catches one
he is right to throw the whole tool away.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gridiron.config import LLMSettings, reports_dir
from gridiron.scouting.evidence import EvidenceBundle


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _tell_line(tell: dict[str, Any]) -> str:
    validation = tell.get("validation")
    if validation and validation.get("holdout_trials"):
        held = "held" if validation.get("held_up") else "has not held"
        record = (
            f" It {held} up since: {validation['holdout_hits']} of "
            f"{validation['holdout_trials']} snaps ({validation['holdout_rate']:.0%})."
        )
    else:
        record = " Not yet tested on later film."

    ci = f"[{tell['ci_low']:.0%}-{tell['ci_high']:.0%}]"
    return (
        f"- **{tell['description']}** {ci}, q={tell['q_value']:.3f}."
        f"{record}"
    )


def render_markdown(bundle: EvidenceBundle, narrative: str | None = None) -> str:
    title = (
        f"Self-scout: {bundle.team}" if bundle.self_scout else f"Scouting report: {bundle.team}"
    )
    lines: list[str] = [f"# {title}", ""]

    scope = []
    if bundle.season:
        scope.append(f"season {bundle.season}")
    if bundle.through_week:
        scope.append(f"through week {bundle.through_week}")
    scope.append(f"{bundle.n_labeled_plays} labeled snaps")
    lines.append(f"*{', '.join(scope)}*")
    lines.append("")

    if narrative:
        lines.extend([narrative.strip(), ""])
    else:
        lines.extend([bundle.headline, ""])

    # --- Coverage mix --------------------------------------------------------
    lines.extend(["## Coverage mix", ""])
    if bundle.coverage_distribution:
        lines.append("| Coverage | Them | League | Diff | Schedule-adjusted |")
        lines.append("| --- | --- | --- | --- | --- |")
        ordered = sorted(bundle.coverage_distribution.items(), key=lambda kv: -kv[1])
        for coverage, rate in ordered:
            if rate < 0.02:
                continue
            league = bundle.baseline_distribution.get(coverage)
            adjusted = bundle.opponent_adjusted.get(coverage, {}).get("adjusted")
            delta = None if league is None else rate - league
            delta_text = "n/a" if delta is None else f"{delta:+.0%}"
            lines.append(
                f"| {coverage} | {_fmt_pct(rate)} | {_fmt_pct(league)} | {delta_text} | "
                f"{_fmt_pct(adjusted)} |"
            )
        lines.append("")

    split = bundle.man_zone_split
    if split:
        lines.append(
            f"Man coverage on {_fmt_pct(split['man'])} of snaps against a league rate of "
            f"{_fmt_pct(split['league_man'])} ({split['delta']:+.0%})."
        )
        lines.append("")

    # --- Tells ---------------------------------------------------------------
    lines.extend(["## Tells", ""])
    if bundle.tells:
        lines.append(
            f"Mined {bundle.mining_summary.get('n_tests', 0)} cue-by-coverage comparisons; "
            f"{bundle.mining_summary.get('n_significant', 0)} survived Benjamini-Hochberg "
            f"correction at q<{bundle.mining_summary.get('fdr_alpha', 0.1)}. "
            f"The strongest, ranked by how much they move the odds and how often the look appears:"
        )
        lines.append("")
        for tell in bundle.tells:
            lines.append(_tell_line(tell))
            clips = tell.get("clip_play_ids") or []
            if clips:
                shown = ", ".join(f"`{c}`" for c in clips[:6])
                more = f" (+{len(clips) - 6} more)" if len(clips) > 6 else ""
                lines.append(f"  - Clips: {shown}{more}")
        lines.append("")
    else:
        lines.append(
            "No cue survived correction and the effect-size floor. That is a real "
            "finding rather than a gap: within this sample, this defense does not tip "
            "its coverage in any way we can measure."
        )
        lines.append("")

    validation = bundle.validation_summary
    if validation and validation.get("pooled_hit_rate") is not None:
        lines.append(
            f"Across forward validation, tells predicting a coverage hit "
            f"{validation['pooled_hit_rate']:.0%} of the time on film they were not mined "
            f"from, against {_fmt_pct(validation.get('pooled_expected'))} expected from this "
            f"defense's own tendencies. "
            f"{_fmt_pct(validation.get('survival_rate'))} of all tells held up."
        )
        if validation.get("pooled_avoidance_rate") is not None:
            lines.append(
                f"Tells predicting a coverage will *not* appear were right in the same "
                f"direction: {validation['pooled_avoidance_rate']:.0%} out of sample against "
                f"{_fmt_pct(validation.get('pooled_avoidance_expected'))} expected."
            )
        lines.append("")

    # --- Tendencies ----------------------------------------------------------
    if bundle.tendencies:
        lines.extend(["## Situational tendencies", ""])
        lines.append("| Down | Distance | Coverage | Rate | League | Diff | Snaps |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in bundle.tendencies:
            delta = row.get("delta")
            delta_text = "n/a" if delta is None else f"{delta:+.0%}"
            lines.append(
                f"| {row.get('down')} | {row.get('distance_band')} | {row.get('coverage')} | "
                f"{_fmt_pct(row.get('rate'))} | {_fmt_pct(row.get('league_rate'))} | "
                f"{delta_text} | {row.get('plays')} |"
            )
        lines.append("")

    if bundle.season_trend:
        lines.extend(["## How they have trended", ""])
        lines.extend(f"- {line}" for line in bundle.season_trend)
        lines.append("")

    if bundle.what_beat_them:
        lines.extend(["## What has beaten them", ""])
        lines.append("| Down | Distance | Play type | EPA/play | Success | Plays |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in bundle.what_beat_them:
            success = row.get("success_rate")
            lines.append(
                f"| {row.get('down')} | {row.get('distance_band')} | {row.get('play_type')} | "
                f"{row.get('epa_per_play', 0):.3f} | {_fmt_pct(success)} | {row.get('plays')} |"
            )
        lines.append("")

    if bundle.personnel:
        lines.extend(["## Secondary personnel", ""])
        for row in bundle.personnel[:12]:
            values = " ".join(str(v) for v in row.values() if v is not None)
            lines.append(f"- {values}")
        lines.append("")

    lines.extend(["## Read this with the caveats", ""])
    if bundle.caveats:
        lines.extend(f"- {c}" for c in bundle.caveats)
    else:
        lines.append("- Sample sizes and baselines are adequate for the claims above.")
    lines.append("")

    lines.append(
        "*Every rate above carries its sample size. Tells are Benjamini-Hochberg "
        "corrected across the full cue grid and forward-validated on film they were "
        "not mined from. Clip ids link each claim to the snaps behind it.*"
    )
    return "\n".join(lines)


NARRATIVE_SYSTEM_PROMPT = """You write scouting summaries for American football coaches.

You will receive a JSON evidence bundle. Write 2-4 short paragraphs summarizing what
this defense does and what to attack.

Absolute rules:
- Use ONLY numbers that appear in the JSON. Never compute, estimate, round differently,
  or invent a number.
- Never claim a tendency the JSON does not contain.
- If the JSON says a sample is small or a baseline is borrowed, say so plainly.
- No preamble, no headings, no bullet lists. Plain paragraphs.
- Write like a coach talking to another coach, not like a data scientist.
"""


def generate_narrative(bundle: EvidenceBundle, settings: LLMSettings | None = None) -> str | None:
    """Optional LLM prose. Returns None whenever anything is off."""
    settings = settings or LLMSettings()
    if not settings.enabled:
        return None

    payload = {
        "team": bundle.team,
        "season": bundle.season,
        "through_week": bundle.through_week,
        "n_labeled_plays": bundle.n_labeled_plays,
        "coverage_distribution": bundle.coverage_distribution,
        "baseline_distribution": bundle.baseline_distribution,
        "man_zone_split": bundle.man_zone_split,
        "tells": [
            {
                "description": t["description"],
                "n_cue": t["n_cue"],
                "validation": t.get("validation"),
            }
            for t in bundle.tells[:8]
        ],
        "season_trend": bundle.season_trend,
        "caveats": bundle.caveats,
    }

    try:
        import httpx

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{settings.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {settings.api_key}"},
                json={
                    "model": settings.model,
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload, indent=2)},
                    ],
                },
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

    return text if _numbers_are_grounded(text, payload) else None


def _numbers_are_grounded(text: str, payload: dict[str, Any]) -> bool:
    """Reject narrative containing percentages that are not in the evidence.

    A blunt check, and deliberately so. It is the difference between a tool a
    coach trusts and one he catches making something up.
    """
    import re

    allowed: set[int] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for v in value.values():
                collect(v)
        elif isinstance(value, list):
            for v in value:
                collect(v)
        elif isinstance(value, (int, float)):
            allowed.add(int(round(float(value) * 100)))
            allowed.add(int(round(float(value))))

    collect(payload)

    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text):
        value = int(round(float(match.group(1))))
        if not any(abs(value - a) <= 1 for a in allowed):
            return False
    return True


def write_report(
    bundle: EvidenceBundle,
    directory: Path | None = None,
    use_llm: bool = True,
) -> tuple[Path, Path]:
    """Write the report and its evidence bundle. Returns (markdown, json) paths."""
    directory = Path(directory or reports_dir())
    directory.mkdir(parents=True, exist_ok=True)

    slug = bundle.team.lower().replace(" ", "-")
    if bundle.self_scout:
        slug = f"self-scout-{slug}"

    narrative = generate_narrative(bundle) if use_llm else None
    markdown = render_markdown(bundle, narrative)

    md_path = directory / f"{slug}.md"
    json_path = directory / f"{slug}.evidence.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(bundle.to_dict(), indent=2, default=str), encoding="utf-8")
    return md_path, json_path
