"""PFF ingestion.

**PFF sells no public developer API.** Their data is an enterprise product, and
`premium.pff.com/api/v1/...` is an internal front-end endpoint that only answers
to a logged-in browser session. Automating against it violates PFF's terms, and
if this tool ever reaches a university staff, that is exactly the kind of thing
that gets it thrown out of the building.

So the working implementation is a **drop folder**. Export whatever CSVs your PFF
tier gives you into `data/pff/`, and this module sniffs the headers, works out
what kind of report each file is, and maps the columns into the internal schema.
That path is identical whether you have a personal PFF+ subscription or a team
PFF Ultimate license - and Ultimate's per-snap coverage labels would be the single
most valuable dataset in this entire project, because they are real ground truth
for college football, which is the one thing the public world cannot give you.

`PffSessionSource` is deliberately a stub. The endpoint shape is recorded so the
decision stays informed, and it stays yours.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from gridiron.config import pff_dir
from gridiron.ingest.schema import normalize_team, to_play_context
from gridiron.taxonomy import COVERAGES

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _canon(column: str) -> str:
    return _NON_ALNUM.sub("_", str(column).strip().lower()).strip("_")


# Header aliases, canonicalized. PFF's exports vary by product and by year, so
# matching is done on a normalized key rather than an exact string.
COLUMN_ALIASES: dict[str, set[str]] = {
    "team": {"team", "team_name", "franchise", "club", "defense", "defensive_team", "offense_team_name"},
    "opponent": {"opponent", "opp", "against", "offense", "offensive_team"},
    "season": {"season", "year", "season_year"},
    "week": {"week", "week_number", "wk"},
    "game_id": {"game_id", "gameid", "game", "pff_game_id"},
    "play_id": {"play_id", "playid", "pff_play_id", "snap_id"},
    "player": {"player", "player_name", "name", "display_name"},
    "player_id": {"player_id", "playerid", "pff_id", "pff_player_id"},
    "position": {"position", "pos", "player_position"},
    "jersey": {"jersey", "jersey_number", "number", "uniform"},
    "coverage": {
        "coverage",
        "coverage_scheme",
        "defensive_coverage",
        "pass_coverage",
        "coverage_type",
        "scheme",
    },
    "man_snaps": {"man_snaps", "snaps_man", "man_coverage_snaps", "man"},
    "zone_snaps": {"zone_snaps", "snaps_zone", "zone_coverage_snaps", "zone"},
    "snaps": {"snaps", "snap_count", "total_snaps", "plays"},
    "grade": {"grade", "grades_defense", "grade_defense", "overall_grade", "pff_grade"},
    "grade_coverage": {"grades_coverage_defense", "coverage_grade", "grade_coverage"},
    "down": {"down"},
    "distance": {"distance", "yards_to_go", "ytg", "togo"},
    "yards_gained": {"yards", "yards_gained", "gain"},
    "epa": {"epa", "expected_points_added"},
}

# PFF's scheme vocabulary mapped onto ours.
PFF_COVERAGE_MAP: dict[str, str] = {
    "cover 0": "Cover 0 Man",
    "cover 0 man": "Cover 0 Man",
    "c0": "Cover 0 Man",
    "cover 1": "Cover 1 Man",
    "cover 1 man": "Cover 1 Man",
    "cover 1 double": "Cover 1 Man",
    "c1": "Cover 1 Man",
    "cover 2": "Cover 2 Zone",
    "cover 2 zone": "Cover 2 Zone",
    "c2": "Cover 2 Zone",
    "cover 2 man": "Cover 2 Man",
    "2 man": "Cover 2 Man",
    "man cover 2": "Cover 2 Man",
    "cover 3": "Cover 3 Zone",
    "cover 3 zone": "Cover 3 Zone",
    "cover 3 cloud": "Cover 3 Zone",
    "cover 3 seam": "Cover 3 Zone",
    "c3": "Cover 3 Zone",
    "cover 4": "Cover 4 Zone",
    "quarters": "Cover 4 Zone",
    "c4": "Cover 4 Zone",
    "cover 6": "Cover 6 Zone",
    "quarter quarter half": "Cover 6 Zone",
    "c6": "Cover 6 Zone",
    "prevent": "Prevent",
}


def map_coverage(raw: Any) -> str | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if text in COVERAGES:
        return text
    key = _NON_ALNUM.sub(" ", text.lower()).strip()
    return PFF_COVERAGE_MAP.get(key)


@dataclass
class SniffResult:
    path: Path
    kind: str  # "per_snap" | "coverage_summary" | "player_grades" | "unknown"
    mapping: dict[str, str] = field(default_factory=dict)  # internal -> actual column
    n_rows: int = 0
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def has_play_level_coverage(self) -> bool:
        return self.kind == "per_snap" and "coverage" in self.mapping


def sniff(path: Path) -> SniffResult:
    """Work out what a PFF export actually contains from its header row."""
    path = Path(path)
    try:
        head = pd.read_csv(path, nrows=200)
    except Exception as exc:
        return SniffResult(path=path, kind="unknown", notes=[f"unreadable: {exc}"])

    canon_to_actual = {_canon(c): c for c in head.columns}
    mapping: dict[str, str] = {}
    for internal, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in canon_to_actual:
                mapping[internal] = canon_to_actual[alias]
                break

    n_rows = sum(1 for _ in open(path, encoding="utf-8", errors="ignore")) - 1

    has_coverage = "coverage" in mapping
    coverage_is_label = False
    if has_coverage:
        sample = head[mapping["coverage"]].dropna().astype(str).head(50)
        recognized = sum(1 for v in sample if map_coverage(v))
        coverage_is_label = len(sample) > 0 and recognized / len(sample) >= 0.5

    if coverage_is_label and ("play_id" in mapping or "game_id" in mapping):
        kind = "per_snap"
        confidence = 0.95
    elif {"man_snaps", "zone_snaps"} & set(mapping):
        kind = "coverage_summary"
        confidence = 0.8
    elif "player" in mapping and ("grade" in mapping or "grade_coverage" in mapping):
        kind = "player_grades"
        confidence = 0.8
    elif coverage_is_label:
        kind = "coverage_summary"
        confidence = 0.6
    else:
        kind = "unknown"
        confidence = 0.2

    notes = []
    if kind == "per_snap":
        notes.append(
            "Per-snap coverage labels found. This is the highest-value PFF file "
            "type: it is real ground truth for training and for the national baseline."
        )
    if kind == "unknown":
        notes.append(
            "Could not identify this export. Add its header names to COLUMN_ALIASES "
            "in ingest/pff.py and it will be picked up next run."
        )

    return SniffResult(
        path=path,
        kind=kind,
        mapping=mapping,
        n_rows=max(n_rows, 0),
        confidence=confidence,
        notes=notes,
    )


class PffSource(Protocol):
    """Anything that can supply PFF data to the rest of the system."""

    def available(self) -> bool: ...

    def load(self) -> dict[str, pd.DataFrame]: ...


class PffDropFolderSource:
    """Reads whatever PFF exports are sitting in `data/pff/`."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory or pff_dir())

    def files(self) -> list[Path]:
        return sorted(
            [p for p in self.directory.glob("**/*") if p.suffix.lower() in (".csv", ".tsv")]
        )

    def available(self) -> bool:
        return bool(self.files())

    def inspect(self) -> list[SniffResult]:
        return [sniff(p) for p in self.files()]

    def load(self) -> dict[str, pd.DataFrame]:
        results = self.inspect()
        buckets: dict[str, list[pd.DataFrame]] = {}
        for result in results:
            if result.kind == "unknown":
                continue
            try:
                df = pd.read_csv(result.path)
            except Exception:
                continue
            df = df.rename(columns={v: k for k, v in result.mapping.items()})
            df["_source_file"] = result.path.name
            buckets.setdefault(result.kind, []).append(df)

        return {
            kind: pd.concat(frames, ignore_index=True) for kind, frames in buckets.items() if frames
        }

    def play_context(self) -> pd.DataFrame:
        """Per-snap exports mapped into the shared play-context schema."""
        loaded = self.load()
        per_snap = loaded.get("per_snap")
        if per_snap is None or per_snap.empty:
            return to_play_context([], "pff")

        rows: list[dict[str, Any]] = []
        for _, row in per_snap.iterrows():
            coverage = map_coverage(row.get("coverage"))
            play_id = row.get("play_id") or row.get("game_id")
            rows.append(
                {
                    "play_id": f"pff-{play_id}",
                    "season": row.get("season"),
                    "week": row.get("week"),
                    "defense_team": normalize_team(row.get("team")),
                    "offense_team": normalize_team(row.get("opponent")),
                    "down": row.get("down"),
                    "distance": row.get("distance"),
                    "yards_gained": row.get("yards_gained"),
                    "epa": row.get("epa"),
                    "coverage": coverage,
                    "coverage_source": "pff" if coverage else None,
                    "source": "pff",
                }
            )
        return to_play_context(rows, "pff")

    def coverage_rates(self) -> pd.DataFrame:
        """Team-level man/zone and scheme splits from summary exports."""
        loaded = self.load()
        summary = loaded.get("coverage_summary")
        if summary is None or summary.empty:
            return pd.DataFrame()

        if "coverage" in summary.columns:
            summary = summary.copy()
            summary["coverage"] = summary["coverage"].map(map_coverage)
            summary = summary.dropna(subset=["coverage"])

        keys = [c for c in ("team", "season", "week", "coverage") if c in summary.columns]
        if not keys:
            return pd.DataFrame()
        value_cols = [c for c in ("snaps", "man_snaps", "zone_snaps") if c in summary.columns]
        if not value_cols:
            return summary
        grouped = summary.groupby(keys, dropna=False)[value_cols].sum().reset_index()
        if "team" in grouped.columns:
            grouped["team"] = grouped["team"].map(normalize_team)
        return grouped


class PffSessionSource:
    """Deliberately unimplemented.

    PFF's Premium Stats front end is backed by endpoints of the shape

        GET https://premium.pff.com/api/v1/facet/{unit}/{report}
            ?league={nfl|ncaa}&season={year}&week={week}&franchiseId={id}

    which require the session cookies of a logged-in PFF+ account. It is an
    undocumented internal API, and driving it programmatically is against PFF's
    terms of service. It is recorded here so the tradeoff is visible rather than
    rediscovered, and it stays unimplemented on purpose. Use the drop folder.
    """

    endpoint_shape = "https://premium.pff.com/api/v1/facet/{unit}/{report}"

    def available(self) -> bool:
        return False

    def load(self) -> dict[str, pd.DataFrame]:
        raise NotImplementedError(
            "Automating premium.pff.com violates PFF's terms of service. Export CSVs "
            "from the PFF UI into data/pff/ and use PffDropFolderSource instead."
        )


def describe_drop_folder(directory: Path | None = None) -> str:
    """Human-readable summary of what is sitting in the drop folder."""
    source = PffDropFolderSource(directory)
    results = source.inspect()
    if not results:
        return (
            f"No PFF exports found in {source.directory}. Drop CSV exports there and "
            "they will be picked up automatically."
        )
    lines = [f"{len(results)} file(s) in {source.directory}:"]
    for result in results:
        lines.append(
            f"  {result.path.name}: {result.kind} ({result.n_rows} rows, "
            f"{len(result.mapping)} columns mapped)"
        )
        lines.extend(f"    {note}" for note in result.notes)
    return "\n".join(lines)
