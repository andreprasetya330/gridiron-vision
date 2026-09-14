"""Coverage and role vocabularies, shared by every stage of the pipeline."""

from __future__ import annotations

COVERAGES: list[str] = [
    "Cover 0 Man",
    "Cover 1 Man",
    "Cover 2 Man",
    "Cover 2 Zone",
    "Cover 3 Zone",
    "Cover 4 Zone",
    "Cover 6 Zone",
    "Prevent",
]

COVERAGE_INDEX: dict[str, int] = {name: i for i, name in enumerate(COVERAGES)}

MAN_COVERAGES = {"Cover 0 Man", "Cover 1 Man", "Cover 2 Man"}
ZONE_COVERAGES = {c for c in COVERAGES if c not in MAN_COVERAGES}

# How many deep defenders each shell nominally plays. Used for sanity checks and
# for the "shell" coarse label, which is far easier to predict than the full
# 8-class problem and is often all a coach actually needs pre-snap.
DEEP_DEFENDERS: dict[str, int] = {
    "Cover 0 Man": 0,
    "Cover 1 Man": 1,
    "Cover 2 Man": 2,
    "Cover 2 Zone": 2,
    "Cover 3 Zone": 3,
    "Cover 4 Zone": 4,
    "Cover 6 Zone": 3,  # quarter-quarter-half: 2 quarters + 1 half
    "Prevent": 4,
}

SHELLS: list[str] = ["0-high", "1-high", "2-high", "3+-high"]


def shell_of(coverage: str) -> str:
    n = DEEP_DEFENDERS[coverage]
    if n == 0:
        return "0-high"
    if n == 1:
        return "1-high"
    if n == 2:
        return "2-high"
    return "3+-high"


def is_man(coverage: str) -> bool:
    return coverage in MAN_COVERAGES


DEFENDER_ROLES: list[str] = ["man", "underneath_zone", "deep_zone", "blitz"]
ROLE_INDEX: dict[str, int] = {name: i for i, name in enumerate(DEFENDER_ROLES)}

# Offensive and defensive position groups we care about. Film-derived tracks will
# often have no position at all, so everything downstream must tolerate None.
DEFENSIVE_POSITIONS = ["DL", "LB", "CB", "S"]
OFFENSIVE_POSITIONS = ["QB", "RB", "TE", "WR", "OL"]

PLAYERS_PER_SIDE = 11
