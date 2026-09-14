"""Field geometry.

College and NFL fields differ in exactly one way that matters for coverage: hash
width. NFL hashes sit 70'9" from each sideline (18.5 ft apart); college hashes sit
60 ft from each sideline (40 ft apart). That changes where the ball is spotted,
which changes what counts as the field side versus the boundary side, which is the
single biggest driver of safety rotation. A model trained on NFL tracking data will
misread college alignments unless the coordinates are expressed in a way that
accounts for it, so hash width is a first-class parameter everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FEET_PER_YARD = 3.0

FIELD_LENGTH_YD = 120.0  # including both 10-yard end zones
FIELD_WIDTH_YD = 160.0 / FEET_PER_YARD  # 53.333
FIELD_CENTER_Y = FIELD_WIDTH_YD / 2.0

League = Literal["nfl", "ncaa"]


@dataclass(frozen=True)
class FieldSpec:
    league: League
    hash_from_sideline_ft: float

    @property
    def hash_from_sideline_yd(self) -> float:
        return self.hash_from_sideline_ft / FEET_PER_YARD

    @property
    def hash_separation_yd(self) -> float:
        return FIELD_WIDTH_YD - 2 * self.hash_from_sideline_yd

    @property
    def left_hash_y(self) -> float:
        return self.hash_from_sideline_yd

    @property
    def right_hash_y(self) -> float:
        return FIELD_WIDTH_YD - self.hash_from_sideline_yd

    def hash_of(self, ball_y: float, tolerance: float = 1.5) -> Literal["left", "right", "middle"]:
        """Which hash the ball is on, given an absolute y in yards from the left sideline."""
        if abs(ball_y - self.left_hash_y) <= tolerance:
            return "left"
        if abs(ball_y - self.right_hash_y) <= tolerance:
            return "right"
        return "middle"

    def field_side_width(self, ball_y: float) -> float:
        """Width in yards of the wide (field) side of the formation."""
        return max(ball_y, FIELD_WIDTH_YD - ball_y)

    def boundary_side_width(self, ball_y: float) -> float:
        return min(ball_y, FIELD_WIDTH_YD - ball_y)

    def hash_asymmetry(self, ball_y: float) -> float:
        """How lopsided the field is at this spot, in yards. 0 in the middle."""
        return self.field_side_width(ball_y) - self.boundary_side_width(ball_y)


NFL_FIELD = FieldSpec(league="nfl", hash_from_sideline_ft=70.75)
NCAA_FIELD = FieldSpec(league="ncaa", hash_from_sideline_ft=60.0)

FIELDS: dict[str, FieldSpec] = {"nfl": NFL_FIELD, "ncaa": NCAA_FIELD}


def get_field(league: str) -> FieldSpec:
    key = league.lower().strip()
    if key not in FIELDS:
        raise ValueError(f"Unknown league {league!r}; expected one of {sorted(FIELDS)}")
    return FIELDS[key]
