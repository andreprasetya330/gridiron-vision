"""A football-realistic synthetic season.

This exists for three reasons:

1. Big Data Bowl data needs Kaggle credentials and a few gigabytes. The whole
   system should run the day you clone it.
2. The tell-mining engine makes strong statistical claims, and the only way to
   know it works is to plant tells of a known size and check that it finds those
   and not a pile of noise. Real data cannot do that - you never know the truth.
3. The vision pipeline needs a target to be validated against.

The generative story is deliberately the honest one: the defense picks a coverage
first, then expresses an alignment. A "tell" is a team-specific quirk in how they
express a given coverage, so it is planted as P(trait | coverage, team) deviating
from the league rate. That is exactly what the miner is supposed to recover.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Iterator

import numpy as np

from gridiron.fields import FIELD_CENTER_Y, FIELD_WIDTH_YD, get_field
from gridiron.taxonomy import COVERAGES
from gridiron.tracking.schema import (
    N_FRAMES,
    SNAP_INDEX,
    TIME_GRID,
    PlayerTrack,
    PlayTracks,
    Situation,
    TrackQuality,
    frame_at,
)

# Traits are the observable quirks a cue extractor will later try to measure.
TRAITS = [
    "press_corners",
    "late_rotation",
    "heavy_box",
    "deep_safeties",
    "nickel_creep",
]

# League-average P(trait | coverage). These are the national baseline the miner
# compares each team against, so they need to be plausible rather than uniform.
BASE_TRAIT_PROB: dict[str, dict[str, float]] = {
    "press_corners": {
        "Cover 0 Man": 0.85,
        "Cover 1 Man": 0.70,
        "Cover 2 Man": 0.65,
        "Cover 2 Zone": 0.35,
        "Cover 3 Zone": 0.25,
        "Cover 4 Zone": 0.20,
        "Cover 6 Zone": 0.22,
        "Prevent": 0.05,
    },
    "late_rotation": {
        "Cover 0 Man": 0.20,
        "Cover 1 Man": 0.35,
        "Cover 2 Man": 0.20,
        "Cover 2 Zone": 0.22,
        "Cover 3 Zone": 0.45,
        "Cover 4 Zone": 0.18,
        "Cover 6 Zone": 0.30,
        "Prevent": 0.05,
    },
    "heavy_box": {
        "Cover 0 Man": 0.80,
        "Cover 1 Man": 0.55,
        "Cover 2 Man": 0.30,
        "Cover 2 Zone": 0.25,
        "Cover 3 Zone": 0.35,
        "Cover 4 Zone": 0.20,
        "Cover 6 Zone": 0.20,
        "Prevent": 0.02,
    },
    "deep_safeties": {
        "Cover 0 Man": 0.02,
        "Cover 1 Man": 0.25,
        "Cover 2 Man": 0.55,
        "Cover 2 Zone": 0.60,
        "Cover 3 Zone": 0.35,
        "Cover 4 Zone": 0.65,
        "Cover 6 Zone": 0.60,
        "Prevent": 0.95,
    },
    "nickel_creep": {
        "Cover 0 Man": 0.75,
        "Cover 1 Man": 0.40,
        "Cover 2 Man": 0.30,
        "Cover 2 Zone": 0.20,
        "Cover 3 Zone": 0.25,
        "Cover 4 Zone": 0.15,
        "Cover 6 Zone": 0.18,
        "Prevent": 0.02,
    },
}

# League base rates for coverage calls, before situation adjusts them.
BASE_COVERAGE_PROB: dict[str, float] = {
    "Cover 0 Man": 0.04,
    "Cover 1 Man": 0.18,
    "Cover 2 Man": 0.06,
    "Cover 2 Zone": 0.14,
    "Cover 3 Zone": 0.32,
    "Cover 4 Zone": 0.17,
    "Cover 6 Zone": 0.07,
    "Prevent": 0.02,
}


@dataclass
class PlantedTell:
    """A team-specific quirk that betrays a coverage.

    `probability` is P(trait | coverage) for this team. The league rate is in
    BASE_TRAIT_PROB, and the gap between them is the effect size the mining
    engine has to detect through the noise of a ~65-snap sample.
    """

    trait: str
    coverage: str
    probability: float
    note: str = ""

    @property
    def league_probability(self) -> float:
        return BASE_TRAIT_PROB[self.trait][self.coverage]

    @property
    def effect_size(self) -> float:
        return self.probability - self.league_probability


@dataclass
class TeamProfile:
    name: str
    coverage_bias: dict[str, float] = dc_field(default_factory=dict)
    tells: list[PlantedTell] = dc_field(default_factory=list)
    # A mid-season coordinator change makes early-season history worthless. One
    # team has one so the season-context code has something real to handle.
    coordinator_change_week: int | None = None
    post_change_bias: dict[str, float] = dc_field(default_factory=dict)

    def coverage_distribution(self, week: int) -> dict[str, float]:
        bias = dict(self.coverage_bias)
        if self.coordinator_change_week is not None and week >= self.coordinator_change_week:
            bias = dict(self.post_change_bias)
        probs = {c: BASE_COVERAGE_PROB[c] * bias.get(c, 1.0) for c in COVERAGES}
        total = sum(probs.values())
        return {c: p / total for c, p in probs.items()}

    def trait_probability(self, trait: str, coverage: str) -> float:
        for tell in self.tells:
            if tell.trait == trait and tell.coverage == coverage:
                return tell.probability
        return BASE_TRAIT_PROB[trait][coverage]


def default_teams() -> list[TeamProfile]:
    """A small league with deliberately varied, discoverable personalities."""
    return [
        TeamProfile(
            name="Washington",
            coverage_bias={"Cover 3 Zone": 1.4, "Cover 1 Man": 1.2, "Cover 4 Zone": 0.8},
            tells=[
                PlantedTell(
                    trait="late_rotation",
                    coverage="Cover 3 Zone",
                    probability=0.82,
                    note="Rolls the field safety down late when spinning to Cover 3",
                ),
                PlantedTell(
                    trait="press_corners",
                    coverage="Cover 1 Man",
                    probability=0.92,
                    note="Corners press almost every time they play Cover 1",
                ),
            ],
        ),
        TeamProfile(
            name="Oregon",
            coverage_bias={"Cover 4 Zone": 1.8, "Cover 2 Zone": 1.3, "Cover 0 Man": 0.5},
            tells=[
                PlantedTell(
                    trait="deep_safeties",
                    coverage="Cover 4 Zone",
                    probability=0.90,
                    note="Safeties bail to 13+ yards before the snap in quarters",
                ),
            ],
        ),
        TeamProfile(
            name="Oregon State",
            coverage_bias={"Cover 0 Man": 2.5, "Cover 1 Man": 1.5, "Cover 4 Zone": 0.5},
            tells=[
                PlantedTell(
                    trait="nickel_creep",
                    coverage="Cover 0 Man",
                    probability=0.95,
                    note="Nickel walks down inside 4 yards before every zero blitz",
                ),
                PlantedTell(
                    trait="heavy_box",
                    coverage="Cover 0 Man",
                    probability=0.95,
                ),
            ],
        ),
        TeamProfile(
            name="Stanford",
            coverage_bias={"Cover 2 Zone": 1.6, "Cover 6 Zone": 1.8},
            tells=[
                PlantedTell(
                    trait="press_corners",
                    coverage="Cover 2 Zone",
                    probability=0.78,
                    note="Corners jam hard in Cover 2, then sink to the flat",
                ),
            ],
        ),
        TeamProfile(
            name="California",
            coverage_bias={"Cover 3 Zone": 1.2},
            tells=[],  # A control team with no planted tells. The miner should
            # report essentially nothing here, which is the harder test.
        ),
        TeamProfile(
            name="Arizona State",
            coverage_bias={"Cover 1 Man": 1.6, "Cover 3 Zone": 0.7},
            coordinator_change_week=8,
            post_change_bias={"Cover 4 Zone": 2.2, "Cover 2 Zone": 1.5, "Cover 1 Man": 0.4},
            tells=[
                PlantedTell(
                    trait="late_rotation",
                    coverage="Cover 1 Man",
                    probability=0.75,
                ),
            ],
        ),
    ]


def _situation_multipliers(down: int, distance: float, yardline: float) -> dict[str, float]:
    """How the situation bends coverage selection, league-wide."""
    m = {c: 1.0 for c in COVERAGES}
    long_yardage = distance >= 8
    short_yardage = distance <= 3

    if down == 3 and long_yardage:
        m["Cover 2 Zone"] *= 1.4
        m["Cover 4 Zone"] *= 1.5
        m["Cover 3 Zone"] *= 1.1
        m["Cover 0 Man"] *= 0.6
    if down == 3 and short_yardage:
        m["Cover 0 Man"] *= 2.2
        m["Cover 1 Man"] *= 1.6
        m["Cover 4 Zone"] *= 0.5
    if down in (1, 2) and short_yardage:
        m["Cover 1 Man"] *= 1.2
        m["Cover 3 Zone"] *= 1.1
    if yardline >= 80:  # defense backed up near its own goal line
        m["Cover 0 Man"] *= 1.8
        m["Cover 1 Man"] *= 1.4
        m["Cover 4 Zone"] *= 0.4
        m["Prevent"] *= 0.1
    if yardline <= 25:  # offense pinned deep
        m["Cover 3 Zone"] *= 1.2
    return m


PERSONNEL_GROUPS = ["11", "11", "11", "12", "10", "21"]


def _personnel_counts(personnel: str) -> tuple[int, int, int]:
    """(running backs, tight ends, wide receivers) for a personnel string."""
    rb = int(personnel[0])
    te = int(personnel[1])
    wr = 5 - rb - te
    return rb, te, wr


@dataclass
class _Skill:
    """An offensive skill player with a route, in normalized coordinates."""

    label: str
    position: str
    x: float
    y: float
    route: str


def _place_offense(rng: np.random.Generator, personnel: str, room_left: float, room_right: float):
    rb, te, wr = _personnel_counts(personnel)
    players: list[_Skill] = []

    shotgun = rng.random() < 0.75
    qb_x = -6.0 if shotgun else -1.5
    players.append(_Skill("QB", "QB", qb_x, rng.normal(0, 0.3), "dropback"))

    for i, dy in enumerate([-3.4, -1.7, 0.0, 1.7, 3.4]):
        players.append(_Skill(f"OL{i}", "OL", -0.6, dy + rng.normal(0, 0.15), "block"))

    routes = ["go", "out", "in", "slant", "curl", "post", "corner"]

    # Split receivers between the two sides, respecting how much room each has.
    n_wide = wr + te
    left_share = room_left / max(room_left + room_right, 1e-6)
    n_left = int(np.clip(round(n_wide * left_share + rng.normal(0, 0.5)), 1, n_wide - 1))
    n_right = n_wide - n_left

    def spread(count: int, room: float, sign: int) -> list[float]:
        if count <= 0:
            return []
        outermost = min(room - 2.0, 24.0)
        inner = 4.0
        if count == 1:
            return [sign * float(rng.uniform(inner + 4, outermost))]
        stops = np.linspace(inner, outermost, count)
        return [sign * float(s + rng.normal(0, 0.8)) for s in stops]

    wide_positions = [(y, -1) for y in spread(n_left, room_left, -1)]
    wide_positions += [(y, 1) for y in spread(n_right, room_right, 1)]

    te_left = te
    for idx, (y, _sign) in enumerate(wide_positions):
        is_te = te_left > 0 and abs(y) < 8.0
        if is_te:
            te_left -= 1
            players.append(_Skill(f"TE{idx}", "TE", -0.6, y, str(rng.choice(routes))))
        else:
            on_line = rng.random() < 0.5
            players.append(
                _Skill(f"WR{idx}", "WR", -0.6 if on_line else -1.4, y, str(rng.choice(routes)))
            )

    for i in range(rb):
        side = -1 if i == 0 else 1
        players.append(
            _Skill(f"RB{i}", "RB", qb_x - 1.0, side * rng.uniform(0.8, 2.2), "checkdown")
        )

    # Trim or pad to exactly 11.
    while len(players) > 11:
        for i, p in enumerate(players):
            if p.position in ("RB", "TE"):
                players.pop(i)
                break
        else:
            players.pop()
    while len(players) < 11:
        players.append(_Skill(f"WRx{len(players)}", "WR", -0.6, rng.uniform(-18, 18), "go"))

    return players


MOTION_PROB = 0.30
MOTION_START_T = -1.7
MOTION_END_T = -0.35

# Motion response is a loud man indicator, not a perfect one, and the difference
# matters. Man defenses banjo or pass the mover off; zone defenses sometimes lock
# and travel. If chasing meant man with certainty the classifier would learn a
# rule that no real defense obeys, and the tell would look far more reliable in
# testing than it ever is on Saturday.
MAN_TRAVEL_PROB = 0.84
ZONE_TRAVEL_PROB = 0.12


@dataclass
class _Motion:
    """A receiver sent across the formation before the snap.

    This matters more than its share of snaps suggests. Whether a defender travels
    with the mover is the loudest man-coverage indicator available before the ball
    moves, so it is the one cue a coach can act on from the sideline.
    """

    label: str
    start_y: float
    end_y: float

    @property
    def travel(self) -> float:
        return abs(self.end_y - self.start_y)


def _choose_motion(rng: np.random.Generator, offense: list["_Skill"]) -> _Motion | None:
    candidates = [p for p in offense if p.position in ("WR", "TE") and abs(p.y) > 4.0]
    if not candidates or rng.random() >= MOTION_PROB:
        return None
    mover = candidates[int(rng.integers(len(candidates)))]
    side = np.sign(mover.y) or 1.0
    if rng.random() < 0.55:
        # Jet across the formation - a long, unmistakable trip.
        end_y = -side * rng.uniform(3.0, min(abs(mover.y) + 4.0, 16.0))
    else:
        # A shorter shift down toward the ball.
        end_y = mover.y - side * rng.uniform(4.5, 7.5)
    return _Motion(label=mover.label, start_y=float(mover.y), end_y=float(end_y))


def _motion_ramp() -> np.ndarray:
    """0 before the motion, 1 from the moment it settles through the snap.

    Smoothstep rather than linear so the mover accelerates and decelerates, which
    keeps the velocity-based features from seeing an impulse that no human makes.
    """
    i0, i1 = frame_at(MOTION_START_T), frame_at(MOTION_END_T)
    ramp = np.zeros(SNAP_INDEX + 1, dtype=np.float64)
    s = np.linspace(0.0, 1.0, i1 - i0 + 1)
    ramp[i0 : i1 + 1] = s * s * (3.0 - 2.0 * s)
    ramp[i1:] = 1.0
    return ramp


@dataclass
class _Defender:
    label: str
    position: str
    x: float
    y: float
    role: str
    assignment: str | None = None
    landmark: tuple[float, float] | None = None


def _safety_alignment(
    rng: np.random.Generator, coverage: str, traits: dict[str, bool], field_sign: int
) -> list[_Defender]:
    """Place the two safeties, which is where most of the coverage signal lives."""
    deep_bonus = 2.5 if traits["deep_safeties"] else 0.0
    d: list[_Defender] = []

    if coverage == "Cover 0 Man":
        for i, sign in enumerate((-1, 1)):
            d.append(
                _Defender(
                    f"S{i}",
                    "S",
                    rng.uniform(4.0, 8.0),
                    sign * rng.uniform(3.0, 9.0),
                    "man",
                )
            )
    elif coverage == "Cover 1 Man":
        d.append(
            _Defender("S0", "S", rng.uniform(11.0, 15.0) + deep_bonus, rng.normal(0, 2.0), "deep_zone")
        )
        robber_role = "blitz" if rng.random() < 0.2 else "underneath_zone"
        d.append(
            _Defender("S1", "S", rng.uniform(4.0, 8.5), rng.normal(0, 4.0), robber_role)
        )
    elif coverage in ("Cover 2 Man", "Cover 2 Zone"):
        for i, sign in enumerate((-1, 1)):
            d.append(
                _Defender(
                    f"S{i}",
                    "S",
                    rng.uniform(11.0, 14.5) + deep_bonus,
                    sign * rng.uniform(7.0, 11.0),
                    "deep_zone",
                )
            )
    elif coverage == "Cover 3 Zone":
        # The signature look: two-high shell that spins to one-high. Whether they
        # show it late is exactly the tell we plant for Washington.
        if traits["late_rotation"]:
            d.append(_Defender("S0", "S", rng.uniform(11.0, 14.0), -6.0 + rng.normal(0, 2), "deep_zone"))
            d.append(_Defender("S1", "S", rng.uniform(10.0, 13.0), 6.0 + rng.normal(0, 2), "underneath_zone"))
        else:
            d.append(_Defender("S0", "S", rng.uniform(12.0, 15.0) + deep_bonus, rng.normal(0, 2.5), "deep_zone"))
            d.append(_Defender("S1", "S", rng.uniform(6.0, 9.5), rng.normal(0, 5.0), "underneath_zone"))
    elif coverage == "Cover 4 Zone":
        for i, sign in enumerate((-1, 1)):
            d.append(
                _Defender(
                    f"S{i}",
                    "S",
                    rng.uniform(9.5, 13.0) + deep_bonus,
                    sign * rng.uniform(6.0, 9.5),
                    "deep_zone",
                )
            )
    elif coverage == "Cover 6 Zone":
        # Quarters to the field, half to the boundary - an asymmetric shell.
        d.append(
            _Defender("S0", "S", rng.uniform(10.0, 12.5) + deep_bonus, field_sign * rng.uniform(6.0, 9.0), "deep_zone")
        )
        d.append(
            _Defender("S1", "S", rng.uniform(12.5, 16.0) + deep_bonus, -field_sign * rng.uniform(8.0, 12.0), "deep_zone")
        )
    else:  # Prevent
        for i, sign in enumerate((-1, 1)):
            d.append(
                _Defender(f"S{i}", "S", rng.uniform(18.0, 25.0), sign * rng.uniform(8.0, 13.0), "deep_zone")
            )
    return d


def _place_defense(
    rng: np.random.Generator,
    coverage: str,
    traits: dict[str, bool],
    offense: list[_Skill],
    field_sign: int,
) -> list[_Defender]:
    defenders: list[_Defender] = []

    n_dl = 4 if rng.random() < 0.75 else 3
    dl_spread = np.linspace(-3.2, 3.2, n_dl)
    for i, y in enumerate(dl_spread):
        defenders.append(
            _Defender(f"DL{i}", "DL", rng.uniform(0.8, 1.6), float(y + rng.normal(0, 0.4)), "blitz")
        )

    safeties = _safety_alignment(rng, coverage, traits, field_sign)

    # Corners take the widest receiver on each side.
    wides = sorted([p for p in offense if p.position in ("WR", "TE")], key=lambda p: p.y)
    left_wides = [p for p in wides if p.y < 0]
    right_wides = [p for p in wides if p.y >= 0]

    press = traits["press_corners"]
    man_coverage = coverage in ("Cover 0 Man", "Cover 1 Man", "Cover 2 Man")

    corner_targets = []
    if left_wides:
        corner_targets.append(left_wides[0])
    if right_wides:
        corner_targets.append(right_wides[-1])

    for i, target in enumerate(corner_targets):
        if press:
            depth = rng.uniform(1.0, 3.0)
        elif coverage in ("Cover 3 Zone", "Cover 4 Zone", "Prevent"):
            depth = rng.uniform(6.0, 9.5)
        else:
            depth = rng.uniform(4.0, 7.0)
        # Leverage: man corners sit inside on a two-high shell, outside otherwise;
        # zone corners keep outside leverage to protect the sideline.
        lev = -1.0 if man_coverage else 1.0
        offset = lev * np.sign(target.y or 1) * rng.uniform(0.5, 1.6)
        role = "man" if man_coverage else ("deep_zone" if coverage in ("Cover 3 Zone", "Cover 4 Zone", "Prevent") else "underneath_zone")
        defenders.append(
            _Defender(
                f"CB{i}",
                "CB",
                depth,
                float(target.y + offset),
                role,
                assignment=target.label if man_coverage else None,
            )
        )

    covered = {d.assignment for d in defenders if d.assignment}
    remaining_skill = [
        p for p in offense if p.position in ("WR", "TE", "RB") and p.label not in covered
    ]
    remaining_skill.sort(key=lambda p: abs(p.y), reverse=True)

    n_defenders_so_far = len(defenders) + len(safeties)
    n_box = 11 - n_defenders_so_far

    slot_index = 0
    for i in range(n_box):
        if i < len(remaining_skill) and abs(remaining_skill[i].y) > 5.0:
            target = remaining_skill[i]
            depth = rng.uniform(3.0, 6.0)
            if traits["nickel_creep"] and slot_index == 0:
                depth = rng.uniform(1.5, 3.5)
            role = "man" if man_coverage else "underneath_zone"
            if coverage == "Cover 0 Man" and traits["heavy_box"] and rng.random() < 0.4:
                role = "blitz"
            defenders.append(
                _Defender(
                    f"NB{slot_index}",
                    "CB",
                    depth,
                    float(target.y + rng.normal(0, 1.0)),
                    role,
                    assignment=target.label if role == "man" else None,
                )
            )
            slot_index += 1
        else:
            depth = rng.uniform(3.5, 6.0)
            if traits["heavy_box"]:
                depth = rng.uniform(2.0, 4.5)
            role = "underneath_zone"
            if man_coverage and i < len(remaining_skill):
                role = "man"
            if coverage == "Cover 0 Man" and rng.random() < 0.5:
                role = "blitz"
            elif traits["heavy_box"] and rng.random() < 0.25:
                role = "blitz"
            assignment = (
                remaining_skill[i].label
                if role == "man" and i < len(remaining_skill)
                else None
            )
            defenders.append(
                _Defender(
                    f"LB{i}",
                    "LB",
                    depth,
                    float(rng.uniform(-7.0, 7.0)),
                    role,
                    assignment=assignment,
                )
            )

    defenders.extend(safeties)
    return defenders[:11]


def _zone_landmark(defender: _Defender, coverage: str, rng: np.random.Generator) -> tuple[float, float]:
    """Where a zone defender ends up ~2.5s after the snap."""
    if defender.role == "deep_zone":
        if coverage == "Cover 3 Zone":
            if defender.position == "CB":
                return 18.0 + rng.normal(0, 1.5), defender.y * 1.15
            return 17.0 + rng.normal(0, 1.5), rng.normal(0, 2.0)
        if coverage == "Cover 4 Zone":
            return 15.0 + rng.normal(0, 1.5), defender.y * 1.1
        if coverage in ("Cover 2 Zone", "Cover 2 Man"):
            return 18.0 + rng.normal(0, 2.0), np.sign(defender.y or 1) * 11.0
        if coverage == "Cover 6 Zone":
            return 16.0 + rng.normal(0, 1.5), defender.y * 1.1
        if coverage == "Prevent":
            return 24.0 + rng.normal(0, 2.0), defender.y
        return 15.0 + rng.normal(0, 2.0), defender.y
    if defender.role == "underneath_zone":
        return 7.0 + rng.normal(0, 1.5), defender.y * 1.2 + rng.normal(0, 1.5)
    return defender.x, defender.y


def _pick_zone_bumper(defense: list["_Defender"], motion: _Motion | None) -> str | None:
    """The underneath zone defender who slides over as motion crosses his face."""
    if motion is None:
        return None
    near_side = [
        d for d in defense if d.role == "underneath_zone" and d.x <= 8.0
    ]
    if not near_side:
        return None
    return min(near_side, key=lambda d: abs(d.y - motion.end_y)).label


def _simulate(
    rng: np.random.Generator,
    offense: list[_Skill],
    defense: list[_Defender],
    coverage: str,
    traits: dict[str, bool],
    motion: _Motion | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Produce trajectories on TIME_GRID for every player."""
    off_xy: dict[str, np.ndarray] = {}
    def_xy: dict[str, np.ndarray] = {}

    post = TIME_GRID >= 0
    t_post = np.clip(TIME_GRID, 0.0, None)
    motion_ramp = _motion_ramp()

    route_end: dict[str, tuple[float, float]] = {}
    for p in offense:
        # The mover runs his route from where the motion left him, not from where
        # he lined up, so everything post-snap keys off `py` rather than `p.y`.
        moving = motion is not None and p.label == motion.label
        py = motion.end_y if moving else p.y

        traj = np.zeros((N_FRAMES, 2), dtype=np.float32)
        traj[:, 0] = p.x
        traj[:, 1] = py
        if moving:
            traj[: SNAP_INDEX + 1, 1] = motion.start_y + motion_ramp * (
                motion.end_y - motion.start_y
            )

        # Pre-snap: a little bit of settling on top of any motion.
        jitter = rng.normal(0, 0.06, size=(SNAP_INDEX + 1, 2)).cumsum(axis=0) * 0.3
        traj[: SNAP_INDEX + 1] += jitter

        if p.position == "OL":
            traj[post, 0] = p.x - 0.8 * np.tanh(t_post[post])
        elif p.position == "QB":
            traj[post, 0] = p.x - 3.0 * np.tanh(t_post[post] * 1.2)
        elif p.position == "RB":
            traj[post, 0] = p.x + 1.5 * t_post[post]
            traj[post, 1] = py + np.sign(py or 1) * 2.0 * np.tanh(t_post[post])
        else:
            speed = rng.uniform(7.0, 8.8)
            if p.route == "go":
                traj[post, 0] = p.x + speed * t_post[post]
                traj[post, 1] = py + rng.normal(0, 0.3) * t_post[post]
            elif p.route == "post":
                traj[post, 0] = p.x + speed * t_post[post]
                traj[post, 1] = py - np.sign(py or 1) * 2.6 * np.clip(t_post[post] - 1.0, 0, None)
            elif p.route == "corner":
                traj[post, 0] = p.x + speed * 0.9 * t_post[post]
                traj[post, 1] = py + np.sign(py or 1) * 2.8 * np.clip(t_post[post] - 1.0, 0, None)
            elif p.route in ("out", "in"):
                sign = 1 if p.route == "out" else -1
                traj[post, 0] = p.x + speed * np.clip(t_post[post], 0, 1.2) + 0.5 * np.clip(t_post[post] - 1.2, 0, None)
                traj[post, 1] = py + sign * np.sign(py or 1) * 6.0 * np.clip(t_post[post] - 1.2, 0, None)
            elif p.route == "slant":
                traj[post, 0] = p.x + speed * 0.75 * t_post[post]
                traj[post, 1] = py - np.sign(py or 1) * 3.2 * np.clip(t_post[post] - 0.4, 0, None)
            else:  # curl / checkdown
                hitch = np.clip(t_post[post], 0, 1.4)
                traj[post, 0] = p.x + speed * hitch - 1.0 * np.clip(t_post[post] - 1.4, 0, None)
                traj[post, 1] = py + rng.normal(0, 0.2)

        off_xy[p.label] = traj
        route_end[p.label] = (float(traj[-1, 0]), float(traj[-1, 1]))

    zone_bumper = _pick_zone_bumper(defense, motion)
    man_travels = motion is not None and rng.random() < MAN_TRAVEL_PROB
    zone_travels = motion is not None and rng.random() < ZONE_TRAVEL_PROB

    for d in defense:
        traj = np.zeros((N_FRAMES, 2), dtype=np.float32)
        traj[:, 0] = d.x
        traj[:, 1] = d.y

        jitter = rng.normal(0, 0.05, size=(SNAP_INDEX + 1, 2)).cumsum(axis=0) * 0.3
        traj[: SNAP_INDEX + 1] += jitter

        # Motion response. A man defender travels with the mover almost the whole
        # way; a zone defense just bumps the coverage over a yard or so. That gap
        # is the tell, and it is generated here rather than assumed downstream.
        if motion is not None:
            shift = motion.end_y - motion.start_y
            covers_mover = d.role == "man" and d.assignment == motion.label
            travels = (covers_mover and man_travels) or (d.label == zone_bumper and zone_travels)
            if travels:
                traj[: SNAP_INDEX + 1, 1] += motion_ramp * shift * rng.uniform(0.82, 1.0)
                traj[: SNAP_INDEX + 1, 0] += motion_ramp * rng.normal(0, 0.4)
            elif covers_mover or d.label == zone_bumper:
                traj[: SNAP_INDEX + 1, 1] += motion_ramp * np.sign(shift) * rng.uniform(0.6, 1.5)

        # Late rotation is a pre-snap movement, which is what makes it a tell you
        # can act on before the ball is snapped.
        if traits["late_rotation"] and d.position == "S":
            rot_start = np.searchsorted(TIME_GRID, -1.2)
            n_rot = SNAP_INDEX - rot_start + 1
            if n_rot > 0:
                ramp = np.linspace(0.0, 1.0, n_rot)
                if d.role == "deep_zone":
                    traj[rot_start : SNAP_INDEX + 1, 0] += ramp * rng.uniform(1.5, 3.5)
                    traj[rot_start : SNAP_INDEX + 1, 1] += ramp * (-d.y * 0.75)
                else:
                    traj[rot_start : SNAP_INDEX + 1, 0] -= ramp * rng.uniform(3.0, 5.5)
                    traj[rot_start : SNAP_INDEX + 1, 1] += ramp * rng.uniform(-2.0, 2.0)
            start_x = float(traj[SNAP_INDEX, 0])
            start_y = float(traj[SNAP_INDEX, 1])
        else:
            start_x, start_y = float(traj[SNAP_INDEX, 0]), float(traj[SNAP_INDEX, 1])

        if d.role == "blitz":
            qb_end = route_end.get("QB", (-8.0, 0.0))
            frac = np.clip(t_post[post] / 2.0, 0, 1)
            traj[post, 0] = start_x + (qb_end[0] - start_x) * frac
            traj[post, 1] = start_y + (qb_end[1] - start_y) * frac
        elif d.role == "man" and d.assignment and d.assignment in off_xy:
            target = off_xy[d.assignment]
            lag = int(round(0.25 * (N_FRAMES / (TIME_GRID[-1] - TIME_GRID[0]))))
            cushion = rng.uniform(0.8, 2.0)
            for i in range(SNAP_INDEX, N_FRAMES):
                src = max(SNAP_INDEX, i - lag)
                tx, ty = target[src, 0], target[src, 1]
                traj[i, 0] = tx + cushion
                traj[i, 1] = ty + rng.normal(0, 0.35)
            # Blend from the actual snap position so the track is continuous.
            blend = np.linspace(0, 1, N_FRAMES - SNAP_INDEX)
            traj[SNAP_INDEX:, 0] = (1 - blend) * start_x + blend * traj[SNAP_INDEX:, 0]
            traj[SNAP_INDEX:, 1] = (1 - blend) * start_y + blend * traj[SNAP_INDEX:, 1]
        else:
            lx, ly = _zone_landmark(d, coverage, rng)
            frac = np.clip(t_post[post] / 2.5, 0, 1)
            eased = frac ** 0.85
            traj[post, 0] = start_x + (lx - start_x) * eased
            traj[post, 1] = start_y + (ly - start_y) * eased

        traj[post] += rng.normal(0, 0.12, size=(int(post.sum()), 2))
        def_xy[d.label] = traj

    return off_xy, def_xy


@dataclass
class SyntheticConfig:
    seasons: tuple[int, ...] = (2025,)
    weeks: int = 12
    plays_per_game: int = 62
    seed: int = 7
    league: str = "ncaa"
    teams: list[TeamProfile] = dc_field(default_factory=default_teams)
    # Fraction of plays that get film-like degradation: missing defenders,
    # coordinate jitter. Keeps the models honest about what film will look like.
    degraded_fraction: float = 0.0


def generate_play(
    rng: np.random.Generator,
    defense_profile: TeamProfile,
    offense_team: str,
    season: int,
    week: int,
    play_index: int,
    league: str = "ncaa",
    degrade: bool = False,
) -> PlayTracks:
    field = get_field(league)

    down = int(rng.choice([1, 2, 3, 4], p=[0.42, 0.31, 0.24, 0.03]))
    distance = float(np.clip(rng.gamma(shape=3.0, scale=3.0), 1, 25)) if down > 1 else 10.0
    yardline = float(np.clip(rng.normal(45, 22), 2, 97))
    quarter = int(rng.integers(1, 5))
    seconds_remaining = float(rng.uniform(0, 900))
    score_margin = int(np.clip(rng.normal(0, 11), -35, 35))
    personnel = str(rng.choice(PERSONNEL_GROUPS))

    hash_choice = rng.choice(["left", "right", "middle"], p=[0.4, 0.4, 0.2])
    if hash_choice == "left":
        ball_y_abs = field.left_hash_y
    elif hash_choice == "right":
        ball_y_abs = field.right_hash_y
    else:
        ball_y_abs = FIELD_CENTER_Y
    play_direction = "right" if rng.random() < 0.5 else "left"
    ball_y_center = float(ball_y_abs - FIELD_CENTER_Y)
    if play_direction == "left":
        ball_y_center = -ball_y_center

    # Room to each side in the normalized frame (offense's right is +y).
    room_right = FIELD_WIDTH_YD / 2 - ball_y_center
    room_left = FIELD_WIDTH_YD / 2 + ball_y_center
    field_sign = 1 if room_right > room_left else -1

    base = defense_profile.coverage_distribution(week)
    mult = _situation_multipliers(down, distance, yardline)
    weights = np.array([base[c] * mult[c] for c in COVERAGES], dtype=np.float64)
    weights /= weights.sum()
    coverage = str(rng.choice(COVERAGES, p=weights))

    traits = {
        t: bool(rng.random() < defense_profile.trait_probability(t, coverage)) for t in TRAITS
    }

    offense = _place_offense(rng, personnel, room_left, room_right)
    # The defense aligns to the pre-motion formation, which is exactly why motion
    # forces it to declare itself.
    defense = _place_defense(rng, coverage, traits, offense, field_sign)
    motion = _choose_motion(rng, offense)
    off_xy, def_xy = _simulate(rng, offense, defense, coverage, traits, motion)

    players: list[PlayerTrack] = []
    for p in offense:
        traj = off_xy[p.label]
        players.append(
            PlayerTrack(
                track_id=f"O_{p.label}",
                side="offense",
                x=traj[:, 0],
                y=traj[:, 1],
                position=p.position,
                is_ball_carrier=p.position == "QB",
            )
        )
    for d in defense:
        traj = def_xy[d.label]
        players.append(
            PlayerTrack(
                track_id=f"D_{d.label}",
                side="defense",
                x=traj[:, 0],
                y=traj[:, 1],
                position=d.position,
                role=d.role,
            )
        )

    quality = TrackQuality(
        defenders_detected=sum(1 for p in players if p.side == "defense"),
        offense_detected=sum(1 for p in players if p.side == "offense"),
    )

    if degrade:
        players, quality = _degrade(rng, players)

    situation = Situation(
        down=down,
        distance=round(distance, 1),
        yardline=round(yardline, 1),
        quarter=quarter,
        seconds_remaining=round(seconds_remaining, 1),
        score_margin=score_margin,
        offense_personnel=personnel,
        ball_y_from_center=round(ball_y_center, 2),
        hash_side=field.hash_of(ball_y_abs),
        league=league,
    )

    return PlayTracks(
        play_id=f"{season}-W{week:02d}-{defense_profile.name.replace(' ', '')}-{play_index:04d}",
        source="synthetic",
        players=players,
        situation=situation,
        quality=quality,
        season=season,
        week=week,
        defense_team=defense_profile.name,
        offense_team=offense_team,
        coverage=coverage,
        coverage_source="synthetic",
        game_id=f"{season}-W{week:02d}-{defense_profile.name}-{offense_team}",
    )


def _degrade(
    rng: np.random.Generator, players: list[PlayerTrack]
) -> tuple[list[PlayerTrack], TrackQuality]:
    """Make a play look like it came off film rather than out of a chip."""
    notes: list[str] = []
    defenders = [p for p in players if p.side == "defense"]

    n_drop = int(rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15]))
    dropped = set()
    if n_drop:
        # Deep players are the ones that actually fall out of frame.
        deepest = sorted(defenders, key=lambda p: -np.nanmax(p.x))
        for p in deepest[:n_drop]:
            dropped.add(p.track_id)
        notes.append(f"{n_drop} defender(s) out of frame")

    kept = [p for p in players if p.track_id not in dropped]
    for p in kept:
        p.x = p.x + rng.normal(0, 0.35, size=p.x.shape).astype(np.float32)
        p.y = p.y + rng.normal(0, 0.35, size=p.y.shape).astype(np.float32)
        # Occlusion gaps.
        if rng.random() < 0.25:
            start = int(rng.integers(0, N_FRAMES - 6))
            span = int(rng.integers(2, 7))
            p.x[start : start + span] = np.nan
            p.y[start : start + span] = np.nan

    n_def = sum(1 for p in kept if p.side == "defense")
    quality = TrackQuality(
        defenders_detected=n_def,
        offense_detected=sum(1 for p in kept if p.side == "offense"),
        mean_detection_conf=float(rng.uniform(0.62, 0.93)),
        registration_error_yd=float(abs(rng.normal(0, 0.9))),
        frames_with_full_defense=float(rng.uniform(0.55, 1.0)) if n_drop else 1.0,
        notes=notes,
    )
    return kept, quality


def generate_season(config: SyntheticConfig | None = None) -> Iterator[PlayTracks]:
    config = config or SyntheticConfig()
    rng = np.random.default_rng(config.seed)
    names = [t.name for t in config.teams]

    for season in config.seasons:
        for week in range(1, config.weeks + 1):
            for defense_profile in config.teams:
                opponents = [n for n in names if n != defense_profile.name]
                offense_team = str(rng.choice(opponents))
                for i in range(config.plays_per_game):
                    degrade = rng.random() < config.degraded_fraction
                    yield generate_play(
                        rng,
                        defense_profile,
                        offense_team,
                        season,
                        week,
                        i,
                        league=config.league,
                        degrade=degrade,
                    )


def planted_tell_summary(config: SyntheticConfig | None = None) -> list[dict[str, Any]]:
    """Ground truth for the mining engine's evaluation."""
    config = config or SyntheticConfig()
    rows = []
    for team in config.teams:
        for tell in team.tells:
            rows.append(
                {
                    "team": team.name,
                    "trait": tell.trait,
                    "coverage": tell.coverage,
                    "team_rate": tell.probability,
                    "league_rate": tell.league_probability,
                    "effect_size": round(tell.effect_size, 3),
                    "note": tell.note,
                }
            )
    return rows
