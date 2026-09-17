"""Who is on the field, which team they are on, and what the defenders are doing.

The important decision here is *when* to decide. A 1080p All-22 frame gives a
player a torso roughly 18x25 pixels, which is not enough pixels to read the
stripes on an official's shirt - measured on Sugar Bowl film, the stripe signal
is indistinguishable from the grass bleeding around a player's shoulders. Any
per-frame appearance test at that scale is a coin flip, and a coin flip applied
every frame deletes real players at random, which fragments their tracks.

So nothing is classified per frame. Appearance is accumulated over a track's
entire life and the decision is made once, on the median, where the signal is
stable. What survives at this resolution is gross colour: on that same film
(measured in scripts/appearance_study.py) the two officials sit at Lab L=24 and
L=30 while every player sits between 75 and 208, because a black-and-white
striped shirt blurs to dark grey while both teams wear a bright jersey.

Officials are therefore found as a *small, much darker* cluster rather than by a
fixed luminance cutoff. The size test is what keeps a team in black jerseys from
being deleted wholesale: eleven dark tracks are a team, two are the crew.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gridiron.vision.teams import jersey_feature

# Luminance gap that has to open up below the darkest jersey before a track is
# called an official. Measured on Sugar Bowl film the crew runs from L=24 to
# L=36 and the darkest player sits at L=46, so the real gap is 10 while players
# are spaced ~2 apart. Absolute rather than relative on purpose: a gap measured
# as a multiple of typical spacing shrinks and grows with how many tracks the
# clip happens to produce, which is not a property of anyone's uniform.
OFFICIAL_DARK_MARGIN = 8.0
# A gap alone is not enough: on a clip with no crew in frame, the darkest player
# always has *someone* below him. An officiating shirt is near-black, so require
# absolute darkness too. Measured crew runs L=24 to L=47.
OFFICIAL_MAX_LUMA = 70.0
# A college crew is seven or eight officials, and most are in a wide All-22
# frame. Beyond that share it is a team wearing dark, not the crew.
OFFICIAL_MAX_SHARE = 0.30
OFFICIAL_MAX_COUNT = 8


@dataclass
class TrackProfile:
    """One track's appearance, aggregated over every frame it was seen in."""

    track_id: int
    samples: int
    lab: np.ndarray  # median grass-masked jersey colour
    height_px: float

    @property
    def luma(self) -> float:
        return float(self.lab[0])


@dataclass
class TrackRoster:
    """The per-track verdict: official, or a player on one of two teams."""

    officials: set[int] = field(default_factory=set)
    team_of: dict[int, int] = field(default_factory=dict)
    centers: np.ndarray | None = None  # chroma (Lab a/b) centres, not full Lab
    separation: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        # Separation is measured in chroma (Lab a/b) units, where the gap
        # between two real uniforms is ~20 and anything under ~6 means the two
        # teams are wearing the same hue and every downstream number is suspect.
        return float(np.clip((self.separation - 6.0) / 14.0, 0.0, 1.0))

    def team(self, track_id: int) -> int | None:
        return self.team_of.get(track_id)


def build_track_profiles(
    frames_meta: list[dict[str, Any]],
    raw_frames: list[np.ndarray],
    stride: int = 2,
    min_samples: int = 3,
    keep_ids: set[int] | None = None,
    frames: set[int] | None = None,
) -> dict[int, TrackProfile]:
    """Median jersey colour per track across the whole clip.

    `stride` subsamples frames because consecutive frames of the same player are
    nearly identical; every second frame is plenty to build a stable median.

    `keep_ids` should be the tracks the homography puts on the field, and
    `frames` the window around the snap. Both exist to keep the sideline out.
    Profiling the whole clip lets the crowd in twice over: a wide shot catches
    the stands, and once the camera follows the play it fills the frame with the
    bench. A bench full of dark team coats fills in the luminance gap that
    separates the officials from the players, and the crew stops being findable.
    """
    samples: dict[int, list[np.ndarray]] = {}
    heights: dict[int, list[float]] = {}
    for j in range(0, min(len(frames_meta), len(raw_frames)), max(1, stride)):
        if frames is not None and j not in frames:
            continue
        frame = raw_frames[j]
        for tracked in frames_meta[j].get("tracked") or []:
            if keep_ids is not None and tracked.track_id not in keep_ids:
                continue
            feature = jersey_feature(frame, tracked.detection)
            if not np.all(np.isfinite(feature)):
                continue
            samples.setdefault(tracked.track_id, []).append(feature)
            heights.setdefault(tracked.track_id, []).append(tracked.detection.height)

    profiles: dict[int, TrackProfile] = {}
    for track_id, feats in samples.items():
        if len(feats) < min_samples:
            continue
        stack = np.stack(feats)
        profiles[track_id] = TrackProfile(
            track_id=track_id,
            samples=len(feats),
            lab=np.median(stack, axis=0).astype(np.float32),
            height_px=float(np.median(heights[track_id])),
        )
    return profiles


def classify_tracks(profiles: dict[int, TrackProfile]) -> TrackRoster:
    """Split tracks into officials and two teams using track-level colour."""
    from sklearn.cluster import KMeans

    roster = TrackRoster()
    if len(profiles) < 6:
        roster.notes.append("too few tracks to separate teams by colour")
        return roster

    ids = sorted(profiles)
    roster.officials = _find_officials(ids, profiles, roster.notes)
    player_ids = [i for i in ids if i not in roster.officials]
    if len(player_ids) < 6:
        roster.notes.append("too few player tracks left to separate teams")
        return roster

    # Teams are separated on chroma alone, with luminance deliberately dropped.
    # Half this field is in stadium shadow, so L spans 25-252 while jersey hue
    # spans about +/-20. Clustering on all three channels therefore splits sun
    # from shade and calls it a team - which is exactly how a defense ends up
    # with four players on it. Hue is the thing a uniform actually fixes.
    player_features = np.stack([profiles[i].lab[1:] for i in player_ids])
    teams = KMeans(n_clusters=2, n_init=10, random_state=0).fit(player_features)
    roster.team_of = {
        tid: int(label) for tid, label in zip(player_ids, teams.labels_, strict=True)
    }
    roster.centers = teams.cluster_centers_
    roster.separation = float(
        np.linalg.norm(teams.cluster_centers_[0] - teams.cluster_centers_[1])
    )
    if roster.confidence < 0.35:
        roster.notes.append(
            f"jersey colours are only {roster.separation:.0f} Lab units apart; team "
            "assignment is unreliable on this film"
        )
    return roster


def _find_officials(
    ids: list[int], profiles: dict[int, TrackProfile], notes: list[str]
) -> set[int]:
    """Officials are the tracks separated from the players by a luminance gap.

    Not a cluster and not a fixed cutoff. Two officials among sixty tracks will
    never win their own KMeans cluster, and a fixed cutoff breaks the moment a
    team wears navy. What is stable is the *gap*: a black-and-white shirt blurs
    to dark grey at this resolution, so officials sit well below the darkest
    jersey with nothing in between.

    The size cap is the safety net. Eleven dark tracks are a team in black; two
    are the crew.
    """
    if len(ids) < 8:
        return set()

    by_luma = sorted(ids, key=lambda t: profiles[t].luma)
    lumas = [profiles[t].luma for t in by_luma]
    spacing = np.diff(lumas)
    limit = min(OFFICIAL_MAX_COUNT, max(1, int(OFFICIAL_MAX_SHARE * len(ids))))
    candidate_gaps = spacing[:limit]
    if candidate_gaps.size == 0:
        return set()

    cut = int(np.argmax(candidate_gaps))
    best_gap = float(candidate_gaps[cut])
    if best_gap < OFFICIAL_DARK_MARGIN:
        return set()
    if lumas[cut] > OFFICIAL_MAX_LUMA:
        return set()

    officials = set(by_luma[: cut + 1])
    notes.append(
        f"{len(officials)} track(s) read as officials: a {best_gap:.0f} Lab-unit gap "
        f"separates them from the darkest jersey"
    )
    return officials


def assign_defensive_roles(players: list[Any], snap_index: int = 20) -> None:
    """Label rushers vs coverage from alignment at the snap, then the rush.

    A rusher is on the line, inside the tackle box, or someone who closes to the
    LOS after the ball is snapped. That is the player who used to get painted as
    a coverage defender because 'defense' was the only tag the overlay had.
    """
    defenders = [p for p in players if getattr(p, "side", None) == "defense"]
    if not defenders:
        return
    snaps: list[tuple[Any, float, float]] = []
    for player in defenders:
        if snap_index >= len(player.x):
            continue
        x = float(player.x[snap_index])
        y = float(player.y[snap_index])
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        snaps.append((player, x, y))
    if not snaps:
        return

    widths = [abs(y) for _p, x, y in snaps if 0.0 <= x <= 5.0]
    box = float(np.clip(max(widths) if widths else 8.0, 6.0, 10.0))
    late = min(snap_index + 15, max(len(p.x) for p, _x, _y in snaps) - 1)

    for player, x, y in snaps:
        x_late = float(player.x[late]) if late < len(player.x) else float("nan")
        crashed = np.isfinite(x_late) and x < 8.0 and x_late < 2.5 and abs(y) <= box + 2.0
        if crashed or (x < 4.5 and abs(y) <= box):
            player.role = "blitz"
        elif x >= 9.0:
            player.role = "deep_zone"
        elif abs(y) >= 8.0:
            player.role = "man"
        else:
            player.role = "underneath_zone"


def near_snap(player: Any, snap_index: int, window: int = 5) -> bool:
    lo = max(0, snap_index - window)
    hi = min(len(player.x), snap_index + window + 1)
    return bool(np.isfinite(player.x[lo:hi]).any())


def correct_sides_by_alignment(players: list[Any], snap_index: int = 20) -> None:
    """Jersey colour is necessary but not sufficient.

    A safety whose colour landed in the wrong cluster becomes a 'receiver'.
    Anyone standing clearly on the defensive side of the ball at the snap is a
    defender, colour be damned.
    """
    for player in players:
        if snap_index >= len(player.x):
            continue
        x = float(player.x[snap_index])
        if not np.isfinite(x):
            continue
        if player.side == "offense" and x > 3.5:
            player.side = "defense"
            if player.track_id.startswith("O_"):
                player.track_id = "D_" + player.track_id[2:]
        elif player.side == "defense" and x < -2.0:
            player.side = "offense"
            player.role = None
            if player.track_id.startswith("D_"):
                player.track_id = "O_" + player.track_id[2:]


def cap_roster(players: list[Any], snap_index: int = 20, per_side: int = 11) -> list[Any]:
    """Keep the eleven on each side nearest a typical alignment landmark."""
    kept: list[Any] = []
    targets = {"defense": (5.0, 0.0), "offense": (-2.0, 0.0)}
    for side, target in targets.items():
        group: list[tuple[Any, float]] = []
        for player in players:
            if player.side != side:
                continue
            if snap_index >= len(player.x):
                continue
            x = float(player.x[snap_index])
            y = float(player.y[snap_index])
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            group.append((player, (x - target[0]) ** 2 + (y - target[1]) ** 2))
        group.sort(key=lambda item: item[1])
        kept.extend(p for p, _d in group[:per_side])
    return kept


def finalize_film_players(players: list[Any], snap_index: int = 20) -> list[Any]:
    players = [p for p in players if near_snap(p, snap_index)]
    correct_sides_by_alignment(players, snap_index)
    players = cap_roster(players, snap_index)
    assign_defensive_roles(players, snap_index)
    return players
