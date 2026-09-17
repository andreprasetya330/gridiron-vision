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
from itertools import combinations
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


# Tracks that are neither team. A college crew plus whoever the homography drags
# in off the sideline; beyond this share it is a team, not the edge of the field.
OUTSIDER_MAX_SHARE = 0.25
# Below this many tracks there is not enough of the field on screen to trust an
# unsupervised split, so colour - which needs no population - takes over.
MIN_TRACKS_FOR_EMBEDDING = 8
# Tracks per cluster. Twenty-two players and a crew of seven divide into groups
# this size; asking for finer clusters splits teams by pose and lighting.
TRACKS_PER_CLUSTER = 8


def classify_tracks(
    profiles: dict[int, TrackProfile],
    embeddings: dict[int, np.ndarray] | None = None,
) -> TrackRoster:
    """Split tracks into non-players and two teams.

    Prefers SigLIP embeddings when they are available and falls back to colour.
    On Sugar Bowl film the difference is not subtle (scripts/embedding_study.py):
    colour split the two teams 8/23 and filed a referee and three people off the
    sideline as players, while the embedding split them 13/16 and put all four
    outsiders in their own cluster. A median colour cannot see a striped shirt;
    an embedding can.
    """
    if embeddings:
        roster = _classify_by_embedding(embeddings)
        if roster.team_of:
            return roster
    return _classify_by_colour(profiles)


def _classify_by_embedding(embeddings: dict[int, np.ndarray]) -> TrackRoster:
    """Cluster tracks in SigLIP space: two teams, plus everyone who is neither.

    Deliberately asks for more clusters than there are teams. On Sugar Bowl film
    the extra groups come out as the officiating crew and a clump of people the
    homography drags in off the sideline - two kinds of non-player that no
    number of team clusters can express.

    Each extra cluster is then judged against the distance between the two teams
    themselves. A cluster further from both teams than the teams are from each
    other is not a team's worth of players; anything nearer is a lighting or
    pose split and gets folded back into whichever team it sits beside. Using
    the teams' own separation as the yardstick keeps the test working on film
    where the uniforms are similar and film where they are not.
    """
    from sklearn.cluster import KMeans

    roster = TrackRoster()
    ids = sorted(embeddings)
    if len(ids) < MIN_TRACKS_FOR_EMBEDDING:
        roster.notes.append("too few tracks to separate teams by appearance")
        return roster

    vectors = np.stack([embeddings[i] for i in ids]).astype(np.float32)
    # A cluster should be a squad, not a fragment. Asking for more clusters than
    # the track count supports splits the teams themselves, at which point the
    # two largest clusters are half-teams and the yardstick collapses with them.
    k = int(np.clip(len(ids) // TRACKS_PER_CLUSTER, 2, 4))
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(vectors)

    sizes = {int(c): int((labels == c).sum()) for c in {int(v) for v in labels.tolist()}}
    if len(sizes) < 2:
        roster.notes.append("appearance clustering collapsed; falling back to colour")
        return roster
    centres = {c: vectors[labels == c].mean(axis=0) for c in sizes}

    ranked = sorted(sizes, key=lambda c: -sizes[c])
    team_a, team_b = ranked[0], ranked[1]
    yardstick = float(np.linalg.norm(centres[team_a] - centres[team_b]))

    team_of: dict[int, int] = {}
    outsiders: set[int] = set()
    for cluster in sizes:
        members = [ids[i] for i, label in enumerate(labels) if label == cluster]
        if cluster in (team_a, team_b):
            side = 0 if cluster == team_a else 1
            team_of.update(dict.fromkeys(members, side))
            continue
        to_a = float(np.linalg.norm(centres[cluster] - centres[team_a]))
        to_b = float(np.linalg.norm(centres[cluster] - centres[team_b]))
        if min(to_a, to_b) > yardstick:
            outsiders.update(members)
        else:
            team_of.update(dict.fromkeys(members, 0 if to_a < to_b else 1))

    if len(outsiders) > max(1, int(OUTSIDER_MAX_SHARE * len(ids))):
        # Discarding a quarter of the film means the clustering found something
        # other than teams-plus-crew. Keep everyone and let colour decide.
        roster.notes.append("appearance clustering flagged too many non-players")
        return roster

    roster.officials = outsiders
    roster.team_of = team_of
    if outsiders:
        roster.notes.append(
            f"{len(outsiders)} track(s) look like neither team (officials or sideline)"
        )

    sides: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for tid, team in team_of.items():
        sides[team].append(embeddings[tid])
    if sides[0] and sides[1]:
        a = np.mean(np.stack(sides[0]), axis=0)
        b = np.mean(np.stack(sides[1]), axis=0)
        cosine = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6))
        # Reported on the scale the colour path uses so `confidence` means one
        # thing downstream regardless of which classifier produced the roster.
        roster.separation = float(np.clip((1.0 - cosine) * 60.0, 0.0, 60.0))
    return roster


def _classify_by_colour(profiles: dict[int, TrackProfile]) -> TrackRoster:
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


def build_track_embeddings(
    frames_meta: list[dict[str, Any]],
    raw_frames: list[np.ndarray],
    keep_ids: set[int] | None = None,
    frames: set[int] | None = None,
    stride: int = 2,
    max_per_track: int = 24,
    min_samples: int = 3,
) -> dict[int, np.ndarray]:
    """One mean SigLIP vector per track, or an empty dict if unavailable.

    Every crop of a track is embedded and averaged rather than embedding a
    single representative frame, for the same reason the colour profile takes a
    median: any one frame can catch a player mid-turn, occluded, or blurred.
    """
    from gridiron.vision.embed import SiglipEmbedder, available, player_crop

    if not available():
        return {}

    crops: dict[int, list[np.ndarray]] = {}
    for j in range(0, min(len(frames_meta), len(raw_frames)), max(1, stride)):
        if frames is not None and j not in frames:
            continue
        for tracked in frames_meta[j].get("tracked") or []:
            tid = tracked.track_id
            if keep_ids is not None and tid not in keep_ids:
                continue
            bucket = crops.setdefault(tid, [])
            if len(bucket) < max_per_track:
                bucket.append(player_crop(raw_frames[j], tracked.detection))

    track_ids = [tid for tid, items in crops.items() if len(items) >= min_samples]
    if len(track_ids) < 8:
        return {}

    flat: list[np.ndarray] = []
    owner: list[int] = []
    for tid in track_ids:
        for crop in crops[tid]:
            if crop.size >= 48:
                flat.append(crop)
                owner.append(tid)
    if not flat:
        return {}

    try:
        features = SiglipEmbedder().embed(flat)
    except Exception:
        # A missing checkpoint or no network should cost accuracy, not the run.
        return {}

    owners = np.array(owner)
    out: dict[int, np.ndarray] = {}
    for tid in track_ids:
        rows = features[owners == tid]
        if rows.size == 0:
            continue
        mean = rows.mean(axis=0)
        out[tid] = mean / max(float(np.linalg.norm(mean)), 1e-6)
    return out


# A fragment and its continuation, at 30fps and in yards. A player who vanishes
# behind a pile for half a second has not crossed the field in the meantime.
MERGE_MAX_GAP_S = 0.5
MERGE_MAX_DISTANCE_YD = 6.0
MERGE_MIN_SIMILARITY = 0.90


def merge_fragments(
    frames_meta: list[dict[str, Any]],
    embeddings: dict[int, np.ndarray],
    roster: TrackRoster,
    fps: float,
) -> dict[int, int]:
    """Stitch track fragments back into players. Returns old id -> kept id.

    A box tracker loses a player behind a pile and calls him someone new when he
    comes out; on Sugar Bowl film the offense ends up with fifteen ids for eleven
    players, and the short fragments then fail the minimum-observations rule and
    drop out of the play entirely.

    Roboflow's pipeline avoids this with SAM2, which is the honest fix and also
    the slowest stage they have. Since the appearance vectors are already
    computed for team classification, the cheap version is to rejoin fragments
    that look alike and could plausibly be the same body.

    Appearance alone is not enough - two linemen in one uniform are nearly
    identical vectors. What makes it safe is requiring the fragments to be the
    same team, never overlap in time, and end and start within a few yards of
    each other. Measured on this clip, every cross-team false match sat more
    than fifteen yards apart, and every true match inside four.
    """
    spans: dict[int, list[int]] = {}
    endpoints: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
    for j, meta in enumerate(frames_meta):
        for tid, position in meta["field_positions"].items():
            if tid not in spans:
                spans[tid] = []
                endpoints[tid] = (position, position)
            spans[tid].append(j)
            endpoints[tid] = (endpoints[tid][0], position)

    max_gap = max(1, int(round(MERGE_MAX_GAP_S * fps)))
    candidates: list[tuple[float, int, int]] = []
    for a, b in combinations(sorted(spans), 2):
        if a not in embeddings or b not in embeddings:
            continue
        if roster.team_of.get(a) != roster.team_of.get(b):
            continue
        first, second = (a, b) if spans[a][-1] <= spans[b][0] else (b, a)
        gap = spans[second][0] - spans[first][-1]
        if not 0 < gap <= max_gap:
            continue
        tail, head = endpoints[first][1], endpoints[second][0]
        if float(np.hypot(tail[0] - head[0], tail[1] - head[1])) > MERGE_MAX_DISTANCE_YD:
            continue
        similarity = float(np.dot(embeddings[first], embeddings[second]))
        if similarity >= MERGE_MIN_SIMILARITY:
            candidates.append((similarity, first, second))

    # Best match first, so a fragment joins the player it most resembles rather
    # than whichever id happened to be compared first.
    candidates.sort(reverse=True)
    parent = {tid: tid for tid in spans}
    extent = {tid: (spans[tid][0], spans[tid][-1]) for tid in spans}

    def root(tid: int) -> int:
        while parent[tid] != tid:
            parent[tid] = parent[parent[tid]]
            tid = parent[tid]
        return tid

    for _similarity, first, second in candidates:
        ra, rb = root(first), root(second)
        if ra == rb:
            continue
        (a0, a1), (b0, b1) = extent[ra], extent[rb]
        if a0 <= b1 and b0 <= a1:
            # The two chains are on screen at once, so they are two players.
            continue
        keep, drop = (ra, rb) if a0 <= b0 else (rb, ra)
        parent[drop] = keep
        extent[keep] = (min(a0, b0), max(a1, b1))

    return {tid: root(tid) for tid in spans if root(tid) != tid}


def apply_merges(frames_meta: list[dict[str, Any]], merges: dict[int, int]) -> None:
    """Rewrite every reference to a merged track so it names the kept track."""
    if not merges:
        return
    for meta in frames_meta:
        for tracked in meta["tracked"]:
            if tracked.track_id in merges:
                tracked.track_id = merges[tracked.track_id]
        positions = {}
        for tid, position in meta["field_positions"].items():
            positions[merges.get(tid, tid)] = position
        meta["field_positions"] = positions


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
