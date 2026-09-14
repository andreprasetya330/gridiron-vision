"""The pre-snap cue vocabulary.

Two hard rules shape this module:

1. **Everything is defined geometrically, never by position label.** Film gives you
   eleven moving dots and no depth chart. If "safety depth" meant "the depth of the
   player labeled FS", every cue would evaporate the moment you pointed the system
   at real video. So a safety is "a defender deeper than 10 yards", a corner is "a
   perimeter defender aligned over a wide receiver", and the same code runs on Big
   Data Bowl and on a Hudl cutup.

2. **Every cue has both a numeric and a categorical form.** The random forest wants
   numbers. The tell miner wants discrete buckets, because "P(Cover 3 | safety at
   12.4 yards)" is not a testable proposition but "P(Cover 3 | two-high shell)" is.

Cues are measured at the snap and over the 1.5 seconds before it. Nothing after the
snap is allowed in here - a tell you can only see post-snap is not a tell, it is a
replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gridiron.tracking.schema import PlayTracks, PlayerTrack, frame_at

SNAP_T = 0.0
EARLY_T = -1.5

DEEP_THRESHOLD_YD = 10.0
BOX_DEPTH_YD = 5.0
BOX_WIDTH_YD = 8.0
LOS_DEPTH_YD = 2.0
PRESS_DEPTH_YD = 3.0
WIDE_RECEIVER_Y = 5.0

CUE_KEYS: list[str] = [
    "shell",
    "deep_defender_count",
    "deepest_depth_band",
    "safety_depth_band",
    "safety_split_band",
    "late_rotation",
    "rotation_direction",
    "press_corners",
    "corner_leverage",
    "corner_depth_band",
    "box_count_band",
    "los_defenders_band",
    "slot_defender_depth_band",
    "defense_spread_band",
    "linebacker_depth_band",
    "motion_response",
    "field_safety_bias",
    "down",
    "distance_band",
    "field_zone",
    "hash_side",
    "personnel",
    "score_state",
]

NUMERIC_CUE_KEYS: list[str] = [
    "deep_defender_count_n",
    "deepest_depth_n",
    "safety_depth_mean_n",
    "safety_depth_max_n",
    "safety_split_n",
    "safety_y_asymmetry_n",
    "rotation_magnitude_n",
    "rotation_lateral_n",
    "rotation_depth_n",
    "press_corner_count_n",
    "corner_depth_mean_n",
    "corner_leverage_mean_n",
    "box_count_n",
    "los_defender_count_n",
    "slot_defender_depth_n",
    "defense_spread_n",
    "defense_width_n",
    "linebacker_depth_mean_n",
    "motion_response_n",
    "field_safety_bias_n",
    "defenders_observed_n",
    "down_n",
    "distance_n",
    "yardline_n",
    "score_margin_n",
]


@dataclass
class CueSet:
    play_id: str
    categorical: dict[str, str] = field(default_factory=dict)
    numeric: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"play_id": self.play_id}
        row.update(self.categorical)
        row.update(self.numeric)
        row.update(self.meta)
        return row


def _xy(track: PlayerTrack, t: float) -> tuple[float, float]:
    return track.at(t)


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if np.isfinite(v)]


def _band(value: float, edges: list[float], labels: list[str]) -> str:
    if not np.isfinite(value):
        return "unknown"
    for edge, label in zip(edges, labels[:-1]):
        if value < edge:
            return label
    return labels[-1]


def _defender_snapshot(play: PlayTracks, t: float) -> list[tuple[PlayerTrack, float, float]]:
    out = []
    for d in play.defense:
        x, y = _xy(d, t)
        if np.isfinite(x) and np.isfinite(y):
            out.append((d, x, y))
    return out


def _receiver_positions(play: PlayTracks, t: float) -> list[float]:
    ys = []
    for p in play.offense:
        x, y = _xy(p, t)
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        # Linemen sit inside; a receiver is anyone split out from the formation.
        if abs(y) >= WIDE_RECEIVER_Y:
            ys.append(y)
    return sorted(ys)


def _find_corners(
    snap: list[tuple[PlayerTrack, float, float]], receivers: list[float]
) -> list[tuple[PlayerTrack, float, float, float]]:
    """The outside-most defensive back on each side, with the receiver he is over.

    Identified by the corner's own alignment rather than by proximity to a
    receiver. Proximity looks natural and fails in two ways that matter: the two
    deep safeties in a two-high shell sit laterally near a wide split and get
    mistaken for corners, and pre-snap motion moves the widest receiver away from
    the corner covering him, so a zone corner who correctly stays put stops being
    recognised at all. Both failures land hardest on Cover 2, which is exactly
    where press depth is the most informative cue there is.

    Returns (track, depth, y, nearest receiver y) per side.
    """
    usable = [item for item in snap if item[1] <= 14.0]
    if len(usable) >= 5:
        # Dropping the two deepest is a scheme-independent way of saying "not a
        # safety": it holds for one-high, two-high, quarters and prevent alike,
        # and needs no knowledge of the coverage, which is the thing being
        # predicted.
        usable = sorted(usable, key=lambda item: -item[1])[2:]

    out: list[tuple[PlayerTrack, float, float, float]] = []
    for side in (-1.0, 1.0):
        on_side = [item for item in usable if np.sign(item[2]) == side and abs(item[2]) >= 4.0]
        if not on_side:
            continue
        corner = max(on_side, key=lambda item: abs(item[2]))
        side_receivers = [y for y in receivers if np.sign(y) == side]
        if not side_receivers:
            continue
        target_y = min(side_receivers, key=lambda y: abs(y - corner[2]))
        out.append((corner[0], corner[1], corner[2], target_y))
    return out


def extract_cues(play: PlayTracks) -> CueSet:
    snap = _defender_snapshot(play, SNAP_T)
    early = {d.track_id: _xy(d, EARLY_T) for d in play.defense}
    receivers = _receiver_positions(play, SNAP_T)

    cat: dict[str, str] = {}
    num: dict[str, float] = {}

    num["defenders_observed_n"] = float(len(snap))

    depths = [x for _, x, _ in snap]
    lateral = [y for _, _, y in snap]

    # --- Shell: how many defenders are playing deep, and how deep ------------
    deep = [(d, x, y) for d, x, y in snap if x >= DEEP_THRESHOLD_YD]
    deep_count = len(deep)
    num["deep_defender_count_n"] = float(deep_count)
    num["deepest_depth_n"] = float(max(depths)) if depths else np.nan

    cat["deep_defender_count"] = str(min(deep_count, 3)) if deep_count < 3 else "3+"
    cat["shell"] = {0: "0-high", 1: "1-high", 2: "2-high"}.get(deep_count, "3+-high")
    cat["deepest_depth_band"] = _band(
        num["deepest_depth_n"], [9.0, 12.0, 15.0, 19.0], ["<9", "9-12", "12-15", "15-19", "19+"]
    )

    if deep:
        deep_depths = [x for _, x, _ in deep]
        deep_ys = [y for _, _, y in deep]
        num["safety_depth_mean_n"] = float(np.mean(deep_depths))
        num["safety_depth_max_n"] = float(np.max(deep_depths))
        num["safety_split_n"] = float(max(deep_ys) - min(deep_ys)) if len(deep) > 1 else 0.0
        num["safety_y_asymmetry_n"] = float(np.mean(deep_ys))
    else:
        # No deep defender at all is itself a strong signal, so fall back to the
        # deepest two rather than emitting NaN and losing the play.
        fallback = sorted(snap, key=lambda item: -item[1])[:2]
        num["safety_depth_mean_n"] = float(np.mean([x for _, x, _ in fallback])) if fallback else np.nan
        num["safety_depth_max_n"] = float(max(depths)) if depths else np.nan
        num["safety_split_n"] = 0.0
        num["safety_y_asymmetry_n"] = float(np.mean([y for _, _, y in fallback])) if fallback else 0.0

    cat["safety_depth_band"] = _band(
        num["safety_depth_mean_n"], [7.0, 10.0, 13.0, 16.0], ["<7", "7-10", "10-13", "13-16", "16+"]
    )
    cat["safety_split_band"] = _band(
        num["safety_split_n"], [2.0, 10.0, 18.0], ["stacked", "narrow", "wide", "very-wide"]
    )

    # Which way the deep defenders are shaded relative to the wide side.
    field_sign = np.sign(play.situation.ball_y_from_center) or 1.0
    # The wide side is opposite the ball's offset from center.
    wide_sign = -field_sign
    bias = num["safety_y_asymmetry_n"] * wide_sign
    num["field_safety_bias_n"] = float(bias) if np.isfinite(bias) else 0.0
    cat["field_safety_bias"] = _band(
        num["field_safety_bias_n"], [-2.0, 2.0], ["boundary", "balanced", "field"]
    )

    # --- Late rotation -------------------------------------------------------
    rot_mag = rot_lat = rot_depth = 0.0
    for d, x, y in snap:
        x0, y0 = early.get(d.track_id, (np.nan, np.nan))
        if not (np.isfinite(x0) and np.isfinite(y0)):
            continue
        if max(x, x0) < 6.0:
            continue  # only secondary movement counts as rotation
        dx, dy = x - x0, y - y0
        mag = float(np.hypot(dx, dy))
        if mag > rot_mag:
            rot_mag, rot_lat, rot_depth = mag, float(dy), float(dx)

    num["rotation_magnitude_n"] = rot_mag
    num["rotation_lateral_n"] = rot_lat
    num["rotation_depth_n"] = rot_depth
    cat["late_rotation"] = "yes" if rot_mag >= 2.5 else "no"
    if rot_mag < 2.5:
        cat["rotation_direction"] = "none"
    elif abs(rot_depth) > abs(rot_lat):
        cat["rotation_direction"] = "down" if rot_depth < 0 else "back"
    else:
        cat["rotation_direction"] = "to-field" if rot_lat * wide_sign > 0 else "to-boundary"

    # --- Corners -------------------------------------------------------------
    corners = _find_corners(snap, receivers)

    press_count = sum(1 for _, x, _, _ in corners if x <= PRESS_DEPTH_YD)
    num["press_corner_count_n"] = float(press_count)
    num["corner_depth_mean_n"] = (
        float(np.mean([x for _, x, _, _ in corners])) if corners else np.nan
    )
    # Leverage sign: positive means the defender is outside the receiver, toward
    # the sideline, which is how zone corners protect the boundary.
    leverages = []
    for _, _, y, target_y in corners:
        outside = (abs(y) - abs(target_y)) * (1.0 if target_y != 0 else 1.0)
        leverages.append(outside)
    num["corner_leverage_mean_n"] = float(np.mean(leverages)) if leverages else np.nan

    cat["press_corners"] = str(press_count) if press_count <= 1 else "2+"
    cat["corner_depth_band"] = _band(
        num["corner_depth_mean_n"], [3.0, 5.5, 8.0], ["press", "soft", "off", "deep-off"]
    )
    cat["corner_leverage"] = _band(
        num["corner_leverage_mean_n"], [-0.4, 0.4], ["inside", "head-up", "outside"]
    )

    # --- Box and front -------------------------------------------------------
    box = [(d, x, y) for d, x, y in snap if x <= BOX_DEPTH_YD and abs(y) <= BOX_WIDTH_YD]
    los = [(d, x, y) for d, x, y in snap if x <= LOS_DEPTH_YD]
    num["box_count_n"] = float(len(box))
    num["los_defender_count_n"] = float(len(los))
    cat["box_count_band"] = _band(num["box_count_n"], [6.0, 7.0, 8.0], ["light", "6", "7", "8+"])
    cat["los_defenders_band"] = _band(
        num["los_defender_count_n"], [4.0, 5.0], ["3-", "4", "5+"]
    )

    linebackers = [
        (d, x, y) for d, x, y in snap if 2.0 < x < DEEP_THRESHOLD_YD and abs(y) <= BOX_WIDTH_YD
    ]
    num["linebacker_depth_mean_n"] = (
        float(np.mean([x for _, x, _ in linebackers])) if linebackers else np.nan
    )
    cat["linebacker_depth_band"] = _band(
        num["linebacker_depth_mean_n"], [3.5, 5.0, 6.5], ["walked-up", "tight", "normal", "deep"]
    )

    # --- Slot / nickel -------------------------------------------------------
    slot_ys = [y for y in receivers if abs(y) < 12.0]
    slot_depth = np.nan
    if slot_ys:
        target = slot_ys[0] if abs(slot_ys[0]) < abs(slot_ys[-1]) else slot_ys[-1]
        candidates = [
            (x, abs(y - target)) for _, x, y in snap if abs(y - target) <= 4.0 and x <= 10.0
        ]
        if candidates:
            slot_depth = float(min(candidates, key=lambda c: c[1])[0])
    num["slot_defender_depth_n"] = slot_depth
    cat["slot_defender_depth_band"] = _band(
        slot_depth, [3.0, 5.0, 7.0], ["creeping", "tight", "normal", "off"]
    )

    # --- Spread --------------------------------------------------------------
    finite_lateral = _finite(lateral)
    num["defense_spread_n"] = float(np.std(finite_lateral)) if finite_lateral else np.nan
    num["defense_width_n"] = (
        float(max(finite_lateral) - min(finite_lateral)) if finite_lateral else np.nan
    )
    cat["defense_spread_band"] = _band(
        num["defense_spread_n"], [7.0, 10.0, 13.0], ["compressed", "normal", "wide", "very-wide"]
    )

    # --- Motion response -----------------------------------------------------
    mover = None
    best_motion = 0.0
    for p in play.offense:
        x0, y0 = _xy(p, EARLY_T)
        x1, y1 = _xy(p, SNAP_T)
        if not all(np.isfinite(v) for v in (x0, y0, x1, y1)):
            continue
        travel = abs(y1 - y0)
        if travel > best_motion:
            best_motion, mover = travel, (y0, y1)

    if mover is None or best_motion < 3.0:
        cat["motion_response"] = "no-motion"
        num["motion_response_n"] = np.nan
    else:
        y0, y1 = mover
        direction = np.sign(y1 - y0)
        traveled = 0.0
        for d, x, y in snap:
            x0, y0d = early.get(d.track_id, (np.nan, np.nan))
            if not np.isfinite(y0d):
                continue
            if np.sign(y - y0d) == direction and abs(y - y0d) >= 2.0 and x <= 10.0:
                traveled = max(traveled, abs(y - y0d))
        num["motion_response_n"] = float(traveled)
        # A defender chasing motion across the formation is the loudest man
        # coverage indicator that exists pre-snap.
        cat["motion_response"] = "chased" if traveled >= 0.55 * best_motion else "zone-bump"

    # --- Situation -----------------------------------------------------------
    s = play.situation
    num["down_n"] = float(s.down) if s.down else np.nan
    num["distance_n"] = float(s.distance) if s.distance is not None else np.nan
    num["yardline_n"] = float(s.yardline) if s.yardline is not None else np.nan
    num["score_margin_n"] = float(s.score_margin) if s.score_margin is not None else np.nan

    cat["down"] = str(s.down) if s.down else "unknown"
    cat["distance_band"] = _band(
        num["distance_n"], [3.0, 7.0, 11.0], ["short", "medium", "long", "very-long"]
    )
    cat["field_zone"] = _band(
        num["yardline_n"], [20.0, 50.0, 80.0], ["backed-up", "own-territory", "plus-territory", "red-zone"]
    )
    cat["hash_side"] = s.hash_side or "unknown"
    cat["personnel"] = s.offense_personnel or "unknown"
    cat["score_state"] = _band(
        num["score_margin_n"], [-8.0, -1.0, 8.0], ["trailing-big", "trailing", "close", "leading-big"]
    )

    return CueSet(
        play_id=play.play_id,
        categorical=cat,
        numeric=num,
        meta={
            "season": play.season,
            "week": play.week,
            "game_id": play.game_key,
            "defense_team": play.defense_team,
            "offense_team": play.offense_team,
            "coverage": play.coverage,
            "quality_score": round(play.quality.score, 3),
            "usable": play.quality.usable,
            "source": play.source,
        },
    )


def cues_to_frame(plays: list[PlayTracks]) -> pd.DataFrame:
    rows = [extract_cues(play).as_row() for play in plays]
    df = pd.DataFrame(rows)
    for key in CUE_KEYS:
        if key not in df.columns:
            df[key] = "unknown"
        df[key] = df[key].fillna("unknown").astype(str)
    for key in NUMERIC_CUE_KEYS:
        if key not in df.columns:
            df[key] = np.nan
        df[key] = pd.to_numeric(df[key], errors="coerce")
    return df
