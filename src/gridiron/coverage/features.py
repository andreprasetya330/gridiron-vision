"""Feature engineering for coverage classification.

Two feature sets, and the difference between them is the whole product:

- **pre-snap** uses only alignment and pre-snap movement. This answers "what are
  they about to run", which is the question a coach actually wants answered.
- **post-snap** adds the first 2.5 seconds after the snap, where man coverage and
  zone coverage stop looking alike. This is what the film overlay displays, and it
  is a much easier problem.

The post-snap discriminators are the classic ones. In man, a defender's velocity
tracks one specific receiver and the distance between them stays roughly constant.
In zone, defenders drift to landmarks, spread out evenly, and their movement
correlates with each other rather than with any receiver.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from gridiron.cues.vocabulary import NUMERIC_CUE_KEYS, extract_cues
from gridiron.tracking.schema import PlayTracks, PlayerTrack

FeatureMode = Literal["presnap", "postsnap"]

EARLY_POST_T = 1.0
LATE_POST_T = 2.5

POSTSNAP_KEYS: list[str] = [
    "deep_count_1s",
    "deep_count_25s",
    "depth_gain_mean",
    "depth_gain_max",
    "mirror_score_mean",
    "mirror_score_max",
    "man_distance_stability",
    "nearest_receiver_dist_mean",
    "zone_spacing_std",
    "deep_spread_25s",
    "underneath_count_25s",
    "pressure_count",
    "qb_converge_min_dist",
    "defense_spread_25s",
    "defense_depth_mean_25s",
    "widest_deep_y",
    "deep_pair_symmetry",
    "coverage_rotation_post",
]


def _vel(track: PlayerTrack, t: float) -> np.ndarray:
    vx, vy = track.velocity(t)
    return np.array([vx, vy], dtype=np.float64)


def _pos(track: PlayerTrack, t: float) -> np.ndarray:
    x, y = track.at(t)
    return np.array([x, y], dtype=np.float64)


def presnap_features(play: PlayTracks) -> dict[str, float]:
    cues = extract_cues(play)
    return {k: float(cues.numeric.get(k, np.nan)) for k in NUMERIC_CUE_KEYS}


def postsnap_features(play: PlayTracks) -> dict[str, float]:
    defense = play.defense
    offense = [p for p in play.offense if p.position in (None, "WR", "TE", "RB")]
    out: dict[str, float] = {k: np.nan for k in POSTSNAP_KEYS}
    if not defense:
        return out

    snap_pos = {d.track_id: _pos(d, 0.0) for d in defense}
    late_pos = {d.track_id: _pos(d, LATE_POST_T) for d in defense}
    early_pos = {d.track_id: _pos(d, EARLY_POST_T) for d in defense}

    late_depths = [p[0] for p in late_pos.values() if np.isfinite(p[0])]
    early_depths = [p[0] for p in early_pos.values() if np.isfinite(p[0])]

    out["deep_count_1s"] = float(sum(1 for d in early_depths if d >= 12.0))
    out["deep_count_25s"] = float(sum(1 for d in late_depths if d >= 14.0))
    out["underneath_count_25s"] = float(sum(1 for d in late_depths if 3.0 <= d < 14.0))
    out["defense_depth_mean_25s"] = float(np.mean(late_depths)) if late_depths else np.nan

    gains = []
    for tid, snap in snap_pos.items():
        late = late_pos.get(tid)
        if late is None or not (np.isfinite(snap[0]) and np.isfinite(late[0])):
            continue
        gains.append(late[0] - snap[0])
    out["depth_gain_mean"] = float(np.mean(gains)) if gains else np.nan
    out["depth_gain_max"] = float(np.max(gains)) if gains else np.nan

    # --- Man versus zone -----------------------------------------------------
    mirror_scores: list[float] = []
    nearest_dists: list[float] = []
    stability: list[float] = []

    for d in defense:
        best_mirror = -1.0
        best_dist = np.inf
        best_stability = np.nan
        for r in offense:
            dists = []
            cos_sims = []
            for t in (0.5, 1.0, 1.5, 2.0, 2.5):
                dp, rp = _pos(d, t), _pos(r, t)
                if not (np.all(np.isfinite(dp)) and np.all(np.isfinite(rp))):
                    continue
                dists.append(float(np.linalg.norm(dp - rp)))
                dv, rv = _vel(d, t), _vel(r, t)
                nd, nr = np.linalg.norm(dv), np.linalg.norm(rv)
                if nd > 0.5 and nr > 0.5:
                    cos_sims.append(float(np.dot(dv, rv) / (nd * nr)))
            if not dists or not cos_sims:
                continue
            mean_dist = float(np.mean(dists))
            # Proximity-weighted velocity agreement: high only when the defender
            # both stays close to the receiver and moves the way he moves.
            proximity = float(np.exp(-mean_dist / 4.0))
            score = proximity * float(np.mean(cos_sims))
            if score > best_mirror:
                best_mirror = score
                best_dist = mean_dist
                best_stability = float(np.std(dists))
        if best_mirror > -1.0:
            mirror_scores.append(best_mirror)
            nearest_dists.append(best_dist)
            if np.isfinite(best_stability):
                stability.append(best_stability)

    out["mirror_score_mean"] = float(np.mean(mirror_scores)) if mirror_scores else np.nan
    out["mirror_score_max"] = float(np.max(mirror_scores)) if mirror_scores else np.nan
    out["nearest_receiver_dist_mean"] = float(np.mean(nearest_dists)) if nearest_dists else np.nan
    # Low variance in defender-receiver distance is man coverage's signature.
    out["man_distance_stability"] = float(np.mean(stability)) if stability else np.nan

    # --- Zone geometry -------------------------------------------------------
    deep = [p for p in late_pos.values() if np.isfinite(p[0]) and p[0] >= 14.0]
    if len(deep) >= 2:
        ys = sorted(p[1] for p in deep)
        gaps = np.diff(ys)
        # Evenly spaced deep defenders means a zone shell dividing the field.
        out["zone_spacing_std"] = float(np.std(gaps))
        out["deep_spread_25s"] = float(ys[-1] - ys[0])
        out["widest_deep_y"] = float(max(abs(ys[0]), abs(ys[-1])))
        out["deep_pair_symmetry"] = float(abs(ys[0] + ys[-1]))
    else:
        out["zone_spacing_std"] = np.nan
        out["deep_spread_25s"] = 0.0 if deep else np.nan
        out["widest_deep_y"] = float(abs(deep[0][1])) if deep else np.nan
        out["deep_pair_symmetry"] = np.nan

    lateral = [p[1] for p in late_pos.values() if np.isfinite(p[1])]
    out["defense_spread_25s"] = float(np.std(lateral)) if lateral else np.nan

    # --- Pressure ------------------------------------------------------------
    qb = next((p for p in play.offense if p.position == "QB"), None)
    if qb is not None:
        qb_pos = _pos(qb, 1.5)
        close = 0
        min_dist = np.inf
        for d in defense:
            dp = _pos(d, 1.5)
            if not (np.all(np.isfinite(dp)) and np.all(np.isfinite(qb_pos))):
                continue
            dist = float(np.linalg.norm(dp - qb_pos))
            min_dist = min(min_dist, dist)
            if dist <= 4.0:
                close += 1
        out["pressure_count"] = float(close)
        out["qb_converge_min_dist"] = float(min_dist) if np.isfinite(min_dist) else np.nan

    snap_deep = sum(1 for p in snap_pos.values() if np.isfinite(p[0]) and p[0] >= 10.0)
    out["coverage_rotation_post"] = float(out["deep_count_25s"] - snap_deep)

    return out


def feature_columns(mode: FeatureMode) -> list[str]:
    if mode == "presnap":
        return list(NUMERIC_CUE_KEYS)
    return list(NUMERIC_CUE_KEYS) + list(POSTSNAP_KEYS)


def build_feature_frame(plays: list[PlayTracks], mode: FeatureMode = "postsnap") -> pd.DataFrame:
    rows = []
    for play in plays:
        row: dict[str, object] = {
            "play_id": play.play_id,
            "season": play.season,
            "week": play.week,
            "game_id": play.game_key,
            "defense_team": play.defense_team,
            "offense_team": play.offense_team,
            "coverage": play.coverage,
            "source": play.source,
            "quality_score": play.quality.score,
            "usable": play.quality.usable,
        }
        row.update(presnap_features(play))
        if mode == "postsnap":
            row.update(postsnap_features(play))
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in feature_columns(mode):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
