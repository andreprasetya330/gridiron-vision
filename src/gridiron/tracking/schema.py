"""The play tracking contract.

Everything in this project - Big Data Bowl rows, a processed film clip, a Hudl
export - is converted into a `PlayTracks` and nothing downstream knows or cares
where it came from. That is what lets a model trained on NFL tracking data score
college film.

Coordinate convention (all units yards, seconds):

    x   downfield, relative to the line of scrimmage, positive in the direction
        the offense is attacking. Defenders are therefore at x > 0 pre-snap and
        the offense at x <= 0.
    y   lateral, relative to the ball, positive toward the offense's right.
    t   seconds relative to the snap. Negative is pre-snap.

Every play is resampled onto one shared time grid so a play is a dense
(players, frames) array. Missing observations are NaN rather than interpolated
guesses, because a defender who walked out of frame is information, not noise.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from gridiron.taxonomy import PLAYERS_PER_SIDE

FPS = 10.0
PRE_SNAP_SECONDS = 2.0
POST_SNAP_SECONDS = 3.0

def make_time_grid(pre: float = PRE_SNAP_SECONDS, post: float = POST_SNAP_SECONDS) -> np.ndarray:
    """10 Hz grid from `-pre` to `+post`, always including the snap at 0.0."""
    return np.round(np.arange(-pre, post + 1e-9, 1.0 / FPS), 2)


TIME_GRID: np.ndarray = make_time_grid()
N_FRAMES = len(TIME_GRID)
SNAP_INDEX = int(np.argmin(np.abs(TIME_GRID)))

Side = Literal["offense", "defense"]


def frame_at(seconds: float) -> int:
    """Index into TIME_GRID nearest to a time in seconds relative to the snap."""
    return int(np.argmin(np.abs(TIME_GRID - seconds)))


@dataclass
class Situation:
    """Game context. Everything is optional because film alone will not tell you."""

    down: int | None = None
    distance: float | None = None
    yardline: float | None = None  # yards from the offense's own goal line, 0-100
    quarter: int | None = None
    seconds_remaining: float | None = None
    score_margin: int | None = None  # from the defense's perspective
    offense_personnel: str | None = None  # e.g. "11", "12", "21"
    ball_y_from_center: float = 0.0  # + toward the offense's right
    hash_side: Literal["left", "right", "middle"] | None = None
    league: str = "nfl"
    # BDB 2025 (and any later corpus with a long pre-snap window) can fill these.
    # 2021 plays leave them None; cues treat that as "unknown", not "no motion".
    pre_snap_motion_yards: float | None = None
    motion_at_snap: bool | None = None
    shift_since_lineset: bool | None = None
    motion_since_lineset: bool | None = None

    @property
    def field_side(self) -> Literal["left", "right", "none"]:
        """Which way the wide side of the field is, from the offense's view."""
        if abs(self.ball_y_from_center) < 1.0:
            return "none"
        return "left" if self.ball_y_from_center > 0 else "right"


@dataclass
class TrackQuality:
    """How much to trust this play.

    A play where the pipeline only ever saw 9 defenders should be flagged loudly
    rather than silently scored, because the two it missed are usually the
    safeties, and the safeties are the whole ballgame for coverage.
    """

    defenders_detected: int = PLAYERS_PER_SIDE
    offense_detected: int = PLAYERS_PER_SIDE
    mean_detection_conf: float = 1.0
    registration_error_yd: float = 0.0
    frames_with_full_defense: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return (
            self.defenders_detected >= PLAYERS_PER_SIDE - 1
            and self.registration_error_yd <= 2.0
            and self.frames_with_full_defense >= 0.6
        )

    @property
    def score(self) -> float:
        """0-1 summary used to weight plays and to sort the review queue."""
        completeness = self.defenders_detected / PLAYERS_PER_SIDE
        registration = float(np.clip(1.0 - self.registration_error_yd / 5.0, 0.0, 1.0))
        return float(
            np.clip(
                0.4 * completeness
                + 0.3 * registration
                + 0.2 * self.frames_with_full_defense
                + 0.1 * self.mean_detection_conf,
                0.0,
                1.0,
            )
        )


@dataclass
class PlayerTrack:
    track_id: str
    side: Side
    x: np.ndarray  # (N_FRAMES,) yards, NaN where unobserved
    y: np.ndarray
    jersey: int | None = None
    position: str | None = None
    role: str | None = None  # ground-truth defender role when known
    is_ball_carrier: bool = False

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)
        if self.x.ndim != 1 or self.x.shape != self.y.shape:
            raise ValueError(
                f"track {self.track_id}: x/y must be 1-D and the same length, "
                f"got x={self.x.shape} y={self.y.shape}"
            )
        if self.x.shape[0] < N_FRAMES:
            raise ValueError(
                f"track {self.track_id}: expected at least {N_FRAMES} frames, "
                f"got x={self.x.shape} y={self.y.shape}"
            )

    def at(self, seconds: float) -> tuple[float, float]:
        i = frame_at(seconds)
        return float(self.x[i]), float(self.y[i])

    def velocity(self, seconds: float, window: float = 0.3) -> tuple[float, float]:
        """Central-difference velocity in yards/second."""
        half = max(1, int(round(window * FPS / 2)))
        i = frame_at(seconds)
        lo, hi = max(0, i - half), min(N_FRAMES - 1, i + half)
        dt = TIME_GRID[hi] - TIME_GRID[lo]
        if dt <= 0:
            return 0.0, 0.0
        dx = self.x[hi] - self.x[lo]
        dy = self.y[hi] - self.y[lo]
        if not np.isfinite(dx) or not np.isfinite(dy):
            return 0.0, 0.0
        return float(dx / dt), float(dy / dt)

    def displacement(self, t0: float, t1: float) -> tuple[float, float]:
        x0, y0 = self.at(t0)
        x1, y1 = self.at(t1)
        return x1 - x0, y1 - y0

    @property
    def observed_fraction(self) -> float:
        return float(np.isfinite(self.x).mean())


def infer_game_id(
    play_id: str,
    *,
    source: str | None = None,
    season: int | None = None,
    week: int | None = None,
    defense_team: str | None = None,
    offense_team: str | None = None,
) -> str | None:
    """Recover a game key from a play id when the field was never stored.

    Big Data Bowl ids are `bdb-{gameId}-{playId}`. Everything else groups as
    season-week-matchup, which is the grain a held-out split has to respect:
    two plays from the same game share personnel and game plan.
    """
    if play_id.startswith("bdb"):
        parts = play_id.split("-")
        if len(parts) >= 3 and parts[1].isdigit():
            return parts[1]
    if week is not None and defense_team and offense_team:
        return f"{season or 0}-W{int(week):02d}-{defense_team}-{offense_team}"
    return None


@dataclass
class PlayTracks:
    play_id: str
    source: str  # "bdb" | "film" | "synthetic" | "hudl" | "pff"
    players: list[PlayerTrack]
    situation: Situation = field(default_factory=Situation)
    quality: TrackQuality = field(default_factory=TrackQuality)
    season: int | None = None
    week: int | None = None
    game_id: str | None = None
    defense_team: str | None = None
    offense_team: str | None = None
    coverage: str | None = None  # ground-truth label when known
    coverage_source: str | None = None  # "pff" | "bdb" | "manual" | "model"
    video_path: str | None = None
    snap_frame_in_video: int | None = None
    video_fps: float | None = None
    # Enough to put boxes back on the film. The homography is image-to-field in
    # yards; origin_x / origin_y are the line of scrimmage and the ball, which
    # is the origin the tracks were normalized to. Without these the overlay can
    # only guess, and a guessed overlay is worse than none.
    homography: list[list[float]] | None = None
    origin_x: float | None = None
    origin_y: float | None = None
    play_direction: str = "right"
    video_width: int | None = None
    video_height: int | None = None
    # Coverage scoring always reads the first N_FRAMES (the locked ± window).
    # Film plays may extend `time_grid` past +3s so the overlay can follow the
    # rest of the clip instead of cutting it off.
    time_grid: np.ndarray = field(default_factory=lambda: TIME_GRID.copy())

    def __post_init__(self) -> None:
        self.time_grid = np.asarray(self.time_grid, dtype=np.float64)
        n = int(self.time_grid.shape[0])
        for player in self.players:
            if player.x.shape[0] != n:
                raise ValueError(
                    f"track {player.track_id}: {player.x.shape[0]} samples, "
                    f"time_grid has {n}"
                )
        if not self.game_id:
            self.game_id = infer_game_id(
                self.play_id,
                source=self.source,
                season=self.season,
                week=self.week,
                defense_team=self.defense_team,
                offense_team=self.offense_team,
            )

    @property
    def game_key(self) -> str | None:
        return self.game_id or infer_game_id(
            self.play_id,
            source=self.source,
            season=self.season,
            week=self.week,
            defense_team=self.defense_team,
            offense_team=self.offense_team,
        )

    @property
    def defense(self) -> list[PlayerTrack]:
        return [p for p in self.players if p.side == "defense"]

    @property
    def offense(self) -> list[PlayerTrack]:
        return [p for p in self.players if p.side == "offense"]

    def array(self, side: Side | None = None) -> np.ndarray:
        """(n_players, N_FRAMES, 2) stack, sorted for determinism."""
        players = self.players if side is None else [p for p in self.players if p.side == side]
        players = sorted(players, key=lambda p: p.track_id)
        if not players:
            return np.zeros((0, N_FRAMES, 2), dtype=np.float32)
        return np.stack(
            [np.stack([p.x[:N_FRAMES], p.y[:N_FRAMES]], axis=-1) for p in players],
            axis=0,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "play_id": self.play_id,
            "source": self.source,
            "season": self.season,
            "week": self.week,
            "game_id": self.game_id,
            "defense_team": self.defense_team,
            "offense_team": self.offense_team,
            "coverage": self.coverage,
            "coverage_source": self.coverage_source,
            "video_path": self.video_path,
            "snap_frame_in_video": self.snap_frame_in_video,
            "video_fps": self.video_fps,
            "homography": self.homography,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "play_direction": self.play_direction,
            "video_width": self.video_width,
            "video_height": self.video_height,
            "situation": asdict(self.situation),
            "quality": {
                **asdict(self.quality),
                "score": round(self.quality.score, 3),
                "usable": self.quality.usable,
            },
            "time_grid": [round(float(t), 2) for t in self.time_grid],
            "players": [
                {
                    "track_id": p.track_id,
                    "side": p.side,
                    "jersey": p.jersey,
                    "position": p.position,
                    "role": p.role,
                    "is_ball_carrier": p.is_ball_carrier,
                    # None instead of NaN so the JSON stays valid and the UI can
                    # render a gap rather than an invented position.
                    "x": [None if not np.isfinite(v) else round(float(v), 2) for v in p.x],
                    "y": [None if not np.isfinite(v) else round(float(v), 2) for v in p.y],
                }
                for p in self.players
            ],
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlayTracks:
        def to_arr(values: list[Any]) -> np.ndarray:
            return np.array(
                [np.nan if v is None else float(v) for v in values], dtype=np.float32
            )

        players = [
            PlayerTrack(
                track_id=p["track_id"],
                side=p["side"],
                x=to_arr(p["x"]),
                y=to_arr(p["y"]),
                jersey=p.get("jersey"),
                position=p.get("position"),
                role=p.get("role"),
                is_ball_carrier=p.get("is_ball_carrier", False),
            )
            for p in payload["players"]
        ]
        quality_payload = dict(payload.get("quality", {}))
        quality_payload.pop("score", None)
        quality_payload.pop("usable", None)
        return cls(
            play_id=payload["play_id"],
            source=payload["source"],
            players=players,
            situation=Situation(**payload.get("situation", {})),
            quality=TrackQuality(**quality_payload),
            season=payload.get("season"),
            week=payload.get("week"),
            game_id=payload.get("game_id"),
            defense_team=payload.get("defense_team"),
            offense_team=payload.get("offense_team"),
            coverage=payload.get("coverage"),
            coverage_source=payload.get("coverage_source"),
            video_path=payload.get("video_path"),
            snap_frame_in_video=payload.get("snap_frame_in_video"),
            video_fps=payload.get("video_fps"),
            homography=payload.get("homography"),
            origin_x=payload.get("origin_x"),
            origin_y=payload.get("origin_y"),
            play_direction=payload.get("play_direction", "right"),
            video_width=payload.get("video_width"),
            video_height=payload.get("video_height"),
            time_grid=np.array(payload.get("time_grid", TIME_GRID.tolist()), dtype=np.float64),
        )


def save_play(play: PlayTracks, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(play.to_dict()), encoding="utf-8")


def load_play(path: Path) -> PlayTracks:
    return PlayTracks.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_plays(plays: Iterable[PlayTracks], directory: Path) -> int:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    count = 0
    for play in plays:
        save_play(play, directory / f"{play.play_id}.json")
        count += 1
    return count


def clear_plays(directory: Path) -> int:
    """Remove play JSON from a corpus directory. Leaves predictions.json alone."""
    from gridiron.config import iter_play_json

    count = 0
    for path in iter_play_json(directory, recursive=False):
        path.unlink()
        count += 1
    return count


def load_plays(
    directory: Path,
    limit: int | None = None,
    *,
    recursive: bool = False,
) -> list[PlayTracks]:
    from gridiron.config import iter_play_json

    paths = iter_play_json(directory, recursive=recursive)
    if limit is not None:
        paths = paths[:limit]
    return [load_play(p) for p in paths]


def load_corpus(source: str = "auto", limit: int | None = None) -> list[PlayTracks]:
    """Load the active play corpus. See `config.resolve_play_directory`."""
    from gridiron.config import resolve_play_directory

    directory, recursive, _resolved = resolve_play_directory(source)
    return load_plays(directory, limit=limit, recursive=recursive)


def resample_to_grid(
    times: np.ndarray,
    values: np.ndarray,
    max_gap: float = 0.4,
    edge_tolerance: float = 0.06,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Put an irregular observation series onto a 10 Hz grid.

    Interpolates across short gaps but refuses to bridge long ones - a defender
    missing for a second should stay missing rather than become a straight line
    through the middle of a rotation.

    `edge_tolerance` covers the boundary case where a series nominally spans the
    whole grid but floating-point drift leaves its last sample a hair short of
    the last tick. Dropping the final frame over 1e-15 of rounding would be a
    silent, recurring data loss, so the nearest observation is used instead.
    """
    grid = TIME_GRID if grid is None else np.asarray(grid, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(times) & np.isfinite(values)
    times, values = times[ok], values[ok]
    if times.size == 0:
        return np.full(grid.shape[0], np.nan, dtype=np.float32)
    order = np.argsort(times)
    times, values = times[order], values[order]

    out = np.interp(grid, times, values, left=np.nan, right=np.nan)

    just_before = (grid < times[0]) & (grid >= times[0] - edge_tolerance)
    just_after = (grid > times[-1]) & (grid <= times[-1] + edge_tolerance)
    out[just_before] = values[0]
    out[just_after] = values[-1]

    # Blank out grid points that sit inside a gap wider than max_gap.
    idx = np.searchsorted(times, grid)
    for i in range(grid.shape[0]):
        lo = idx[i] - 1
        hi = idx[i]
        if lo < 0 or hi >= times.size:
            continue
        if times[hi] - times[lo] > max_gap:
            out[i] = np.nan
    return out.astype(np.float32)


def smooth_track(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Median-filter observed samples. Does not invent positions across NaNs."""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or window < 3:
        return values
    out = values.copy()
    half = window // 2
    for i in range(values.size):
        if not np.isfinite(values[i]):
            continue
        sl = values[max(0, i - half) : min(values.size, i + half + 1)]
        finite = sl[np.isfinite(sl)]
        if finite.size >= 2:
            out[i] = float(np.median(finite))
    return out
