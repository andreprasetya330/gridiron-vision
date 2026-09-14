"""NFL Big Data Bowl 2021 loaders.

BDB 2021 is the reason this project can exist without hand-labeling thousands of
clips: it ships 10 Hz tracking for every player on every passing play, and the
community has published per-play coverage labels on top of it.

Expected layout under `data/raw/bdb/`:

    games.csv, players.csv, plays.csv, week1.csv ... week17.csv
    coverages_week1.csv          (Kaggle: tombliss/additional-data-coverage-schemes-for-week-1)
    coverages_2018.csv           (optional, the fuller label set)

Nothing here fails hard on missing files - it reports what it found and lets the
caller fall back to synthetic data.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from gridiron.config import bdb_dir
from gridiron.fields import FIELD_CENTER_Y, NFL_FIELD
from gridiron.taxonomy import COVERAGES
from gridiron.tracking.normalize import ball_y_from_center, normalize_xy
from gridiron.tracking.schema import (
    PlayerTrack,
    PlayTracks,
    Situation,
    TrackQuality,
    resample_to_grid,
)

KAGGLE_COMPETITION = "nfl-big-data-bowl-2021"
KAGGLE_COVERAGE_DATASET = "tombliss/additional-data-coverage-schemes-for-week-1"

# The community label sets use a handful of aliases for the same shells.
COVERAGE_ALIASES: dict[str, str] = {
    "3 Seam": "Cover 3 Zone",
    "Cover 3 Seam": "Cover 3 Zone",
    "Cover 1 Double": "Cover 1 Man",
    "Cover 3 Cloud": "Cover 3 Zone",
    "Cover 3 Sky": "Cover 3 Zone",
    "Cover 3 Buzz": "Cover 3 Zone",
    "Quarters": "Cover 4 Zone",
    "Cover 6": "Cover 6 Zone",
    "Cover 2": "Cover 2 Zone",
    "Cover 0": "Cover 0 Man",
    "Cover 1": "Cover 1 Man",
    "2 Man": "Cover 2 Man",
    "Man Cover 2": "Cover 2 Man",
}

POSITION_GROUPS: dict[str, str] = {
    "QB": "QB", "RB": "RB", "FB": "RB", "HB": "RB",
    "WR": "WR", "TE": "TE",
    "C": "OL", "G": "OL", "T": "OL", "OL": "OL",
    "CB": "CB", "DB": "CB", "S": "S", "FS": "S", "SS": "S",
    "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
}


def normalize_coverage_label(raw: str | float | None) -> str | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    label = str(raw).strip()
    label = COVERAGE_ALIASES.get(label, label)
    return label if label in COVERAGES else None


def _copy_csvs(source: Path, destination: Path) -> int:
    """Flatten downloaded CSVs into the BDB directory the loader expects."""
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in Path(source).rglob("*.csv"):
        target = destination / path.name
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
        copied += 1
    return copied


def _extract_zips(directory: Path) -> None:
    import zipfile

    for archive in directory.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(directory)


def download(directory: Path | None = None) -> Path:
    """Fetch BDB 2021 tracking plus week-1 coverage labels.

    Prefers `kagglehub` (Kaggle's current download API). Falls back to the
    kaggle CLI if the package is missing. Either path needs Kaggle credentials
    and accepted competition rules.
    """
    directory = Path(directory or bdb_dir())
    directory.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
    except ImportError:
        kagglehub = None

    if kagglehub is not None:
        try:
            competition_path = Path(kagglehub.competition_download(KAGGLE_COMPETITION))
        except Exception as exc:
            name = exc.__class__.__name__
            message = str(exc).lower()
            if name == "UnauthenticatedError" or "not authenticated" in message:
                raise RuntimeError(
                    "Kaggle is not authenticated. Create an API token at "
                    "https://www.kaggle.com/settings/api, save it as "
                    "~/.kaggle/access_token, accept the nfl-big-data-bowl-2021 "
                    "competition rules, then re-run `gridiron bdb download`."
                ) from exc
            if "403" in message or "permission" in message or "accepted the competition rules" in message:
                raise RuntimeError(
                    "Kaggle authenticated, but BDB 2021 is 403 Forbidden. Open "
                    "https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/rules "
                    "while logged in as this Kaggle user, accept the rules, then "
                    "re-run `gridiron bdb download`."
                ) from exc
            raise
        _copy_csvs(competition_path, directory)
        try:
            labels_path = Path(kagglehub.dataset_download(KAGGLE_COVERAGE_DATASET))
            _copy_csvs(labels_path, directory)
        except Exception:
            # Tracking is still usable unlabeled; status() reports the gap.
            pass
        _extract_zips(directory)
        return directory

    commands = [
        ["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION, "-p", str(directory)],
        ["kaggle", "datasets", "download", "-d", KAGGLE_COVERAGE_DATASET, "-p", str(directory)],
    ]
    for cmd in commands:
        subprocess.run(cmd, check=True)
    _extract_zips(directory)
    return directory


def available(directory: Path | None = None) -> dict[str, bool | int | str]:
    directory = Path(directory or bdb_dir())
    weeks = sorted(directory.glob("week*.csv"))
    coverage_files = sorted(directory.glob("coverages*.csv"))
    summary = coverage_label_summary(directory)
    return {
        "games": (directory / "games.csv").exists(),
        "plays": (directory / "plays.csv").exists(),
        "players": (directory / "players.csv").exists(),
        "weeks": bool(weeks),
        "n_weeks": len(weeks),
        "coverages": bool(coverage_files),
        "n_coverage_files": len(coverage_files),
        "n_labeled_plays": summary["n_labeled"],
        "labeled_weeks": ",".join(str(w) for w in summary["weeks"]) or "none",
    }


def coverage_label_summary(directory: Path | None = None) -> dict:
    directory = Path(directory or bdb_dir())
    try:
        labels = _load_coverage_labels(directory)
    except Exception:
        return {"n_labeled": 0, "weeks": []}
    weeks: list[int] = []
    games_path = directory / "games.csv"
    if not labels.empty and games_path.exists():
        try:
            games = pd.read_csv(games_path, usecols=lambda c: c.lower() in {"gameid", "week"})
            cols = {c.lower(): c for c in games.columns}
            games = games.rename(columns={cols["gameid"]: "gameId", cols["week"]: "week"})
            merged = labels.merge(games[["gameId", "week"]], on="gameId", how="left")
            weeks = sorted(int(w) for w in merged["week"].dropna().unique())
        except Exception:
            weeks = []
    return {"n_labeled": int(len(labels)), "weeks": weeks}


def _load_coverage_labels(directory: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(directory.glob("coverages*.csv")):
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        if "coverage" not in cols or "gameid" not in cols or "playid" not in cols:
            continue
        out = df[[cols["gameid"], cols["playid"], cols["coverage"]]].copy()
        out.columns = ["gameId", "playId", "coverage"]
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["gameId", "playId", "coverage"])
    labels = pd.concat(frames, ignore_index=True)
    labels["coverage"] = labels["coverage"].map(normalize_coverage_label)
    return labels.dropna(subset=["coverage"]).drop_duplicates(["gameId", "playId"])


def iter_plays(
    directory: Path | None = None,
    weeks: list[int] | None = None,
    labeled_only: bool = True,
    limit: int | None = None,
) -> Iterator[PlayTracks]:
    """Stream BDB plays as `PlayTracks`."""
    directory = Path(directory or bdb_dir())
    status = available(directory)
    if not (status["plays"] and status["games"] and status["weeks"]):
        raise FileNotFoundError(
            f"Big Data Bowl files not found under {directory}. "
            "Run `gridiron bdb download` or use `gridiron demo` for synthetic data."
        )

    games = pd.read_csv(directory / "games.csv")
    plays = pd.read_csv(directory / "plays.csv")
    labels = _load_coverage_labels(directory)
    if not labels.empty:
        plays = plays.merge(labels, on=["gameId", "playId"], how="left")
    else:
        plays["coverage"] = None

    game_meta = games.set_index("gameId")[
        ["season", "week", "homeTeamAbbr", "visitorTeamAbbr"]
    ].to_dict("index")

    week_files = sorted(directory.glob("week*.csv"))
    emitted = 0

    for week_file in week_files:
        week_num = int("".join(ch for ch in week_file.stem if ch.isdigit()) or 0)
        if weeks and week_num not in weeks:
            continue

        tracking = pd.read_csv(week_file)
        tracking["position"] = tracking["position"].map(
            lambda p: POSITION_GROUPS.get(str(p), None)
        )

        for (game_id, play_id), frame in tracking.groupby(["gameId", "playId"], sort=False):
            play_row = plays[(plays.gameId == game_id) & (plays.playId == play_id)]
            if play_row.empty:
                continue
            play_row = play_row.iloc[0]
            coverage = normalize_coverage_label(play_row.get("coverage"))
            if labeled_only and coverage is None:
                continue

            play = _build_play(frame, play_row, game_meta.get(game_id, {}), coverage)
            if play is None:
                continue
            yield play
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _build_play(
    frame: pd.DataFrame,
    play_row: pd.Series,
    game_meta: dict,
    coverage: str | None,
) -> PlayTracks | None:
    snap_frames = frame.loc[frame.get("event") == "ball_snap", "frameId"]
    if snap_frames.empty:
        return None
    snap_frame_id = int(snap_frames.iloc[0])

    play_direction = str(frame["playDirection"].iloc[0])

    ball = frame[frame["team"] == "football"]
    snap_ball = ball[ball.frameId == snap_frame_id]
    if snap_ball.empty:
        return None
    los_x = float(snap_ball["x"].iloc[0])
    ball_y = float(snap_ball["y"].iloc[0])

    possession = str(play_row.get("possessionTeam", ""))
    home = str(game_meta.get("homeTeamAbbr", ""))
    visitor = str(game_meta.get("visitorTeamAbbr", ""))
    offense_side = "home" if possession == home else "away"
    defense_team = visitor if offense_side == "home" else home

    # BDB frames are 10 Hz; convert frameId to seconds relative to the snap.
    frame = frame[frame["team"] != "football"].copy()
    frame["t"] = (frame["frameId"] - snap_frame_id) / 10.0

    players: list[PlayerTrack] = []
    for nfl_id, track in frame.groupby("nflId", sort=False):
        side = "offense" if str(track["team"].iloc[0]) == offense_side else "defense"
        x_abs = track["x"].to_numpy()
        y_abs = track["y"].to_numpy()
        x_norm, y_norm = normalize_xy(x_abs, y_abs, los_x, ball_y, play_direction)
        times = track["t"].to_numpy()
        players.append(
            PlayerTrack(
                track_id=f"{'O' if side == 'offense' else 'D'}_{int(nfl_id)}",
                side=side,
                x=resample_to_grid(times, x_norm),
                y=resample_to_grid(times, y_norm),
                jersey=int(track["jerseyNumber"].iloc[0])
                if not pd.isna(track["jerseyNumber"].iloc[0])
                else None,
                position=track["position"].iloc[0],
            )
        )

    if not players:
        return None

    yardline = play_row.get("absoluteYardlineNumber")
    situation = Situation(
        down=int(play_row["down"]) if not pd.isna(play_row.get("down")) else None,
        distance=float(play_row["yardsToGo"]) if not pd.isna(play_row.get("yardsToGo")) else None,
        yardline=float(yardline) if yardline is not None and not pd.isna(yardline) else None,
        quarter=int(play_row["quarter"]) if not pd.isna(play_row.get("quarter")) else None,
        offense_personnel=_personnel_code(play_row.get("personnelO")),
        ball_y_from_center=round(ball_y_from_center(ball_y, play_direction), 2),
        hash_side=NFL_FIELD.hash_of(ball_y),
        league="nfl",
        score_margin=_score_margin(play_row, offense_side),
    )

    n_def = sum(1 for p in players if p.side == "defense")
    n_off = sum(1 for p in players if p.side == "offense")

    return PlayTracks(
        play_id=f"bdb-{int(play_row['gameId'])}-{int(play_row['playId'])}",
        source="bdb",
        players=players,
        situation=situation,
        quality=TrackQuality(defenders_detected=n_def, offense_detected=n_off),
        season=int(game_meta.get("season", 0)) or None,
        week=int(game_meta.get("week", 0)) or None,
        defense_team=defense_team or None,
        offense_team=possession or None,
        coverage=coverage,
        coverage_source="bdb" if coverage else None,
        game_id=str(int(play_row["gameId"])),
    )


def _personnel_code(raw: object) -> str | None:
    """'1 RB, 1 TE, 3 WR' -> '11'."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw)
    rb = te = 0
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk or not chunk[0].isdigit():
            continue
        count = int(chunk[0])
        if "RB" in chunk:
            rb = count
        elif "TE" in chunk:
            te = count
    return f"{rb}{te}"


def _score_margin(play_row: pd.Series, offense_side: str) -> int | None:
    home = play_row.get("preSnapHomeScore")
    away = play_row.get("preSnapVisitorScore")
    if pd.isna(home) or pd.isna(away):
        return None
    # Reported from the defense's perspective.
    return int(away - home) if offense_side == "home" else int(home - away)


def league_coverage_rates(plays: list[PlayTracks]) -> dict[str, float]:
    """Observed coverage distribution, used as the national baseline prior."""
    counts: dict[str, int] = {c: 0 for c in COVERAGES}
    total = 0
    for play in plays:
        if play.coverage in counts:
            counts[play.coverage] += 1
            total += 1
    if total == 0:
        return {c: 1.0 / len(COVERAGES) for c in COVERAGES}
    return {c: n / total for c, n in counts.items()}


def league_rates_path() -> Path:
    from gridiron.config import subdir

    return subdir("cache", "bdb") / "coverage_rates.json"


def save_league_rates(plays: list[PlayTracks], path: Path | None = None) -> Path:
    """Persist the BDB coverage mix so college mining can borrow it."""
    rates = league_coverage_rates(plays)
    payload = {
        "source": "nfl-big-data-bowl",
        "n_plays": sum(1 for p in plays if p.coverage),
        "weeks": sorted({p.week for p in plays if p.week is not None}),
        "rates": rates,
    }
    path = Path(path or league_rates_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_league_rates(path: Path | None = None) -> dict[str, float] | None:
    path = Path(path or league_rates_path())
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("rates", payload)
    rates = {c: float(raw[c]) for c in COVERAGES if c in raw}
    if not rates:
        return None
    total = sum(rates.values())
    if total <= 0:
        return None
    return {c: rates.get(c, 0.0) / total for c in COVERAGES}
