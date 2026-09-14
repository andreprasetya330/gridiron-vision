"""CollegeFootballData API client.

CFBD is the source that actually works today for college opponent data: free key,
bearer-token auth, and endpoints covering games, drives, plays, advanced stats,
PPA/EPA, rosters, and matchup history. It gives you everything except coverage,
which is precisely the gap the film pipeline fills.

Responses are cached to Parquet on disk. The free tier has a monthly call budget
and a scouting report regenerated five times on a Tuesday should not spend it
five times over.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from gridiron.config import subdir
from gridiron.ingest.schema import normalize_team, to_play_context

BASE_URL = "https://api.collegefootballdata.com"
KEY_URL = "https://collegefootballdata.com/key"


class CFBDError(RuntimeError):
    pass


@dataclass
class CFBDConfig:
    api_key: str | None = None
    timeout: float = 30.0
    cache_ttl_hours: float = 24.0
    max_retries: int = 3
    min_interval_seconds: float = 0.35

    def resolve_key(self) -> str:
        key = self.api_key or os.environ.get("CFBD_API_KEY", "").strip()
        if len(key) >= 2 and key[0] == key[-1] and key[0] in {'"', "'"}:
            key = key[1:-1].strip()
        if not key:
            raise CFBDError(
                "No CollegeFootballData API key. Get a free one at "
                f"{KEY_URL} and set CFBD_API_KEY in your .env file."
            )
        return key


class CFBDClient:
    def __init__(self, config: CFBDConfig | None = None) -> None:
        self.config = config or CFBDConfig()
        self.cache_dir = subdir("cache", "cfbd")
        self._last_call = 0.0

    def _cache_path(self, endpoint: str, params: dict[str, Any]) -> Path:
        stem = endpoint.strip("/").replace("/", "_")
        key = "_".join(f"{k}-{v}" for k, v in sorted(params.items()) if v is not None)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in key)[:120]
        return self.cache_dir / f"{stem}__{safe or 'all'}.parquet"

    def _fresh(self, path: Path) -> bool:
        candidate = path if path.exists() else path.with_suffix(".json")
        if not candidate.exists():
            return False
        age_hours = (time.time() - candidate.stat().st_mtime) / 3600
        return age_hours < self.config.cache_ttl_hours

    def _read_cache(self, path: Path) -> pd.DataFrame | None:
        if path.exists():
            return pd.read_parquet(path)
        json_path = path.with_suffix(".json")
        if json_path.exists():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            return pd.json_normalize(payload) if payload else pd.DataFrame()
        return None

    def get(
        self, endpoint: str, params: dict[str, Any] | None = None, use_cache: bool = True
    ) -> pd.DataFrame:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        cache_path = self._cache_path(endpoint, params)
        if use_cache and self._fresh(cache_path):
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached

        headers = {
            "Authorization": f"Bearer {self.config.resolve_key()}",
            "Accept": "application/json",
        }

        last_error: Exception | None = None
        payload: Any = None
        for attempt in range(self.config.max_retries):
            elapsed = time.time() - self._last_call
            if elapsed < self.config.min_interval_seconds:
                time.sleep(self.config.min_interval_seconds - elapsed)
            try:
                with httpx.Client(timeout=self.config.timeout) as client:
                    response = client.get(
                        f"{BASE_URL}/{endpoint.lstrip('/')}", params=params, headers=headers
                    )
                self._last_call = time.time()

                if response.status_code == 401:
                    raise CFBDError("CFBD rejected the API key (401). Check CFBD_API_KEY.")
                if response.status_code == 403:
                    raise CFBDError(
                        "CFBD forbade this endpoint (403). Live scoreboard and some "
                        "analytics routes need a Patreon tier; historical plays/games/stats do not."
                    )
                if response.status_code == 429:
                    raise CFBDError(
                        "CFBD rate limit hit (429). The free tier has a monthly call cap; "
                        "cached data is still available on disk."
                    )
                response.raise_for_status()
                payload = response.json()
                break
            except CFBDError:
                raise
            except Exception as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        else:
            if use_cache:
                cached = self._read_cache(cache_path)
                if cached is not None:
                    return cached
            raise CFBDError(f"CFBD request to /{endpoint} failed: {last_error}")

        df = pd.json_normalize(payload) if payload else pd.DataFrame()
        if payload:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if not df.empty:
                    df.to_parquet(cache_path, index=False)
            except Exception:
                cache_path.with_suffix(".json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
        return df

    def ping(self) -> dict[str, Any]:
        """Cheap authenticated call so `doctor` and `ingest cfbd --ping` can fail fast."""
        conferences = self.get("conferences", use_cache=False)
        return {
            "ok": True,
            "conferences": int(len(conferences)),
            "cache_dir": str(self.cache_dir),
        }

    # --- Convenience wrappers ------------------------------------------------

    def games(self, season: int, team: str | None = None, week: int | None = None) -> pd.DataFrame:
        return self.get("games", {"year": season, "team": team, "week": week})

    def plays(
        self, season: int, week: int, team: str | None = None, defense: str | None = None
    ) -> pd.DataFrame:
        return self.get(
            "plays", {"year": season, "week": week, "team": team, "defense": defense}
        )

    def drives(self, season: int, team: str | None = None) -> pd.DataFrame:
        return self.get("drives", {"year": season, "team": team})

    def advanced_season_stats(self, season: int, team: str | None = None) -> pd.DataFrame:
        return self.get("stats/season/advanced", {"year": season, "team": team})

    def advanced_game_stats(
        self, season: int, team: str | None = None, week: int | None = None
    ) -> pd.DataFrame:
        return self.get("stats/game/advanced", {"year": season, "team": team, "week": week})

    def team_ppa(self, season: int, team: str | None = None) -> pd.DataFrame:
        return self.get("ppa/teams", {"year": season, "team": team})

    def roster(self, season: int, team: str) -> pd.DataFrame:
        return self.get("roster", {"year": season, "team": team})

    def matchup(self, team1: str, team2: str) -> pd.DataFrame:
        return self.get("teams/matchup", {"team1": team1, "team2": team2})

    def betting_lines(self, season: int, team: str | None = None) -> pd.DataFrame:
        return self.get("lines", {"year": season, "team": team})


def _cell(row: pd.Series, *names: str) -> Any:
    for name in names:
        if name in row.index and pd.notna(row.get(name)):
            return row.get(name)
    return None


def _clock_seconds(row: pd.Series) -> float | None:
    minutes = _cell(row, "clock.minutes", "clockMinutes")
    seconds = _cell(row, "clock.seconds", "clockSeconds")
    if minutes is None and seconds is None:
        clock = _cell(row, "clock")
        if isinstance(clock, dict):
            minutes = clock.get("minutes")
            seconds = clock.get("seconds")
    if minutes is None and seconds is None:
        return None
    try:
        return float(minutes or 0) * 60 + float(seconds or 0)
    except (TypeError, ValueError):
        return None


def _personnel_code(raw: object) -> str | None:
    """'1 RB, 1 TE, 3 WR' or '11' -> '11'."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if text.isdigit() and 1 <= len(text) <= 2:
        return text.zfill(2)
    rb = te = 0
    found = False
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk or not chunk[0].isdigit():
            continue
        count = int(chunk[0])
        if "RB" in chunk.upper():
            rb = count
            found = True
        elif "TE" in chunk.upper():
            te = count
            found = True
    return f"{rb}{te}" if found else None


def defensive_plays_to_context(plays: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """CFBD /plays rows -> the shared play-context schema."""
    if plays.empty:
        return to_play_context([], "cfbd")

    rows: list[dict[str, Any]] = []
    for _, row in plays.iterrows():
        offense_points = _cell(row, "offenseScore", "offense_score")
        defense_points = _cell(row, "defenseScore", "defense_score")
        margin = None
        if offense_points is not None and defense_points is not None:
            try:
                margin = int(defense_points) - int(offense_points)
            except (TypeError, ValueError):
                margin = None

        rows.append(
            {
                "play_id": f"cfbd-{_cell(row, 'id', 'playId')}",
                "season": season,
                "week": week,
                "defense_team": normalize_team(_cell(row, "defense", "defenseTeam")),
                "offense_team": normalize_team(_cell(row, "offense", "offenseTeam")),
                "down": _cell(row, "down"),
                "distance": _cell(row, "distance"),
                "yardline": _cell(row, "yardsToGoal", "yards_to_goal"),
                "quarter": _cell(row, "period"),
                "clock_seconds": _clock_seconds(row),
                "score_margin": margin,
                "offense_personnel": _personnel_code(
                    _cell(row, "offensePersonnel", "offense_personnel", "personnel")
                ),
                "play_type": _cell(row, "playType", "play_type"),
                "yards_gained": _cell(row, "yardsGained", "yards_gained"),
                "epa": _cell(row, "ppa", "expectedPointsAdded"),
                "source": "cfbd",
            }
        )
    return to_play_context(rows, "cfbd")


def season_context(
    client: CFBDClient, season: int, team: str, weeks: range | None = None
) -> pd.DataFrame:
    """Pull an opponent's full season of defensive plays."""
    weeks = weeks or range(1, 16)
    frames = []
    for week in weeks:
        try:
            plays = client.plays(season=season, week=week, defense=team)
        except CFBDError as exc:
            message = str(exc).lower()
            if "401" in message or "403" in message or "rejected" in message:
                raise
            break
        if plays.empty:
            continue
        frames.append(defensive_plays_to_context(plays, season, week))
    if not frames:
        return to_play_context([], "cfbd")
    return pd.concat(frames, ignore_index=True)


def opponent_profile(client: CFBDClient, season: int, team: str) -> dict[str, Any]:
    """Everything CFBD knows about a team, shaped for the scouting report."""
    profile: dict[str, Any] = {"team": normalize_team(team), "season": season}

    try:
        advanced = client.advanced_season_stats(season, team)
        if not advanced.empty:
            record = advanced.iloc[0].to_dict()
            profile["defense"] = {
                k.replace("defense.", ""): v
                for k, v in record.items()
                if isinstance(k, str) and k.startswith("defense.")
            }
            profile["offense"] = {
                k.replace("offense.", ""): v
                for k, v in record.items()
                if isinstance(k, str) and k.startswith("offense.")
            }
    except CFBDError as exc:
        profile["errors"] = [str(exc)]

    try:
        games = client.games(season, team)
        if not games.empty:
            profile["games"] = int(len(games))
            if "homeTeam" in games.columns or "home_team" in games.columns:
                home = games.get("homeTeam", games.get("home_team"))
                away = games.get("awayTeam", games.get("away_team"))
                profile["opponents"] = sorted(
                    {
                        str(away.iloc[i] if home.iloc[i] == team else home.iloc[i])
                        for i in range(len(games))
                        if pd.notna(home.iloc[i]) and pd.notna(away.iloc[i])
                    }
                )
    except CFBDError:
        pass

    try:
        ppa = client.team_ppa(season, team)
        if not ppa.empty:
            profile["ppa"] = ppa.iloc[0].to_dict()
    except CFBDError:
        pass

    try:
        roster = client.roster(season, team)
        if not roster.empty:
            profile["roster_size"] = int(len(roster))
    except CFBDError:
        pass

    return profile
