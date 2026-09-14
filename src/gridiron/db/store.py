"""The play database.

DuckDB over Parquet rather than a server. A full season of college football is a
few hundred thousand rows, which DuckDB queries instantly from a laptop with no
daemon to run, no port to open, and no migration to forget. The Parquet files stay
readable by pandas and by anyone who wants to poke at them directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from gridiron.config import data_dir, db_path
from gridiron.taxonomy import COVERAGES

TABLES = {
    "plays": "plays.parquet",
    "cues": "cues.parquet",
    "predictions": "predictions.parquet",
    "tells": "tells.parquet",
}


class PlayStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or data_dir()) / "warehouse"
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, table: str) -> Path:
        if table not in TABLES:
            raise KeyError(f"unknown table {table!r}; expected one of {sorted(TABLES)}")
        return self.root / TABLES[table]

    def write(self, table: str, frame: pd.DataFrame, mode: str = "replace") -> Path:
        path = self.path_for(table)
        if mode == "append" and path.exists():
            existing = pd.read_parquet(path)
            frame = pd.concat([existing, frame], ignore_index=True)
            key = "play_id" if "play_id" in frame.columns else None
            if key:
                frame = frame.drop_duplicates(subset=[key], keep="last")
        # Object columns holding mixed types defeat Parquet; stringify them rather
        # than losing the whole write.
        safe = frame.copy()
        for column in safe.columns:
            if safe[column].dtype == object:
                types = {type(v) for v in safe[column].dropna().head(200)}
                if len(types) > 1:
                    safe[column] = safe[column].astype(str)
        safe.to_parquet(path, index=False)
        return path

    def read(self, table: str) -> pd.DataFrame:
        path = self.path_for(table)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def exists(self, table: str) -> bool:
        return self.path_for(table).exists()

    def connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(str(db_path()))
        for table, filename in TABLES.items():
            path = self.root / filename
            if path.exists():
                con.execute(
                    f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{path.as_posix()}')"
                )
        return con

    def query(self, sql: str) -> pd.DataFrame:
        with self.connect() as con:
            return con.execute(sql).fetchdf()

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for table in TABLES:
            frame = self.read(table)
            out[table] = {
                "rows": len(frame),
                "columns": len(frame.columns),
                "path": str(self.path_for(table)),
            }
        return out


def tendency_table(
    cues: pd.DataFrame,
    team: str,
    by: list[str] | None = None,
    min_plays: int = 8,
) -> pd.DataFrame:
    """Coverage rates broken out by situation, with the league rate beside them.

    A tendency table without the league column is a trap: every defense plays
    Cover 3 a lot on early downs, so "they play Cover 3 on 34% of first downs" is
    not information until you know everyone else does too.
    """
    by = by or ["down", "distance_band"]
    frame = cues[cues["coverage"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()

    missing = [c for c in by if c not in frame.columns]
    if missing:
        raise KeyError(f"cue table has no column(s) {missing}")

    team_frame = frame[frame["defense_team"] == team]
    others = frame[frame["defense_team"] != team]
    if team_frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for keys, group in team_frame.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        if len(group) < min_plays:
            continue

        peer = others
        for column, value in zip(by, keys):
            peer = peer[peer[column] == value]

        for coverage in COVERAGES:
            rate = float((group["coverage"] == coverage).mean())
            league = float((peer["coverage"] == coverage).mean()) if len(peer) else float("nan")
            if rate < 0.05 and (pd.isna(league) or league < 0.05):
                continue
            row = dict(zip(by, keys))
            row.update(
                {
                    "coverage": coverage,
                    "plays": len(group),
                    "rate": round(rate, 4),
                    "league_rate": round(league, 4) if pd.notna(league) else None,
                    "delta": round(rate - league, 4) if pd.notna(league) else None,
                }
            )
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["_abs_delta"] = pd.to_numeric(table["delta"], errors="coerce").abs().fillna(0.0)
    return (
        table.sort_values(["_abs_delta", "plays"], ascending=[False, False])
        .drop(columns="_abs_delta")
        .reset_index(drop=True)
    )
