"""The API behind the overlay.

Play listing is served from artifacts on disk. Film ingest is the exception:
the UI posts a clip, the vision pipeline plus BDB coverage model run, and the
play is filed under the given defense team.
"""

from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from gridiron.config import (
    film_overlays_dir,
    iter_play_json,
    models_dir,
    play_source_counts,
    predictions_path,
    resolve_play_directory,
    reports_dir,
)
from gridiron.taxonomy import COVERAGES, DEFENDER_ROLES, coverage_shell, family_of
from gridiron.tracking.schema import TIME_GRID, load_play
from gridiron.vision.ingest import FilmIngestError, FilmIngestRequest, ingest_film, media_kind, media_type


@lru_cache(maxsize=1)
def _predictions() -> dict[str, dict[str, Any]]:
    path = predictions_path()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {p["play_id"]: p for p in payload}


@lru_cache(maxsize=8)
def _play_paths(source: str, directory: str, recursive: bool) -> dict[str, Path]:
    return {p.stem: p for p in iter_play_json(Path(directory), recursive=recursive)}


def _paths_for(source: str = "auto") -> dict[str, Path]:
    directory, recursive, _resolved = resolve_play_directory(source)
    return _play_paths(source, str(directory), recursive)


def _summary_row(play_id: str, payload: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    quality = payload.get("quality") or {}
    disguise = prediction.get("disguise") or {}
    presnap = prediction.get("presnap") or {}
    return {
        "play_id": play_id,
        "season": payload.get("season"),
        "week": payload.get("week"),
        "defense_team": payload.get("defense_team"),
        "offense_team": payload.get("offense_team"),
        "coverage_truth": payload.get("coverage"),
        "coverage_predicted": prediction.get("coverage"),
        "coverage_presnap": presnap.get("coverage"),
        "confidence": prediction.get("confidence"),
        "quality_score": quality.get("score"),
        "usable": quality.get("usable", prediction.get("usable", True)),
        "source": payload.get("source"),
        "has_video": bool(payload.get("video_path")),
        "disguised": bool(disguise.get("disguised")),
        "disguise_kind": disguise.get("kind"),
        "coverage_family": family_of(prediction.get("coverage") or payload.get("coverage")),
        "coverage_shell": coverage_shell(prediction.get("coverage") or payload.get("coverage")),
        "situation": payload.get("situation", {}),
    }


@lru_cache(maxsize=8)
def _play_summaries(source: str, directory: str, recursive: bool) -> tuple[dict[str, Any], ...]:
    predictions = _predictions()
    rows: list[dict[str, Any]] = []
    for play_id, path in _play_paths(source, directory, recursive).items():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(_summary_row(play_id, payload, predictions.get(play_id, {})))
    return tuple(rows)


def _summaries_for(source: str = "auto") -> tuple[dict[str, Any], ...]:
    directory, recursive, _resolved = resolve_play_directory(source)
    return _play_summaries(source, str(directory), recursive)


def _clear_api_caches() -> None:
    _predictions.cache_clear()
    _play_paths.cache_clear()
    _play_summaries.cache_clear()


def _minimap_path(play_id: str) -> Path:
    return film_overlays_dir() / f"{play_id}_minimap.png"


def _enrich_play(play_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    prediction = _predictions().get(play_id) or payload.get("prediction")
    payload["prediction"] = prediction
    coverage = (prediction or {}).get("coverage") or payload.get("coverage")
    payload["coverage_family"] = family_of(coverage)
    payload["coverage_shell"] = coverage_shell(coverage)
    minimap = _minimap_path(play_id)
    payload["minimap_url"] = f"/api/plays/{play_id}/minimap" if minimap.exists() else None
    payload["media_kind"] = media_kind(payload.get("video_path"))
    return payload


def create_app() -> FastAPI:
    app = FastAPI(
        title="Gridiron Vision",
        version="0.1.0",
        description="Coverage detection and tell mining for football film",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        _directory, _rec, resolved = resolve_play_directory("auto")
        paths = _paths_for("auto")
        return {
            "status": "ok",
            "plays": len(paths),
            "corpus": resolved,
            "sources": {k: v for k, v in play_source_counts().items() if k != "_root"},
            "predictions": len(_predictions()),
            "models": [p.name for p in models_dir().glob("*")],
            "coverages": COVERAGES,
            "roles": DEFENDER_ROLES,
            "time_grid": TIME_GRID.tolist(),
        }

    @app.get("/api/plays")
    def list_plays(
        team: str | None = None,
        coverage: str | None = None,
        week: int | None = None,
        usable_only: bool = False,
        disguised_only: bool = False,
        source: str = Query("auto"),
        limit: int = Query(200, le=2000),
        offset: int = 0,
    ) -> dict[str, Any]:
        rows = [
            row
            for row in _summaries_for(source)
            if (not team or row["defense_team"] == team)
            and (
                not coverage
                or row["coverage_predicted"] == coverage
                or row["coverage_truth"] == coverage
            )
            and (week is None or row["week"] == week)
            and (not usable_only or row["usable"])
            and (not disguised_only or row["disguised"])
        ]
        return {"total": len(rows), "plays": rows[offset : offset + limit], "corpus": source}

    @app.get("/api/plays/{play_id}")
    def get_play(play_id: str) -> dict[str, Any]:
        path = _paths_for("auto").get(play_id) or _paths_for("all").get(play_id)
        if path is None:
            raise HTTPException(status_code=404, detail=f"unknown play {play_id}")
        play = load_play(path)
        return _enrich_play(play_id, play.to_dict())

    @app.get("/api/plays/{play_id}/video")
    def get_video(play_id: str) -> FileResponse:
        path = _paths_for("auto").get(play_id) or _paths_for("all").get(play_id)
        if path is None:
            raise HTTPException(status_code=404, detail=f"unknown play {play_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        video_path = payload.get("video_path")
        if not video_path or not Path(video_path).exists():
            raise HTTPException(status_code=404, detail="no video on disk for this play")
        media = Path(video_path)
        return FileResponse(media, media_type=media_type(media))

    @app.get("/api/plays/{play_id}/minimap")
    def get_minimap(play_id: str) -> FileResponse:
        path = _minimap_path(play_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="no minimap for this play")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/film/ingest")
    async def ingest_clip(
        file: UploadFile = File(...),
        defense_team: str = Form(...),
        offense_team: str | None = Form(None),
        league: str = Form("ncaa"),
        down: int | None = Form(None),
        distance: float | None = Form(None),
        week: int | None = Form(None),
        season: int | None = Form(None),
    ) -> dict[str, Any]:
        filename = Path(file.filename or "clip.bin")
        suffix = filename.suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
            tmp_path = Path(handle.name)
        try:
            result = ingest_film(
                FilmIngestRequest(
                    path=tmp_path,
                    original_name=filename.name,
                    defense_team=defense_team,
                    offense_team=offense_team,
                    league=league,
                    down=down,
                    distance=distance,
                    week=week,
                    season=season,
                )
            )
        except FilmIngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            tmp_path.unlink(missing_ok=True)
            await file.close()

        _clear_api_caches()
        plays = [_enrich_play(play.play_id, play.to_dict()) for play in result.plays]
        pred_by_id = {row["play_id"]: row for row in result.predictions}
        for payload in plays:
            if payload["play_id"] in pred_by_id:
                payload["prediction"] = pred_by_id[payload["play_id"]]
                coverage = payload["prediction"].get("coverage")
                payload["coverage_family"] = family_of(coverage)
                payload["coverage_shell"] = coverage_shell(coverage)
        return {"plays": plays, "play_ids": [p["play_id"] for p in plays]}

    @app.get("/api/teams")
    def list_teams(source: str = Query("auto")) -> dict[str, Any]:
        teams: dict[str, int] = {}
        for row in _summaries_for(source):
            team = row.get("defense_team")
            if team:
                teams[str(team)] = teams.get(str(team), 0) + 1
        return {"teams": [{"team": t, "plays": n} for t, n in sorted(teams.items())], "corpus": source}

    @app.get("/api/reports")
    def list_reports() -> dict[str, Any]:
        directory = reports_dir()
        return {
            "reports": [
                {"slug": p.stem, "path": str(p), "bytes": p.stat().st_size}
                for p in sorted(directory.glob("*.md"))
            ]
        }

    @app.get("/api/reports/{slug}")
    def get_report(slug: str) -> dict[str, Any]:
        md_path = reports_dir() / f"{slug}.md"
        json_path = reports_dir() / f"{slug}.evidence.json"
        if not md_path.exists():
            raise HTTPException(status_code=404, detail=f"no report named {slug}")
        evidence = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else None
        return {
            "slug": slug,
            "markdown": md_path.read_text(encoding="utf-8"),
            "evidence": evidence,
        }

    return app


app = create_app()
