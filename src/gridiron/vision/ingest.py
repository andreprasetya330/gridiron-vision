"""File a film clip into the play library and score coverage from its tracks."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gridiron.config import (
    film_uploads_dir,
    plays_dir,
    predictions_path,
)
from gridiron.coverage.bridge import (
    load_default_models,
    overlay_predictions_from_tracks,
    upsert_predictions,
)
from gridiron.tracking.schema import PlayTracks, save_plays

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".webm"}
ALLOWED_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES


class FilmIngestError(ValueError):
    """The clip could not be turned into a scored play."""


@dataclass
class FilmIngestRequest:
    path: Path
    original_name: str
    defense_team: str
    offense_team: str | None = None
    league: str = "ncaa"
    down: int | None = None
    distance: float | None = None
    week: int | None = None
    season: int | None = None
    backend: str = "auto"


@dataclass
class FilmIngestResult:
    plays: list[PlayTracks]
    predictions: list[dict[str, Any]]


def slug(value: str, fallback: str = "team") -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or fallback


def media_kind(path: str | Path | None) -> str | None:
    if not path:
        return None
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return None


def media_type(path: Path) -> str:
    return {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".m4v": "video/mp4",
        ".avi": "video/x-msvideo",
        ".webm": "video/webm",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(path.suffix.lower(), "application/octet-stream")


def ingest_film(request: FilmIngestRequest) -> FilmIngestResult:
    """Copy the clip, run the vision pipeline, tag the team, and score coverage."""
    defense = request.defense_team.strip()
    if not defense:
        raise FilmIngestError("defense team is required so the play can be filed")

    source = Path(request.path)
    suffix = source.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise FilmIngestError(
            f"unsupported file type {suffix or '(none)'}. Use an MP4 clip or a still image."
        )
    if not source.exists():
        raise FilmIngestError(f"file not found: {source}")

    from gridiron.vision.pipeline import FilmPipeline, PipelineConfig

    play_id = _unique_play_id(defense, request.original_name or source.name)
    stored = film_uploads_dir() / f"{play_id}{suffix}"
    if source.resolve() != stored.resolve():
        shutil.copy2(source, stored)

    stride = 1
    pipeline = FilmPipeline(
        PipelineConfig(
            league=request.league,
            single_play=True,
            stride=stride,
            backend=request.backend,
        )
    )
    try:
        plays = pipeline.process(stored, play_id=play_id)
    except Exception as exc:
        raise FilmIngestError(f"vision pipeline failed: {exc}") from exc

    if not plays:
        raise FilmIngestError(
            "no usable play extracted. Common causes: the homography could not be "
            "solved, or offense and defense could not be separated."
        )

    offense = (request.offense_team or "").strip() or None
    for play in plays:
        play.defense_team = defense
        play.offense_team = offense
        play.week = request.week
        play.season = request.season
        play.video_path = str(stored)
        play.situation.league = request.league
        if request.down is not None:
            play.situation.down = request.down
        if request.distance is not None:
            play.situation.distance = request.distance

    save_plays(plays, plays_dir("film"))
    post_model, pre_model = load_default_models()
    predictions: list[dict[str, Any]] = []
    if post_model is not None or pre_model is not None:
        predictions = overlay_predictions_from_tracks(plays, post_model, pre_model)
        if predictions:
            upsert_predictions(predictions, predictions_path())
    return FilmIngestResult(plays=plays, predictions=predictions)


def _unique_play_id(defense_team: str, original_name: str) -> str:
    stem = slug(Path(original_name).stem, fallback="clip")[:40]
    base = f"{slug(defense_team)}-{stem}"
    directory = plays_dir("film")
    uploads = film_uploads_dir()
    candidate = base
    n = 0
    while (directory / f"{candidate}.json").exists() or any(uploads.glob(f"{candidate}.*")):
        n += 1
        candidate = f"{base}-{n:02d}"
    return candidate
