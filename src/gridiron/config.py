"""Project paths and runtime configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var, "").strip()
    return Path(raw).expanduser().resolve() if raw else default


def _env_float(var: str, default: float) -> float:
    raw = os.environ.get(var, "").strip()
    if not raw:
        return default
    return float(raw)


@lru_cache(maxsize=1)
def data_dir() -> Path:
    return _env_path("GRIDIRON_DATA_DIR", PROJECT_ROOT / "data")


def subdir(*parts: str) -> Path:
    """Return a data subdirectory, creating it on demand."""
    path = data_dir().joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir() -> Path:
    return subdir("raw")


def bdb_dir(edition: str = "2021") -> Path:
    """Raw Big Data Bowl CSVs. 2021 and 2025 must not share a folder."""
    if normalize_bdb_edition(edition) == "2025":
        return subdir("raw", "bdb2025")
    return subdir("raw", "bdb")


def normalize_bdb_edition(edition: str | int | None) -> str:
    text = str(edition or "2021").strip().lower().replace("bdb", "")
    if text in {"2025", "25"}:
        return "2025"
    return "2021"


def bdb_play_source(edition: str = "2021") -> str:
    return "bdb2025" if normalize_bdb_edition(edition) == "2025" else "bdb"


def pff_dir() -> Path:
    return subdir("pff")


PREDICTIONS_FILENAME = "predictions.json"

# Real data first. `auto` picks the first of these that actually has plays, so a
# leftover synthetic season cannot silently contaminate a Big Data Bowl training
# run (or the other way around).
PLAY_SOURCE_PRIORITY: tuple[str, ...] = ("bdb", "bdb2025", "film", "pff", "hudl", "synthetic")


def plays_dir(source: str | None = None) -> Path:
    """Play JSON lives under `data/plays/<source>/` so corpora cannot mix.

    Passing no source returns the parent directory, which is also where
    `predictions.json` lives - one overlay file, keyed by play id.
    """
    if source:
        return subdir("plays", source)
    return subdir("plays")


def predictions_path() -> Path:
    return plays_dir() / PREDICTIONS_FILENAME


def iter_play_json(directory: Path, recursive: bool = False) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(
        p for p in directory.glob(pattern) if p.is_file() and p.name != PREDICTIONS_FILENAME
    )


def play_source_counts() -> dict[str, int]:
    root = plays_dir()
    counts = {name: len(iter_play_json(root / name)) for name in PLAY_SOURCE_PRIORITY}
    counts["_root"] = len(iter_play_json(root))
    return counts


def resolve_play_directory(source: str = "auto") -> tuple[Path, bool, str]:
    """Pick which play files to load.

    Returns `(directory, recursive, resolved_source)`. `auto` prefers real data
    over the synthetic season. `all` walks every source together - only for
    explicit mixes, never the default.
    """
    requested = (source or "auto").strip().lower()
    if requested in {"all", "mix"}:
        return plays_dir(), True, "all"
    if requested != "auto":
        return plays_dir(requested), False, requested

    counts = play_source_counts()
    for name in PLAY_SOURCE_PRIORITY:
        if counts.get(name, 0):
            return plays_dir(name), False, name
    return plays_dir(), False, "legacy"


def film_dir() -> Path:
    return subdir("film")


def film_uploads_dir() -> Path:
    return subdir("film", "uploads")


def film_overlays_dir() -> Path:
    return subdir("film", "overlays")


def models_dir() -> Path:
    return subdir("models")


def reports_dir() -> Path:
    return subdir("reports")


def db_path() -> Path:
    return data_dir() / "gridiron.duckdb"


class RoboflowSettings:
    """Hosted film workflow: player detection, then local field mapping.

    Roboflow returns boxes. The pipeline solves a per-frame homography from
    painted yard lines. The UGA calibration polygon is fallback only.
    """

    def __init__(self) -> None:
        self.api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
        self.api_url = os.environ.get(
            "ROBOFLOW_API_URL", "https://serverless.roboflow.com"
        ).strip()
        self.workspace = os.environ.get("ROBOFLOW_WORKSPACE", "andre-4cotb").strip()
        self.workflow_id = os.environ.get(
            "ROBOFLOW_WORKFLOW_ID", "defensive-coverage-analysis-1789781275490"
        ).strip()
        self.player_workflow_id = os.environ.get(
            "ROBOFLOW_PLAYER_WORKFLOW_ID", "american-football-player-trackin"
        ).strip()
        self.detect_only = os.environ.get("ROBOFLOW_DETECT_ONLY", "1").strip() not in {
            "0",
            "false",
            "False",
            "no",
        }
        self.use_cache = os.environ.get("ROBOFLOW_USE_CACHE", "1").strip() not in {
            "0",
            "false",
            "False",
            "no",
        }
        self.player_confidence = _env_float("ROBOFLOW_PLAYER_CONFIDENCE", 0.4)
        self.player_iou_threshold = _env_float("ROBOFLOW_PLAYER_IOU", 0.3)
        self.timeout_s = _env_float("ROBOFLOW_TIMEOUT", 60.0)
        self.max_retries = max(1, int(os.environ.get("ROBOFLOW_MAX_RETRIES", "3") or 3))

    @property
    def enabled(self) -> bool:
        workflow = self.player_workflow_id if self.detect_only else self.workflow_id
        return bool(self.api_key and self.workspace and workflow)


class LLMSettings:
    """Optional LLM used only for narrative prose in scouting reports.

    The evidence bundle is always computed deterministically. If no endpoint is
    configured the report renderer falls back to templated prose, which is
    verbose but never wrong.
    """

    def __init__(self) -> None:
        self.base_url = os.environ.get("GRIDIRON_LLM_BASE_URL", "").strip()
        self.api_key = os.environ.get("GRIDIRON_LLM_API_KEY", "").strip()
        self.model = os.environ.get("GRIDIRON_LLM_MODEL", "gpt-4o-mini").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key)
