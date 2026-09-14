"""Source-agnostic ingestion: CFBD, nflverse, PFF exports."""

from gridiron.ingest.schema import (
    PLAY_CONTEXT_COLUMNS,
    normalize_team,
    to_play_context,
)

__all__ = ["PLAY_CONTEXT_COLUMNS", "normalize_team", "to_play_context"]
