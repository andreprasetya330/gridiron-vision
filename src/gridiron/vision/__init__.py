"""Film -> field coordinates.

Everything in this package is optional at import time. The vision extras
(`uv sync --extra vision`) pull ultralytics, opencv, and supervision, which is a
large tree that the analytics half of the project does not need.
"""

VISION_EXTRA_HINT = (
    "Vision dependencies are not installed. Run `uv sync --extra vision` to add "
    "opencv-python, ultralytics, and supervision."
)


def require(module: str):
    """Import an optional vision dependency with a useful error message."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(f"{VISION_EXTRA_HINT} (missing {module})") from exc
