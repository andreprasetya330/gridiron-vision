"""Convert absolute field coordinates into the normalized play frame.

Raw sources give you a spot on a 120x53.3 field plus a play direction. The models
want the ball at the origin with the offense always attacking +x, so that a play
going left in the first quarter and a play going right in the third look
identical. This module is the only place that transformation lives.
"""

from __future__ import annotations

import numpy as np

from gridiron.fields import FIELD_CENTER_Y, FieldSpec


def normalize_xy(
    x_abs: np.ndarray | float,
    y_abs: np.ndarray | float,
    los_x: float,
    ball_y: float,
    play_direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Absolute field coords -> (downfield, lateral) relative to the ball.

    `play_direction` is "right" when the offense attacks increasing x.
    """
    x_abs = np.asarray(x_abs, dtype=np.float64)
    y_abs = np.asarray(y_abs, dtype=np.float64)
    if play_direction == "right":
        return (x_abs - los_x).astype(np.float32), (y_abs - ball_y).astype(np.float32)
    if play_direction == "left":
        return (los_x - x_abs).astype(np.float32), (ball_y - y_abs).astype(np.float32)
    raise ValueError(f"play_direction must be 'left' or 'right', got {play_direction!r}")


def ball_y_from_center(ball_y: float, play_direction: str) -> float:
    """Signed lateral ball spot, positive toward the offense's right.

    This is what tells you which side is the field and which is the boundary,
    and it has to be sign-flipped with play direction like everything else.
    """
    offset = ball_y - FIELD_CENTER_Y
    return float(offset if play_direction == "right" else -offset)


def denormalize_xy(
    x_norm: np.ndarray | float,
    y_norm: np.ndarray | float,
    los_x: float,
    ball_y: float,
    play_direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `normalize_xy`, used to draw predictions back onto a real field."""
    x_norm = np.asarray(x_norm, dtype=np.float64)
    y_norm = np.asarray(y_norm, dtype=np.float64)
    if play_direction == "right":
        return (x_norm + los_x).astype(np.float32), (y_norm + ball_y).astype(np.float32)
    return (los_x - x_norm).astype(np.float32), (ball_y - y_norm).astype(np.float32)


def hash_side_for(field: FieldSpec, ball_y: float, play_direction: str) -> str:
    """Hash side expressed from the offense's point of view."""
    side = field.hash_of(ball_y)
    if side == "middle" or play_direction == "right":
        return side
    return "right" if side == "left" else "left"


def mirror_play(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip a play across the middle of the field.

    Used as training augmentation. Coverage labels are mirror-invariant except
    for Cover 6, which is quarters to one side and half to the other, so callers
    handling Cover 6 must flip the strength label alongside the coordinates.
    """
    return x, -y
