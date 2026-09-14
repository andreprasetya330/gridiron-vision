"""Snap detection and play segmentation.

The snap is the anchor for everything. Cue extraction reads the 1.5 seconds before
it, the coverage model reads the 2.5 seconds after it, and an error of half a
second turns a two-high shell rotating late into a two-high shell that never
rotated.

The signal is unmistakable once you look for it: twenty-two people stand nearly
still, and then all of them move at once. Total motion energy across the frame is
flat and then spikes, and the spike is the snap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class SnapCandidate:
    frame_index: int
    score: float
    stillness_before: float
    confidence: float = 0.0


@dataclass
class PlaySegment:
    start_frame: int
    snap_frame: int
    end_frame: int
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)


def motion_energy(positions_by_frame: list[dict[int, tuple[float, float]]]) -> np.ndarray:
    """Mean per-player displacement between consecutive frames.

    Computed from tracked positions rather than raw pixels, so a camera pan does
    not read as a snap - a pan moves everyone identically, and subtracting the
    median displacement removes it.
    """
    n = len(positions_by_frame)
    energy = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        previous, current = positions_by_frame[i - 1], positions_by_frame[i]
        shared = set(previous) & set(current)
        if not shared:
            continue
        deltas = np.array(
            [
                [current[t][0] - previous[t][0], current[t][1] - previous[t][1]]
                for t in shared
            ]
        )
        # Remove global camera motion before measuring player motion.
        deltas = deltas - np.median(deltas, axis=0)
        energy[i] = float(np.mean(np.linalg.norm(deltas, axis=1)))
    return energy


def find_snap(
    energy: np.ndarray,
    fps: float = 30.0,
    stillness_window: float = 0.7,
    min_ratio: float = 2.5,
) -> SnapCandidate | None:
    """Locate the frame where a still formation explodes into motion."""
    if energy.size < 10:
        return None

    smoothed = _smooth(energy, max(3, int(fps * 0.1)))
    window = max(2, int(stillness_window * fps))

    best: SnapCandidate | None = None
    for i in range(window, len(smoothed) - 2):
        before = smoothed[i - window : i]
        after = smoothed[i : i + max(2, int(fps * 0.3))]
        if before.size == 0 or after.size == 0:
            continue
        stillness = float(np.mean(before))
        burst = float(np.mean(after))
        if stillness <= 1e-6:
            ratio = burst / 1e-6
        else:
            ratio = burst / stillness
        if ratio < min_ratio:
            continue
        score = ratio * burst
        if best is None or score > best.score:
            best = SnapCandidate(frame_index=i, score=score, stillness_before=stillness)

    if best is None:
        return None
    best.confidence = float(np.clip(np.log1p(best.score) / 4.0, 0.0, 1.0))
    return best


def segment_plays(
    energy: np.ndarray,
    fps: float = 30.0,
    min_gap_seconds: float = 8.0,
    pre_snap_seconds: float = 3.0,
    post_snap_seconds: float = 5.0,
) -> list[PlaySegment]:
    """Split a continuous film reel into individual plays.

    Hudl cutups arrive one play per clip and skip this entirely. A full-game
    recording needs it, and the same stillness-then-burst signature works, with a
    minimum spacing so a single snap does not register several times.
    """
    if energy.size < int(fps * 4):
        return []

    smoothed = _smooth(energy, max(3, int(fps * 0.15)))
    threshold = float(np.median(smoothed) + 2.5 * _mad(smoothed))
    min_gap = int(min_gap_seconds * fps)

    segments: list[PlaySegment] = []
    last_snap = -min_gap
    for i in range(int(fps), len(smoothed) - 1):
        if smoothed[i] < threshold or i - last_snap < min_gap:
            continue
        before = smoothed[max(0, i - int(fps * 0.7)) : i]
        if before.size and float(np.mean(before)) * 2.0 > smoothed[i]:
            continue
        start = max(0, i - int(pre_snap_seconds * fps))
        end = min(len(smoothed) - 1, i + int(post_snap_seconds * fps))
        segments.append(
            PlaySegment(
                start_frame=start,
                snap_frame=i,
                end_frame=end,
                confidence=float(np.clip(smoothed[i] / max(threshold, 1e-6) / 3.0, 0, 1)),
            )
        )
        last_snap = i
    return segments


def refine_with_ball(
    snap_frame: int, ball_positions: dict[int, tuple[float, float]], fps: float = 30.0
) -> int:
    """Nudge the snap frame using ball motion, when the ball was detected.

    The ball is small, fast, and frequently occluded, so it refines a snap estimate
    but never establishes one.
    """
    if not ball_positions:
        return snap_frame
    window = int(fps * 0.5)
    frames = sorted(f for f in ball_positions if abs(f - snap_frame) <= window)
    if len(frames) < 3:
        return snap_frame

    best_frame, best_delta = snap_frame, 0.0
    for a, b in zip(frames, frames[1:]):
        pa, pb = ball_positions[a], ball_positions[b]
        delta = float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))
        if delta > best_delta:
            best_delta, best_frame = delta, a
    return best_frame


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values.astype(np.float64)
    kernel = np.ones(window) / window
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def _mad(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median))) or 1e-6
