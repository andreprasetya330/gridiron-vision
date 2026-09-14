"""Tensorization and augmentation for the neural coverage model.

A play is a *set* of players, not a sequence, so the encoder must be permutation
invariant. Each player becomes one token carrying its whole trajectory; the model
attends across tokens.

The augmentations are not generic ML hygiene, they are a domain-gap strategy. Film
loses defenders behind other defenders, jitters coordinates through homography
error, and shifts the apparent line of scrimmage. Training with track dropout and
coordinate jitter is what stops a model trained on clean chip data from falling
apart the first time it sees video.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from gridiron.taxonomy import COVERAGE_INDEX, COVERAGES, DEFENDER_ROLES, ROLE_INDEX
from gridiron.tracking.schema import TIME_GRID, PlayTracks, frame_at

MAX_PLAYERS = 24
SAMPLE_TIMES: list[float] = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
PRESNAP_TIMES: list[float] = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.0]

# Per timestep: x, y, observed-mask. Plus per-player: side, is-defense, observed frac.
PER_STEP_FEATURES = 3
PER_PLAYER_FEATURES = 3


@dataclass
class AugmentConfig:
    enabled: bool = True
    dropout_prob: float = 0.35  # chance a play drops any defenders at all
    max_dropped: int = 2
    jitter_yd: float = 0.35
    mirror_prob: float = 0.5
    los_shift_yd: float = 0.4
    time_jitter_frames: int = 1


def play_to_tensor(
    play: PlayTracks,
    times: list[float] | None = None,
    rng: np.random.Generator | None = None,
    augment: AugmentConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (features[MAX_PLAYERS, F], mask[MAX_PLAYERS], role_targets[MAX_PLAYERS])."""
    times = times or SAMPLE_TIMES
    aug = augment if (augment and augment.enabled and rng is not None) else None

    players = sorted(play.players, key=lambda p: (p.side != "defense", p.track_id))

    drop: set[str] = set()
    if aug and rng.random() < aug.dropout_prob:
        defenders = [p for p in players if p.side == "defense"]
        n = int(rng.integers(1, aug.max_dropped + 1))
        if len(defenders) > n:
            # Deep defenders are the ones film actually loses.
            deepest = sorted(defenders, key=lambda p: -np.nanmax(np.where(np.isfinite(p.x), p.x, -np.inf)))
            for p in deepest[:n]:
                drop.add(p.track_id)

    mirror = bool(aug and rng.random() < aug.mirror_prob)
    los_shift = float(rng.normal(0, aug.los_shift_yd)) if aug else 0.0
    frame_shift = int(rng.integers(-aug.time_jitter_frames, aug.time_jitter_frames + 1)) if aug else 0

    n_steps = len(times)
    feat_dim = n_steps * PER_STEP_FEATURES + PER_PLAYER_FEATURES
    features = np.zeros((MAX_PLAYERS, feat_dim), dtype=np.float32)
    mask = np.zeros(MAX_PLAYERS, dtype=np.float32)
    roles = np.full(MAX_PLAYERS, -100, dtype=np.int64)

    slot = 0
    for p in players:
        if slot >= MAX_PLAYERS or p.track_id in drop:
            continue
        vec = np.zeros(feat_dim, dtype=np.float32)
        for i, t in enumerate(times):
            idx = int(np.clip(frame_at(t) + frame_shift, 0, len(TIME_GRID) - 1))
            x, y = float(p.x[idx]), float(p.y[idx])
            observed = 1.0 if (np.isfinite(x) and np.isfinite(y)) else 0.0
            if observed:
                x = x - los_shift
                if mirror:
                    y = -y
                if aug:
                    x += float(rng.normal(0, aug.jitter_yd))
                    y += float(rng.normal(0, aug.jitter_yd))
            else:
                x = y = 0.0
            base = i * PER_STEP_FEATURES
            # Scaled into roughly unit range so the encoder starts well conditioned.
            vec[base] = x / 15.0
            vec[base + 1] = y / 15.0
            vec[base + 2] = observed

        is_def = 1.0 if p.side == "defense" else 0.0
        vec[-3] = is_def
        vec[-2] = 1.0 - is_def
        vec[-1] = p.observed_fraction

        features[slot] = vec
        mask[slot] = 1.0
        if p.side == "defense" and p.role in ROLE_INDEX:
            roles[slot] = ROLE_INDEX[p.role]
        slot += 1

    return features, mask, roles


class CoverageDataset(Dataset):
    def __init__(
        self,
        plays: list[PlayTracks],
        mode: str = "postsnap",
        augment: AugmentConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.plays = [p for p in plays if p.coverage in COVERAGE_INDEX]
        self.times = PRESNAP_TIMES if mode == "presnap" else SAMPLE_TIMES
        self.augment = augment
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.plays)

    def __getitem__(self, index: int):
        play = self.plays[index]
        features, mask, roles = play_to_tensor(
            play, times=self.times, rng=self._rng, augment=self.augment
        )
        return {
            "features": torch.from_numpy(features),
            "mask": torch.from_numpy(mask),
            "roles": torch.from_numpy(roles),
            "coverage": torch.tensor(COVERAGE_INDEX[play.coverage], dtype=torch.long),
            "weight": torch.tensor(max(play.quality.score, 0.2), dtype=torch.float32),
        }

    @property
    def feature_dim(self) -> int:
        return len(self.times) * PER_STEP_FEATURES + PER_PLAYER_FEATURES

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights, softened.

        Cover 3 is a third of all snaps and Prevent is two percent. Without this
        the model learns to guess Cover 3 and stops.
        """
        counts = np.zeros(len(COVERAGES), dtype=np.float64)
        for play in self.plays:
            counts[COVERAGE_INDEX[play.coverage]] += 1
        counts = np.maximum(counts, 1.0)
        weights = (counts.sum() / counts) ** 0.5
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32)
