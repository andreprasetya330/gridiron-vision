"""Set transformer over the 22 player tracks.

Two heads share one encoder:

- a **coverage head** predicting the team call from a pooled play embedding
- a **role head** predicting man / underneath zone / deep zone / blitz for each
  defender token

The role head earns its place twice over. It is what makes the film overlay useful
- a label on each defender beats a single team-level guess by a mile - and it acts
as dense supervision that makes the coverage head better, because eleven role
labels per play carry far more signal than one coverage label.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from gridiron.taxonomy import COVERAGES, DEFENDER_ROLES


@dataclass
class ModelConfig:
    feature_dim: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.15
    n_coverages: int = len(COVERAGES)
    n_roles: int = len(DEFENDER_ROLES)


class CoverageNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = nn.Sequential(
            nn.Linear(config.feature_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)

        # Attention pooling rather than mean pooling: the safeties decide the
        # coverage and the offensive line never does, so let the model weight them.
        self.pool_query = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )

        self.coverage_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.n_coverages),
        )
        self.role_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.n_roles),
        )

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """features (B, P, F), mask (B, P) with 1 for real players."""
        tokens = self.embed(features)
        pad = mask < 0.5
        # A play with zero valid tokens would make attention produce NaNs; keep
        # the first slot alive as a harmless sentinel.
        all_pad = pad.all(dim=1)
        if all_pad.any():
            pad = pad.clone()
            pad[all_pad, 0] = False

        encoded = self.encoder(tokens, src_key_padding_mask=pad)

        query = self.pool_query.expand(encoded.size(0), -1, -1)
        pooled, _ = self.pool_attn(query, encoded, encoded, key_padding_mask=pad)
        pooled = pooled.squeeze(1)

        return self.coverage_head(pooled), self.role_head(encoded)


def compute_loss(
    coverage_logits: torch.Tensor,
    role_logits: torch.Tensor,
    coverage_target: torch.Tensor,
    role_target: torch.Tensor,
    weight: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    role_loss_weight: float = 0.4,
    label_smoothing: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    per_play = F.cross_entropy(
        coverage_logits,
        coverage_target,
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    # Weight each play by its track quality: a play where two defenders were never
    # seen should not pull the model as hard as a clean one.
    coverage_loss = (per_play * weight).sum() / weight.sum().clamp_min(1e-6)

    role_loss = torch.tensor(0.0, device=coverage_logits.device)
    if (role_target != -100).any():
        role_loss = F.cross_entropy(
            role_logits.reshape(-1, role_logits.size(-1)),
            role_target.reshape(-1),
            ignore_index=-100,
        )

    total = coverage_loss + role_loss_weight * role_loss
    return total, {
        "loss": float(total.detach()),
        "coverage_loss": float(coverage_loss.detach()),
        "role_loss": float(role_loss.detach()),
    }
