"""Training loop for the set transformer, with honest confidence calibration.

Calibration is not a nicety here. The overlay prints a probability next to a
coverage name, and a coach will read "82%" as a claim about the world. Raw neural
network softmax outputs are systematically overconfident, so a temperature is
fitted on held-out data and applied to everything the UI ever sees.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gridiron.coverage.baseline import CoverageMetrics, evaluate_predictions
from gridiron.coverage.dataset import AugmentConfig, CoverageDataset, play_to_tensor
from gridiron.coverage.model import CoverageNet, ModelConfig, compute_loss
from gridiron.taxonomy import COVERAGES, DEFENDER_ROLES
from gridiron.tracking.schema import PlayTracks


@dataclass
class TrainConfig:
    mode: str = "postsnap"
    epochs: int = 40
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    role_loss_weight: float = 0.4
    patience: int = 8
    seed: int = 0
    device: str | None = None


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _loader(dataset: CoverageDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,  # Windows + small tensors: workers cost more than they save
    )


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """One-parameter temperature scaling, fitted by LBFGS on the validation set."""
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().detach().clamp(0.25, 8.0))


@torch.no_grad()
def _collect(model: CoverageNet, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_targets, role_correct, role_total = [], [], 0, 0
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        coverage_logits, role_logits = model(features, mask)
        all_logits.append(coverage_logits.cpu())
        all_targets.append(batch["coverage"])

        roles = batch["roles"].to(device)
        valid = roles != -100
        if valid.any():
            pred = role_logits.argmax(dim=-1)
            role_correct += int((pred[valid] == roles[valid]).sum())
            role_total += int(valid.sum())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    role_acc = role_correct / role_total if role_total else float("nan")
    return logits, targets, role_acc


class TrainedCoverageNet:
    """A trained model plus everything needed to score a play honestly."""

    def __init__(
        self,
        model: CoverageNet,
        mode: str,
        temperature: float = 1.0,
        metrics: dict | None = None,
    ) -> None:
        self.model = model
        self.mode = mode
        self.temperature = temperature
        self.metrics = metrics or {}
        self.classes = list(COVERAGES)
        self.roles = list(DEFENDER_ROLES)

    @torch.no_grad()
    def predict_play(self, play: PlayTracks, device: torch.device | None = None) -> dict:
        from gridiron.coverage.dataset import PRESNAP_TIMES, SAMPLE_TIMES

        device = device or next(self.model.parameters()).device
        times = PRESNAP_TIMES if self.mode == "presnap" else SAMPLE_TIMES
        features, mask, _ = play_to_tensor(play, times=times)
        f = torch.from_numpy(features).unsqueeze(0).to(device)
        m = torch.from_numpy(mask).unsqueeze(0).to(device)

        self.model.eval()
        coverage_logits, role_logits = self.model(f, m)
        probs = torch.softmax(coverage_logits / self.temperature, dim=-1)[0].cpu().numpy()
        role_probs = torch.softmax(role_logits, dim=-1)[0].cpu().numpy()

        # Map role predictions back onto the defenders in their sorted order,
        # matching how play_to_tensor filled the slots.
        players = sorted(play.players, key=lambda p: (p.side != "defense", p.track_id))
        roles: dict[str, dict] = {}
        for slot, p in enumerate(players[: len(mask)]):
            if mask[slot] < 0.5 or p.side != "defense":
                continue
            dist = role_probs[slot]
            roles[p.track_id] = {
                "role": self.roles[int(dist.argmax())],
                "confidence": float(dist.max()),
                "probabilities": {r: float(v) for r, v in zip(self.roles, dist)},
            }

        order = np.argsort(-probs)
        return {
            "play_id": play.play_id,
            "coverage": self.classes[int(order[0])],
            "confidence": float(probs[order[0]]),
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            "runner_up": self.classes[int(order[1])],
            "roles": roles,
            "quality_score": round(play.quality.score, 3),
            "usable": play.quality.usable,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": asdict(self.model.config),
                "mode": self.mode,
                "temperature": self.temperature,
                "metrics": self.metrics,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device | None = None) -> TrainedCoverageNet:
        device = device or pick_device()
        payload = torch.load(path, map_location=device, weights_only=False)
        model = CoverageNet(ModelConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        return cls(model, payload["mode"], payload["temperature"], payload.get("metrics"))


def train(
    train_plays: list[PlayTracks],
    val_plays: list[PlayTracks],
    config: TrainConfig | None = None,
    verbose: bool = True,
) -> tuple[TrainedCoverageNet, CoverageMetrics]:
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    device = pick_device(config.device)

    train_ds = CoverageDataset(train_plays, mode=config.mode, augment=AugmentConfig(), seed=config.seed)
    val_ds = CoverageDataset(val_plays, mode=config.mode, augment=None, seed=config.seed)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("need labeled plays in both train and validation sets")

    train_loader = _loader(train_ds, config.batch_size, shuffle=True)
    val_loader = _loader(val_ds, config.batch_size, shuffle=False)

    model = CoverageNet(ModelConfig(feature_dim=train_ds.feature_dim)).to(device)
    class_weights = train_ds.class_weights().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = max(1, config.epochs * len(train_loader))
    warmup = max(1, int(total_steps * config.warmup_frac))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    step = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            coverage = batch["coverage"].to(device)
            roles = batch["roles"].to(device)
            weight = batch["weight"].to(device)

            coverage_logits, role_logits = model(features, mask)
            loss, parts = compute_loss(
                coverage_logits,
                role_logits,
                coverage,
                roles,
                weight,
                class_weights=class_weights,
                role_loss_weight=config.role_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            epoch_losses.append(parts["loss"])

        logits, targets, role_acc = _collect(model, val_loader, device)
        val_loss = float(torch.nn.functional.cross_entropy(logits, targets))
        val_acc = float((logits.argmax(dim=-1) == targets).float().mean())

        if verbose:
            print(
                f"epoch {epoch + 1:3d}/{config.epochs}  "
                f"train {np.mean(epoch_losses):.4f}  val {val_loss:.4f}  "
                f"acc {val_acc:.3f}  role_acc {role_acc:.3f}"
            )

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                if verbose:
                    print(f"early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    logits, targets, role_acc = _collect(model, val_loader, device)
    temperature = fit_temperature(logits, targets)

    probs = torch.softmax(logits / temperature, dim=-1).numpy()
    y_true = np.array([COVERAGES[i] for i in targets.numpy()])
    metrics = evaluate_predictions(y_true, probs, list(COVERAGES))

    trained = TrainedCoverageNet(
        model,
        mode=config.mode,
        temperature=temperature,
        metrics={**metrics.to_dict(), "role_accuracy": round(float(role_acc), 4)},
    )
    return trained, metrics


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
