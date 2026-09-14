"""Gradient boosting baseline for coverage classification.

Histogram gradient boosting is the right first model here: it eats NaNs natively,
which matters enormously because film-derived plays genuinely have missing
defenders, and it trains in seconds so you can iterate on features rather than on
learning rates.

Evaluation reports more than accuracy, because 8-class accuracy alone hides what a
coach cares about. A model that confuses Cover 3 with Cover 1 is far worse than one
that confuses Cover 2 Zone with Cover 6, and man/zone and shell accuracy expose
that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, log_loss

from gridiron.coverage.features import FeatureMode, feature_columns
from gridiron.taxonomy import COVERAGES, is_man, shell_of
from gridiron.tracking.schema import PlayTracks


@dataclass
class CoverageMetrics:
    n: int
    accuracy: float
    macro_f1: float
    top2_accuracy: float
    man_zone_accuracy: float
    shell_accuracy: float
    log_loss: float
    majority_accuracy: float
    per_class_f1: dict[str, float] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    @property
    def lift_over_majority(self) -> float:
        return self.accuracy - self.majority_accuracy

    def summary(self) -> str:
        return (
            f"n={self.n}  acc={self.accuracy:.3f} (majority {self.majority_accuracy:.3f}, "
            f"+{self.lift_over_majority:.3f})  macroF1={self.macro_f1:.3f}  "
            f"top2={self.top2_accuracy:.3f}  man/zone={self.man_zone_accuracy:.3f}  "
            f"shell={self.shell_accuracy:.3f}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "top2_accuracy": round(self.top2_accuracy, 4),
            "man_zone_accuracy": round(self.man_zone_accuracy, 4),
            "shell_accuracy": round(self.shell_accuracy, 4),
            "log_loss": round(self.log_loss, 4),
            "majority_accuracy": round(self.majority_accuracy, 4),
            "lift_over_majority": round(self.lift_over_majority, 4),
            "per_class_f1": {k: round(v, 4) for k, v in self.per_class_f1.items()},
            "labels": self.labels,
            "confusion": self.confusion,
        }


def evaluate_predictions(
    y_true: np.ndarray, proba: np.ndarray, classes: list[str]
) -> CoverageMetrics:
    order = np.argsort(-proba, axis=1)
    top1 = np.array([classes[i] for i in order[:, 0]])
    top2 = [{classes[i] for i in row[:2]} for row in order]

    accuracy = float((top1 == y_true).mean())
    top2_acc = float(np.mean([t in s for t, s in zip(y_true, top2)]))
    man_zone = float(np.mean([is_man(p) == is_man(t) for p, t in zip(top1, y_true)]))
    shell = float(np.mean([shell_of(p) == shell_of(t) for p, t in zip(top1, y_true)]))

    values, counts = np.unique(y_true, return_counts=True)
    majority = float(counts.max() / counts.sum())

    present = sorted(set(y_true) | set(top1))
    f1_per = f1_score(y_true, top1, average=None, labels=present, zero_division=0)

    try:
        ll = float(log_loss(y_true, proba, labels=classes))
    except ValueError:
        ll = float("nan")

    cm = confusion_matrix(y_true, top1, labels=classes)

    return CoverageMetrics(
        n=len(y_true),
        accuracy=accuracy,
        macro_f1=float(f1_score(y_true, top1, average="macro", zero_division=0)),
        top2_accuracy=top2_acc,
        man_zone_accuracy=man_zone,
        shell_accuracy=shell,
        log_loss=ll,
        majority_accuracy=majority,
        per_class_f1={c: float(v) for c, v in zip(present, f1_per)},
        confusion=cm.tolist(),
        labels=list(classes),
    )


class CoverageBaseline:
    """Gradient boosting over engineered features."""

    def __init__(self, mode: FeatureMode = "postsnap", **kwargs: Any) -> None:
        self.mode: FeatureMode = mode
        self.columns = feature_columns(mode)
        # Set at fit time: the subset of `columns` that had at least one observed
        # value. Everything downstream must use this, not `columns`.
        self.fitted_columns: list[str] = list(self.columns)
        self.classes: list[str] = list(COVERAGES)
        params = {
            "max_iter": 400,
            "learning_rate": 0.06,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "random_state": 0,
        }
        params.update(kwargs)
        self.model = HistGradientBoostingClassifier(**params)

    def _matrix(self, df: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
        columns = self.fitted_columns if columns is None else columns
        missing = [c for c in columns if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        return df[columns].to_numpy(dtype=np.float64)

    def fit(self, df: pd.DataFrame) -> CoverageBaseline:
        labeled = df[df["coverage"].notna()].copy()
        if labeled.empty:
            raise ValueError("no labeled plays to train on")

        # A feature can be entirely missing on a given corpus - a cue that needs
        # pre-snap motion is undefined on a set of plays with none, and a film
        # batch where registration failed loses whole feature families. Gradient
        # boosting handles NaN per row, but a column with zero observed values
        # has no bin edges to compute and raises deep inside the binner, so it
        # gets dropped here and remembered so scoring uses the same columns.
        full = self._matrix(labeled, self.columns)
        observed = np.isfinite(full).any(axis=0)
        self.fitted_columns = [c for c, keep in zip(self.columns, observed) if keep]
        if not self.fitted_columns:
            raise ValueError("every feature is missing on this corpus")

        X = full[:, observed]
        y = labeled["coverage"].to_numpy()
        self.model.fit(X, y)
        self.classes = list(self.model.classes_)
        return self

    @property
    def dropped_columns(self) -> list[str]:
        """Features that were unusable at fit time. Worth surfacing, not hiding."""
        return [c for c in self.columns if c not in set(self.fitted_columns)]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._matrix(df.copy()))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(df)
        return np.array([self.classes[i] for i in proba.argmax(axis=1)])

    def evaluate(self, df: pd.DataFrame) -> CoverageMetrics:
        labeled = df[df["coverage"].notna()].copy()
        proba = self.predict_proba(labeled)
        return evaluate_predictions(labeled["coverage"].to_numpy(), proba, self.classes)

    def permutation_importance(
        self, df: pd.DataFrame, n_repeats: int = 3, seed: int = 0
    ) -> pd.DataFrame:
        """Which features actually carry the signal.

        Permutation importance rather than impurity importance, because impurity
        importance is biased toward high-cardinality continuous features and half
        of these are counts.
        """
        from sklearn.inspection import permutation_importance as sk_perm

        labeled = df[df["coverage"].notna()].copy()
        X = self._matrix(labeled)
        y = labeled["coverage"].to_numpy()
        result = sk_perm(self.model, X, y, n_repeats=n_repeats, random_state=seed, scoring="accuracy")
        return (
            pd.DataFrame(
                {
                    "feature": self.fitted_columns,
                    "importance": result.importances_mean,
                    "std": result.importances_std,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "mode": self.mode,
                "columns": self.columns,
                "fitted_columns": self.fitted_columns,
                "classes": self.classes,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> CoverageBaseline:
        payload = joblib.load(path)
        obj = cls(mode=payload["mode"])
        obj.model = payload["model"]
        obj.columns = payload["columns"]
        obj.fitted_columns = payload.get("fitted_columns", payload["columns"])
        obj.classes = payload["classes"]
        return obj


def split_by_game(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole games. Used when the corpus is too short for a week split.

    Game ids from Big Data Bowl are chronological (`YYYYMMDDxx`), so sorting and
    taking the last slice is still a forward split, just at a coarser grain.
    """
    keys = df["game_id"].astype(str)
    games = sorted(k for k in keys.unique() if k and k != "nan")
    if len(games) < 2:
        cut = int(len(df) * (1.0 - test_fraction))
        cut = min(max(cut, 1), len(df) - 1) if len(df) > 1 else 0
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()
    n_test = max(1, int(round(len(games) * test_fraction)))
    n_test = min(n_test, len(games) - 1)
    test_games = set(games[-n_test:])
    train = df[~keys.isin(test_games)].copy()
    test = df[keys.isin(test_games)].copy()
    return train, test


def split_by_week(df: pd.DataFrame, holdout_weeks: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on early weeks, test on late ones.

    A random split leaks: two plays from the same game share personnel, weather,
    and game plan, so random splits flatter the model badly. Splitting forward in
    time is both honest and how the tool is actually used.

    When there are not enough weeks (the public BDB coverage labels are only
    week 1), fall back to holding out whole games rather than cutting the play
    list 80/20.
    """
    if "week" in df.columns and not df["week"].isna().all():
        weeks = sorted(df["week"].dropna().unique())
        if len(weeks) > holdout_weeks:
            test_weeks = set(weeks[-holdout_weeks:])
            train = df[~df["week"].isin(test_weeks)].copy()
            test = df[df["week"].isin(test_weeks)].copy()
            if len(train) and len(test):
                return train, test
    if "game_id" in df.columns and df["game_id"].notna().any():
        return split_by_game(df)
    cut = int(len(df) * 0.8)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def split_plays(
    plays: list[PlayTracks], holdout_weeks: int = 3
) -> tuple[list[PlayTracks], list[PlayTracks]]:
    """Same week-then-game split, for the neural net which trains on PlayTracks."""
    if not plays:
        return [], []
    frame = pd.DataFrame(
        {
            "play_id": [p.play_id for p in plays],
            "week": [p.week for p in plays],
            "game_id": [p.game_key for p in plays],
        }
    )
    train_df, test_df = split_by_week(frame, holdout_weeks=holdout_weeks)
    test_ids = set(test_df["play_id"])
    train = [p for p in plays if p.play_id not in test_ids]
    test = [p for p in plays if p.play_id in test_ids]
    return train, test


def describe_split(train: pd.DataFrame | list, test: pd.DataFrame | list) -> str:
    """One-line summary of which grain the split used."""

    def _frame(obj: pd.DataFrame | list) -> pd.DataFrame:
        if isinstance(obj, pd.DataFrame):
            return obj
        return pd.DataFrame(
            {
                "week": [getattr(p, "week", None) for p in obj],
                "game_id": [getattr(p, "game_key", None) for p in obj],
            }
        )

    train_df, test_df = _frame(train), _frame(test)
    train_weeks = set(train_df["week"].dropna()) if "week" in train_df.columns else set()
    test_weeks = set(test_df["week"].dropna()) if "week" in test_df.columns else set()
    if train_weeks and test_weeks and not train_weeks & test_weeks:
        return (
            f"week split: train weeks {sorted(int(w) for w in train_weeks)} "
            f"({len(train_df)} plays), holdout weeks {sorted(int(w) for w in test_weeks)} "
            f"({len(test_df)} plays)"
        )
    train_games = set(train_df["game_id"].dropna()) if "game_id" in train_df.columns else set()
    test_games = set(test_df["game_id"].dropna()) if "game_id" in test_df.columns else set()
    if train_games or test_games:
        return (
            f"game split: {len(train_games)} games / {len(train_df)} plays train, "
            f"{len(test_games)} games / {len(test_df)} plays holdout"
        )
    return f"positional split: {len(train_df)} train / {len(test_df)} holdout"
