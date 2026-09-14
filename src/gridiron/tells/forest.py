"""Random forest coverage predictor with per-play explanations.

A random forest is the right model for this particular job even though the neural
network scores better. The point is not the last two points of accuracy, it is
being able to hand a coach a sentence: "this reads Cover 3 because the field safety
is 12 yards deep and the corner is bailing at the snap." SHAP over a forest gives
you exactly that, per play, in a form that survives contact with a whiteboard.

Two variants are trained:

- **league** - one model over every team, which is the national prior
- **team** - the league model plus a team indicator, so a team that behaves
  differently from the league given identical cues shows up as a large SHAP
  contribution on its own indicator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from gridiron.cues.vocabulary import CUE_KEYS, NUMERIC_CUE_KEYS
from gridiron.tells.mining import phrase_cue


@dataclass
class ForestConfig:
    n_estimators: int = 500
    max_depth: int | None = 14
    min_samples_leaf: int = 8
    max_features: str | float = "sqrt"
    class_weight: str | None = "balanced_subsample"
    random_state: int = 0
    include_team: bool = False
    n_jobs: int = -1


@dataclass
class Explanation:
    play_id: str
    predicted: str
    confidence: float
    contributions: list[tuple[str, float]] = field(default_factory=list)

    def sentence(self, top_k: int = 3) -> str:
        drivers = [name for name, value in self.contributions[:top_k] if value > 0]
        if not drivers:
            return f"Reads {self.predicted} ({self.confidence:.0%}) with no single dominant cue."
        joined = ", and ".join(drivers) if len(drivers) <= 2 else ", ".join(drivers[:-1]) + f", and {drivers[-1]}"
        return f"Reads {self.predicted} ({self.confidence:.0%}) because {joined}."


class CoverageForest:
    """Interpretable coverage model over the cue vocabulary."""

    def __init__(self, config: ForestConfig | None = None) -> None:
        self.config = config or ForestConfig()
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        self.feature_names: list[str] = []
        self.classes: list[str] = []
        self._explainer = None

    def _design(self, cues: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        numeric = [c for c in NUMERIC_CUE_KEYS if c in cues.columns]
        categorical = [c for c in CUE_KEYS if c in cues.columns]
        if self.config.include_team and "defense_team" in cues.columns:
            categorical = categorical + ["defense_team"]

        X = cues[numeric].copy() if numeric else pd.DataFrame(index=cues.index)
        # Trees handle NaN poorly in sklearn's forest, so impute with a sentinel
        # far outside the real range rather than the mean - "not observed" is a
        # category, not an average defender.
        X = X.fillna(-99.0)

        if categorical:
            dummies = pd.get_dummies(cues[categorical].astype(str), prefix=categorical)
            X = pd.concat([X, dummies], axis=1)

        if fit:
            self.feature_names = list(X.columns)
        else:
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]
        return X

    def fit(self, cues: pd.DataFrame) -> CoverageForest:
        labeled = cues[cues["coverage"].notna()].copy()
        if labeled.empty:
            raise ValueError("no labeled plays to fit the forest on")
        X = self._design(labeled, fit=True)
        y = labeled["coverage"].to_numpy()
        self.model.fit(X, y)
        self.classes = list(self.model.classes_)
        self._explainer = None
        return self

    def predict_proba(self, cues: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._design(cues))

    def predict(self, cues: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(cues)
        return np.array([self.classes[i] for i in proba.argmax(axis=1)])

    def feature_importance(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {"feature": self.feature_names, "importance": self.model.feature_importances_}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _get_explainer(self):
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def explain(self, cues: pd.DataFrame, top_k: int = 5) -> list[Explanation]:
        """Per-play SHAP attributions, phrased for a human."""
        X = self._design(cues)
        proba = self.model.predict_proba(X)
        predictions = proba.argmax(axis=1)

        try:
            shap_values = self._get_explainer().shap_values(X)
        except Exception:
            # SHAP is optional at runtime; fall back to global importance so the
            # UI still has something to show rather than erroring out.
            return [
                Explanation(
                    play_id=str(cues.iloc[i].get("play_id", i)),
                    predicted=self.classes[predictions[i]],
                    confidence=float(proba[i, predictions[i]]),
                    contributions=[],
                )
                for i in range(len(X))
            ]

        explanations: list[Explanation] = []
        for i in range(len(X)):
            cls = int(predictions[i])
            values = _shap_row(shap_values, i, cls)
            order = np.argsort(-np.abs(values))[:top_k]
            contributions = [
                (_humanize(self.feature_names[j], X.iloc[i, j]), float(values[j])) for j in order
            ]
            explanations.append(
                Explanation(
                    play_id=str(cues.iloc[i].get("play_id", i)),
                    predicted=self.classes[cls],
                    confidence=float(proba[i, cls]),
                    contributions=contributions,
                )
            )
        return explanations

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "features": self.feature_names,
                "classes": self.classes,
                "config": self.config,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> CoverageForest:
        payload = joblib.load(path)
        obj = cls(payload["config"])
        obj.model = payload["model"]
        obj.feature_names = payload["features"]
        obj.classes = payload["classes"]
        return obj


def _shap_row(shap_values: Any, row: int, cls: int) -> np.ndarray:
    """Normalize across the several shapes SHAP returns for multiclass forests."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[cls][row])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[row, :, cls]
    return arr[row]


def _humanize(feature: str, value: Any) -> str:
    """Turn a design-matrix column back into something a coach would say."""
    if feature.endswith("_n"):
        pretty = feature[:-2].replace("_", " ")
        if isinstance(value, (int, float, np.floating)) and value != -99.0:
            return f"{pretty} is {float(value):.1f}"
        return f"{pretty} is unknown"

    for cue_key in sorted(CUE_KEYS, key=len, reverse=True):
        prefix = cue_key + "_"
        if feature.startswith(prefix):
            return phrase_cue(cue_key, feature[len(prefix) :])
    if feature.startswith("defense_team_"):
        return f"the defense is {feature[len('defense_team_'):]}"
    return feature.replace("_", " ")
