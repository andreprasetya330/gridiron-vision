"""Locked success criteria for the BDB post-snap coverage model.

These gates are the contract for the training loop. Do not edit the numbers
to make a run look finished. A criterion scores 8-10 only if it clears the
gate; the loop is done only when every criterion is >= 8.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gridiron.config import models_dir, predictions_path, PROJECT_ROOT
from gridiron.taxonomy import COVERAGES

LOCKED_VERSION = 1
LOCKED_DATE = "2026-09-14"
LOCKED_TARGET = "baseline_postsnap on BDB, forward week holdout"

# Pass line is 8. These values are frozen.
GATE8 = {
    "train_n": 5000,
    "test_n": 1500,
    "holdout_weeks": 3,
    "accuracy_lift": 0.15,
    "man_zone": 0.82,
    "shell": 0.70,
    "macro_f1": 0.40,
    "top2": 0.80,
    "min_predicted_classes": 6,
    "min_support_for_f1": 80,
    "min_f1_supported": 0.15,
    "max_predicted_share": 0.55,
    "min_predictions": 100,
}
GATE10 = {
    "accuracy_lift": 0.25,
    "man_zone": 0.90,
    "shell": 0.82,
    "macro_f1": 0.55,
    "top2": 0.90,
}

CRITERION_IDS = (
    "C1_forward_split",
    "C2_accuracy_lift",
    "C3_man_zone",
    "C4_shell",
    "C5_macro_f1",
    "C6_top2",
    "C7_class_coverage",
    "C8_artifacts",
    "C9_no_collapse",
    "C10_tests",
)


def score_higher_is_better(value: float, gate8: float, gate10: float) -> int:
    if value >= gate10:
        return 10
    if value >= gate8:
        span = max(gate10 - gate8, 1e-9)
        return min(10, 8 + int(((value - gate8) / span) * 2 + 1e-9))
    frac = max(0.0, value / max(gate8, 1e-9))
    return max(1, min(7, int(1 + frac * 7)))


def score_lower_is_better(value: float, gate8: float, gate10: float) -> int:
    return score_higher_is_better(-value, -gate8, -gate10)


@dataclass
class CriterionScore:
    id: str
    score: int
    evidence: str
    gate8: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "evidence": self.evidence,
            "gate8": self.gate8,
        }


def metrics_path(mode: str = "postsnap") -> Path:
    return models_dir() / f"baseline_{mode}_metrics.json"


def model_path(mode: str = "postsnap") -> Path:
    return models_dir() / f"baseline_{mode}.joblib"


def evaluate_report(report: dict[str, Any], *, run_tests: bool = True) -> list[CriterionScore]:
    metrics = report.get("metrics") or {}
    train_weeks = set(report.get("train_weeks") or [])
    test_weeks = set(report.get("test_weeks") or [])
    train_n = int(report.get("train_n") or 0)
    test_n = int(report.get("test_n") or metrics.get("n") or 0)
    overlap = sorted(train_weeks & test_weeks)
    holdout_weeks = len(test_weeks)

    if overlap:
        c1 = 1
        c1_ev = f"week leakage: {overlap}"
    elif holdout_weeks < GATE8["holdout_weeks"] or train_n < GATE8["train_n"] or test_n < GATE8["test_n"]:
        c1 = 4
        c1_ev = (
            f"weeks train={sorted(train_weeks)} test={sorted(test_weeks)} "
            f"n={train_n}/{test_n} (need {GATE8['holdout_weeks']} holdout weeks, "
            f"n>={GATE8['train_n']}/{GATE8['test_n']})"
        )
    else:
        c1 = 10 if holdout_weeks >= 3 and train_n >= 8000 else 8
        c1_ev = f"week split train={sorted(int(w) for w in train_weeks)} test={sorted(int(w) for w in test_weeks)} n={train_n}/{test_n}"

    lift = float(metrics.get("lift_over_majority") or 0.0)
    man_zone = float(metrics.get("man_zone_accuracy") or 0.0)
    shell = float(metrics.get("shell_accuracy") or 0.0)
    macro_f1 = float(metrics.get("macro_f1") or 0.0)
    top2 = float(metrics.get("top2_accuracy") or 0.0)
    per_class = metrics.get("per_class_f1") or {}
    true_counts = report.get("true_class_counts") or {}
    pred_counts = report.get("predicted_class_counts") or {}
    pred_total = sum(pred_counts.values()) or 1
    pred_share = max(pred_counts.values()) / pred_total if pred_counts else 1.0
    predicted_classes = len([c for c, n in pred_counts.items() if n > 0])

    weak_supported = []
    for name, n in true_counts.items():
        if int(n) >= GATE8["min_support_for_f1"] and float(per_class.get(name, 0.0)) < GATE8["min_f1_supported"]:
            weak_supported.append(f"{name} n={n} f1={per_class.get(name, 0):.3f}")
    if predicted_classes >= GATE8["min_predicted_classes"] and not weak_supported:
        c7 = 10 if predicted_classes >= 8 else 8
        c7_ev = f"predicted {predicted_classes}/8 classes; supported F1 ok"
    else:
        c7 = 5 if predicted_classes >= 4 else 2
        c7_ev = f"predicted {predicted_classes}/8; weak={weak_supported or 'none'}"

    c9 = 10 if pred_share <= GATE8["max_predicted_share"] else max(1, int(7 * (1.0 - pred_share) / (1.0 - GATE8["max_predicted_share"])))
    c9_ev = f"max predicted share={pred_share:.3f} (gate <= {GATE8['max_predicted_share']})"

    artifact_bits = []
    joblib = Path(report.get("model_path") or model_path())
    metrics_file = Path(report.get("metrics_path") or metrics_path())
    preds = predictions_path()
    if joblib.exists():
        try:
            from gridiron.coverage.baseline import CoverageBaseline

            CoverageBaseline.load(joblib)
            artifact_bits.append("model_loads")
        except Exception as exc:
            artifact_bits.append(f"model_load_failed:{exc}")
    else:
        artifact_bits.append("model_missing")
    if metrics_file.exists():
        artifact_bits.append("metrics_json")
    else:
        artifact_bits.append("metrics_missing")
    pred_n = 0
    if preds.exists():
        try:
            payload = json.loads(preds.read_text(encoding="utf-8"))
            pred_n = len(payload) if isinstance(payload, list) else len(payload)
        except Exception:
            pred_n = 0
    if pred_n >= GATE8["min_predictions"]:
        artifact_bits.append(f"predictions={pred_n}")
    else:
        artifact_bits.append(f"predictions_short={pred_n}")
    c8 = 10 if {"model_loads", "metrics_json"} <= set(artifact_bits) and pred_n >= GATE8["min_predictions"] else (
        6 if joblib.exists() else 2
    )
    if "model_load_failed" in " ".join(artifact_bits):
        c8 = min(c8, 3)

    test_score, test_ev = 1, "tests not run"
    if run_tests:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_coverage_model.py",
                "tests/test_bdb.py",
                "-m",
                "not slow",
                "-q",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            test_score, test_ev = 10, "pytest coverage+bdb (not slow) passed"
        else:
            test_score, test_ev = 3, (proc.stdout + proc.stderr)[-800:]

    return [
        CriterionScore("C1_forward_split", c1, c1_ev, f"disjoint weeks, holdout>={GATE8['holdout_weeks']}, n>={GATE8['train_n']}/{GATE8['test_n']}"),
        CriterionScore("C2_accuracy_lift", score_higher_is_better(lift, GATE8["accuracy_lift"], GATE10["accuracy_lift"]), f"lift={lift:.3f}", f">= {GATE8['accuracy_lift']}"),
        CriterionScore("C3_man_zone", score_higher_is_better(man_zone, GATE8["man_zone"], GATE10["man_zone"]), f"man/zone={man_zone:.3f}", f">= {GATE8['man_zone']}"),
        CriterionScore("C4_shell", score_higher_is_better(shell, GATE8["shell"], GATE10["shell"]), f"shell={shell:.3f}", f">= {GATE8['shell']}"),
        CriterionScore("C5_macro_f1", score_higher_is_better(macro_f1, GATE8["macro_f1"], GATE10["macro_f1"]), f"macroF1={macro_f1:.3f}", f">= {GATE8['macro_f1']}"),
        CriterionScore("C6_top2", score_higher_is_better(top2, GATE8["top2"], GATE10["top2"]), f"top2={top2:.3f}", f">= {GATE8['top2']}"),
        CriterionScore("C7_class_coverage", c7, c7_ev, f">= {GATE8['min_predicted_classes']} classes; F1>={GATE8['min_f1_supported']} if support>={GATE8['min_support_for_f1']}"),
        CriterionScore("C8_artifacts", c8, "; ".join(artifact_bits), f"loadable joblib + metrics json + >= {GATE8['min_predictions']} predictions"),
        CriterionScore("C9_no_collapse", c9, c9_ev, f"max class share <= {GATE8['max_predicted_share']}"),
        CriterionScore("C10_tests", test_score, test_ev, "pytest tests/test_coverage_model.py tests/test_bdb.py -m 'not slow'"),
    ]


def weakest(scores: list[CriterionScore]) -> CriterionScore:
    return min(scores, key=lambda s: (s.score, s.id))


def all_pass(scores: list[CriterionScore]) -> bool:
    return all(s.score >= 8 for s in scores)


def summary_table(scores: list[CriterionScore]) -> str:
    lines = [
        f"LOCKED v{LOCKED_VERSION} {LOCKED_DATE} — {LOCKED_TARGET}",
        f"{'id':22} {'score':>5}  evidence",
    ]
    for s in scores:
        flag = "PASS" if s.score >= 8 else "FAIL"
        lines.append(f"{s.id:22} {s.score:2d}/10 {flag}  {s.evidence}")
    failed = [s.id for s in scores if s.score < 8]
    lines.append("DONE" if not failed else "NOT DONE — below 8: " + ", ".join(failed))
    lines.append(f"weakest: {weakest(scores).id} ({weakest(scores).score})")
    return "\n".join(lines)
