"""Roboflow workflow parsing, without hitting the hosted API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gridiron.taxonomy import COVERAGES
from gridiron.vision.roboflow import (
    MINIMAP_WIDTH_PX,
    coverage_prediction_row,
    map_coverage_label,
    minimap_px_to_field,
    parse_coverage,
    parse_field_players,
    parse_workflow_result,
)
from gridiron.vision.pipeline import FilmPipeline, PipelineConfig, _associate_field_players


def test_map_cover_0_to_taxonomy():
    assert map_coverage_label("Cover 0") == "Cover 0 Man"
    assert map_coverage_label("cover_3") == "Cover 3 Zone"
    assert map_coverage_label("Cover 2 Man") == "Cover 2 Man"


def test_minimap_pixels_are_ten_per_yard():
    x, y = minimap_px_to_field(100, 266.5)
    assert x == pytest.approx(10.0)
    assert y == pytest.approx(26.65)


def test_parse_coverage_from_classification_payload():
    call = parse_coverage(
        {
            "top": "Cover 0",
            "confidence": 0.95,
            "predictions": [
                {"class": "Cover 0", "confidence": 0.95},
                {"class": "Cover 1", "confidence": 0.03},
                {"class": "Cover 3", "confidence": 0.02},
            ],
        }
    )
    assert call.coverage == "Cover 0 Man"
    assert call.confidence == pytest.approx(0.95)
    assert call.probabilities["Cover 0 Man"] == pytest.approx(0.95)
    assert call.runner_up == "Cover 1 Man"


def test_parse_field_players_drops_officials():
    players = parse_field_players(
        {
            "players": [
                {"class": "defense_player", "x": 700, "y": 200},
                {"class": "official", "x": 10, "y": 10},
                {
                    "class": "offense_player",
                    "field_x_normalized": 0.4,
                    "field_y_normalized": 0.5,
                },
            ]
        }
    )
    assert len(players) == 2
    assert players[0].side == "defense"
    assert players[0].field_x == pytest.approx(70.0)
    assert players[1].side == "offense"
    assert players[1].field_x == pytest.approx(0.4 * MINIMAP_WIDTH_PX / 10.0)


def test_coverage_prediction_row_fills_taxonomy():
    call = parse_coverage({"top": "Cover 0", "confidence": 0.95, "predictions": []})
    row = coverage_prediction_row("uga-1", call, quality_score=0.4, usable=False)
    assert row["coverage"] == "Cover 0 Man"
    assert row["source"] == "roboflow"
    assert row["experimental"] is True
    assert set(row["probabilities"]) == set(COVERAGES)


def _stub_frame(n_each: int = 11):
    from gridiron.vision.roboflow import FieldPlayer, CoverageCall, WorkflowFrame

    players = []
    for i in range(n_each):
        players.append(
            FieldPlayer(
                class_name="offense_player",
                field_x=48.0 + i * 0.15,
                field_y=12.0 + i * 2.4,
                minimap_x=480,
                minimap_y=120,
                confidence=0.9,
            )
        )
        players.append(
            FieldPlayer(
                class_name="defense_player",
                field_x=56.0 + (i % 3) * 0.8,
                field_y=10.0 + i * 2.5,
                minimap_x=560,
                minimap_y=100,
                confidence=0.88,
            )
        )
    H = np.eye(3).tolist()
    return WorkflowFrame(
        coverage=CoverageCall(
            coverage="Cover 0 Man",
            confidence=0.95,
            probabilities={name: (0.95 if name == "Cover 0 Man" else 0.01) for name in COVERAGES},
            raw_top="Cover 0",
        ),
        players=players,
        homography=H,
        image_width=64,
        image_height=48,
        notes=["test"],
    )


class _FakeWorkflow:
    def run_frame(self, frame):
        return _stub_frame()


def test_associate_keeps_side_and_nearest_id():
    from gridiron.vision.roboflow import FieldPlayer

    prev = {1: (10.0, 10.0), 2: (20.0, 20.0)}
    sides = {1: "offense", 2: "defense"}
    incoming = [
        FieldPlayer("defense_player", 20.4, 19.7, 0, 0),
        FieldPlayer("offense_player", 10.2, 10.1, 0, 0),
    ]
    assigned, next_id = _associate_field_players(incoming, prev, sides, 3)
    assert assigned[2].side == "defense"
    assert assigned[1].side == "offense"
    assert next_id == 3


def test_pipeline_still_uses_workflow_stub(tmp_path: Path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    image = tmp_path / "snap.jpg"
    cv2.imwrite(str(image), np.zeros((48, 64, 3), dtype=np.uint8))
    monkeypatch.setenv("GRIDIRON_DATA_DIR", str(tmp_path / "data"))
    from gridiron import config as cfg

    cfg.data_dir.cache_clear()

    pipeline = FilmPipeline(
        PipelineConfig(backend="roboflow", league="ncaa"),
        workflow=_FakeWorkflow(),
    )
    plays = pipeline.process(image, play_id="uga-still")
    assert len(plays) == 1
    play = plays[0]
    assert play.source == "film"
    assert play.quality.defenders_detected >= 9
    assert play.quality.offense_detected >= 9
    assert pipeline.predictions[0]["coverage"] == "Cover 0 Man"
    assert "experimental" in "".join(play.quality.notes).lower() or play.quality.notes


def test_parse_workflow_result_reads_listed_outputs():
    parsed = parse_workflow_result(
        [
            {
                "coverage_predictions": {
                    "top": "Cover 0",
                    "confidence": 0.95,
                    "predictions": [{"class": "Cover 0", "confidence": 0.95}],
                },
                "field_coordinates": {
                    "players": [{"class": "defense_player", "x": 200, "y": 100}],
                    "calibration_polygon": [[775, 58], [1813, 85], [1845, 993], [328, 874]],
                    "minimap_width": 1200,
                    "minimap_height": 533,
                },
                "player_predictions": {
                    "predictions": [
                        {
                            "x": 100,
                            "y": 200,
                            "width": 40,
                            "height": 80,
                            "class": "defense_player",
                            "confidence": 0.91,
                        }
                    ],
                    "image": {"width": 1920, "height": 1080},
                },
                "minimap": None,
                "output_image": None,
            }
        ]
    )
    assert parsed.coverage.coverage == "Cover 0 Man"
    assert parsed.players[0].field_x == pytest.approx(20.0)
    assert parsed.image_width == 1920
