"""Player workflow contract, grounded in a live american-football-player-trackin run."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridiron.vision.roboflow import (
    PLAYER_WORKFLOW_OUTPUT_KEYS,
    compact_detections,
    detection_output,
    parse_player_result,
    player_workflow_parameters,
    run_player_workflow,
)

# Shape captured from workflows_run on a UGA All-22 still (1920x1080).
LIVE_PLAYER_RESULT = {
    "predictions": {
        "image": {"width": 1920, "height": 1080},
        "predictions": [
            {
                "width": 47.0,
                "height": 66.0,
                "x": 867.5,
                "y": 313.0,
                "confidence": 0.90,
                "class_id": 1,
                "class": "offense_player",
                "detection_id": "61a798c4-0e54-42bf-81a6-0747a580e280",
                "parent_id": "image",
                "points": [[1, 2], [3, 4]],
            },
            {
                "width": 28.0,
                "height": 79.0,
                "x": 1035.0,
                "y": 511.5,
                "confidence": 0.89,
                "class_id": 0,
                "class": "defense_player",
                "detection_id": "4bed693d-fea3-46b9-adee-8a9877f807ab",
                "parent_id": "image",
            },
            {
                "width": 32.0,
                "height": 70.0,
                "x": 1122.0,
                "y": 485.0,
                "confidence": 0.88,
                "class_id": 2,
                "class": "official",
                "detection_id": "35f7eab3-1ca4-406a-b668-3f5c56cf482d",
                "parent_id": "image",
            },
        ],
    },
    "inference_id": "242c3384-67db-4c06-8d4f-c1c426a88bad",
    "model_id": "andre-4cotb/american-football-player-trackin-1-rfdetr-small-t1",
}

SAMPLE_IMAGE = Path(__file__).resolve().parents[1] / "uga-play.jpg"


def test_parameters_match_declared_workflow_inputs():
    params = player_workflow_parameters()
    assert set(params) == {"confidence", "iou_threshold", "class_agnostic_nms", "max_detections"}
    assert params["class_agnostic_nms"] is False


def test_detection_output_finds_live_key_without_hardcoding():
    key, payload = detection_output(LIVE_PLAYER_RESULT)
    assert key == "predictions"
    rows = compact_detections(payload)
    assert all("points" not in row for row in rows)
    assert {row["class"] for row in rows} == {"offense_player", "defense_player", "official"}


def test_parse_player_result_reads_live_output_keys():
    parsed = parse_player_result([LIVE_PLAYER_RESULT], image_size=(1920, 1080))
    assert set(LIVE_PLAYER_RESULT) >= set(PLAYER_WORKFLOW_OUTPUT_KEYS)
    assert parsed.image_width == 1920
    assert any(p.side == "offense" for p in parsed.players)
    assert any(p.side == "defense" for p in parsed.players)
    assert all(p.class_name != "official" for p in parsed.players)
    assert parsed.detections
    assert parsed.model_id and "rfdetr" in parsed.model_id
    from gridiron.vision.roboflow import overlay_boxes_from_detections

    boxes = overlay_boxes_from_detections(parsed.detections)
    assert {row["class_name"] for row in boxes} == {
        "offense_player",
        "defense_player",
        "official",
    }
    assert all(row["box"] and len(row["box"]) == 4 for row in boxes)


@pytest.mark.skipif(not SAMPLE_IMAGE.exists(), reason="uga-play.jpg is local-only")
def test_run_player_workflow_returns_declared_outputs():
    pytest.importorskip("inference_sdk")
    from gridiron.config import RoboflowSettings

    if not RoboflowSettings().api_key:
        pytest.skip("ROBOFLOW_API_KEY is not set")
    payload = run_player_workflow(SAMPLE_IMAGE)
    missing = [key for key in PLAYER_WORKFLOW_OUTPUT_KEYS if key not in payload]
    assert not missing, f"missing workflow outputs: {missing}"
    key, detections = detection_output(payload)
    assert key is not None
    assert compact_detections(detections)
    assert "rfdetr" in str(payload.get("model_id") or "").lower() or payload.get("model_id")
