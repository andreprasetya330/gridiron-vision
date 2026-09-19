"""Roboflow hosted workflow as the film vision backend.

The saved workflow (`defensive-coverage-analysis-...`) does three jobs that
used to be local YOLO + line registration + a tracking-data coverage model:

1. Detect `offense_player` / `defense_player` / `official`
2. Project feet through a UGA-broadcast homography onto a 120-yard minimap
3. Classify the shell from that minimap

The homography is calibrated to one camera. Other pans, zooms, or venues need
a new polygon (or automatic landmark registration). Coverage labels from the
workflow are an experimental baseline, not a coach-validated call.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gridiron.config import RoboflowSettings
from gridiron.fields import FIELD_LENGTH_YD, FIELD_WIDTH_YD
from gridiron.taxonomy import COVERAGES
from gridiron.vision import require

# Matches the Football_Field_Minimap block in the published workflow.
MINIMAP_WIDTH_PX = 1200
MINIMAP_HEIGHT_PX = 533
ENDZONE_PX = 100
PX_PER_YD = 10.0

# Image-space corners of the playing surface on the UGA calibration shot
# (goal line / sideline intersections), in the order used by the workflow:
# near-left, near-right, far-right, far-left from the camera's view of the
# 100-yard rectangle (end zones excluded).
UGA_CALIBRATION_POLYGON = (
    (775.0, 58.0),
    (1813.0, 85.0),
    (1845.0, 993.0),
    (328.0, 874.0),
)
UGA_CALIBRATION_IMAGE_SIZE = (1920, 1080)

UGA_CALIBRATION_NOTE = (
    "Field homography is calibrated to the UGA camera view. Other angles, pans, "
    "or zoom levels need a separate calibration or automatic landmark registration."
)
EXPERIMENTAL_COVERAGE_NOTE = (
    "Workflow coverage is an experimental baseline, not coach-validated or "
    "calibrated across varied coverages."
)

_CLASS_ALIASES = {
    "cover 0": "Cover 0 Man",
    "cover 0 man": "Cover 0 Man",
    "cover 0 blitz": "Cover 0 Man",
    "0-high": "Cover 0 Man",
    "cover 1": "Cover 1 Man",
    "cover 1 man": "Cover 1 Man",
    "cover 1 robber": "Cover 1 Man",
    "1-high": "Cover 1 Man",
    "cover 2 man": "Cover 2 Man",
    "cover 2": "Cover 2 Zone",
    "cover 2 zone": "Cover 2 Zone",
    "tampa 2": "Cover 2 Zone",
    "2-high": "Cover 2 Zone",
    "cover 3": "Cover 3 Zone",
    "cover 3 zone": "Cover 3 Zone",
    "cover 4": "Cover 4 Zone",
    "cover 4 zone": "Cover 4 Zone",
    "quarters": "Cover 4 Zone",
    "cover 6": "Cover 6 Zone",
    "cover 6 zone": "Cover 6 Zone",
    "quarter quarter half": "Cover 6 Zone",
    "prevent": "Prevent",
}


@dataclass
class CoverageCall:
    coverage: str | None
    confidence: float
    probabilities: dict[str, float]
    raw_top: str | None = None

    @property
    def runner_up(self) -> str | None:
        ranked = sorted(self.probabilities.items(), key=lambda item: item[1], reverse=True)
        if len(ranked) < 2:
            return None
        return ranked[1][0]


@dataclass
class FieldPlayer:
    class_name: str
    field_x: float
    field_y: float
    minimap_x: float
    minimap_y: float
    confidence: float = 0.0
    image_box: tuple[float, float, float, float] | None = None

    @property
    def side(self) -> str | None:
        name = self.class_name.lower()
        if "offense" in name:
            return "offense"
        if "defense" in name:
            return "defense"
        return None


@dataclass
class WorkflowFrame:
    coverage: CoverageCall
    players: list[FieldPlayer]
    homography: list[list[float]] | None
    image_width: int | None = None
    image_height: int | None = None
    minimap_png: bytes | None = None
    output_png: bytes | None = None
    notes: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def map_coverage_label(raw: str | None) -> str | None:
    if not raw:
        return None
    key = re.sub(r"[_\-]+", " ", str(raw).strip().lower())
    key = re.sub(r"\s+", " ", key)
    if key in _CLASS_ALIASES:
        return _CLASS_ALIASES[key]
    for name in COVERAGES:
        if name.lower() == key:
            return name
    return None


def minimap_px_to_field(x_px: float, y_px: float) -> tuple[float, float]:
    """Minimap pixels → absolute field yards (including end zones)."""
    return float(x_px) / PX_PER_YD, float(y_px) / PX_PER_YD


def image_to_field_homography(
    image_width: int | None = None,
    image_height: int | None = None,
    polygon: list[list[float]] | tuple[tuple[float, float], ...] | None = None,
) -> np.ndarray:
    """Image pixels → field yards, using the workflow's UGA polygon."""
    cv2 = require("cv2")
    src = np.asarray(polygon if polygon is not None else UGA_CALIBRATION_POLYGON, dtype=np.float32)
    cal_w, cal_h = UGA_CALIBRATION_IMAGE_SIZE
    if image_width and image_height and (image_width != cal_w or image_height != cal_h):
        src = src * np.array(
            [image_width / float(cal_w), image_height / float(cal_h)], dtype=np.float32
        )
    dst = np.asarray(
        [
            [ENDZONE_PX / PX_PER_YD, 0.0],
            [(MINIMAP_WIDTH_PX - ENDZONE_PX) / PX_PER_YD, 0.0],
            [(MINIMAP_WIDTH_PX - ENDZONE_PX) / PX_PER_YD, MINIMAP_HEIGHT_PX / PX_PER_YD],
            [ENDZONE_PX / PX_PER_YD, MINIMAP_HEIGHT_PX / PX_PER_YD],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def _unwrap_result(result: Any) -> dict[str, Any]:
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return first
    if isinstance(result, dict):
        return result
    raise ValueError(f"unexpected workflow result type: {type(result)!r}")


def _decode_image_bytes(payload: Any) -> bytes | None:
    if payload is None:
        return None
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    if isinstance(payload, dict):
        value = payload.get("value") or payload.get("data")
        if isinstance(value, str):
            blob = value
            if "," in blob and blob.strip().startswith("data:"):
                blob = blob.split(",", 1)[1]
            return base64.b64decode(blob)
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        nested = payload.get("image") or payload.get("numpy_image")
        if nested is not None and nested is not payload:
            return _decode_image_bytes(nested)
        return None
    if isinstance(payload, str):
        blob = payload
        if "," in blob and blob.strip().startswith("data:"):
            blob = blob.split(",", 1)[1]
        try:
            return base64.b64decode(blob)
        except Exception:
            return None
    return None


def _as_probability_map(payload: Any) -> tuple[str | None, float, dict[str, float]]:
    if payload is None:
        return None, 0.0, {}
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        rows = payload
        top_name = str(rows[0].get("class") or rows[0].get("class_name") or "")
        top_conf = float(rows[0].get("confidence") or 0.0)
    elif isinstance(payload, dict):
        top_name = str(payload.get("top") or payload.get("predicted_classes", [""])[0] or "")
        top_conf = float(payload.get("confidence") or 0.0)
        rows = payload.get("predictions") or payload.get("predicted_classes") or []
        if isinstance(rows, dict):
            mapped = {
                map_coverage_label(name) or name: float(conf)
                for name, conf in rows.items()
            }
            mapped = {k: v for k, v in mapped.items() if k in COVERAGES}
            if mapped:
                top = max(mapped, key=mapped.get)
                return top, float(mapped[top]), {c: float(mapped.get(c, 0.0)) for c in COVERAGES}
            rows = []
        if not top_name and isinstance(rows, list) and rows and isinstance(rows[0], dict):
            top_name = str(rows[0].get("class") or rows[0].get("class_name") or "")
            top_conf = float(rows[0].get("confidence") or 0.0)
    else:
        return None, 0.0, {}

    scores: dict[str, float] = {name: 0.0 for name in COVERAGES}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            mapped = map_coverage_label(str(row.get("class") or row.get("class_name") or ""))
            if mapped:
                scores[mapped] = max(scores[mapped], float(row.get("confidence") or 0.0))
    mapped_top = map_coverage_label(top_name)
    if mapped_top and scores.get(mapped_top, 0.0) == 0.0:
        scores[mapped_top] = top_conf
    if mapped_top:
        top_conf = max(top_conf, scores.get(mapped_top, 0.0))
    return mapped_top, top_conf, scores


def parse_coverage(payload: Any) -> CoverageCall:
    coverage, confidence, probabilities = _as_probability_map(payload)
    raw_top = None
    if isinstance(payload, dict):
        raw_top = payload.get("top")
        if raw_top is None and isinstance(payload.get("predictions"), list) and payload["predictions"]:
            raw_top = payload["predictions"][0].get("class")
    elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
        raw_top = payload[0].get("class")
    return CoverageCall(
        coverage=coverage,
        confidence=float(confidence),
        probabilities=probabilities,
        raw_top=str(raw_top) if raw_top else None,
    )


def parse_field_players(payload: Any, detections: list[dict[str, Any]] | None = None) -> list[FieldPlayer]:
    if payload is None:
        return []
    records = payload.get("players", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return []
    boxes = detections or []
    players: list[FieldPlayer] = []
    for i, row in enumerate(records):
        if not isinstance(row, dict):
            continue
        class_name = str(row.get("class") or row.get("class_name") or "player")
        if "official" in class_name.lower() or "referee" in class_name.lower():
            continue
        if "x" in row:
            mx, my = float(row["x"]), float(row["y"])
            fx, fy = minimap_px_to_field(mx, my)
        elif "field_x_normalized" in row:
            mx = float(row["field_x_normalized"]) * MINIMAP_WIDTH_PX
            my = float(row["field_y_normalized"]) * MINIMAP_HEIGHT_PX
            fx, fy = minimap_px_to_field(mx, my)
        else:
            continue
        box = None
        conf = float(row.get("confidence") or 0.0)
        if i < len(boxes):
            det = boxes[i]
            conf = float(det.get("confidence") or conf)
            if {"x", "y", "width", "height"} <= det.keys():
                cx, cy = float(det["x"]), float(det["y"])
                w, h = float(det["width"]), float(det["height"])
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            elif {"x1", "y1", "x2", "y2"} <= det.keys():
                box = (float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"]))
        players.append(
            FieldPlayer(
                class_name=class_name,
                field_x=fx,
                field_y=fy,
                minimap_x=mx,
                minimap_y=my,
                confidence=conf,
                image_box=box,
            )
        )
    return players


def _detection_rows(payload: Any) -> tuple[list[dict[str, Any]], int | None, int | None]:
    if payload is None:
        return [], None, None
    image = None
    rows: list[dict[str, Any]]
    if isinstance(payload, dict):
        image = payload.get("image")
        raw = payload.get("predictions", payload.get("detections", []))
        rows = raw if isinstance(raw, list) else []
    elif isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
    else:
        return [], None, None
    width = height = None
    if isinstance(image, dict):
        width = image.get("width")
        height = image.get("height")
    return rows, int(width) if width else None, int(height) if height else None


def parse_workflow_result(result: Any, image_size: tuple[int, int] | None = None) -> WorkflowFrame:
    payload = _unwrap_result(result)
    coverage = parse_coverage(payload.get("coverage_predictions"))
    detections, det_w, det_h = _detection_rows(payload.get("player_predictions"))
    field_payload = payload.get("field_coordinates") or {}
    players = parse_field_players(field_payload, detections)
    polygon = None
    if isinstance(field_payload, dict):
        polygon = field_payload.get("calibration_polygon")
    width = det_w or (image_size[0] if image_size else None)
    height = det_h or (image_size[1] if image_size else None)
    notes = [UGA_CALIBRATION_NOTE, EXPERIMENTAL_COVERAGE_NOTE]
    homography = None
    try:
        H = image_to_field_homography(width, height, polygon)
        homography = H.tolist()
    except Exception as exc:
        notes.append(f"could not reconstruct image-to-field homography: {exc}")

    return WorkflowFrame(
        coverage=coverage,
        players=players,
        homography=homography,
        image_width=width,
        image_height=height,
        minimap_png=_decode_image_bytes(payload.get("minimap")),
        output_png=_decode_image_bytes(payload.get("output_image")),
        notes=notes,
        raw={k: v for k, v in payload.items() if k not in {"minimap", "output_image"}},
    )


def coverage_prediction_row(
    play_id: str,
    call: CoverageCall,
    *,
    quality_score: float,
    usable: bool,
) -> dict[str, Any]:
    coverage = call.coverage or "Cover 3 Zone"
    probabilities = {name: float(call.probabilities.get(name, 0.0)) for name in COVERAGES}
    if coverage in probabilities and probabilities[coverage] == 0.0 and call.confidence:
        probabilities[coverage] = float(call.confidence)
    ranked = sorted(probabilities, key=lambda name: probabilities[name], reverse=True)
    return {
        "play_id": play_id,
        "coverage": coverage,
        "confidence": round(float(call.confidence), 4),
        "probabilities": probabilities,
        "runner_up": ranked[1] if len(ranked) > 1 else ranked[0],
        "roles": {},
        "quality_score": round(float(quality_score), 3),
        "usable": usable,
        "source": "roboflow",
        "experimental": True,
        "notes": [EXPERIMENTAL_COVERAGE_NOTE, UGA_CALIBRATION_NOTE],
    }


class WorkflowClient:
    def __init__(self, settings: RoboflowSettings | None = None) -> None:
        self.settings = settings or RoboflowSettings()
        if not self.settings.enabled:
            raise RuntimeError(
                "ROBOFLOW_API_KEY is not set. Copy .env.example to .env and add the key "
                "from Workspace Settings → API Keys."
            )
        self._client = None

    def _sdk(self):
        if self._client is None:
            try:
                from inference_sdk import InferenceConfiguration, InferenceHTTPClient
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "inference-sdk is not installed. Run `uv sync --extra vision`."
                ) from exc
            self._client = InferenceHTTPClient(
                api_url=self.settings.api_url,
                api_key=self.settings.api_key,
            ).configure(InferenceConfiguration(api_key_transport="header"))
        return self._client

    def run_image(self, image: str | Path | np.ndarray) -> WorkflowFrame:
        size = None
        if isinstance(image, np.ndarray) and image.ndim >= 2:
            size = (int(image.shape[1]), int(image.shape[0]))
        result = self._sdk().run_workflow(
            workspace_name=self.settings.workspace,
            workflow_id=self.settings.workflow_id,
            images={"image": image if not isinstance(image, Path) else str(image)},
            use_cache=self.settings.use_cache,
        )
        return parse_workflow_result(result, image_size=size)

    def run_frame(self, frame_bgr: np.ndarray) -> WorkflowFrame:
        rgb = frame_bgr[:, :, ::-1] if frame_bgr.ndim == 3 else frame_bgr
        return self.run_image(rgb)
