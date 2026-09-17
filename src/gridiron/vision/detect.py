"""Player and ball detection.

A stock COCO YOLO already finds football players reasonably well, because a player
is a person. What it does not do is separate players from referees, coaches, and
the crowd behind the bench, and those false positives poison team clustering and
box counts. So the workflow is: start with the COCO person class to get moving,
then fine-tune on a few hundred frames from the film you actually intend to
process.

`prepare_finetune_dataset` writes the YOLO directory layout and a data.yaml so the
labeling round trip is one command in each direction.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gridiron.config import models_dir
from gridiron.vision import require

CLASS_NAMES = ["player", "referee", "ball"]
COCO_PERSON_CLASS = 0
COCO_SPORTS_BALL_CLASS = 32


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str = "player"

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def foot_point(self) -> tuple[float, float]:
        """Where the player meets the ground.

        This is the point that gets projected through the homography. Using the
        box center instead puts every player a yard and a half downfield of where
        he actually is, and that error is larger than the differences the coverage
        model is trying to read.
        """
        return (self.x1 + self.x2) / 2, self.y2

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass
class DetectorConfig:
    """Defaults measured on 1080p All-22, not inherited from COCO benchmarks.

    Recall is the ceiling on everything downstream, because a body that is never
    detected cannot be tracked and a track that dies becomes a new identity. On
    Sugar Bowl film (scripts/detector_sweep.py), against roughly 26 people on
    screen, mean kept detections per frame were:

        yolov8m @1280 conf .18 -> 15.4
        yolo11x @1280 conf .18 -> 20.0
        yolo11x @1920 conf .10 -> 24.0

    Hence the larger model at native resolution. The low confidence floor is
    deliberate: BoT-SORT runs a second association pass over low-scoring boxes,
    so a weak detection is useful evidence rather than noise.
    """

    weights: str = "yolo11x.pt"
    confidence: float = 0.15
    iou: float = 0.5
    image_size: int = 1920  # All-22 is wide and players are small; do not downscale
    device: str | None = None
    max_detections: int = 80
    custom_model: bool = False


class PlayerDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            ultralytics = require("ultralytics")
            weights = self.config.weights
            local = models_dir() / weights
            self._model = ultralytics.YOLO(str(local) if local.exists() else weights)
        return self._model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.image_size,
            device=self.config.device,
            max_det=self.config.max_detections,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            name = str(names.get(class_id, class_id))
            if not self.config.custom_model:
                if class_id == COCO_PERSON_CLASS:
                    name = "player"
                elif class_id == COCO_SPORTS_BALL_CLASS:
                    name = "ball"
                else:
                    continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(x1, y1, x2, y2, float(box.conf.item()), class_id, name)
            )
        return detections

    def detect_batch(self, frames: Iterable[np.ndarray]) -> list[list[Detection]]:
        return [self.detect(frame) for frame in frames]


def filter_sideline_detections(
    detections: list[Detection], frame_height: int, min_height_ratio: float = 0.02
) -> list[Detection]:
    """Drop the crowd, the chain gang, and anyone standing on the sideline.

    Two cheap geometric filters do most of the work: anything too small is in the
    stands, and anything whose size is wildly out of line with its neighbours at
    similar image height is not on the field. Perspective means a real player's
    apparent height varies smoothly with his y coordinate.

    The outlier gate is deliberately loose. This runs before registration, so it
    is guessing about the field, while the pipeline's `_on_field` check runs
    *after* the homography and can reject a body by where it actually stands. A
    tight gate here was throwing away several real players per frame, and a
    missing player costs far more than a surviving one that `_on_field` will
    drop a moment later.
    """
    if not detections:
        return []

    keep = [d for d in detections if d.height >= min_height_ratio * frame_height]
    keep = [
        d
        for d in keep
        if 0.08 * frame_height < d.foot_point[1] < 0.88 * frame_height
    ]
    if len(keep) < 6:
        return keep

    heights = np.array([d.height for d in keep])
    centers_y = np.array([d.center[1] for d in keep])

    # Fit apparent height as a linear function of image row, then reject outliers.
    try:
        slope, intercept = np.polyfit(centers_y, heights, 1)
        expected = slope * centers_y + intercept
        residual = np.abs(heights - expected)
        threshold = max(5.0 * float(np.median(residual)), 0.03 * frame_height)
        return [d for d, r in zip(keep, residual) if r <= threshold]
    except Exception:
        return keep


def prepare_finetune_dataset(
    output_dir: Path, class_names: list[str] | None = None
) -> Path:
    """Create the YOLO dataset skeleton and data.yaml for hand labeling."""
    output_dir = Path(output_dir)
    names = class_names or CLASS_NAMES
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(names)}",
                f"names: {names}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def finetune(
    data_yaml: Path,
    base_weights: str = "yolov8m.pt",
    epochs: int = 60,
    image_size: int = 1280,
    device: str | None = None,
    batch: int = 4,
    workers: int = 2,
    **kwargs: Any,
) -> Any:
    """Fine-tune a detector on labeled frames.

    The batch default is small on purpose. All-22 wants a large `image_size` -
    players are a few dozen pixels tall and shrinking the frame erases them - and
    memory scales with the product, so the combination that matters most here is
    exactly the one that runs out of memory first. An 8 GB laptop GPU will not
    hold ultralytics' default batch of 16 at this resolution. Raise it on a card
    with room; lower `image_size` only as a last resort.
    """
    ultralytics = require("ultralytics")
    model = ultralytics.YOLO(base_weights)
    return model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        device=device,
        batch=batch,
        workers=workers,
        project=str(models_dir()),
        name="player-detector",
        exist_ok=True,
        **kwargs,
    )
