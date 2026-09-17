"""How many of the ~26 people on the field does each detector setting actually find?

Recall is the ceiling on everything downstream: a body that is never detected
cannot be tracked, and a track that dies every few frames becomes a new identity.

    uv run python scripts/detector_sweep.py data/film/clips/uga-defense-1.mp4
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from gridiron.vision.detect import DetectorConfig, PlayerDetector, filter_sideline_detections

video = Path(sys.argv[1] if len(sys.argv) > 1 else "data/film/clips/uga-defense-1.mp4")
SAMPLES = [40, 60, 68, 80, 100, 130, 160]

capture = cv2.VideoCapture(str(video))
frames = []
for idx in SAMPLES:
    capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = capture.read()
    if ok:
        frames.append(frame)
capture.release()
print(f"{len(frames)} sample frames from {video.name}\n")

TRIALS = [
    ("yolov8m  1280 c.18 (current)", "yolov8m.pt", 1280, 0.18),
    ("yolov8m  1920 c.10", "yolov8m.pt", 1920, 0.10),
    ("yolo11x  1280 c.18", "yolo11x.pt", 1280, 0.18),
    ("yolo11x  1920 c.10", "yolo11x.pt", 1920, 0.10),
    ("yolo11x  1920 c.05", "yolo11x.pt", 1920, 0.05),
]

print(f"{'setting':>30} {'raw':>6} {'kept':>6} {'lost to filter':>15} {'s/frame':>8}")
for label, weights, imgsz, conf in TRIALS:
    try:
        det = PlayerDetector(
            DetectorConfig(weights=weights, image_size=imgsz, confidence=conf, max_detections=80)
        )
        raw_counts, kept_counts = [], []
        start = time.time()
        for frame in frames:
            raw = [d for d in det.detect(frame) if d.class_name == "player"]
            kept = [
                d
                for d in filter_sideline_detections(raw, frame.shape[0])
                if d.class_name == "player"
            ]
            raw_counts.append(len(raw))
            kept_counts.append(len(kept))
        elapsed = (time.time() - start) / max(len(frames), 1)
        raw_m, kept_m = float(np.mean(raw_counts)), float(np.mean(kept_counts))
        print(
            f"{label:>30} {raw_m:>6.1f} {kept_m:>6.1f} {raw_m - kept_m:>15.1f} {elapsed:>8.2f}"
        )
    except Exception as exc:  # noqa: BLE001 - a missing weight file should not stop the sweep
        print(f"{label:>30}  skipped: {type(exc).__name__}: {exc}")
