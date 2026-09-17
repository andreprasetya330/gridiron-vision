"""Measure what actually separates officials from players on a given clip.

Prints grass-masked jersey colour per detection and writes a labeled contact
sheet of full detection boxes, so thresholds in `gridiron.vision.identity` are
chosen from this film rather than from a guess about what an official looks like.

    uv run python scripts/appearance_study.py data/film/clips/uga-defense-1.mp4 68
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans

from gridiron.vision.detect import PlayerDetector, filter_sideline_detections
from gridiron.vision.teams import jersey_feature

video = Path(sys.argv[1] if len(sys.argv) > 1 else "data/film/clips/uga-defense-1.mp4")
center = int(sys.argv[2]) if len(sys.argv) > 2 else 68

capture = cv2.VideoCapture(str(video))
capture.set(cv2.CAP_PROP_POS_FRAMES, center)
ok, frame = capture.read()
capture.release()
if not ok:
    raise SystemExit(f"could not read frame {center} of {video}")

detector = PlayerDetector()
detections = filter_sideline_detections(detector.detect(frame), frame.shape[0])
detections = [d for d in detections if d.class_name == "player"]
print(f"frame {center}: {len(detections)} player detections")

feats = []
tiles = []
for i, d in enumerate(detections):
    feat = jersey_feature(frame, d)
    lab_a = float(feat[1]) - 128.0
    lab_b = float(feat[2]) - 128.0
    feats.append((i, float(feat[0]), lab_a, lab_b, float(np.hypot(lab_a, lab_b))))

    x1, y1 = int(max(0, d.x1)), int(max(0, d.y1))
    x2, y2 = int(min(frame.shape[1], d.x2)), int(min(frame.shape[0], d.y2))
    box = frame[y1:y2, x1:x2]
    if box.size == 0:
        box = np.zeros((8, 8, 3), np.uint8)
    tile = cv2.resize(box, (64, 128), interpolation=cv2.INTER_CUBIC)
    tile = cv2.copyMakeBorder(tile, 18, 2, 2, 2, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(tile, str(i), (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    tiles.append(tile)

print(f"{'idx':>4} {'L':>7} {'a':>7} {'b':>7} {'chroma':>7}")
for i, L, a, b, chroma in sorted(feats, key=lambda r: -r[4]):
    print(f"{i:>4} {L:>7.1f} {a:>7.1f} {b:>7.1f} {chroma:>7.1f}")

# Two-team clustering, then distance of every track to its nearest team centre.
X = np.array([[f[1], f[2], f[3]] for f in feats], dtype=np.float32)
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
dists = np.linalg.norm(X[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
nearest = dists.min(axis=1)
print("\ncluster sizes:", np.bincount(km.labels_))
print("centres:", np.round(km.cluster_centers_, 1).tolist())
print(f"\n{'idx':>4} {'cluster':>8} {'dist':>7}   (outliers = likely officials)")
for i in np.argsort(-nearest):
    print(f"{feats[i][0]:>4} {km.labels_[i]:>8} {nearest[i]:>7.1f}")

per_row = 10
grid = []
for start in range(0, len(tiles), per_row):
    chunk = tiles[start : start + per_row]
    while len(chunk) < per_row:
        chunk.append(np.zeros_like(tiles[0]))
    grid.append(np.hstack(chunk))
sheet = np.vstack(grid)
out = Path("data/cache/appearance_sheet.png")
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), sheet)
print(f"\ncontact sheet -> {out.resolve()}")
