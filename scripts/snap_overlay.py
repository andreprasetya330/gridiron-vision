"""Draw the roster verdict on the snap frame so it can be checked by eye.

Numbers can say the team split is 13/5 without saying which thirteen. This
paints every track at the snap with what the classifier decided.

    uv run python scripts/snap_overlay.py data/film/clips/uga-defense-1.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from gridiron.vision.detect import PlayerDetector, filter_sideline_detections
from gridiron.vision.identity import build_track_profiles, classify_tracks
from gridiron.vision.pipeline import _on_field
from gridiron.vision.registration import LineRegistrar, SmoothedRegistration
from gridiron.vision.track import build_tracker

video = Path(sys.argv[1] if len(sys.argv) > 1 else "data/film/clips/uga-defense-1.mp4")
target = int(sys.argv[2]) if len(sys.argv) > 2 else 101

detector = PlayerDetector()
capture = cv2.VideoCapture(str(video))
fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
tracker = build_tracker(frame_rate=int(round(fps)))
registrar = LineRegistrar("ncaa")
smoother = SmoothedRegistration()

frames_meta: list[dict] = []
raw_frames: list[np.ndarray] = []
index = 0
while index <= target:
    ok, frame = capture.read()
    if not ok:
        break
    dets = filter_sideline_detections(detector.detect(frame), frame.shape[0])
    tracked = tracker.update([d for d in dets if d.class_name == "player"], index, frame)
    registration = smoother.update(registrar.register(frame))
    on_field = {}
    if registration.homography is not None and tracked:
        mapped = registration.to_field(np.array([t.detection.foot_point for t in tracked]))
        for t, point in zip(tracked, mapped, strict=True):
            if np.all(np.isfinite(point)) and _on_field(float(point[0]), float(point[1])):
                on_field[t.track_id] = (float(point[0]), float(point[1]))
    frames_meta.append({"tracked": tracked, "field_positions": on_field})
    raw_frames.append(frame)
    index += 1
capture.release()

ids: set[int] = set()
for meta in frames_meta:
    ids.update(meta["field_positions"])
profiles = build_track_profiles(frames_meta, raw_frames, keep_ids=ids)
roster = classify_tracks(profiles)

meta = frames_meta[-1]
canvas = raw_frames[-1].copy()
COLOR = {0: (80, 220, 80), 1: (80, 140, 250), None: (170, 170, 170)}
for t in meta["tracked"]:
    tid = t.track_id
    d = t.detection
    on = tid in meta["field_positions"]
    if tid in roster.officials:
        color, tag = (0, 0, 255), f"{tid} REF"
    else:
        team = roster.team_of.get(tid)
        color = COLOR[team]
        tag = f"{tid} T{team if team is not None else '?'}"
    if not on:
        tag += " off"
    thickness = 2 if on else 1
    cv2.rectangle(canvas, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, thickness)
    luma = profiles[tid].luma if tid in profiles else float("nan")
    cv2.putText(
        canvas, f"{tag} L{luma:.0f}", (int(d.x1), int(d.y1) - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
    )

out = Path("data/cache/snap_overlay.png")
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), canvas)
print(f"frame {target}: {len(meta['tracked'])} tracked, {len(meta['field_positions'])} on field")
print(f"officials {sorted(roster.officials)}")
if roster.centers is not None:
    print("chroma centres:", np.round(roster.centers, 1).tolist())
print(f"-> {out.resolve()}")
