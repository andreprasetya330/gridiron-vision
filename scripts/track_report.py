"""Report tracking stability and the official/team verdict for a clip.

Answers the two questions you cannot get from the play JSON alone: is the
tracker holding identities, and who did the roster classifier throw away.

    uv run python scripts/track_report.py data/film/clips/uga-defense-1.mp4
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
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000

detector = PlayerDetector()
capture = cv2.VideoCapture(str(video))
fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
tracker = build_tracker(frame_rate=int(round(fps)))
registrar = LineRegistrar("ncaa")
smoother = SmoothedRegistration()
print(f"tracker: {type(tracker).__name__}  fps={fps:.0f}")

frames_meta: list[dict] = []
raw_frames: list[np.ndarray] = []
index = 0
while index < limit:
    ok, frame = capture.read()
    if not ok:
        break
    dets = filter_sideline_detections(detector.detect(frame), frame.shape[0])
    players = [d for d in dets if d.class_name == "player"]
    tracked = tracker.update(players, index, frame)

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

n = len(frames_meta)
spans: dict[int, list[int]] = {}
for j, meta in enumerate(frames_meta):
    for tid in meta["field_positions"]:
        spans.setdefault(tid, []).append(j)

lengths = np.array([len(v) for v in spans.values()])
raw_per_frame = np.array([len(m["tracked"]) for m in frames_meta])
per_frame = np.array([len(m["field_positions"]) for m in frames_meta])
print(f"\nframes              {n}")
print(f"detected/frame      mean {raw_per_frame.mean():.1f}  (before homography)")
print(f"on-field/frame      mean {per_frame.mean():.1f}  min {per_frame.min()}  max {per_frame.max()}")
print(f"distinct on-field   {len(spans)}   (22 players + crew is the floor)")
print(f"track length        median {np.median(lengths):.0f}  mean {lengths.mean():.0f}  max {lengths.max()}")
print(f"tracks >= 50% of clip: {(lengths >= 0.5 * n).sum()}")
print(f"tracks <  10% of clip: {(lengths < 0.1 * n).sum()}  (fragments)")

on_field_ids = set(spans)
profiles = build_track_profiles(frames_meta, raw_frames, keep_ids=on_field_ids)
roster = classify_tracks(profiles)
print(f"\nprofiles {len(profiles)}  officials {sorted(roster.officials)}")
for note in roster.notes:
    print(f"  note: {note}")
sizes: dict[int, int] = {}
for team in roster.team_of.values():
    sizes[team] = sizes.get(team, 0) + 1
print(f"team sizes {sizes}  separation {roster.separation:.1f}  conf {roster.confidence:.2f}")
if roster.centers is not None:
    print("team centres (Lab):", np.round(roster.centers, 1).tolist())

print(f"\n{'track':>6} {'luma':>7} {'frames':>7} {'verdict':>10}")
for tid in sorted(profiles, key=lambda t: profiles[t].luma):
    p = profiles[tid]
    verdict = "OFFICIAL" if tid in roster.officials else f"team {roster.team_of.get(tid, '-')}"
    print(f"{tid:>6} {p.luma:>7.1f} {len(spans.get(tid, [])):>7} {verdict:>10}")
