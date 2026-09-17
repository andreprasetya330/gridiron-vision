"""How many players of each team the pipeline can see, frame by frame.

Answers the question a play summary cannot: when the roster comes up short, is
it because a player was never detected, because the homography threw him off
the field, or because the snap landed on a frame where the camera had not yet
found everyone.

    uv run python scripts/roster_timeline.py data/film/clips/uga-defense-1.mp4
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from gridiron.vision.detect import PlayerDetector, filter_sideline_detections
from gridiron.vision.identity import (
    build_track_embeddings,
    build_track_profiles,
    classify_tracks,
)
from gridiron.vision.pipeline import PRE_SNAP_SECONDS, _on_field
from gridiron.vision.registration import LineRegistrar, SmoothedRegistration
from gridiron.vision.snap import find_snap, motion_energy
from gridiron.vision.track import build_tracker

video = Path(sys.argv[1] if len(sys.argv) > 1 else "data/film/clips/uga-defense-1.mp4")
cache = Path("data/cache") / f"{video.stem}-timeline.pkl"

if cache.exists():
    state = pickle.loads(cache.read_bytes())
    positions, fps = state["positions"], state["fps"]
    roster, snap_first, snap_clean = (
        state["roster"],
        state["snap_first"],
        state["snap_clean"],
    )
else:
    detector = PlayerDetector()
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = build_tracker(frame_rate=int(round(fps)))
    registrar, smoother = LineRegistrar("ncaa"), SmoothedRegistration()

    frames_meta: list[dict] = []
    raw_frames: list[np.ndarray] = []
    index = 0
    while True:
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

    first = find_snap(motion_energy([m["field_positions"] for m in frames_meta]), fps=fps)
    snap_first = first.frame_index if first else len(frames_meta) // 3
    lo = max(0, snap_first - int(fps * PRE_SNAP_SECONDS))
    hi = min(len(frames_meta), snap_first + int(fps * 3.0) + 1)
    window = set(range(lo, hi))
    keep = {tid for j in window for tid in frames_meta[j]["field_positions"]}

    roster = classify_tracks(
        build_track_profiles(frames_meta, raw_frames, keep_ids=keep, frames=window),
        build_track_embeddings(frames_meta, raw_frames, keep_ids=keep, frames=window),
    )
    positions = [
        {t: p for t, p in m["field_positions"].items() if t not in roster.officials}
        for m in frames_meta
    ]
    clean = find_snap(motion_energy(positions), fps=fps)
    snap_clean = clean.frame_index if clean else snap_first
    cache.write_bytes(
        pickle.dumps(
            {
                "positions": positions,
                "fps": fps,
                "roster": roster,
                "snap_first": snap_first,
                "snap_clean": snap_clean,
            }
        )
    )
    print(f"cached -> {cache}")

print(f"snap before cleaning {snap_first}, after cleaning {snap_clean}")
print(f"outsiders {sorted(roster.officials)}")
print(f"team sizes {dict(Counter(roster.team_of.values()))}\n")

print(" frame | team0 | team1 | total | missing from team0")
team0 = {t for t, v in roster.team_of.items() if v == 0}
team1 = {t for t, v in roster.team_of.items() if v == 1}
for j in range(max(0, snap_clean - 24), min(len(positions), snap_clean + 40), 2):
    seen = set(positions[j])
    a, b = seen & team0, seen & team1
    mark = " <- snap" if abs(j - snap_clean) < 2 else ""
    print(
        f"  {j:4} | {len(a):5} | {len(b):5} | {len(seen):5} | "
        f"{sorted(team0 - seen)}{mark}"
    )
