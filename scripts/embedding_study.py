"""Lab colour vs SigLIP embeddings for separating teams and officials.

Runs both classifiers over the same tracks and writes a contact sheet per
cluster, because the only honest check is whether the groups look like teams.

    uv run python scripts/embedding_study.py data/film/clips/uga-defense-1.mp4
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans

from gridiron.vision.detect import PlayerDetector, filter_sideline_detections
from gridiron.vision.embed import SiglipEmbedder, player_crop
from gridiron.vision.identity import build_track_profiles, classify_tracks
from gridiron.vision.pipeline import _on_field
from gridiron.vision.registration import LineRegistrar, SmoothedRegistration
from gridiron.vision.snap import find_snap, motion_energy
from gridiron.vision.track import build_tracker

video = Path(sys.argv[1] if len(sys.argv) > 1 else "data/film/clips/uga-defense-1.mp4")
CACHE = Path("data/cache") / f"{video.stem}-embeddings.npz"

detector = PlayerDetector()
capture = cv2.VideoCapture(str(video))
fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
tracker = build_tracker(frame_rate=int(round(fps)))
registrar = LineRegistrar("ncaa")
smoother = SmoothedRegistration()

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

energy = motion_energy([m["field_positions"] for m in frames_meta])
snap = find_snap(energy, fps=fps)
snap_idx = snap.frame_index if snap else len(frames_meta) // 3
lo, hi = max(0, snap_idx - int(fps * 2)), min(len(frames_meta), snap_idx + int(fps * 3))
window = set(range(lo, hi))
print(f"{len(frames_meta)} frames, snap {snap_idx}, window {lo}-{hi}")

on_field_ids: set[int] = set()
for j in window:
    on_field_ids.update(frames_meta[j]["field_positions"])

# --- colour baseline -------------------------------------------------------
profiles = build_track_profiles(frames_meta, raw_frames, keep_ids=on_field_ids, frames=window)
lab_roster = classify_tracks(profiles)
print(f"\n[Lab]    tracks {len(profiles)}  officials {sorted(lab_roster.officials)}")
print(f"[Lab]    team sizes {dict(Counter(lab_roster.team_of.values()))}")
for note in lab_roster.notes:
    print(f"[Lab]    note: {note}")

# --- SigLIP ----------------------------------------------------------------
crops_by_track: dict[int, list[np.ndarray]] = defaultdict(list)
for j in sorted(window):
    if j % 2:
        continue
    for t in frames_meta[j]["tracked"]:
        if t.track_id in on_field_ids and len(crops_by_track[t.track_id]) < 24:
            crops_by_track[t.track_id].append(player_crop(raw_frames[j], t.detection))

track_ids = sorted(k for k, v in crops_by_track.items() if len(v) >= 3)
flat, owner = [], []
for tid in track_ids:
    for crop in crops_by_track[tid]:
        flat.append(crop)
        owner.append(tid)
print(f"\n[SigLIP] embedding {len(flat)} crops from {len(track_ids)} tracks ...")

embedder = SiglipEmbedder()
features = embedder.embed(flat)
owner_arr = np.array(owner)

# One vector per track: the mean of its crops, renormalised. Clustering the
# crops individually was tried and is much worse - with only a few hundred crops
# from one play, the split falls along pose and lighting rather than team.
track_vecs = np.stack([features[owner_arr == tid].mean(axis=0) for tid in track_ids])
track_vecs /= np.clip(np.linalg.norm(track_vecs, axis=1, keepdims=True), 1e-6, None)

siglip_roster = classify_tracks({}, dict(zip(track_ids, track_vecs, strict=True)))
print(f"[SigLIP] outsiders {sorted(siglip_roster.officials)}")
print(f"[SigLIP] team sizes {dict(Counter(siglip_roster.team_of.values()))}")
for note in siglip_roster.notes:
    print(f"[SigLIP] note: {note}")

for k in (2, 3, 4):
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(track_vecs)
    print(f"[SigLIP] raw k={k} cluster sizes {dict(Counter(labels.tolist()))}")

# --- contact sheets --------------------------------------------------------
def sheet(groups: dict[str, list[int]], path: Path, pick: dict[int, np.ndarray]) -> None:
    rows = []
    for name, ids in groups.items():
        tiles = []
        for tid in ids[:14]:
            tile = cv2.resize(pick[tid], (48, 96), interpolation=cv2.INTER_CUBIC)
            tile = cv2.copyMakeBorder(tile, 18, 2, 2, 2, cv2.BORDER_CONSTANT, value=(20, 20, 20))
            cv2.putText(tile, str(tid), (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            tiles.append(tile)
        if not tiles:
            continue
        while len(tiles) < 14:
            tiles.append(np.zeros_like(tiles[0]))
        row = np.hstack(tiles)
        banner = np.zeros((22, row.shape[1], 3), np.uint8)
        cv2.putText(banner, name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        rows.append(np.vstack([banner, row]))
    if rows:
        cv2.imwrite(str(path), np.vstack(rows))
        print(f"-> {path.resolve()}")


best = {tid: max(crops_by_track[tid], key=lambda c: c.shape[0]) for tid in track_ids}
out = Path("data/cache")
out.mkdir(parents=True, exist_ok=True)

# Detection and tracking cost minutes; clustering experiments cost seconds.
np.savez(CACHE, features=features, owner=owner_arr, track_ids=np.array(track_ids))
print(f"-> cached embeddings {CACHE.resolve()}")

lab_groups: dict[str, list[int]] = defaultdict(list)
for tid in track_ids:
    if tid in lab_roster.officials:
        lab_groups["Lab: OFFICIAL"].append(tid)
    else:
        lab_groups[f"Lab: team {lab_roster.team_of.get(tid, '?')}"].append(tid)
sheet(lab_groups, out / "cluster_lab.png", best)

sig_groups: dict[str, list[int]] = defaultdict(list)
for tid in track_ids:
    if tid in siglip_roster.officials:
        sig_groups["SigLIP: NOT A PLAYER"].append(tid)
    else:
        sig_groups[f"SigLIP: team {siglip_roster.team_of.get(tid, '?')}"].append(tid)
sheet(dict(sorted(sig_groups.items())), out / "cluster_siglip.png", best)
