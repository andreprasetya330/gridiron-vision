"""Draw a finished play back onto its film, the way the UI renders it.

Every other diagnostic here inspects a stage in isolation. This one reads the
emitted play JSON - the same file the API serves - and projects it back through
the homography, so what you see is what shipped: sides, roles, and whether
anyone on screen is wearing stripes.

    uv run python scripts/play_overlay.py data/plays/film/uga-defense-1-000.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

SIDE_COLOR = {"offense": (200, 200, 200), "defense": (80, 200, 255)}
ROLE_COLOR = {
    "blitz": (60, 120, 249),
    "man": (240, 160, 60),
    "deep_zone": (120, 220, 120),
    "underneath_zone": (200, 140, 240),
}

play_path = Path(
    sys.argv[1] if len(sys.argv) > 1 else "data/plays/film/uga-defense-1-000.json"
)
play = json.loads(play_path.read_text())

homography = np.array(play["homography"], dtype=np.float64)
inverse = np.linalg.inv(homography)
snap_frame = int(play["snap_frame_in_video"])
origin_x, origin_y = float(play["origin_x"]), float(play["origin_y"])

grid = play.get("time_grid")
snap_tick = int(np.argmin(np.abs(np.array(grid)))) if grid else 20
# A play carries a single homography, measured at the snap, so only the snap
# frame is guaranteed to line up. Later ticks drift as the camera pans, which
# is a property of the contract rather than of the tracking - pass a list of
# tenths of a second to look at them anyway.
offsets = [int(a) for a in sys.argv[2:]] or [0]

capture = cv2.VideoCapture(play["video_path"])
panels = []
for offset in offsets:
    tick = snap_tick + offset
    frame_index = snap_frame + int(round(offset / 10.0 * play["video_fps"]))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        continue

    for player in play["players"]:
        if tick >= len(player["x"]):
            continue
        nx, ny = player["x"][tick], player["y"][tick]
        if nx is None or ny is None:
            continue
        field = np.array([[[nx + origin_x, ny + origin_y]]], dtype=np.float64)
        px, py = cv2.perspectiveTransform(field, inverse)[0][0]
        if not (0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]):
            continue

        role = player.get("role")
        colour = ROLE_COLOR.get(role) or SIDE_COLOR[player["side"]]
        px, py = int(px), int(py)
        cv2.rectangle(frame, (px - 22, py - 78), (px + 22, py + 6), colour, 2)
        label = player["track_id"] if not role else f"{player['track_id']} {role}"
        cv2.rectangle(frame, (px - 22, py - 96), (px - 22 + 9 * len(label), py - 78), colour, -1)
        cv2.putText(
            frame, label, (px - 19, py - 83), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1
        )

    banner = f"{play['play_id']}  frame {frame_index}  t={offset / 10:+.1f}s"
    cv2.rectangle(frame, (0, 0), (760, 40), (0, 0, 0), -1)
    cv2.putText(frame, banner, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    panels.append(frame if len(offsets) == 1 else cv2.resize(frame, (960, 540)))
capture.release()

out = Path("data/cache") / f"{play['play_id']}-overlay.png"
rows = [np.hstack(panels[i : i + 2]) for i in range(0, len(panels) - 1, 2)]
cv2.imwrite(str(out), np.vstack(rows) if rows else panels[0])
print(f"-> {out.resolve()}")
