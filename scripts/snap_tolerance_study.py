"""Pick the formation-refinement tolerance against a known-truth snap.

The synthetic clip is the only film where the true snap frame is known exactly,
so the tolerance is chosen here rather than by eye on broadcast footage.

    uv run python scripts/snap_tolerance_study.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from gridiron.tracking.synthetic import SyntheticConfig, generate_season
from gridiron.vision import pipeline as pl
from gridiron.vision.pipeline import PipelineConfig, process_video
from gridiron.vision.registration import FixedRegistrar
from gridiron.vision.render import CameraSpec, TruthBoxDetector, render_play

tmp = Path(tempfile.mkdtemp())
play = next(iter(generate_season(SyntheticConfig(weeks=1, plays_per_game=1, seed=7))))
out = tmp / "play.mp4"
truth = render_play(play, out, camera=CameraSpec.around(55.0), los_x=55.0)
print(f"truth snap frame: {truth.snap_frame}\n")

original = pl._refine_with_formation
print(f"{'tol':>6} {'ahead':>6} {'snap':>6} {'error':>6} {'defenders':>10} {'offense':>8}")
for tolerance in [1.05, 1.15, 1.25, 1.4]:
    for ahead in [0.0, 0.15, 0.25, 0.4]:
        pl._refine_with_formation = (
            lambda fm, s, fps, _t=tolerance, _a=ahead: original(
                fm, s, fps, tolerance=_t, lookahead_s=_a
            )
        )
        got = process_video(
            out,
            PipelineConfig(league="ncaa", single_play=True),
            play_id="film-000",
            detector=TruthBoxDetector(truth),
            registrar=FixedRegistrar.from_homography(truth.image_to_field),
        )[0]
        d = sum(p.side == "defense" for p in got.players)
        o = sum(p.side == "offense" for p in got.players)
        err = got.snap_frame_in_video - truth.snap_frame
        print(
            f"{tolerance:>6.2f} {ahead:>6.2f} {got.snap_frame_in_video:>6} "
            f"{err:>+6} {d:>10} {o:>8}"
        )

pl._refine_with_formation = original
