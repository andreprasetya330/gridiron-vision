"""The pipeline's job is one JSON contract. These tests pin that contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from gridiron.tracking.schema import PlayTracks, load_play, save_play
from gridiron.vision.pipeline import PipelineConfig, process_video
from gridiron.vision.registration import FixedRegistrar
from gridiron.vision.render import CameraSpec, TruthBoxDetector, render_play
from gridiron.tracking.synthetic import SyntheticConfig, generate_season


@pytest.fixture(scope="module")
def rendered_play(tmp_path_factory):
    play = next(iter(generate_season(SyntheticConfig(weeks=1, plays_per_game=1, seed=7))))
    out = tmp_path_factory.mktemp("film") / "play.mp4"
    truth = render_play(play, out, camera=CameraSpec.around(55.0), los_x=55.0)
    return play, truth, out


def test_pipeline_emits_the_play_contract(rendered_play):
    play, truth, video = rendered_play
    recovered = process_video(
        video,
        PipelineConfig(league="ncaa", single_play=True),
        play_id="film-000",
        detector=TruthBoxDetector(truth),
        registrar=FixedRegistrar.from_homography(truth.image_to_field),
    )
    assert len(recovered) == 1
    got = recovered[0]

    assert got.play_id == "film-000"
    assert got.source == "film"
    assert got.video_path == str(video)
    assert got.homography is not None
    assert got.origin_x is not None and got.origin_y is not None
    assert got.video_width and got.video_height
    assert got.snap_frame_in_video == pytest.approx(truth.snap_frame, abs=2)

    defenders = [p for p in got.players if p.side == "defense"]
    offense = [p for p in got.players if p.side == "offense"]
    assert len(defenders) >= 9
    assert len(offense) >= 9
    assert got.quality.usable or got.quality.defenders_detected >= 9


def test_play_json_round_trips_overlay_fields(tmp_path: Path):
    play = PlayTracks(
        play_id="x",
        source="film",
        players=[],
        homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        origin_x=55.0,
        origin_y=26.6,
        play_direction="right",
        video_width=1280,
        video_height=720,
    )
    path = tmp_path / "x.json"
    save_play(play, path)
    loaded = load_play(path)
    assert loaded.homography == play.homography
    assert loaded.origin_x == pytest.approx(55.0)
    assert loaded.video_width == 1280
    play.vision_model = "andre-4cotb/american-football-player-trackin-1-rfdetr-small-t1"
    play.vision_frames = [
        {
            "t": 0.0,
            "video_frame": 0,
            "boxes": [
                {
                    "class_name": "defense_player",
                    "side": "defense",
                    "confidence": 0.9,
                    "box": [10.0, 20.0, 40.0, 80.0],
                }
            ],
            "players": [
                {
                    "track_id": "D_1",
                    "side": "defense",
                    "class_name": "defense_player",
                    "minimap_x": 560.0,
                    "minimap_y": 200.0,
                }
            ],
        }
    ]
    play.minimap_width = 1200
    play.minimap_height = 533
    save_play(play, path)
    loaded = load_play(path)
    assert loaded.vision_model == play.vision_model
    assert loaded.vision_frames == play.vision_frames
    assert loaded.minimap_width == 1200
    # Old payloads without these fields must still load.
    raw = path.read_text(encoding="utf-8")
    stripped = PlayTracks.from_dict(
        {k: v for k, v in __import__("json").loads(raw).items() if k != "homography"}
    )
    assert stripped.homography is None
