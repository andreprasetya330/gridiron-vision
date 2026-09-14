"""Registration is the stage whose errors look like coverage mistakes.

A two-yard homography error moves a safety from twelve deep to ten, which is
the difference between Cover 4 and Cover 3. So these tests pin the geometry
against a known camera, not against whether the picture looks plausible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from gridiron.vision.evaluate import registration_error
from gridiron.vision.registration import (
    FixedRegistrar,
    LineRegistrar,
    SmoothedRegistration,
    _line_params,
    _merge_collinear,
    manual_homography,
)


def test_manual_homography_is_exact_on_its_own_points():
    image = np.array([[100.0, 400.0], [700.0, 410.0], [620.0, 120.0], [180.0, 110.0]])
    field = np.array([[40.0, 0.0], [80.0, 0.0], [80.0, 53.3], [40.0, 53.3]])
    registration = manual_homography(image, field)
    assert registration.valid, registration.notes
    recovered = registration.to_field(image)
    assert np.mean(np.linalg.norm(recovered - field, axis=1)) < 0.05


def test_line_registrar_recovers_the_field_plane(tmp_path: Path):
    from gridiron.tracking.synthetic import SyntheticConfig, generate_season
    from gridiron.vision.render import CameraSpec, render_play

    play = next(iter(generate_season(SyntheticConfig(weeks=1, plays_per_game=1, seed=4))))
    video = tmp_path / "play.mp4"
    truth = render_play(play, video, camera=CameraSpec.around(55.0), los_x=55.0)

    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    assert ok

    estimated = LineRegistrar(league="ncaa").register(frame)
    assert estimated.homography is not None, estimated.notes

    error = registration_error(estimated, truth.image_to_field, frame.shape)
    # The 5-yard downfield ambiguity is excluded from usable(); what remains
    # is the error that actually moves a safety relative to the line.
    assert error.usable, error.as_dict()
    assert error.lateral_rmse_yd < 0.5
    assert error.downfield_rmse_yd < 0.5


def test_merge_collinear_joins_fragments_of_one_line():
    # Two pieces of the same painted line, hundreds of pixels apart along it.
    a = np.array([10.0, 10.0, 10.0, 80.0])
    b = np.array([11.0, 200.0, 11.0, 400.0])
    other = np.array([120.0, 10.0, 120.0, 400.0])
    merged = _merge_collinear([a, b, other], offset_tol=12.0)
    assert len(merged) == 2


def test_line_params_are_direction_invariant():
    forward = np.array([10.0, 20.0, 110.0, 40.0])
    backward = np.array([110.0, 40.0, 10.0, 20.0])
    a1, r1 = _line_params(forward)
    a2, r2 = _line_params(backward)
    assert min(abs(a1 - a2), 180 - abs(a1 - a2)) < 1.0
    assert abs(r1 - r2) < 1.0


def test_smoothed_registration_survives_a_failed_frame():
    good = manual_homography(
        np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]),
        np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]),
    )
    smoother = SmoothedRegistration(window=5)
    smoother.update(good)
    recovered = smoother.update(
        type(good)(homography=None, method="lines", notes=["failed"])
    )
    assert recovered.homography is not None
    assert "smoothed" in recovered.method


def test_fixed_registrar_is_constant():
    H = np.eye(3)
    registrar = FixedRegistrar.from_homography(H)
    a = registrar.register(np.zeros((10, 10, 3), dtype=np.uint8))
    b = registrar.register(np.zeros((10, 10, 3), dtype=np.uint8))
    assert np.allclose(a.homography, b.homography)
    assert a.valid
