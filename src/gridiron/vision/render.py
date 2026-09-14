"""Render tracking data back into video, with the answer key attached.

The vision pipeline has an obvious validation problem: on real film you never know
the truth. You can look at an overlay and say it seems about right, but "seems
about right" is how a two-yard registration error survives to production and
quietly turns every Cover 4 into a Cover 3.

So this module runs the pipeline backwards. It takes a play whose coordinates are
known exactly, puts a camera on it, and writes an MP4 plus the ground truth for
every stage: the true homography, the true snap frame, the true box and team for
every player in every frame. Running the pipeline on that video and comparing
gives a real number for registration error in yards, identity switches, snap
offset, and the coverage-model accuracy drop from film noise.

What this does and does not prove
---------------------------------
It does prove the geometry, the tracking, the team logic, the snap detection, the
coordinate conventions, and the JSON contract are correct end to end, and it will
catch a sign flip or an off-by-one in the time grid instantly.

It does not prove the detector works on real football, because rendered players
are ellipses and real players are people. Detector accuracy on real film is a
separate question that needs labeled frames, and `gridiron film label` sets that
up. Keeping the two questions apart is the point: it means a failure here is
always a logic bug, never a "the model just isn't good enough yet" shrug.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from gridiron.fields import FIELD_WIDTH_YD, get_field
from gridiron.tracking.normalize import denormalize_xy
from gridiron.tracking.schema import N_FRAMES, TIME_GRID, PlayTracks
from gridiron.vision import require
from gridiron.vision.detect import Detection

PLAYER_HEIGHT_YD = 2.0

# BGR. Both are strongly saturated and far apart in Lab, which is what the team
# clustering measures. Neither is white, deliberately: a white jersey is the same
# colour as the painted yard lines, and a segmentation that cannot tell a receiver
# from a sideline is testing the renderer's palette rather than the pipeline.
OFFENSE_COLOR = (60, 190, 235)
DEFENSE_COLOR = (131, 46, 75)
PANTS_COLOR = (48, 44, 46)
TURF_COLOR = (58, 104, 52)
LINE_COLOR = (238, 238, 238)


@dataclass
class CameraSpec:
    """An elevated sideline camera, which is what All-22 actually is.

    Modeled as a ground-plane homography rather than a full projection because
    players stand on the ground: their feet are the only points that matter for
    position, and for those the homography is exact.
    """

    width: int = 1280
    height: int = 720
    # Downfield window the camera covers, in absolute field yards. All-22 is
    # zoomed to the play rather than the stadium, so this is deliberately tight:
    # around 50 yards, which keeps every player in frame through a deep route.
    x_min: float = 30.0
    x_max: float = 80.0
    camera_height_yd: float = 22.0
    # How much narrower the far sideline appears than the near one. 1.0 would be
    # a plan view with no perspective at all.
    far_shrink: float = 0.56
    near_row: float = 0.93
    far_row: float = 0.24

    @classmethod
    def around(cls, los_x: float, downfield_yd: float = 50.0, **kwargs) -> CameraSpec:
        """A camera framed on the line of scrimmage, biased downfield.

        Weighted 40/60 because the interesting half of a coverage snap happens in
        front of the defense, and a safety fifteen yards deep leaving the frame is
        the failure that makes a play unusable.
        """
        return cls(
            x_min=los_x - 0.4 * downfield_yd,
            x_max=los_x + 0.6 * downfield_yd,
            **kwargs,
        )


@dataclass
class RenderedTruth:
    """Everything the pipeline is supposed to work out for itself."""

    play_id: str
    video_path: str
    fps: float
    snap_frame: int
    league: str
    los_x: float
    ball_y: float
    play_direction: str
    homography_image_to_field: list[list[float]]
    # frame index -> list of {track_id, side, box, field_xy}
    frames: dict[str, list[dict]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: Path) -> RenderedTruth:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def image_to_field(self) -> np.ndarray:
        return np.array(self.homography_image_to_field, dtype=np.float64)


def build_camera(camera: CameraSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return (field->image, image->field) homographies for the ground plane."""
    cv2 = require("cv2")
    w, h = camera.width, camera.height

    near_y, far_y = 0.0, FIELD_WIDTH_YD
    field_pts = np.array(
        [
            [camera.x_min, near_y],
            [camera.x_max, near_y],
            [camera.x_max, far_y],
            [camera.x_min, far_y],
        ],
        dtype=np.float32,
    )

    near_v = camera.near_row * h
    far_v = camera.far_row * h
    near_half = 0.47 * w
    far_half = near_half * camera.far_shrink
    cx = w / 2.0

    image_pts = np.array(
        [
            [cx - near_half, near_v],
            [cx + near_half, near_v],
            [cx + far_half, far_v],
            [cx - far_half, far_v],
        ],
        dtype=np.float32,
    )

    field_to_image = cv2.getPerspectiveTransform(field_pts, image_pts)
    return field_to_image, np.linalg.inv(field_to_image)


def _project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    cv2 = require("cv2")
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, homography.astype(np.float64)).reshape(-1, 2)


def _horizon_row(field_to_image: np.ndarray) -> float:
    """Image row where the ground plane vanishes.

    Apparent height of anything standing on the ground is proportional to its
    distance below this row, which is the cheapest correct way to size players
    under perspective.
    """
    far = _project(field_to_image, np.array([[60.0, 4000.0]]))[0]
    return float(far[1])


def _draw_field(canvas: np.ndarray, field_to_image: np.ndarray, league: str) -> None:
    cv2 = require("cv2")
    spec = get_field(league)
    canvas[:] = TURF_COLOR

    for yard in range(0, 121, 5):
        p = _project(field_to_image, np.array([[yard, 0.0], [yard, FIELD_WIDTH_YD]]))
        thickness = 3 if yard % 10 == 0 else 2
        cv2.line(
            canvas,
            tuple(np.round(p[0]).astype(int)),
            tuple(np.round(p[1]).astype(int)),
            LINE_COLOR,
            thickness,
            cv2.LINE_AA,
        )

    for y in (0.0, FIELD_WIDTH_YD):
        p = _project(field_to_image, np.array([[0.0, y], [120.0, y]]))
        cv2.line(
            canvas,
            tuple(np.round(p[0]).astype(int)),
            tuple(np.round(p[1]).astype(int)),
            LINE_COLOR,
            3,
            cv2.LINE_AA,
        )

    # Hash marks, one per yard, at the league's hash width.
    for y in (spec.left_hash_y, spec.right_hash_y):
        for yard in range(10, 111):
            p = _project(field_to_image, np.array([[yard, y - 0.35], [yard, y + 0.35]]))
            cv2.line(
                canvas,
                tuple(np.round(p[0]).astype(int)),
                tuple(np.round(p[1]).astype(int)),
                LINE_COLOR,
                2,
                cv2.LINE_AA,
            )


def _draw_player(
    canvas: np.ndarray, foot: np.ndarray, height_px: float, color: tuple[int, int, int]
) -> tuple[float, float, float, float]:
    cv2 = require("cv2")
    u, v = float(foot[0]), float(foot[1])
    h = max(height_px, 8.0)
    w = max(h * 0.42, 4.0)

    torso_top = v - h
    torso_bottom = v - 0.32 * h
    # Legs first so the torso paints over them.
    cv2.line(
        canvas,
        (int(round(u)), int(round(torso_bottom))),
        (int(round(u)), int(round(v))),
        PANTS_COLOR,
        max(int(round(w * 0.34)), 2),
        cv2.LINE_AA,
    )
    cv2.ellipse(
        canvas,
        (int(round(u)), int(round((torso_top + torso_bottom) / 2))),
        (int(round(w / 2)), int(round((torso_bottom - torso_top) / 2))),
        0,
        0,
        360,
        color,
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(
        canvas,
        (int(round(u)), int(round(torso_top - 0.10 * h))),
        max(int(round(w * 0.34)), 2),
        color,
        -1,
        cv2.LINE_AA,
    )
    return (u - w / 2, torso_top - 0.22 * h, u + w / 2, v)


def render_play(
    play: PlayTracks,
    output_path: Path,
    camera: CameraSpec | None = None,
    fps: float = 30.0,
    los_x: float = 55.0,
    play_direction: str = "right",
    jitter_px: float = 0.0,
    seed: int = 0,
) -> RenderedTruth:
    """Write an MP4 of one play plus the ground truth JSON beside it.

    `jitter_px` adds per-frame detection noise, which is the honest way to ask
    "how much registration error can the coverage model absorb before its answers
    stop meaning anything".
    """
    cv2 = require("cv2")
    camera = camera or CameraSpec()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    field_to_image, image_to_field = build_camera(camera)
    horizon = _horizon_row(field_to_image)
    ball_y = play.situation.ball_y_from_center + FIELD_WIDTH_YD / 2.0
    if play_direction == "left":
        ball_y = FIELD_WIDTH_YD - ball_y

    # Video frames run at `fps`; the play's own samples are on TIME_GRID. Sampling
    # the grid at video rate rather than the other way round keeps the renderer
    # honest about the pipeline having to rediscover the time base.
    duration = float(TIME_GRID[-1] - TIME_GRID[0])
    n_video_frames = int(round(duration * fps)) + 1
    video_times = TIME_GRID[0] + np.arange(n_video_frames) / fps
    snap_frame = int(round(-TIME_GRID[0] * fps))

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (camera.width, camera.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output_path}")

    background = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    _draw_field(background, field_to_image, play.situation.league)

    truth = RenderedTruth(
        play_id=play.play_id,
        video_path=str(output_path),
        fps=fps,
        snap_frame=snap_frame,
        league=play.situation.league,
        los_x=los_x,
        ball_y=float(ball_y),
        play_direction=play_direction,
        homography_image_to_field=image_to_field.tolist(),
    )

    try:
        for frame_index, t in enumerate(video_times):
            canvas = background.copy()
            entries: list[dict] = []

            # Painter's algorithm: far players first, so near ones occlude them.
            drawable = []
            for player in play.players:
                x_norm, y_norm = _sample(player, t)
                if not (np.isfinite(x_norm) and np.isfinite(y_norm)):
                    continue
                fx, fy = denormalize_xy(x_norm, y_norm, los_x, ball_y, play_direction)
                drawable.append((player, float(fx), float(fy)))

            for player, fx, fy in sorted(drawable, key=lambda item: -item[2]):
                foot = _project(field_to_image, np.array([[fx, fy]]))[0]
                if jitter_px:
                    foot = foot + rng.normal(0, jitter_px, size=2)
                height_px = PLAYER_HEIGHT_YD / camera.camera_height_yd * (foot[1] - horizon)
                if height_px <= 4 or not (0 <= foot[0] < camera.width):
                    continue
                color = OFFENSE_COLOR if player.side == "offense" else DEFENSE_COLOR
                box = _draw_player(canvas, foot, height_px, color)
                entries.append(
                    {
                        "track_id": player.track_id,
                        "side": player.side,
                        "role": player.role,
                        "box": [round(float(v), 2) for v in box],
                        "field_xy": [round(fx, 3), round(fy, 3)],
                    }
                )

            truth.frames[str(frame_index)] = entries
            writer.write(canvas)
    finally:
        writer.release()

    truth_path = output_path.with_suffix(".truth.json")
    truth_path.write_text(json.dumps(truth.to_dict(), indent=2), encoding="utf-8")
    return truth


def _sample(player, t: float) -> tuple[float, float]:
    """Linear read of a track at an arbitrary time, without extrapolating."""
    if t <= TIME_GRID[0]:
        return float(player.x[0]), float(player.y[0])
    if t >= TIME_GRID[-1]:
        return float(player.x[N_FRAMES - 1]), float(player.y[N_FRAMES - 1])
    ok = np.isfinite(player.x) & np.isfinite(player.y)
    if not ok.any():
        return float("nan"), float("nan")
    x = np.interp(t, TIME_GRID[ok], player.x[ok], left=np.nan, right=np.nan)
    y = np.interp(t, TIME_GRID[ok], player.y[ok], left=np.nan, right=np.nan)
    return float(x), float(y)


class TruthBoxDetector:
    """Replays the ground-truth boxes, optionally with noise.

    This is not cheating, it is isolation. With detection held perfect, any error
    the pipeline produces has to come from registration, tracking, team logic,
    the snap search, or a coordinate convention - and those are the parts where a
    bug is silent and expensive. Detector recall is a real question too, but it is
    a different one, answered by `RenderedPlayerDetector` and ultimately by
    labeled frames of real film.

    `box_jitter_px` and `miss_rate` let the same harness answer "how much
    detection noise can the coverage model absorb", which is the number that
    decides how good the detector actually has to be.
    """

    def __init__(
        self,
        truth: RenderedTruth,
        box_jitter_px: float = 0.0,
        miss_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.truth = truth
        self.box_jitter_px = box_jitter_px
        self.miss_rate = miss_rate
        self.rng = np.random.default_rng(seed)
        self._frame_index = -1

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # The pipeline hands frames over in order and does not pass the index, so
        # the counter tracks it. Any change to that contract shows up immediately
        # as a validation failure rather than a subtle misalignment.
        self._frame_index += 1
        entries = self.truth.frames.get(str(self._frame_index), [])
        out: list[Detection] = []
        for entry in entries:
            if self.miss_rate and self.rng.random() < self.miss_rate:
                continue
            x1, y1, x2, y2 = entry["box"]
            if self.box_jitter_px:
                dx, dy = self.rng.normal(0, self.box_jitter_px, size=2)
                x1, x2 = x1 + dx, x2 + dx
                y1, y2 = y1 + dy, y2 + dy
            out.append(Detection(x1, y1, x2, y2, 0.95, 0, "player"))
        return out

    def detect_batch(self, frames) -> list[list[Detection]]:
        return [self.detect(frame) for frame in frames]

    def reset(self) -> None:
        self._frame_index = -1


class RenderedPlayerDetector:
    """Finds rendered players by jersey colour.

    A real detector on real pixels, just an easy one. It exists so the validation
    run exercises detection, tracking, and team clustering as a chain rather than
    feeding the tracker ground-truth boxes and pretending the chain was tested.
    """

    def __init__(self, min_height_px: float = 6.0) -> None:
        self.min_height_px = min_height_px

    def player_mask(self, frame: np.ndarray) -> np.ndarray:
        cv2 = require("cv2")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Jerseys are saturated and not green; turf is saturated and green; paint
        # is unsaturated. Pants are too dark for any of those tests, so they come
        # in on value alone - and they matter, because the bottom of the pants is
        # the foot point that gets projected through the homography.
        jersey = (sat >= 80) & ((hue < 30) | (hue > 100))
        pants = val < 70
        mask = ((jersey | pants) * 255).astype(np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8), iterations=2)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 2), np.uint8))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        cv2 = require("cv2")
        mask = self.player_mask(frame)

        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        detections: list[Detection] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if h < self.min_height_px or area < 12 or w > 3 * h:
                continue
            detections.append(
                Detection(float(x), float(y), float(x + w), float(y + h), 0.9, 0, "player")
            )
        return detections

    def detect_batch(self, frames) -> list[list[Detection]]:
        return [self.detect(frame) for frame in frames]
