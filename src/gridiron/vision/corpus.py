"""The film corpus - what film we have, where it came from, and what is labeled.

Two kinds of label live here and they are worth keeping straight, because they
are collected differently and they train different things:

- **Detection labels** are boxes on individual frames. They are expensive per
  unit and cheap in aggregate, because a few hundred frames is enough to finetune
  a detector that already knows what a person looks like.
- **Coverage labels** are one string per play. They are cheap per unit and only
  useful in bulk, and they are what the coverage model actually learns from.

A clip carries provenance because film rights are the constraint that decides
whether any of this is usable. A Hudl cutup shared by a program comes with terms
attached; a public All-22 upload does not come with permission just because it is
reachable. Recording the source at registration means the question gets answered
once, at the point where somebody knows the answer, rather than a year later when
nobody does.

Splits are by clip, never by frame. Adjacent frames of the same snap are nearly
identical images, so a frame-level split reports a detector that has effectively
memorized its validation set.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from gridiron.config import film_dir
from gridiron.vision import require

MANIFEST_NAME = "corpus.json"
ANGLES = ("sideline", "endzone", "broadcast", "synthetic")


@dataclass
class PlayLabel:
    """Ground truth for one play within a clip."""

    play_id: str
    coverage: str | None = None
    start_frame: int | None = None
    snap_frame: int | None = None
    end_frame: int | None = None
    offense: str | None = None
    defense: str | None = None
    notes: str = ""


@dataclass
class ClipRecord:
    clip_id: str
    path: str
    angle: str = "sideline"
    source: str = "unknown"
    rights: str = "unreviewed"
    season: int | None = None
    week: int | None = None
    offense: str | None = None
    defense: str | None = None
    fps: float | None = None
    frame_count: int | None = None
    width: int | None = None
    height: int | None = None
    synthetic: bool = False
    truth_path: str | None = None
    split: str = "train"
    plays: list[PlayLabel] = field(default_factory=list)
    added_at: str = ""

    @property
    def video(self) -> Path:
        return Path(self.path)

    @property
    def duration_s(self) -> float | None:
        if self.fps and self.frame_count:
            return self.frame_count / self.fps
        return None

    @property
    def labeled_coverage(self) -> int:
        return sum(1 for p in self.plays if p.coverage)

    def to_json(self) -> dict:
        out = asdict(self)
        out["plays"] = [asdict(p) for p in self.plays]
        return out

    @classmethod
    def from_json(cls, raw: dict) -> "ClipRecord":
        plays = [PlayLabel(**p) for p in raw.get("plays", [])]
        return cls(**{**raw, "plays": plays})


def probe_video(path: Path) -> dict:
    """Read fps and dimensions without decoding the whole file."""
    cv2 = require("cv2")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        return {
            "fps": float(cap.get(cv2.CAP_PROP_FPS)) or None,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
        }
    finally:
        cap.release()


class FilmCorpus:
    """A manifest of clips on disk, with their labels and provenance."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else film_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / MANIFEST_NAME
        self.clips: dict[str, ClipRecord] = {}
        self.load()

    def load(self) -> "FilmCorpus":
        if self.manifest_path.exists():
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.clips = {
                c["clip_id"]: ClipRecord.from_json(c) for c in raw.get("clips", [])
            }
        return self

    def save(self) -> Path:
        payload = {
            "version": 1,
            "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "clips": [c.to_json() for c in self.clips.values()],
        }
        self.manifest_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return self.manifest_path

    def __len__(self) -> int:
        return len(self.clips)

    def __iter__(self) -> Iterator[ClipRecord]:
        return iter(self.clips.values())

    def get(self, clip_id: str) -> ClipRecord | None:
        return self.clips.get(clip_id)

    def register(
        self,
        path: Path,
        clip_id: str | None = None,
        angle: str = "sideline",
        source: str = "unknown",
        rights: str = "unreviewed",
        season: int | None = None,
        week: int | None = None,
        offense: str | None = None,
        defense: str | None = None,
        synthetic: bool = False,
        truth_path: Path | None = None,
        split: str = "train",
        probe: bool = True,
    ) -> ClipRecord:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if angle not in ANGLES:
            raise ValueError(f"angle must be one of {ANGLES}, got {angle!r}")

        clip_id = clip_id or path.stem
        meta = {}
        if probe:
            try:
                meta = probe_video(path)
            except Exception:
                meta = {}

        record = ClipRecord(
            clip_id=clip_id,
            path=str(path),
            angle=angle,
            source=source,
            rights=rights,
            season=season,
            week=week,
            offense=offense,
            defense=defense,
            synthetic=synthetic,
            truth_path=str(truth_path) if truth_path else None,
            split=split,
            added_at=datetime.now(UTC).isoformat(timespec="seconds"),
            **meta,
        )
        self.clips[clip_id] = record
        return record

    def remove(self, clip_id: str) -> bool:
        return self.clips.pop(clip_id, None) is not None

    def label_play(
        self,
        clip_id: str,
        play_id: str,
        coverage: str | None = None,
        **kwargs,
    ) -> PlayLabel:
        clip = self.clips[clip_id]
        for play in clip.plays:
            if play.play_id == play_id:
                if coverage is not None:
                    play.coverage = coverage
                for key, value in kwargs.items():
                    setattr(play, key, value)
                return play
        play = PlayLabel(play_id=play_id, coverage=coverage, **kwargs)
        clip.plays.append(play)
        return play

    def assign_splits(self, val_fraction: float = 0.25, seed: int = 0) -> dict[str, int]:
        """Split by clip so that near-duplicate frames cannot straddle the split."""
        ids = sorted(self.clips)
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_fraction))) if len(ids) > 1 else 0
        for i, clip_id in enumerate(ids):
            self.clips[clip_id].split = "val" if i < n_val else "train"
        return {
            "train": sum(1 for c in self if c.split == "train"),
            "val": sum(1 for c in self if c.split == "val"),
        }

    def summary(self) -> dict:
        clips = list(self)
        by_angle: dict[str, int] = {}
        by_rights: dict[str, int] = {}
        for clip in clips:
            by_angle[clip.angle] = by_angle.get(clip.angle, 0) + 1
            by_rights[clip.rights] = by_rights.get(clip.rights, 0) + 1
        total_plays = sum(len(c.plays) for c in clips)
        return {
            "clips": len(clips),
            "synthetic_clips": sum(1 for c in clips if c.synthetic),
            "real_clips": sum(1 for c in clips if not c.synthetic),
            "plays": total_plays,
            "plays_with_coverage": sum(c.labeled_coverage for c in clips),
            "minutes": round(
                sum(c.duration_s or 0.0 for c in clips) / 60.0, 1
            ),
            "by_angle": by_angle,
            "by_rights": by_rights,
        }


def _frame_signature(frame: np.ndarray) -> np.ndarray:
    """A tiny grayscale thumbnail, used only to tell near-duplicate frames apart."""
    cv2 = require("cv2")
    small = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return (gray - gray.mean()) / (gray.std() + 1e-6)


def sample_frames(
    video_path: Path,
    max_frames: int = 40,
    stride: int | None = None,
    min_difference: float = 0.02,
) -> list[tuple[int, np.ndarray]]:
    """Pick frames worth labeling: spread out in time and visually distinct.

    Labeling twenty consecutive frames of the same snap costs twenty units of
    effort and buys about one, so near-duplicates are dropped on the way out.

    `min_difference` is deliberately a duplicate guard and not a diversity
    filter. Raising it looks like it should improve the sample and instead
    quietly starves it: a fixed All-22 camera holds most of the frame constant,
    so consecutive frames correlate above 0.9 even while the play develops, and a
    threshold like 0.35 rejects nearly everything and returns one frame per clip
    without complaining. Temporal spread is `stride`'s job.
    """
    cv2 = require("cv2")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if stride is None:
            stride = max(1, total // (max_frames * 2)) if total else 5

        picked: list[tuple[int, np.ndarray]] = []
        signatures: list[np.ndarray] = []
        index = -1
        while len(picked) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            if index % stride:
                continue

            sig = _frame_signature(frame)
            # Both signatures are zero mean and unit variance, so this average is
            # a correlation in [-1, 1] and 1 - correlation reads as "how new".
            if signatures and min_difference > 0:
                similarity = float(np.mean(sig * signatures[-1]))
                if 1.0 - similarity < min_difference:
                    continue

            signatures.append(sig)
            picked.append((index, frame))
        return picked
    finally:
        cap.release()


def export_for_labeling(
    corpus: FilmCorpus,
    output_dir: Path,
    max_frames_per_clip: int = 40,
    clips: Iterable[str] | None = None,
) -> dict:
    """Write a YOLO dataset skeleton plus sampled frames, ready to be labeled.

    Frames land in the split their clip belongs to, and an index records which
    clip and frame each image came from so labels can be traced back to film.
    """
    from gridiron.vision.detect import prepare_finetune_dataset

    cv2 = require("cv2")
    output_dir = Path(output_dir)
    yaml_path = prepare_finetune_dataset(output_dir)

    wanted = set(clips) if clips is not None else None
    index: list[dict] = []
    for clip in corpus:
        if wanted is not None and clip.clip_id not in wanted:
            continue
        if not clip.video.exists():
            continue
        for frame_index, frame in sample_frames(
            clip.video, max_frames=max_frames_per_clip
        ):
            name = f"{clip.clip_id}_{frame_index:06d}.jpg"
            dest = output_dir / "images" / clip.split / name
            cv2.imwrite(str(dest), frame)
            index.append(
                {
                    "image": name,
                    "clip_id": clip.clip_id,
                    "frame": frame_index,
                    "split": clip.split,
                }
            )

    (output_dir / "frames.json").write_text(
        json.dumps({"frames": index}, indent=2), encoding="utf-8"
    )
    return {
        "data_yaml": str(yaml_path),
        "images": len(index),
        "clips": len({e["clip_id"] for e in index}),
    }


def labeling_status(dataset_dir: Path) -> dict:
    """How much of an exported dataset has actually been labeled."""
    dataset_dir = Path(dataset_dir)
    out: dict = {"splits": {}, "labeled": 0, "images": 0, "empty": 0}
    for split in ("train", "val"):
        images = sorted((dataset_dir / "images" / split).glob("*.jpg"))
        labels_dir = dataset_dir / "labels" / split
        labeled = 0
        empty = 0
        for image in images:
            label = labels_dir / f"{image.stem}.txt"
            if not label.exists():
                continue
            labeled += 1
            if not label.read_text(encoding="utf-8").strip():
                empty += 1
        out["splits"][split] = {
            "images": len(images),
            "labeled": labeled,
            "unlabeled": len(images) - labeled,
        }
        out["images"] += len(images)
        out["labeled"] += labeled
        out["empty"] += empty
    out["fraction"] = out["labeled"] / out["images"] if out["images"] else 0.0
    return out


def write_yolo_labels(
    label_path: Path,
    boxes: Iterable[tuple[float, float, float, float]],
    width: int,
    height: int,
    class_id: int = 0,
) -> Path:
    """Write boxes in YOLO's normalized cx cy w h format."""
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        bw = abs(x2 - x1) / width
        bh = abs(y2 - y1) / height
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return label_path


def seed_synthetic_clips(
    corpus: FilmCorpus,
    n_plays: int = 12,
    output_dir: Path | None = None,
    seed: int = 0,
    jitter_px: float = 0.0,
) -> list[ClipRecord]:
    """Render synthetic plays into the corpus, boxes and coverage labels included.

    This exists because the corpus starts empty and every downstream stage needs
    something to run on. Synthetic film is not a substitute for real film - the
    detector it produces has never seen turf, shadows, or a referee - but it has
    exact ground truth for free, so it is what makes the pipeline testable before
    a single real clip is cleared for use.
    """
    from gridiron.tracking.synthetic import SyntheticConfig, generate_season
    from gridiron.vision.render import CameraSpec, render_play

    output_dir = Path(output_dir) if output_dir else corpus.root / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = SyntheticConfig(weeks=1, plays_per_game=max(1, n_plays), seed=seed)
    records: list[ClipRecord] = []
    for i, play in enumerate(generate_season(cfg)):
        if i >= n_plays:
            break
        clip_path = output_dir / f"{play.play_id}.mp4"
        truth = render_play(
            play,
            clip_path,
            camera=CameraSpec.around(55.0),
            los_x=55.0,
            jitter_px=jitter_px,
            seed=seed + i,
        )
        record = corpus.register(
            clip_path,
            clip_id=play.play_id,
            angle="synthetic",
            source="generated",
            rights="owned",
            season=play.season,
            week=play.week,
            offense=play.offense_team,
            defense=play.defense_team,
            synthetic=True,
            truth_path=clip_path.with_suffix(".truth.json"),
        )
        corpus.label_play(
            record.clip_id,
            play_id=play.play_id,
            coverage=play.coverage,
            snap_frame=truth.snap_frame,
            offense=play.offense_team,
            defense=play.defense_team,
        )
        records.append(record)
    corpus.save()
    return records


def export_synthetic_detection_labels(
    corpus: FilmCorpus, output_dir: Path, max_frames_per_clip: int = 20
) -> dict:
    """Build a fully labeled YOLO dataset from synthetic clips.

    The rendered truth already knows every box, so this is the one corner of the
    corpus where labels cost nothing.
    """
    from gridiron.vision.detect import prepare_finetune_dataset
    from gridiron.vision.render import RenderedTruth

    cv2 = require("cv2")
    output_dir = Path(output_dir)
    yaml_path = prepare_finetune_dataset(output_dir)

    written = 0
    for clip in corpus:
        if not clip.synthetic or not clip.truth_path:
            continue
        truth_file = Path(clip.truth_path)
        if not truth_file.exists() or not clip.video.exists():
            continue
        truth = RenderedTruth.load(truth_file)

        if not truth.frames:
            continue
        indices = sorted(int(k) for k in truth.frames)
        step = max(1, len(indices) // max_frames_per_clip)
        wanted = set(indices[::step])

        cap = cv2.VideoCapture(str(clip.video))
        try:
            index = -1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                index += 1
                if index not in wanted:
                    continue
                entries = truth.frames.get(str(index), [])
                name = f"{clip.clip_id}_{index:06d}"
                cv2.imwrite(
                    str(output_dir / "images" / clip.split / f"{name}.jpg"), frame
                )
                write_yolo_labels(
                    output_dir / "labels" / clip.split / f"{name}.txt",
                    [tuple(e["box"]) for e in entries],
                    width=frame.shape[1],
                    height=frame.shape[0],
                )
                written += 1
        finally:
            cap.release()

    return {"data_yaml": str(yaml_path), "images": written}
