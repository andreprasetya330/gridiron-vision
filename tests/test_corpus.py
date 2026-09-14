"""The corpus is bookkeeping, so the tests are about the bookkeeping going wrong."""

from __future__ import annotations

import pytest

from gridiron.vision.corpus import (
    FilmCorpus,
    export_for_labeling,
    export_synthetic_detection_labels,
    labeling_status,
    sample_frames,
    write_yolo_labels,
)

cv2 = pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def seeded_corpus(tmp_path_factory):
    from gridiron.vision.corpus import seed_synthetic_clips

    root = tmp_path_factory.mktemp("corpus")
    corpus = FilmCorpus(root)
    seed_synthetic_clips(corpus, n_plays=4, seed=5)
    return corpus


def test_manifest_round_trips(seeded_corpus):
    reloaded = FilmCorpus(seeded_corpus.root)
    assert len(reloaded) == len(seeded_corpus)
    assert {c.clip_id for c in reloaded} == {c.clip_id for c in seeded_corpus}
    assert all(c.plays and c.plays[0].coverage for c in reloaded)


def test_seeded_clips_carry_coverage_and_rights(seeded_corpus):
    summary = seeded_corpus.summary()
    assert summary["plays_with_coverage"] == summary["plays"] > 0
    # Generated film is ours; nothing in the corpus should default to unreviewed
    # rights without a human having said so.
    assert summary["by_rights"] == {"owned": summary["clips"]}


def test_register_rejects_unknown_angle(seeded_corpus):
    clip = next(iter(seeded_corpus))
    with pytest.raises(ValueError):
        seeded_corpus.register(clip.video, clip_id="bad", angle="drone")


def test_label_play_updates_rather_than_duplicates(seeded_corpus):
    clip = next(iter(seeded_corpus))
    before = len(clip.plays)
    seeded_corpus.label_play(clip.clip_id, clip.plays[0].play_id, coverage="Cover 1 Man")
    assert len(clip.plays) == before
    assert clip.plays[0].coverage == "Cover 1 Man"


def test_splits_are_assigned_per_clip(seeded_corpus):
    counts = seeded_corpus.assign_splits(val_fraction=0.5, seed=3)
    assert counts["train"] + counts["val"] == len(seeded_corpus)
    assert counts["val"] > 0
    assert {c.split for c in seeded_corpus} <= {"train", "val"}


def test_sampling_meets_its_budget(seeded_corpus):
    """The dedup guard must not silently starve the sample.

    A fixed camera makes consecutive frames correlate strongly, so a threshold
    meant as a duplicate guard turns into a near-total filter if it is set like a
    diversity filter. This pins the default to the useful side of that line.
    """
    clip = next(iter(seeded_corpus))
    frames = sample_frames(clip.video, max_frames=8)
    assert len(frames) == 8
    assert len({index for index, _ in frames}) == 8


def test_export_for_labeling_indexes_every_frame(seeded_corpus, tmp_path):
    out = tmp_path / "tolabel"
    result = export_for_labeling(seeded_corpus, out, max_frames_per_clip=3)

    assert result["images"] == 3 * len(seeded_corpus)
    assert (out / "data.yaml").exists()

    status = labeling_status(out)
    assert status["images"] == result["images"]
    assert status["labeled"] == 0  # nothing is labeled until a human does it

    import json

    index = json.loads((out / "frames.json").read_text(encoding="utf-8"))["frames"]
    assert len(index) == result["images"]
    # Every image must be traceable back to the clip and frame it came from.
    for entry in index:
        assert (out / "images" / entry["split"] / entry["image"]).exists()
        assert seeded_corpus.get(entry["clip_id"]) is not None


def test_synthetic_export_is_fully_labeled(seeded_corpus, tmp_path):
    out = tmp_path / "detector"
    result = export_synthetic_detection_labels(seeded_corpus, out, max_frames_per_clip=4)

    status = labeling_status(out)
    assert result["images"] > 0
    assert status["labeled"] == status["images"]
    assert status["empty"] == 0  # every rendered frame has players in it


def test_yolo_labels_are_normalized(tmp_path):
    path = write_yolo_labels(
        tmp_path / "a.txt", [(10, 20, 30, 60)], width=100, height=200
    )
    fields = path.read_text(encoding="utf-8").split()
    assert fields[0] == "0"
    cx, cy, w, h = (float(v) for v in fields[1:])
    assert (cx, cy) == pytest.approx((0.2, 0.2))
    assert (w, h) == pytest.approx((0.2, 0.2))


def test_degenerate_boxes_are_dropped(tmp_path):
    path = write_yolo_labels(
        tmp_path / "b.txt", [(10, 20, 10, 60), (0, 0, 10, 10)], width=100, height=200
    )
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 1
