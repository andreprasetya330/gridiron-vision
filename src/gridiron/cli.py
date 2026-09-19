"""Command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from gridiron import config as cfg

app = typer.Typer(add_completion=False, help="Gridiron Vision: coverage detection for football film")
bdb_app = typer.Typer(help="NFL Big Data Bowl data")
train_app = typer.Typer(help="Model training")
film_app = typer.Typer(help="Film processing")
ingest_app = typer.Typer(help="External data ingestion")
tells_app = typer.Typer(help="Tell mining and validation")
corpus_app = typer.Typer(help="The film corpus and its labels")

app.add_typer(bdb_app, name="bdb")
app.add_typer(train_app, name="train")
app.add_typer(film_app, name="film")
app.add_typer(ingest_app, name="ingest")
app.add_typer(tells_app, name="tells")
app.add_typer(corpus_app, name="corpus")

console = Console()

SOURCE_HELP = "Play corpus: auto (real data over synthetic), bdb, synthetic, film, or all"


def _load_corpus(source: str = "auto", limit: int | None = None):
    from gridiron.tracking.schema import load_plays

    directory, recursive, resolved = cfg.resolve_play_directory(source)
    plays = load_plays(directory, limit=limit, recursive=recursive)
    return plays, directory, resolved


@app.command()
def doctor() -> None:
    """Check the environment, especially the GPU situation."""
    import sys

    table = Table(title="Gridiron Vision environment")
    table.add_column("Check")
    table.add_column("Result")

    table.add_row("python", sys.version.split()[0])
    table.add_row("data dir", str(cfg.data_dir()))

    try:
        import torch

        table.add_row("torch", torch.__version__)
        table.add_row("cuda available", str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            table.add_row("gpu", f"{name} (sm_{capability[0]}{capability[1]})")
            table.add_row("cuda build", str(torch.version.cuda))
            try:
                torch.zeros(8, device="cuda").sum().item()
                table.add_row("gpu kernels", "[green]working[/green]")
            except Exception as exc:
                table.add_row(
                    "gpu kernels",
                    f"[red]FAILED[/red] {exc}\nBlackwell (sm_120) needs cu128+ wheels.",
                )
    except ImportError:
        table.add_row("torch", "[red]not installed[/red]")

    for module, label in (
        ("cv2", "opencv"),
        ("ultralytics", "ultralytics"),
        ("supervision", "supervision"),
        ("inference_sdk", "inference-sdk"),
        ("shap", "shap"),
    ):
        try:
            __import__(module)
            table.add_row(label, "[green]installed[/green]")
        except ImportError:
            hint = " (uv sync --extra vision)" if module in ("cv2", "ultralytics", "supervision", "inference_sdk") else ""
            table.add_row(label, f"[yellow]missing[/yellow]{hint}")

    import os

    table.add_row("CFBD_API_KEY", "set" if os.environ.get("CFBD_API_KEY") else "[yellow]not set[/yellow]")
    from gridiron.config import RoboflowSettings

    rf = RoboflowSettings()
    table.add_row("ROBOFLOW_API_KEY", "set" if rf.api_key else "[yellow]not set[/yellow]")
    if rf.api_key:
        table.add_row("Roboflow workflow", f"{rf.workspace}/{rf.workflow_id}")
    try:
        from gridiron.ingest.cfbd import CFBDClient, CFBDConfig

        ping = CFBDClient(CFBDConfig(timeout=12.0, max_retries=2)).ping()
        table.add_row("CFBD ping", f"[green]ok[/green] ({ping['conferences']} conferences)")
    except Exception as exc:
        table.add_row("CFBD ping", f"[yellow]{exc}[/yellow]")
    console.print(table)


@app.command()
def demo(
    weeks: int = typer.Option(12, help="Weeks of synthetic season to generate"),
    plays_per_game: int = typer.Option(62),
    degraded: float = typer.Option(0.25, help="Fraction of plays degraded to look like film"),
    epochs: int = typer.Option(25),
    skip_net: bool = typer.Option(False, help="Skip the neural model (faster)"),
    team: str = typer.Option("Washington", help="Team to scout"),
) -> None:
    """Run the whole system end to end on a synthetic season."""
    import numpy as np
    import pandas as pd

    from gridiron.coverage.baseline import CoverageBaseline, describe_split, split_by_week
    from gridiron.coverage.features import build_feature_frame
    from gridiron.cues.vocabulary import cues_to_frame
    from gridiron.db.store import PlayStore
    from gridiron.scouting.evidence import build_evidence
    from gridiron.scouting.report import write_report
    from gridiron.tells.forest import CoverageForest, ForestConfig
    from gridiron.tells.mining import MiningConfig, TellMiner
    from gridiron.tells.validation import forward_validate
    from gridiron.tracking.schema import clear_plays, save_plays
    from gridiron.tracking.synthetic import (
        SyntheticConfig,
        generate_season,
        planted_tell_summary,
    )

    console.rule("[bold]1. Generating synthetic season")
    synth_config = SyntheticConfig(
        weeks=weeks, plays_per_game=plays_per_game, degraded_fraction=degraded
    )
    plays = list(generate_season(synth_config))
    console.print(f"generated {len(plays)} plays across {weeks} weeks")

    directory = cfg.plays_dir("synthetic")
    clear_plays(directory)
    save_plays(plays, directory)
    console.print(f"wrote plays to {directory}")

    console.rule("[bold]2. Extracting cues and features")
    cues = cues_to_frame(plays)
    features = build_feature_frame(plays, mode="postsnap")
    store = PlayStore()
    store.write("cues", cues)
    console.print(f"cue table: {cues.shape[0]} rows x {cues.shape[1]} columns")

    console.rule("[bold]3. Training the gradient boosting baselines")
    train_df, test_df = split_by_week(features, holdout_weeks=3)
    console.print(describe_split(train_df, test_df))

    post = CoverageBaseline(mode="postsnap").fit(train_df)
    post_metrics = post.evaluate(test_df)
    console.print(f"post-snap: {post_metrics.summary()}")
    post.save(cfg.models_dir() / "baseline_postsnap.joblib")

    pre_features = build_feature_frame(plays, mode="presnap")
    pre_train, pre_test = split_by_week(pre_features, holdout_weeks=3)
    pre = CoverageBaseline(mode="presnap").fit(pre_train)
    pre_metrics = pre.evaluate(pre_test)
    console.print(f"pre-snap:  {pre_metrics.summary()}")
    pre.save(cfg.models_dir() / "baseline_presnap.joblib")

    console.rule("[bold]4. Random forest over the cue vocabulary")
    train_ids = set(train_df["play_id"])
    cue_train = cues[cues["play_id"].isin(train_ids)]
    cue_test = cues[~cues["play_id"].isin(train_ids)]
    forest = CoverageForest(ForestConfig(include_team=False)).fit(cue_train)
    forest.save(cfg.models_dir() / "coverage_forest.joblib")

    predicted = forest.predict(cue_test)
    truth = cue_test["coverage"].to_numpy()
    console.print(f"forest accuracy on held-out weeks: {(predicted == truth).mean():.3f}")

    importance = forest.feature_importance().head(8)
    table = Table(title="What the forest keys on")
    table.add_column("Feature")
    table.add_column("Importance", justify="right")
    for _, row in importance.iterrows():
        table.add_row(str(row["feature"]), f"{row['importance']:.4f}")
    console.print(table)

    predictions: list[dict] = []

    if not skip_net:
        console.rule("[bold]5. Training the set transformer")
        from gridiron.coverage.train import TrainConfig, train as train_net

        test_ids = set(test_df["play_id"])
        train_plays = [p for p in plays if p.play_id not in test_ids]
        val_plays = [p for p in plays if p.play_id in test_ids]
        trained, metrics = train_net(
            train_plays, val_plays, TrainConfig(epochs=epochs), verbose=False
        )
        console.print(f"set transformer: {metrics.summary()}")
        console.print(f"role head accuracy: {trained.metrics.get('role_accuracy')}")
        console.print(f"calibration temperature: {trained.temperature:.3f}")
        trained.save(cfg.models_dir() / "coverage_net.pt")

        console.rule("[bold]6. Scoring every play")
        for play in plays:
            predictions.append(trained.predict_play(play))
    else:
        console.rule("[bold]5. Scoring every play with the baseline")
        proba = post.predict_proba(features)
        for i, play in enumerate(plays):
            order = np.argsort(-proba[i])
            predictions.append(
                {
                    "play_id": play.play_id,
                    "coverage": post.classes[int(order[0])],
                    "confidence": float(proba[i][order[0]]),
                    "probabilities": {c: float(v) for c, v in zip(post.classes, proba[i])},
                    "runner_up": post.classes[int(order[1])],
                    "roles": {},
                    "quality_score": round(play.quality.score, 3),
                    "usable": play.quality.usable,
                }
            )

    from gridiron.coverage.bridge import write_predictions

    write_predictions(predictions, cfg.predictions_path())
    console.print(f"wrote {len(predictions)} predictions")

    console.rule("[bold]7. Mining tells")
    mining_config = MiningConfig(min_support=10, min_lift=0.10, fdr_alpha=0.10)
    miner = TellMiner(cues, mining_config)
    all_results = miner.mine_all(current_week=weeks)

    table = Table(title="Tells found per team")
    table.add_column("Team")
    table.add_column("Tests", justify="right")
    table.add_column("Significant", justify="right")
    table.add_column("Reported", justify="right")
    table.add_column("Top tell")
    for name, result in all_results.items():
        top = result.tells[0].description if result.tells else "-"
        table.add_row(name, str(result.n_tests), str(result.n_significant), str(len(result.tells)), top)
    console.print(table)

    tell_rows = [t.to_dict() for r in all_results.values() for t in r.tells]
    if tell_rows:
        store.write("tells", pd.DataFrame(tell_rows))

    console.rule("[bold]8. Did we recover what we planted?")
    planted = planted_tell_summary(synth_config)
    trait_to_cues = {
        "late_rotation": {"late_rotation", "rotation_direction"},
        "press_corners": {"press_corners", "corner_depth_band", "corner_leverage"},
        "deep_safeties": {"safety_depth_band", "shell", "deepest_depth_band", "deep_defender_count"},
        "nickel_creep": {"slot_defender_depth_band"},
        "heavy_box": {"box_count_band", "linebacker_depth_band", "los_defenders_band"},
    }
    table = Table(title="Planted tells vs mined tells")
    table.add_column("Team")
    table.add_column("Planted trait")
    table.add_column("Coverage")
    table.add_column("Effect", justify="right")
    table.add_column("Recovered")
    for row in planted:
        result = all_results.get(row["team"])
        expected_cues = trait_to_cues.get(row["trait"], set())
        found = any(
            t.cue_key in expected_cues and t.coverage == row["coverage"]
            for t in (result.tells if result else [])
        )
        table.add_row(
            row["team"],
            row["trait"],
            row["coverage"],
            f"{row['effect_size']:+.2f}",
            "[green]yes[/green]" if found else "[yellow]no[/yellow]",
        )
    control = all_results.get("California")
    if control is not None:
        table.add_row(
            "California", "[dim]none planted[/dim]", "-", "-",
            f"{len(control.tells)} reported (want few)",
        )
    console.print(table)

    console.rule("[bold]9. Forward validation")
    validation = forward_validate(cues, team, mining_config)
    if validation.predictive:
        console.print(
            f"{team}: {validation.pooled_hit_rate:.0%} pooled out-of-sample hit rate "
            f"vs {validation.pooled_expected:.0%} expected; "
            f"{validation.survival_rate:.0%} of tells held up"
        )
    elif validation.validations:
        console.print(
            f"{team}: only avoidance tells mined; "
            f"{validation.pooled_avoidance_rate:.0%} out of sample "
            f"vs {validation.pooled_avoidance_expected:.0%} expected (lower is correct)"
        )
    else:
        console.print(f"{team}: " + "; ".join(validation.notes))

    console.rule("[bold]10. Scouting reports")
    evidence = build_evidence(cues, team, through_week=weeks, mining_config=mining_config)
    md_path, json_path = write_report(evidence, use_llm=False)
    console.print(f"opponent report: {md_path}")

    self_evidence = build_evidence(
        cues, "Oregon State", through_week=weeks, mining_config=mining_config, self_scout=True
    )
    self_md, _ = write_report(self_evidence, use_llm=False)
    console.print(f"self-scout: {self_md}")

    console.rule("[bold green]Done")
    console.print("Run `gridiron serve` and start the web app in ./web to see the overlay.")


@bdb_app.command("download")
def bdb_download() -> None:
    """Download Big Data Bowl data via the Kaggle CLI."""
    from gridiron.tracking import bdb

    console.print(
        "Uses kagglehub for tracking (accepted BDB 2021 rules). "
        "Coverage labels fall back to the public ngscleanR copies if the old "
        "Kaggle extra dataset is gone."
    )
    path = bdb.download()
    status = bdb.available(path)
    console.print(f"downloaded to {path}")
    console.print(
        f"weeks={status['n_weeks']}  coverages={status['n_coverage_files']}  "
        f"labeled plays={status['n_labeled_plays']}"
    )


@bdb_app.command("status")
def bdb_status() -> None:
    """Report which Big Data Bowl files are present."""
    from gridiron.tracking import bdb

    status = bdb.available()
    table = Table(title=f"Big Data Bowl files in {cfg.bdb_dir()}")
    table.add_column("File")
    table.add_column("Present")
    for key, value in status.items():
        table.add_row(key, str(value))
    console.print(table)
    labeled_weeks = str(status.get("labeled_weeks") or "")
    if labeled_weeks in {"", "none"} or "," not in labeled_weeks:
        console.print(
            "[yellow]Coverage labels look like a single week. The public Tom Bliss "
            "set is week 1 only. Drop additional coverages*.csv files into "
            f"{cfg.bdb_dir()} for a week split; training will hold out whole games "
            "until then.[/yellow]"
        )
    counts = cfg.play_source_counts()
    console.print(
        f"play corpora on disk: "
        + ", ".join(f"{name}={n}" for name, n in counts.items() if n)
    )


@bdb_app.command("build")
def bdb_build(
    weeks: Optional[str] = typer.Option(None, help="Comma-separated week numbers"),
    limit: Optional[int] = typer.Option(None),
    labeled_only: bool = typer.Option(True, help="Skip plays with no coverage label"),
) -> None:
    """Convert raw Big Data Bowl tracking data into normalized plays."""
    from gridiron.tracking import bdb
    from gridiron.tracking.schema import clear_plays, save_plays

    week_list = [int(w) for w in weeks.split(",")] if weeks else None
    plays = list(bdb.iter_plays(weeks=week_list, limit=limit, labeled_only=labeled_only))
    if not plays:
        console.print("[yellow]no labeled plays found[/yellow]")
        raise typer.Exit(1)

    directory = cfg.plays_dir("bdb")
    cleared = clear_plays(directory)
    n = save_plays(plays, directory)
    rates_path = bdb.save_league_rates(plays)
    labeled_weeks = sorted({p.week for p in plays if p.week is not None})
    console.print(f"wrote {n} plays to {directory} (cleared {cleared} stale files)")
    console.print(f"league coverage rates -> {rates_path}")
    if len(labeled_weeks) <= 1:
        console.print(
            f"[yellow]labels cover week(s) {labeled_weeks or 'unknown'} only. "
            "Week-based holdouts and forward-validated tells need more; training "
            "will split by game instead. Additional coverages*.csv files in "
            f"{cfg.bdb_dir()} are picked up automatically.[/yellow]"
        )


@train_app.command("baseline")
def train_baseline(
    mode: str = typer.Option("postsnap", help="presnap or postsnap"),
    holdout_weeks: int = typer.Option(3),
    source: str = typer.Option("auto", help=SOURCE_HELP),
) -> None:
    """Train the gradient boosting baseline on whatever plays are on disk."""
    from gridiron.coverage.baseline import CoverageBaseline, describe_split, split_by_week
    from gridiron.coverage.features import build_feature_frame

    plays, directory, resolved = _load_corpus(source)
    labeled = [p for p in plays if p.coverage]
    if not labeled:
        console.print("[red]no labeled plays; run `gridiron demo` or `gridiron bdb build`[/red]")
        raise typer.Exit(1)

    console.print(f"corpus: {resolved} ({directory})  {len(labeled)} labeled plays")
    features = build_feature_frame(labeled, mode=mode)
    train_df, test_df = split_by_week(features, holdout_weeks=holdout_weeks)
    console.print(describe_split(train_df, test_df))
    model = CoverageBaseline(mode=mode).fit(train_df)
    metrics = model.evaluate(test_df)
    console.print(metrics.summary())

    path = cfg.models_dir() / f"baseline_{mode}.joblib"
    model.save(path)
    predicted = model.predict(test_df)
    from collections import Counter

    from gridiron.coverage.train import save_metrics

    report = {
        "mode": mode,
        "source": resolved,
        "split": describe_split(train_df, test_df),
        "train_n": int(len(train_df)),
        "test_n": int(len(test_df)),
        "train_weeks": sorted(int(w) for w in train_df["week"].dropna().unique()),
        "test_weeks": sorted(int(w) for w in test_df["week"].dropna().unique()),
        "metrics": metrics.to_dict(),
        "true_class_counts": dict(Counter(test_df["coverage"].dropna().astype(str))),
        "predicted_class_counts": dict(Counter(str(x) for x in predicted)),
        "dropped_columns": model.dropped_columns,
        "model_path": str(path),
    }
    metrics_file = cfg.models_dir() / f"baseline_{mode}_metrics.json"
    report["metrics_path"] = str(metrics_file)
    save_metrics(report, metrics_file)
    console.print(f"saved to {path}")
    console.print(f"metrics -> {metrics_file}")


@train_app.command("net")
def train_net_command(
    mode: str = typer.Option("postsnap"),
    epochs: int = typer.Option(40),
    holdout_weeks: int = typer.Option(3),
    source: str = typer.Option("auto", help=SOURCE_HELP),
) -> None:
    """Train the set transformer with the per-defender role head."""
    from gridiron.coverage.baseline import describe_split, split_plays
    from gridiron.coverage.train import TrainConfig, train as run_train

    plays = [p for p in _load_corpus(source)[0] if p.coverage]
    if not plays:
        console.print("[red]no labeled plays on disk[/red]")
        raise typer.Exit(1)

    train_plays, val_plays = split_plays(plays, holdout_weeks=holdout_weeks)
    console.print(describe_split(train_plays, val_plays))
    trained, metrics = run_train(train_plays, val_plays, TrainConfig(mode=mode, epochs=epochs))
    console.print(metrics.summary())

    path = cfg.models_dir() / f"coverage_net_{mode}.pt"
    trained.save(path)
    console.print(f"saved to {path}")


@film_app.command("process")
def film_process(
    video: Path = typer.Argument(..., help="Path to an MP4 or a still image"),
    league: str = typer.Option("ncaa"),
    single_play: bool = typer.Option(True, help="Treat the clip as one play"),
    stride: Optional[int] = typer.Option(
        None, help="Process every Nth frame. Default 3 for Roboflow, 1 for local."
    ),
    backend: str = typer.Option(
        "auto",
        help="Film backend: auto (Roboflow when ROBOFLOW_API_KEY is set), roboflow, or local",
    ),
) -> None:
    """Turn film into per-play tracking JSON via the Roboflow coverage workflow."""
    from gridiron.coverage.bridge import upsert_predictions
    from gridiron.tracking.schema import save_plays
    from gridiron.vision.pipeline import FilmPipeline, PipelineConfig

    resolved_stride = stride
    if resolved_stride is None:
        from gridiron.config import RoboflowSettings

        uses_hosted = backend != "local" and RoboflowSettings().enabled
        resolved_stride = 3 if uses_hosted else 1

    config = PipelineConfig(
        league=league,
        single_play=single_play,
        stride=resolved_stride,
        backend=backend,
    )
    pipeline = FilmPipeline(config)
    plays = pipeline.process(video)
    if not plays:
        console.print(
            "[yellow]no usable plays extracted. Common causes: the homography could "
            "not be solved, or offense and defense could not be separated.[/yellow]"
        )
        raise typer.Exit(1)

    out_dir = cfg.plays_dir("film")
    n = save_plays(plays, out_dir)
    if pipeline.predictions:
        pred_path = upsert_predictions(pipeline.predictions, cfg.predictions_path())
        console.print(f"wrote {len(pipeline.predictions)} workflow coverage call(s) to {pred_path}")
    for play in plays:
        rushers = sum(1 for p in play.players if p.role == "blitz")
        console.print(
            f"{play.play_id}: {play.quality.defenders_detected} defenders, "
            f"{rushers} rushers, snap {play.snap_frame_in_video}, "
            f"{len(play.time_grid)} ticks, "
            f"registration error {play.quality.registration_error_yd:.2f} yd, "
            f"quality {play.quality.score:.2f}"
        )
        for note in play.quality.notes:
            console.print(f"  [yellow]{note}[/yellow]")
    console.print(f"wrote {n} plays")


@film_app.command("dataset")
def film_dataset(output: Path = typer.Option(Path("data/detector"))) -> None:
    """Create the YOLO dataset skeleton for hand labeling."""
    from gridiron.vision.detect import prepare_finetune_dataset

    path = prepare_finetune_dataset(output)
    console.print(f"dataset skeleton at {path.parent}")
    console.print("Put images in images/train and YOLO labels in labels/train, then run:")
    console.print(f"  uv run gridiron film finetune {path}")


@film_app.command("finetune")
def film_finetune(
    data_yaml: Path = typer.Argument(...),
    epochs: int = typer.Option(60),
    weights: str = typer.Option("yolov8m.pt", help="Base weights"),
    image_size: int = typer.Option(1280),
    batch: int = typer.Option(4, help="Lower this first if CUDA runs out of memory"),
) -> None:
    """Fine-tune the player detector."""
    from gridiron.vision.detect import finetune

    finetune(
        data_yaml,
        base_weights=weights,
        epochs=epochs,
        image_size=image_size,
        batch=batch,
    )


@film_app.command("evaluate")
def film_evaluate(
    weights: str = typer.Option(
        "player-detector/weights/best.pt", help="Detector weights"
    ),
    split: str = typer.Option("val", help="Which corpus split to score"),
    frames: int = typer.Option(40, help="Frames per clip"),
    image_size: int = typer.Option(960),
) -> None:
    """Score detection and tracking against corpus ground truth."""
    from gridiron.vision.corpus import FilmCorpus
    from gridiron.vision.detect import DetectorConfig, PlayerDetector
    from gridiron.vision.evaluate import evaluate_corpus

    corpus = FilmCorpus()
    if not len(corpus):
        console.print("[red]corpus is empty; try `gridiron corpus seed`[/red]")
        raise typer.Exit(1)

    detector = PlayerDetector(
        DetectorConfig(weights=weights, custom_model=True, image_size=image_size)
    )
    result = evaluate_corpus(
        corpus, detector, max_frames_per_clip=frames, split=split or None
    )
    if not result["clips"]:
        console.print(f"[yellow]no clips with ground truth in split {split!r}[/yellow]")
        raise typer.Exit(1)

    detection, tracking = result["detection"], result["tracking"]
    table = Table(title=f"Detection and tracking on {result['clips']} {split} clips")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for label, key in [
        ("precision", "precision"),
        ("recall", "recall"),
        ("mAP50", "average_precision"),
        ("mean IoU", "mean_iou"),
    ]:
        table.add_row(label, f"{detection[key]:.3f}")
    table.add_row("ID switches", str(tracking["id_switches"]))
    table.add_row("role-crossing switches", str(tracking["role_crossing_switches"]))
    table.add_row("mostly tracked", f"{tracking['mostly_tracked_fraction']:.3f}")
    table.add_row("usable plays", f"{tracking['usable_fraction']:.0%}")
    console.print(table)

    if tracking["role_crossing_switches"]:
        console.print(
            "[yellow]Identity swapped across a role boundary. That is the kind "
            "that corrupts cue extraction, so treat those plays as suspect.[/yellow]"
        )


@corpus_app.command("add")
def corpus_add(
    video: Path = typer.Argument(..., help="Path to a clip"),
    angle: str = typer.Option("sideline", help="sideline, endzone, or broadcast"),
    source: str = typer.Option(..., help="Where it came from, e.g. 'UW Hudl cutup'"),
    rights: str = typer.Option(
        "unreviewed", help="cleared, owned, unreviewed, or restricted"
    ),
    season: Optional[int] = typer.Option(None),
    week: Optional[int] = typer.Option(None),
    offense: Optional[str] = typer.Option(None),
    defense: Optional[str] = typer.Option(None),
) -> None:
    """Register a clip in the corpus."""
    from gridiron.vision.corpus import FilmCorpus

    corpus = FilmCorpus()
    record = corpus.register(
        video,
        angle=angle,
        source=source,
        rights=rights,
        season=season,
        week=week,
        offense=offense,
        defense=defense,
    )
    corpus.save()
    duration = f"{record.duration_s:.1f}s" if record.duration_s else "unknown length"
    console.print(f"registered {record.clip_id} ({angle}, {duration}, rights={rights})")


@corpus_app.command("seed")
def corpus_seed(
    plays: int = typer.Option(12, help="How many synthetic plays to render"),
    seed: int = typer.Option(0),
) -> None:
    """Render synthetic clips so the pipeline has something to run on."""
    from gridiron.vision.corpus import FilmCorpus, seed_synthetic_clips

    corpus = FilmCorpus()
    records = seed_synthetic_clips(corpus, n_plays=plays, seed=seed)
    console.print(f"rendered {len(records)} synthetic clips into {corpus.root}")
    console.print(
        "[yellow]Synthetic film has exact labels and none of the appearance of "
        "real film. Use it to test the pipeline, not to judge the detector.[/yellow]"
    )


@corpus_app.command("list")
def corpus_list() -> None:
    """Show the corpus and how much of it is labeled."""
    from gridiron.vision.corpus import FilmCorpus

    corpus = FilmCorpus()
    if not len(corpus):
        console.print("[yellow]corpus is empty; try `gridiron corpus seed`[/yellow]")
        return

    table = Table(title=f"Film corpus in {corpus.root}")
    for column in ("Clip", "Angle", "Rights", "Split", "Length", "Plays", "Labeled"):
        table.add_column(column)
    for clip in sorted(corpus, key=lambda c: c.clip_id):
        length = f"{clip.duration_s:.1f}s" if clip.duration_s else "-"
        table.add_row(
            clip.clip_id,
            clip.angle,
            clip.rights,
            clip.split,
            length,
            str(len(clip.plays)),
            str(clip.labeled_coverage),
        )
    console.print(table)

    summary = corpus.summary()
    console.print(
        f"{summary['clips']} clips ({summary['real_clips']} real, "
        f"{summary['synthetic_clips']} synthetic), {summary['minutes']} minutes, "
        f"{summary['plays_with_coverage']}/{summary['plays']} plays labeled"
    )
    if summary["by_rights"].get("unreviewed"):
        console.print(
            f"[yellow]{summary['by_rights']['unreviewed']} clips have unreviewed "
            f"rights. Settle that before anything built on them leaves the lab."
            f"[/yellow]"
        )


@corpus_app.command("export")
def corpus_export(
    output: Path = typer.Option(Path("data/detector"), help="Dataset directory"),
    frames: int = typer.Option(40, help="Frames sampled per clip"),
    val_fraction: float = typer.Option(0.25),
    labeled: bool = typer.Option(
        False, "--labeled", help="Emit synthetic ground-truth boxes instead of blanks"
    ),
) -> None:
    """Export frames as a YOLO dataset for labeling (or with synthetic labels)."""
    from gridiron.vision.corpus import (
        FilmCorpus,
        export_for_labeling,
        export_synthetic_detection_labels,
        labeling_status,
    )

    corpus = FilmCorpus()
    if not len(corpus):
        console.print("[red]corpus is empty[/red]")
        raise typer.Exit(1)

    corpus.assign_splits(val_fraction=val_fraction)
    corpus.save()

    if labeled:
        result = export_synthetic_detection_labels(
            corpus, output, max_frames_per_clip=frames
        )
    else:
        result = export_for_labeling(corpus, output, max_frames_per_clip=frames)

    console.print(f"wrote {result['images']} images to {output}")
    status = labeling_status(output)
    console.print(f"{status['labeled']}/{status['images']} images have labels")
    if not labeled:
        console.print(f"Label them, then: uv run gridiron film finetune {output}/data.yaml")


@corpus_app.command("label")
def corpus_label(
    clip_id: str = typer.Argument(...),
    play_id: str = typer.Option(..., help="Play within the clip"),
    coverage: str = typer.Option(..., help="Ground-truth coverage name"),
    snap_frame: Optional[int] = typer.Option(None),
) -> None:
    """Attach a ground-truth coverage label to a play."""
    from gridiron.taxonomy import COVERAGES
    from gridiron.vision.corpus import FilmCorpus

    if coverage not in COVERAGES:
        console.print(f"[red]unknown coverage {coverage!r}[/red]")
        console.print(f"expected one of: {', '.join(COVERAGES)}")
        raise typer.Exit(1)

    corpus = FilmCorpus()
    if corpus.get(clip_id) is None:
        console.print(f"[red]no clip {clip_id!r} in the corpus[/red]")
        raise typer.Exit(1)

    extra = {"snap_frame": snap_frame} if snap_frame is not None else {}
    corpus.label_play(clip_id, play_id, coverage=coverage, **extra)
    corpus.save()
    console.print(f"{clip_id}/{play_id} labeled {coverage}")


@ingest_app.command("cfbd")
def ingest_cfbd(
    team: Optional[str] = typer.Option(None, help="Team name, e.g. 'Washington'"),
    season: Optional[int] = typer.Option(None),
    weeks: int = typer.Option(15),
    ping: bool = typer.Option(False, "--ping", help="Verify the API key with one cheap call"),
) -> None:
    """Pull a team's defensive plays from CollegeFootballData."""
    from gridiron.db.store import PlayStore
    from gridiron.ingest.cfbd import CFBDClient, CFBDError, opponent_profile, season_context

    client = CFBDClient()
    if ping:
        try:
            result = client.ping()
        except CFBDError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        console.print(f"[green]CFBD authenticated[/green] - {result['conferences']} conferences visible")
        return

    if not team or season is None:
        console.print("[red]--team and --season are required unless you pass --ping[/red]")
        raise typer.Exit(2)

    try:
        context = season_context(client, season, team, range(1, weeks + 1))
    except CFBDError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if context.empty:
        console.print("[yellow]no plays returned[/yellow]")
        raise typer.Exit(1)

    store = PlayStore()
    store.write("plays", context, mode="append")
    console.print(f"ingested {len(context)} defensive plays for {team} ({season})")

    profile = opponent_profile(client, season, team)
    path = cfg.subdir("cache", "profiles") / f"{team.lower().replace(' ', '-')}-{season}.json"
    path.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
    console.print(f"team profile written to {path}")


@ingest_app.command("nflverse")
def ingest_nflverse(
    season: int = typer.Option(...), team: Optional[str] = typer.Option(None)
) -> None:
    """Pull nflfastR play-by-play."""
    from gridiron.db.store import PlayStore
    from gridiron.ingest.nflverse import pbp_to_context

    context = pbp_to_context(season, team)
    PlayStore().write("plays", context, mode="append")
    console.print(f"ingested {len(context)} plays from nflverse {season}")


@ingest_app.command("pff")
def ingest_pff(inspect_only: bool = typer.Option(False, "--inspect")) -> None:
    """Import PFF exports from the drop folder."""
    from gridiron.db.store import PlayStore
    from gridiron.ingest.pff import PffDropFolderSource, describe_drop_folder

    console.print(describe_drop_folder())
    if inspect_only:
        return

    source = PffDropFolderSource()
    if not source.available():
        raise typer.Exit(0)

    context = source.play_context()
    if not context.empty:
        PlayStore().write("plays", context, mode="append")
        labeled = int(context["coverage"].notna().sum())
        console.print(
            f"imported {len(context)} snaps, {labeled} with coverage labels. "
            "Per-snap coverage labels are the highest-value data in this project."
        )

    rates = source.coverage_rates()
    if not rates.empty:
        path = cfg.subdir("cache", "pff") / "coverage_rates.parquet"
        rates.to_parquet(path, index=False)
        console.print(f"coverage summary written to {path}")


@tells_app.command("mine")
def tells_mine(
    team: str = typer.Option(...),
    min_support: int = typer.Option(10),
    min_lift: float = typer.Option(0.10),
    fdr_alpha: float = typer.Option(0.10),
    week: Optional[int] = typer.Option(None, help="Only use film through this week"),
    source: str = typer.Option("auto", help=SOURCE_HELP),
) -> None:
    """Mine one team's tells against the national baseline."""
    from gridiron.cues.vocabulary import cues_to_frame
    from gridiron.tells.mining import TellMiner, config_for_corpus

    plays, directory, resolved = _load_corpus(source)
    cues = cues_to_frame(plays)
    if cues.empty:
        console.print("[red]no plays available[/red]")
        raise typer.Exit(1)

    console.print(f"corpus: {resolved} ({directory})")
    config = config_for_corpus(
        resolved, min_support=min_support, min_lift=min_lift, fdr_alpha=fdr_alpha
    )
    if config.baseline_source != "observed":
        console.print(f"national baseline: borrowed {config.baseline_source}")
    result = TellMiner(cues, config).mine(team, current_week=week)

    console.print(
        f"{result.n_plays} plays, {result.n_tests} comparisons, "
        f"{result.n_significant} significant after FDR correction"
    )
    for note in result.notes:
        console.print(f"[yellow]{note}[/yellow]")

    if not result.tells:
        return

    table = Table(title=f"{team} tells")
    table.add_column("Cue")
    table.add_column("Coverage")
    table.add_column("Them", justify="right")
    table.add_column("Expected", justify="right")
    table.add_column("League", justify="right")
    table.add_column("n", justify="right")
    table.add_column("q", justify="right")
    for tell in result.tells:
        table.add_row(
            f"{tell.cue_key}={tell.cue_value}",
            tell.coverage,
            f"{tell.shrunk_rate:.0%}",
            f"{tell.expected_rate:.0%}",
            f"{tell.baseline_rate:.0%}",
            str(tell.n_cue),
            f"{tell.q_value:.3f}",
        )
    console.print(table)


@tells_app.command("validate")
def tells_validate(
    team: str = typer.Option(...),
    min_train_weeks: int = typer.Option(4),
    source: str = typer.Option("auto", help=SOURCE_HELP),
) -> None:
    """Forward-validate a team's tells on film they were not mined from."""
    from gridiron.cues.vocabulary import cues_to_frame
    from gridiron.tells.mining import config_for_corpus
    from gridiron.tells.validation import forward_validate

    plays, _directory, resolved = _load_corpus(source)
    cues = cues_to_frame(plays)
    result = forward_validate(
        cues, team, config_for_corpus(resolved), min_train_weeks=min_train_weeks
    )
    for note in result.notes:
        console.print(f"[yellow]{note}[/yellow]")
    if not result.validations:
        raise typer.Exit(0)

    table = Table(title=f"{team}: out-of-sample tell performance")
    table.add_column("Cue")
    table.add_column("Coverage")
    table.add_column("Dir")
    table.add_column("Mined", justify="right")
    table.add_column("Holdout", justify="right")
    table.add_column("Record", justify="right")
    table.add_column("Held up")
    for v in result.validations:
        table.add_row(
            f"{v.cue_key}={v.cue_value}",
            v.coverage,
            "runs" if v.direction > 0 else "avoids",
            f"{v.mined_rate:.0%}",
            "n/a" if v.holdout_trials == 0 else f"{v.holdout_rate:.0%}",
            f"{v.holdout_hits}/{v.holdout_trials}",
            "[green]yes[/green]" if v.held_up else "no",
        )
    console.print(table)
    if result.predictive:
        console.print(
            f"predictive tells: {result.pooled_hit_rate:.0%} out-of-sample hit rate "
            f"vs {result.pooled_expected:.0%} expected"
        )
    if result.avoidance:
        console.print(
            f"avoidance tells: {result.pooled_avoidance_rate:.0%} out of sample "
            f"vs {result.pooled_avoidance_expected:.0%} expected (lower is correct)"
        )


@app.command()
def scout(
    team: str = typer.Option(...),
    season: Optional[int] = typer.Option(None),
    week: Optional[int] = typer.Option(None),
    self_scout: bool = typer.Option(False, "--self-scout"),
    use_llm: bool = typer.Option(True, help="Use the LLM for narrative prose if configured"),
    source: str = typer.Option("auto", help=SOURCE_HELP),
) -> None:
    """Build the evidence bundle and write a scouting report."""
    from gridiron.cues.vocabulary import cues_to_frame
    from gridiron.db.store import PlayStore
    from gridiron.scouting.evidence import build_evidence
    from gridiron.scouting.report import write_report
    from gridiron.tells.mining import config_for_corpus

    plays, _directory, resolved = _load_corpus(source)
    cues = cues_to_frame(plays)
    if cues.empty:
        console.print("[red]no plays available[/red]")
        raise typer.Exit(1)

    store = PlayStore()
    play_context = store.read("plays")
    evidence = build_evidence(
        cues,
        team,
        season=season,
        through_week=week,
        mining_config=config_for_corpus(resolved),
        play_context=play_context if not play_context.empty else None,
        self_scout=self_scout,
    )
    md_path, json_path = write_report(evidence, use_llm=use_llm)
    console.print(f"report:   {md_path}")
    console.print(f"evidence: {json_path}")
    console.print("")
    console.print(evidence.headline)
    for tell in evidence.tells[:5]:
        console.print(f"  - {tell['description']}")


@app.command()
def score(
    model_path: Optional[Path] = typer.Option(None, help="coverage_net.pt or baseline_*.joblib"),
    limit: Optional[int] = typer.Option(None),
    source: str = typer.Option("auto", help=SOURCE_HELP),
) -> None:
    """Score every play on disk and write predictions.json for the overlay."""
    from gridiron.coverage.bridge import score_plays, write_predictions

    plays, directory, resolved = _load_corpus(source, limit=limit)
    if not plays:
        console.print("[red]no plays on disk; run `gridiron demo` or `gridiron film process`[/red]")
        raise typer.Exit(1)

    console.print(f"corpus: {resolved} ({directory})  {len(plays)} plays")
    model = _load_scorer(model_path)
    predictions = score_plays(plays, model)
    path = write_predictions(predictions, cfg.predictions_path())
    console.print(f"wrote {len(predictions)} predictions to {path}")


@film_app.command("bridge")
def film_bridge(
    n_plays: int = typer.Option(40, help="How many synthetic snaps to run through the camera"),
    seed: int = typer.Option(0),
) -> None:
    """Measure how much accuracy the camera costs.

    Renders synthetic snaps, recovers them through the vision pipeline, and
    scores the same coverage model on both the clean tracks and the recovered
    ones. The drop is the domain gap. Until real All-22 is labeled, this is the
    number that decides whether the vision pipeline is good enough to train on.
    """
    from gridiron.coverage.baseline import CoverageBaseline, split_plays
    from gridiron.coverage.bridge import apply_film_noise, measure_gap
    from gridiron.coverage.features import build_feature_frame
    from gridiron.tracking.synthetic import SyntheticConfig, generate_season

    plays = [p for p in _load_corpus("auto")[0] if p.coverage]
    if len(plays) < 80:
        plays = [p for p in generate_season(SyntheticConfig(weeks=4, plays_per_game=40, seed=seed)) if p.coverage]

    train_plays, holdout_all = split_plays(plays, holdout_weeks=1)
    if len(holdout_all) < n_plays:
        train_plays, holdout_all = plays[: int(0.7 * len(plays))], plays[int(0.7 * len(plays)) :]
    model = CoverageBaseline(mode="postsnap")
    model.fit(build_feature_frame(train_plays, mode="postsnap"))
    holdout = holdout_all[:n_plays]

    noisy = apply_film_noise(holdout, seed=seed)
    gap = measure_gap(holdout, noisy, model, film_source="augmented")
    console.print(f"augmented stand-in: {gap.summary()}")

    try:
        from gridiron.vision.pipeline import PipelineConfig, process_video
        from gridiron.vision.registration import FixedRegistrar
        from gridiron.vision.render import CameraSpec, TruthBoxDetector, render_play
    except Exception as exc:
        console.print(f"[yellow]vision extras missing, skipping recovered-film gap ({exc})[/yellow]")
        return

    work = cfg.subdir("scratch", "bridge")
    recovered = []
    clean = []
    for i, play in enumerate(holdout[: min(12, n_plays)]):
        video = work / f"{play.play_id}.mp4"
        truth = render_play(play, video, camera=CameraSpec.around(55.0), los_x=55.0)
        got = process_video(
            video,
            PipelineConfig(league=play.situation.league, single_play=True),
            play_id=play.play_id,
            detector=TruthBoxDetector(truth),
            registrar=FixedRegistrar.from_homography(truth.image_to_field),
        )
        if got:
            recovered.append(got[0])
            clean.append(play)

    if recovered:
        film_gap = measure_gap(clean, recovered, model, film_source="recovered")
        console.print(f"recovered film:     {film_gap.summary()}")
        if film_gap.drop > 0.15:
            console.print(
                "[yellow]The camera costs more than 15 points of accuracy. "
                "Coverage calls on film will need the quality flag, and the "
                "model should be retrained with heavier dropout.[/yellow]"
            )


def _load_scorer(model_path: Optional[Path]):
    from gridiron.coverage.baseline import CoverageBaseline
    from gridiron.coverage.train import TrainedCoverageNet

    if model_path is None:
        net = cfg.models_dir() / "coverage_net.pt"
        baseline = cfg.models_dir() / "baseline_postsnap.joblib"
        if net.exists():
            model_path = net
        elif baseline.exists():
            model_path = baseline
        else:
            console.print("[red]no trained model on disk; run `gridiron demo` or `gridiron train`[/red]")
            raise typer.Exit(1)

    if model_path.suffix == ".pt":
        return TrainedCoverageNet.load(model_path)
    return CoverageBaseline.load(model_path)


@app.command()
def serve(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8000)) -> None:
    """Run the API backing the overlay UI."""
    import uvicorn

    uvicorn.run("gridiron.api.main:app", host=host, port=port, reload=False)


@app.command()
def status() -> None:
    """Summarize what is in the play database."""
    from gridiron.db.store import PlayStore

    summary = PlayStore().summary()
    table = Table(title="Play database")
    table.add_column("Table")
    table.add_column("Rows", justify="right")
    table.add_column("Columns", justify="right")
    for name, info in summary.items():
        table.add_row(name, str(info["rows"]), str(info["columns"]))
    console.print(table)

    plays_table = Table(title="Play corpora")
    plays_table.add_column("Source")
    plays_table.add_column("Plays", justify="right")
    for name, n in cfg.play_source_counts().items():
        if n or name != "_root":
            plays_table.add_row(name, str(n))
    console.print(plays_table)
    _directory, _rec, resolved = cfg.resolve_play_directory("auto")
    console.print(f"auto corpus: {resolved}")


if __name__ == "__main__":
    app()
