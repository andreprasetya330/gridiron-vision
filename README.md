# Gridiron Vision

AI coverage detection for American football film. Turns game video into field coordinates,
classifies defensive coverage, mines statistically validated pre-snap tells against a national
baseline, and writes opponent scouting reports where every claim links back to the plays that
produced it.

## The idea

Split vision from coverage. **Roboflow detects players** and the UGA homography
graphs them onto a bird's-eye minimap. Those points become `PlayTracks`. **The
Big Data Bowl coverage model** then reads the points — the same model trained on
NFL tracking — rather than the hosted Qwen image classifier. A still is scored as
a pre-snap look; a video clip gets the post-snap call plus disguise. The
homography is locked to the UGA camera view.

Everything downstream consumes one JSON contract (`PlayTracks`), so film, a Hudl export, or a
PFF export all describe the same thing.

## Quick start

```powershell
# 1. Install (uv handles the Python 3.12 toolchain and the cu128 PyTorch wheels)
uv sync --extra dev --extra vision

# 2. Generate a synthetic season so every stage runs end to end without external downloads
uv run gridiron demo

# 3. Open the overlay
uv run gridiron serve
cd web; npm install; npm run dev
```

`gridiron demo` builds a synthetic but football-realistic season (coverage-conditioned player
alignments, per-team tendencies, deliberately planted tells), trains the models on it, mines the
tells, and writes a scouting report. It exists so you can see the whole system work on day one and
so the tests have ground truth to check against. Swap in real data with `gridiron ingest` and
`gridiron bdb` and nothing downstream changes. Play JSON is stored per corpus
(`data/plays/synthetic`, `data/plays/bdb`, `data/plays/film`) so a leftover demo season cannot
contaminate a Big Data Bowl training run. Commands default to `--source auto`, which prefers real
data over synthetic.

## Roboflow player workflow

Film frames go through the hosted workflow **american-football-player-trackin** in workspace
`andre-4cotb` (RF-DETR small). Copy `.env.example` to `.env` and set `ROBOFLOW_API_KEY` from
[Workspace Settings → API Keys](https://app.roboflow.com/andre-4cotb/settings/api). Never commit
the key.

The Python client is `gridiron.vision.roboflow.run_player_workflow`. It uses
`inference-sdk.InferenceHTTPClient` against `https://serverless.roboflow.com` with header auth
(`Authorization: Bearer`). Do not put the key in the query string or JSON body.

| | |
| --- | --- |
| Input | `image` |
| Parameters | `confidence` (0.4), `iou_threshold` (0.3), `class_agnostic_nms` (false), `max_detections` (1000) |
| Outputs | `predictions` (boxes: `offense_player` / `defense_player` / `official`), `inference_id`, `model_id` |

`gridiron film process` and the web ingest page both call this function, then project feet onto
the UGA minimap and score coverage with the BDB model. Clips are sampled as still frames — live
webcam / RTSP would need Roboflow's WebRTC path, which this repo does not use.

## Commands

| Command | What it does |
| --- | --- |
| `gridiron demo` | Full synthetic pipeline: data, training, tells, report |
| `gridiron bdb download` | Fetch Big Data Bowl 2021 via kagglehub (needs a Kaggle API token) |
| `gridiron bdb status` | Show which BDB files and coverage-label weeks are on disk |
| `gridiron bdb build` | Normalize BDB into `data/plays/bdb/` and persist league coverage rates |
| `gridiron train baseline` | Gradient boosting on engineered features (`--source auto\|bdb\|synthetic\|film`) |
| `gridiron train net` | Set-transformer over the 22 tracks + per-defender role head |
| `gridiron film process <video-or-image>` | Roboflow player detect → minimap points → BDB coverage model |
| `gridiron film evaluate` | Score detection and tracking against corpus ground truth |
| `gridiron film bridge` | Measure how much accuracy the camera costs the coverage model |
| `gridiron film finetune <data.yaml>` | Fine-tune the player detector |
| `gridiron corpus seed` | Render synthetic All-22 clips with labels so the pipeline has film to run on |
| `gridiron corpus add <video>` | Register a real clip (angle, source, rights) |
| `gridiron corpus export` | Write a YOLO dataset of frames to label |
| `gridiron score` | Run the coverage model over plays on disk and write `predictions.json` |
| `gridiron ingest cfbd --ping` | Verify the CollegeFootballData API key |
| `gridiron ingest cfbd --team "Washington" --season 2025` | Pull play context from CollegeFootballData |
| `gridiron ingest pff` | Import whatever PFF exports are sitting in `data/pff/` |
| `gridiron tells mine --team X` | Mine tells vs the national baseline, FDR corrected |
| `gridiron tells validate --team X` | Forward-validate tells and report hit rates |
| `gridiron scout --team X` | Build the evidence bundle and write the report |
| `gridiron scout --self-scout --team Washington` | Find the tells you are giving away |
| `gridiron serve` | FastAPI backend for the overlay UI |

## Data sources

**Works today, free:**
- [CollegeFootballData](https://collegefootballdata.com) - free API key, gives situation, drives,
  advanced stats, PPA/EPA, rosters, matchup history. Everything except coverage.
- nflverse / nflfastR play-by-play, and NFL Big Data Bowl for the coverage labels themselves.

**PFF:** there is no public developer API. `premium.pff.com/api/v1/...` is an internal front-end
endpoint requiring a logged-in session cookie, and automating against it violates their terms. So
`ingest/pff.py` implements a **drop-folder importer** instead: export CSVs from whatever PFF tier
you have into `data/pff/`, and it sniffs headers and maps columns automatically. `PffSessionSource`
is a documented stub, deliberately not implemented.

## Honest limits

- The **national baseline** for college coverage rates has to come from your own processed film
  corpus, which starts empty. Until it grows, the system falls back to NFL rates written by
  `gridiron bdb build` (`data/cache/bdb/coverage_rates.json`) and labels the baseline as
  `borrowed` everywhere it is displayed.
- Training holds out later **weeks** when the corpus has them. The public BDB coverage labels are
  week 1 only, so in that case the split holds out whole **games** instead of cutting the play list
  80/20 (plays from the same game share personnel and game plan).
- **Sample size is the real enemy**, not model accuracy. A defense gives you ~65 snaps a game; with
  ~50 cues x 8 coverages you are running hundreds of comparisons, so chance alone produces
  convincing fake tells every week. The mining engine applies minimum sample thresholds,
  beta-binomial shrinkage, Wilson intervals, and Benjamini-Hochberg FDR correction, then
  forward-validates each tell on later weeks and displays its actual hit rate.
- **Film rights**: Hudl cutups belong to the program. Real UW film needs a coach to say yes.
- **NCAA tech rules**: this is a film-room and scouting tool, not a gameday sideline device.

## Layout

```
src/gridiron/
  tracking/    play representation, normalization, BDB loaders, synthetic generator
  coverage/    features, gradient boosting baseline, set-transformer, training, eval
  vision/      detection, tracking, field registration, corpus, pipeline, eval
  ingest/      CFBD client, nflverse loaders, PFF drop-folder importer, unified schema
  cues/        pre-snap cue vocabulary
  tells/       random forest + SHAP, baseline comparison, FDR, forward validation
  scouting/    evidence bundles and report generation
  db/          DuckDB + Parquet play store
  api/         FastAPI
web/           React overlay
```

## GPU note

The RTX 5060 is Blackwell (sm_120). Only CUDA 12.8+ wheels contain kernels for it. `pyproject.toml`
pins the `pytorch-cu128` index for `torch` and `torchvision`. Verify with `uv run gridiron doctor`.
