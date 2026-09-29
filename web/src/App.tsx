import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { CoveragePanel } from "./components/CoveragePanel";
import { FieldView } from "./components/FieldView";
import { FilmIngest } from "./components/FilmIngest";
import { VideoOverlay } from "./components/VideoOverlay";
import type { Health, Play, PlaySummary } from "./types";

const SNAP_LABEL = "SNAP";

interface LibraryGroup {
  team: string;
  families: { family: string; plays: PlaySummary[] }[];
}

function groupByTeamAndFamily(plays: PlaySummary[]): LibraryGroup[] {
  const teams = new Map<string, Map<string, PlaySummary[]>>();
  for (const play of plays) {
    const team = play.defense_team || "Unassigned";
    const family = play.coverage_family || "Unscored";
    if (!teams.has(team)) teams.set(team, new Map());
    const families = teams.get(team)!;
    if (!families.has(family)) families.set(family, []);
    families.get(family)!.push(play);
  }
  return [...teams.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([team, families]) => ({
      team,
      families: [...families.entries()]
        .sort((a, b) => a[0].localeCompare(b[0]))
        .map(([family, grouped]) => ({ family, plays: grouped })),
    }));
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [source, setSource] = useState<string>("film");
  const [team, setTeam] = useState<string>("");
  const [filtersReady, setFiltersReady] = useState(false);
  const [coverageFilter, setCoverageFilter] = useState<string>("");
  const [disguiseOnly, setDisguiseOnly] = useState(false);
  const [plays, setPlays] = useState<PlaySummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [play, setPlay] = useState<Play | null>(null);
  const [frame, setFrame] = useState(20);
  const [playing, setPlaying] = useState(false);
  const [showTrails, setShowTrails] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [libraryTick, setLibraryTick] = useState(0);
  const [mediaTime, setMediaTime] = useState(0);
  const [mediaDuration, setMediaDuration] = useState(0);
  const [seekTime, setSeekTime] = useState<number | null>(null);

  const rafRef = useRef<number | null>(null);
  const lastTickRef = useRef<number>(0);

  const refreshCatalog = useCallback(() => {
    api
      .health()
      .then(setHealth)
      .catch((e) => setError(String(e)));
    setFiltersReady(true);
  }, []);

  useEffect(() => {
    refreshCatalog();
  }, [refreshCatalog, libraryTick]);

  useEffect(() => {
    if (!filtersReady) return;
    const controller = new AbortController();
    api
      .plays(
        {
          coverage: coverageFilter || undefined,
          disguised_only: disguiseOnly || undefined,
          source,
          limit: 400,
        },
        controller.signal,
      )
      .then((r) => {
        setPlays(r.plays);
        setSelected((current) => {
          if (current && r.plays.some((item) => item.play_id === current)) return current;
          return r.plays[0]?.play_id ?? null;
        });
      })
      .catch((e) => {
        if (e instanceof Error && e.name === "AbortError") return;
        setError(String(e));
      });
    return () => controller.abort();
  }, [filtersReady, coverageFilter, disguiseOnly, source, libraryTick]);

  useEffect(() => {
    if (!selected) {
      setPlay(null);
      return;
    }
    const controller = new AbortController();
    api
      .play(selected, controller.signal)
      .then((p) => {
        setPlay(p);
        setFrame(p.time_grid.findIndex((t) => Math.abs(t) < 1e-6) || 20);
        setMediaTime(0);
        setMediaDuration(0);
        setSeekTime(0);
        setPlaying(false);
      })
      .catch((e) => {
        if (e instanceof Error && e.name === "AbortError") return;
        setError(String(e));
      });
    return () => controller.abort();
  }, [selected]);

  const nFrames = play?.time_grid.length ?? 51;
  const filmClock = Boolean(
    play?.video_path && play.media_kind !== "image" && play.vision_frames?.length,
  );
  const disguiseCount = useMemo(() => plays.filter((p) => p.disguised).length, [plays]);
  const visiblePlays = useMemo(() => {
    const query = team.trim().toLowerCase();
    if (!query) return plays;
    return plays.filter((item) => (item.defense_team || "").toLowerCase().includes(query));
  }, [plays, team]);
  const library = useMemo(() => groupByTeamAndFamily(visiblePlays), [visiblePlays]);

  const handleMediaTime = useCallback((time: number, visionIndex: number) => {
    setMediaTime(time);
    setFrame(visionIndex);
  }, []);

  const tick = useCallback(
    (timestamp: number) => {
      if (timestamp - lastTickRef.current >= 100) {
        lastTickRef.current = timestamp;
        setFrame((f) => (f + 1 >= nFrames ? 0 : f + 1));
      }
      rafRef.current = requestAnimationFrame(tick);
    },
    [nFrames],
  );

  useEffect(() => {
    if (playing && !filmClock) {
      rafRef.current = requestAnimationFrame(tick);
    }
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, tick, filmClock]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === " ") {
        event.preventDefault();
        setPlaying((p) => !p);
      } else if (event.key === "ArrowRight") {
        setFrame((f) => Math.min(nFrames - 1, f + 1));
      } else if (event.key === "ArrowLeft") {
        setFrame((f) => Math.max(0, f - 1));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [nFrames]);

  const currentTime = play?.time_grid[frame] ?? 0;
  const hasVision = Boolean(play?.vision_frames?.length);
  const matchup = play
    ? `${play.defense_team ?? "Defense"} vs ${play.offense_team ?? "offense"}`
    : "Drop film to analyze a play";

  return (
    <div className="page">
      <div className="shell">
        <nav className="rail" aria-label="Workspace">
          <span className="rail-logo">GV</span>
          <span className="rail-dot active" title="Film room" />
        </nav>

        <div className="workspace">
          <header>
            <div>
              <p className="eyebrow">Coverage control center</p>
              <h1>Gridiron Vision</h1>
              <p className="lede">
                Put film in. Get the player-tracking boxes and a minimap of those detections.
              </p>
            </div>
            <div className="header-stats">
              <div className="stat-chip">
                <span>Film plays</span>
                <strong>{health?.sources?.film ?? plays.length}</strong>
              </div>
              <div className="stat-chip">
                <span>Scored</span>
                <strong>{health?.predictions ?? "—"}</strong>
              </div>
              <div className="stat-chip">
                <span>Disguise in view</span>
                <strong>{disguiseCount}</strong>
              </div>
            </div>
          </header>

          <FilmIngest
            onError={setError}
            onAnalyzed={(next, defenseTeam) => {
              setSource("film");
              setTeam(defenseTeam);
              setPlay(next);
              setSelected(next.play_id);
              setFrame(next.time_grid.findIndex((t) => Math.abs(t) < 1e-6) || 20);
              setLibraryTick((value) => value + 1);
            }}
          />

          {error && <div className="error">{error}</div>}

          <div className="layout">
            <aside className="sidebar card">
              <p className="eyebrow">Library</p>
              <label className="field-label">
                Corpus
                <select
                  value={source}
                  onChange={(e) => {
                    setFiltersReady(false);
                    setPlays([]);
                    setPlay(null);
                    setSelected(null);
                    setTeam("");
                    setSource(e.target.value);
                  }}
                >
                  <option value="film">Film</option>
                  <option value="auto">Tracking (BDB)</option>
                </select>
              </label>

              <label className="field-label">
                Defense team
                <input
                  type="text"
                  value={team}
                  onChange={(e) => setTeam(e.target.value)}
                  placeholder="Type a team"
                  autoComplete="off"
                />
              </label>

              <label className="field-label">
                Coverage
                <select value={coverageFilter} onChange={(e) => setCoverageFilter(e.target.value)}>
                  <option value="">All coverages</option>
                  {(health?.coverages ?? []).map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </label>

              <label className="check-label">
                <input
                  type="checkbox"
                  checked={disguiseOnly}
                  onChange={(e) => setDisguiseOnly(e.target.checked)}
                />
                Disguise only
              </label>

              <div className="play-list">
                {filtersReady &&
                  library.map((group) => (
                    <div className="library-team" key={group.team}>
                      <h3 className="library-team-name">{group.team}</h3>
                      {group.families.map((family) => (
                        <div key={`${group.team}-${family.family}`}>
                          <p className="library-family">{family.family}</p>
                          {family.plays.map((summary) => {
                            const correct =
                              summary.coverage_truth && summary.coverage_predicted
                                ? summary.coverage_truth === summary.coverage_predicted
                                : null;
                            return (
                              <button
                                key={summary.play_id}
                                className={`play-item ${selected === summary.play_id ? "active" : ""}`}
                                onClick={() => setSelected(summary.play_id)}
                              >
                                <div className="play-item-top">
                                  <span className="play-cov">
                                    {summary.coverage_predicted ?? "—"}
                                  </span>
                                  {correct !== null && (
                                    <span
                                      className={`dot ${correct ? "ok" : "bad"}`}
                                      title={correct ? "matches label" : "differs from label"}
                                    />
                                  )}
                                </div>
                                <div className="play-item-sub">
                                  {summary.coverage_shell ?? ""}
                                  {summary.situation?.down ? ` · ${summary.situation.down} & ` : ""}
                                  {summary.situation?.distance ?? ""}
                                  {summary.has_video && <span className="flag film">film</span>}
                                  {summary.disguised && (
                                    <span className="flag disguise">disguise</span>
                                  )}
                                </div>
                              </button>
                            );
                          })}
                        </div>
                      ))}
                    </div>
                  ))}
                {!filtersReady && <p className="muted">Loading plays…</p>}
                {filtersReady && plays.length === 0 && (
                  <p className="muted">
                    {source === "film"
                      ? "No film plays yet. Drop a clip above."
                      : "No plays in this filter."}
                  </p>
                )}
              </div>
            </aside>

            <main className="stage">
              {play ? (
                <>
                  <div className="stage-header">
                    <div>
                      <h2>{matchup}</h2>
                      <p className="muted">
                        {[play.coverage_family, play.prediction?.coverage ?? play.coverage]
                          .filter(Boolean)
                          .join(" · ") || play.play_id}
                        {play.situation.down
                          ? ` · ${play.situation.down} and ${play.situation.distance}`
                          : ""}
                      </p>
                    </div>
                    <label className="check-label inline">
                      <input
                        type="checkbox"
                        checked={showTrails}
                        onChange={(e) => setShowTrails(e.target.checked)}
                      />
                      Trails
                    </label>
                  </div>

                  <div className="stage-body">
                    <div className="stage-visuals">
                      {play.video_path && (
                        <VideoOverlay
                          play={play}
                          frame={frame}
                          playing={playing}
                          seekTime={seekTime}
                          onMediaTime={handleMediaTime}
                          onDuration={setMediaDuration}
                        />
                      )}
                      {!hasVision && (
                        <div className="minimap-row">
                          <div className="minimap-pane">
                            <p className="minimap-label">Minimap</p>
                            <FieldView play={play} frame={frame} showTrails={showTrails} />
                          </div>
                          {play.minimap_url && (
                            <div className="minimap-pane">
                              <p className="minimap-label">Bird&apos;s-eye snapshot</p>
                              <img
                                className="minimap-still"
                                src={play.minimap_url}
                                alt="Projected player minimap"
                              />
                            </div>
                          )}
                        </div>
                      )}
                      {!play.video_path && (
                        <p className="muted film-hint">No film clip on this play.</p>
                      )}
                      <div className="scrubber card">
                        <button className="play-button" onClick={() => setPlaying((p) => !p)}>
                          {playing ? "Pause" : "Play"}
                        </button>
                        {filmClock ? (
                          <input
                            type="range"
                            min={0}
                            max={Math.max(mediaDuration, 0.01)}
                            step={0.01}
                            value={Math.min(mediaTime, mediaDuration || mediaTime)}
                            onChange={(e) => {
                              const next = Number(e.target.value);
                              setPlaying(false);
                              setMediaTime(next);
                              setSeekTime(next);
                            }}
                          />
                        ) : (
                          <input
                            type="range"
                            min={0}
                            max={nFrames - 1}
                            value={frame}
                            onChange={(e) => setFrame(Number(e.target.value))}
                          />
                        )}
                        <span className="timecode">
                          {filmClock
                            ? `${mediaTime.toFixed(2)}s`
                            : Math.abs(currentTime) < 1e-6
                              ? SNAP_LABEL
                              : `${currentTime > 0 ? "+" : ""}${currentTime.toFixed(2)}s`}
                        </span>
                      </div>
                    </div>
                    <aside className="details">
                      <CoveragePanel play={play} />
                    </aside>
                  </div>
                </>
              ) : (
                <p className="muted">Drop a clip to see player tracking and the field minimap.</p>
              )}
            </main>
          </div>
        </div>
      </div>
    </div>
  );
}
