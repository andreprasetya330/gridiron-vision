import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { CoveragePanel } from "./components/CoveragePanel";
import { FieldView } from "./components/FieldView";
import { VideoOverlay } from "./components/VideoOverlay";
import type { Health, Play, PlaySummary } from "./types";

const SNAP_LABEL = "SNAP";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [teams, setTeams] = useState<{ team: string; plays: number }[]>([]);
  const [team, setTeam] = useState<string>("");
  const [coverageFilter, setCoverageFilter] = useState<string>("");
  const [plays, setPlays] = useState<PlaySummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [play, setPlay] = useState<Play | null>(null);
  const [frame, setFrame] = useState(20);
  const [playing, setPlaying] = useState(false);
  const [showRoles, setShowRoles] = useState(true);
  const [showTrails, setShowTrails] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const rafRef = useRef<number | null>(null);
  const lastTickRef = useRef<number>(0);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((e) => setError(String(e)));
    api
      .teams()
      .then((r) => {
        setTeams(r.teams);
        if (r.teams.length > 0) setTeam(r.teams[0].team);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    api
      .plays({ team: team || undefined, coverage: coverageFilter || undefined, limit: 400 })
      .then((r) => {
        setPlays(r.plays);
        if (r.plays.length > 0) setSelected(r.plays[0].play_id);
      })
      .catch((e) => setError(String(e)));
  }, [team, coverageFilter]);

  useEffect(() => {
    if (!selected) return;
    api
      .play(selected)
      .then((p) => {
        setPlay(p);
        setFrame(p.time_grid.findIndex((t) => Math.abs(t) < 1e-6) || 20);
      })
      .catch((e) => setError(String(e)));
  }, [selected]);

  const nFrames = play?.time_grid.length ?? 51;

  const tick = useCallback(
    (timestamp: number) => {
      if (timestamp - lastTickRef.current >= 60) {
        lastTickRef.current = timestamp;
        setFrame((f) => (f + 1 >= nFrames ? 0 : f + 1));
      }
      rafRef.current = requestAnimationFrame(tick);
    },
    [nFrames],
  );

  useEffect(() => {
    if (playing) {
      rafRef.current = requestAnimationFrame(tick);
    }
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, tick]);

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

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="logo">GV</span>
          <div>
            <h1>Gridiron Vision</h1>
            <p>Coverage detection and tell mining for football film</p>
          </div>
        </div>
        <div className="header-stats">
          {health && (
            <>
              <span>{health.plays} plays</span>
              <span>{health.predictions} scored</span>
            </>
          )}
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <div className="layout">
        <aside className="sidebar">
          <label className="field-label">
            Defense
            <select value={team} onChange={(e) => setTeam(e.target.value)}>
              <option value="">All teams</option>
              {teams.map((t) => (
                <option key={t.team} value={t.team}>
                  {t.team} ({t.plays})
                </option>
              ))}
            </select>
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

          <div className="play-list">
            {plays.map((summary) => {
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
                    <span className="play-week">W{summary.week ?? "?"}</span>
                    <span className="play-cov">{summary.coverage_predicted ?? "—"}</span>
                    {correct !== null && (
                      <span className={`dot ${correct ? "ok" : "bad"}`} title={correct ? "matches label" : "differs from label"} />
                    )}
                  </div>
                  <div className="play-item-sub">
                    {summary.situation?.down ? `${summary.situation.down} & ` : ""}
                    {summary.situation?.distance ?? ""}
                    {!summary.usable && <span className="flag">low quality</span>}
                  </div>
                </button>
              );
            })}
            {plays.length === 0 && (
              <p className="muted">
                No plays yet. Run <code>uv run gridiron demo</code> to generate a season.
              </p>
            )}
          </div>
        </aside>

        <main className="stage">
          {play ? (
            <>
              <div className="stage-header">
                <div>
                  <h2>{play.play_id}</h2>
                  <p className="muted">
                    {play.defense_team ?? "defense"} vs {play.offense_team ?? "offense"}
                    {play.situation.down
                      ? ` · ${play.situation.down} and ${play.situation.distance}`
                      : ""}
                    {play.situation.hash_side ? ` · ${play.situation.hash_side} hash` : ""}
                    {play.situation.offense_personnel
                      ? ` · ${play.situation.offense_personnel} personnel`
                      : ""}
                  </p>
                </div>
                <div className="toggles">
                  <label>
                    <input
                      type="checkbox"
                      checked={showRoles}
                      onChange={(e) => setShowRoles(e.target.checked)}
                    />
                    Roles
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={showTrails}
                      onChange={(e) => setShowTrails(e.target.checked)}
                    />
                    Trails
                  </label>
                </div>
              </div>

              <VideoOverlay play={play} frame={frame} showRoles={showRoles} />
              <FieldView play={play} frame={frame} showRoles={showRoles} showTrails={showTrails} />

              <div className="scrubber">
                <button className="play-button" onClick={() => setPlaying((p) => !p)}>
                  {playing ? "Pause" : "Play"}
                </button>
                <input
                  type="range"
                  min={0}
                  max={nFrames - 1}
                  value={frame}
                  onChange={(e) => setFrame(Number(e.target.value))}
                />
                <span className="timecode">
                  {Math.abs(currentTime) < 1e-6
                    ? SNAP_LABEL
                    : `${currentTime > 0 ? "+" : ""}${currentTime.toFixed(2)}s`}
                </span>
              </div>

              <div className="legend">
                <span className="key man">Man</span>
                <span className="key deep">Deep zone</span>
                <span className="key under">Underneath zone</span>
                <span className="key blitz">Blitz</span>
                <span className="key off">Offense</span>
              </div>
            </>
          ) : (
            <p className="muted">Select a play.</p>
          )}
        </main>

        <aside className="details">{play && <CoveragePanel play={play} />}</aside>
      </div>
    </div>
  );
}
