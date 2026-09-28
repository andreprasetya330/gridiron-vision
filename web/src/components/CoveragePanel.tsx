import type { Play } from "../types";

interface Props {
  play: Play;
}

export function CoveragePanel({ play }: Props) {
  const prediction = play.prediction;

  if (!prediction) {
    return (
      <div className="coverage-stack">
        <section className="card filing-card">
          <h2>Filed as</h2>
          <div className="filing-path">
            <span className="filing-chip">{play.defense_team ?? "Unassigned team"}</span>
            <span className="filing-sep">/</span>
            <span className="filing-chip">{play.coverage_family ?? "Unscored"}</span>
          </div>
        </section>
        <section className="card">
          <h2>Coverage</h2>
          <p className="muted">
            No prediction stored. Run <code>gridiron train baseline</code> then re-analyze.
          </p>
        </section>
      </div>
    );
  }

  const entries = Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1]);
  const correct = play.coverage !== null ? play.coverage === prediction.coverage : null;
  const disguise = prediction.disguise;
  const look = prediction.presnap?.coverage ?? disguise?.showed;

  return (
    <div className="coverage-stack">
      <section className="card filing-card">
        <h2>Filed as</h2>
        <div className="filing-path">
          <span className="filing-chip">{play.defense_team ?? "Unassigned team"}</span>
          <span className="filing-sep">/</span>
          <span className="filing-chip">{play.coverage_family ?? "Unscored"}</span>
          <span className="filing-sep">/</span>
          <span className="filing-chip strong">
            {prediction.coverage}
            {play.coverage_shell ? ` · ${play.coverage_shell}` : ""}
          </span>
        </div>
        {play.offense_team && (
          <p className="muted filing-vs">vs {play.offense_team}</p>
        )}
      </section>

      <section className="card">
        <h2>Coverage call</h2>
        <div className="look-grid">
          <div>
            <div className="look-label">Showed</div>
            <div className="look-value">{look ?? disguise?.showed_shell ?? "—"}</div>
            {prediction.presnap && (
              <div className="look-conf">{(prediction.presnap.confidence * 100).toFixed(0)}%</div>
            )}
          </div>
          <div>
            <div className="look-label">Ran</div>
            <div className="look-value">{prediction.coverage}</div>
            <div className="look-conf">{(prediction.confidence * 100).toFixed(0)}%</div>
          </div>
        </div>

        {disguise?.disguised && (
          <div className="disguise-badge">
            Disguise · {disguise.kind}
            {disguise.showed_shell && disguise.ran_shell
              ? ` · ${disguise.showed_shell} → ${disguise.ran_shell}`
              : ""}
          </div>
        )}

        {play.coverage && (
          <div className={`truth ${correct ? "correct" : "wrong"}`}>
            Labeled {play.coverage}
            {correct ? " — match" : " — miss"}
          </div>
        )}
      </section>

      <section className="card">
        <h2>Model probabilities</h2>
        <div className="bars">
          {entries.map(([coverage, probability]) => (
            <div className="bar-row" key={coverage}>
              <span className="bar-label">{coverage}</span>
              <div className="bar-track">
                <div
                  className={`bar-fill ${coverage === prediction.coverage ? "top" : ""}`}
                  style={{ width: `${Math.max(probability * 100, 0.5)}%` }}
                />
              </div>
              <span className="bar-value">{(probability * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </section>

      <QualityNotice play={play} />
    </div>
  );
}

function QualityNotice({ play }: Props) {
  const { quality } = play;
  const problems: string[] = [];

  if (quality.defenders_detected < 11) {
    problems.push(
      `Only ${quality.defenders_detected} defenders tracked. Missing safeties hurt Cover 1 / 3 / 4.`,
    );
  }
  if (quality.registration_error_yd > 1.5) {
    problems.push(`Field registration is off by about ${quality.registration_error_yd.toFixed(1)} yards.`);
  }
  if (quality.frames_with_full_defense < 0.7) {
    problems.push(
      `Full defense visible in ${(quality.frames_with_full_defense * 100).toFixed(0)}% of frames.`,
    );
  }
  problems.push(...quality.notes);

  if (problems.length === 0) {
    return (
      <section className="card quality good">
        Tracking quality is clean on this play.
      </section>
    );
  }

  return (
    <section className="card quality warn">
      <strong>Read this with caution</strong>
      <ul>
        {problems.map((problem) => (
          <li key={problem}>{problem}</li>
        ))}
      </ul>
    </section>
  );
}
