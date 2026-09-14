import type { Play } from "../types";

interface Props {
  play: Play;
}

/**
 * Probabilities, the honest version. The confidence shown here is temperature
 * calibrated on held-out plays, and the panel says out loud when the underlying
 * tracking was bad enough that the number should not be trusted.
 */
export function CoveragePanel({ play }: Props) {
  const prediction = play.prediction;

  if (!prediction) {
    return (
      <div className="panel">
        <h2>Coverage</h2>
        <p className="muted">
          No prediction stored for this play. Run <code>gridiron demo</code> or train a model
          and re-score.
        </p>
      </div>
    );
  }

  const entries = Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1]);
  const correct =
    play.coverage !== null ? play.coverage === prediction.coverage : null;

  return (
    <div className="panel">
      <h2>Coverage</h2>

      <div className="call">
        <span className="call-name">{prediction.coverage}</span>
        <span className="call-confidence">{(prediction.confidence * 100).toFixed(0)}%</span>
      </div>

      {play.coverage && (
        <div className={`truth ${correct ? "correct" : "wrong"}`}>
          Labeled {play.coverage}
          {correct ? " — match" : " — miss"}
        </div>
      )}

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

      <QualityNotice play={play} />
    </div>
  );
}

function QualityNotice({ play }: Props) {
  const { quality } = play;
  const problems: string[] = [];

  if (quality.defenders_detected < 11) {
    problems.push(
      `Only ${quality.defenders_detected} defenders were tracked. The players most often lost are the deep safeties, which are exactly the ones that separate Cover 1, 3, and 4.`,
    );
  }
  if (quality.registration_error_yd > 1.5) {
    problems.push(
      `Field registration is off by about ${quality.registration_error_yd.toFixed(1)} yards, so depths are unreliable.`,
    );
  }
  if (quality.frames_with_full_defense < 0.7) {
    problems.push(
      `The full defense was visible in only ${(quality.frames_with_full_defense * 100).toFixed(0)}% of frames.`,
    );
  }
  problems.push(...quality.notes);

  if (problems.length === 0) {
    return <div className="quality good">Tracking quality is clean on this play.</div>;
  }

  return (
    <div className="quality warn">
      <strong>Read this with caution</strong>
      <ul>
        {problems.map((problem) => (
          <li key={problem}>{problem}</li>
        ))}
      </ul>
    </div>
  );
}
