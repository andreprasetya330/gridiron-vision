import { useMemo } from "react";
import type { Play } from "../types";

const ROLE_COLORS: Record<string, string> = {
  man: "#f97316",
  deep_zone: "#38bdf8",
  underneath_zone: "#a78bfa",
  blitz: "#ef4444",
};

const ROLE_LABELS: Record<string, string> = {
  man: "MAN",
  deep_zone: "DEEP",
  underneath_zone: "UNDER",
  blitz: "BLITZ",
};

interface Props {
  play: Play;
  frame: number;
  showRoles: boolean;
  showTrails: boolean;
}

// Field window, in yards relative to the line of scrimmage and the ball.
const X_MIN = -12;
const X_MAX = 30;
const Y_HALF = 28;

const WIDTH = 900;
const HEIGHT = (WIDTH * (X_MAX - X_MIN)) / (Y_HALF * 2);

/**
 * Bird's-eye view. The field is drawn rotated so downfield runs up the screen,
 * which is how coaches draw it on a board and how every coverage diagram in
 * football is oriented.
 */
export function FieldView({ play, frame, showRoles, showTrails }: Props) {
  const toScreen = useMemo(() => {
    return (x: number, y: number): [number, number] => {
      const sx = ((y + Y_HALF) / (Y_HALF * 2)) * WIDTH;
      const sy = HEIGHT - ((x - X_MIN) / (X_MAX - X_MIN)) * HEIGHT;
      return [sx, sy];
    };
  }, []);

  const yardLines = useMemo(() => {
    const lines: { y: number; label: string; major: boolean }[] = [];
    for (let x = Math.ceil(X_MIN / 5) * 5; x <= X_MAX; x += 5) {
      const [, sy] = toScreen(x, 0);
      lines.push({ y: sy, label: x === 0 ? "LOS" : `${x > 0 ? "+" : ""}${x}`, major: x % 10 === 0 });
    }
    return lines;
  }, [toScreen]);

  const trailStart = Math.max(0, frame - 12);

  return (
    <svg
      className="field"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Bird's-eye view of player positions"
    >
      <rect x={0} y={0} width={WIDTH} height={HEIGHT} fill="#0d1f14" rx={8} />

      {yardLines.map((line) => (
        <g key={line.label}>
          <line
            x1={0}
            x2={WIDTH}
            y1={line.y}
            y2={line.y}
            stroke={line.label === "LOS" ? "#facc15" : "#ffffff"}
            strokeOpacity={line.label === "LOS" ? 0.85 : line.major ? 0.22 : 0.1}
            strokeWidth={line.label === "LOS" ? 2 : 1}
            strokeDasharray={line.label === "LOS" ? "8 6" : undefined}
          />
          <text x={8} y={line.y - 5} className="yard-label">
            {line.label}
          </text>
        </g>
      ))}

      {/* Hash marks: college hashes sit 40 feet apart, NFL 18.5. */}
      {[-1, 1].map((sign) => {
        const hashY = play.situation.league === "nfl" ? 3.08 : 6.67;
        const [sx] = toScreen(0, sign * hashY - play.situation.ball_y_from_center);
        return (
          <line
            key={sign}
            x1={sx}
            x2={sx}
            y1={0}
            y2={HEIGHT}
            stroke="#ffffff"
            strokeOpacity={0.08}
            strokeDasharray="4 10"
          />
        );
      })}

      {showTrails &&
        play.players.map((player) => {
          const points: string[] = [];
          for (let i = trailStart; i <= frame; i += 1) {
            const px = player.x[i];
            const py = player.y[i];
            if (px === null || py === null) continue;
            const [sx, sy] = toScreen(px, py);
            points.push(`${sx.toFixed(1)},${sy.toFixed(1)}`);
          }
          if (points.length < 2) return null;
          const role = play.prediction?.roles[player.track_id]?.role ?? player.role;
          const color =
            player.side === "offense" ? "#e2e8f0" : (showRoles && role && ROLE_COLORS[role]) || "#f87171";
          return (
            <polyline
              key={`trail-${player.track_id}`}
              points={points.join(" ")}
              fill="none"
              stroke={color}
              strokeOpacity={0.35}
              strokeWidth={2}
            />
          );
        })}

      {play.players.map((player) => {
        const px = player.x[frame];
        const py = player.y[frame];
        if (px === null || py === null) return null;
        const [sx, sy] = toScreen(px, py);

        const predictedRole = play.prediction?.roles[player.track_id];
        const role = predictedRole?.role ?? player.role;
        const isDefense = player.side === "defense";
        const color = isDefense
          ? (showRoles && role && ROLE_COLORS[role]) || "#f87171"
          : "#e2e8f0";

        return (
          <g key={player.track_id}>
            <circle
              cx={sx}
              cy={sy}
              r={isDefense ? 9 : 8}
              fill={color}
              fillOpacity={isDefense ? 0.9 : 0.55}
              stroke="#020617"
              strokeWidth={1.5}
            />
            {player.jersey !== null && (
              <text x={sx} y={sy + 3.5} className="jersey">
                {player.jersey}
              </text>
            )}
            {isDefense && showRoles && role && (
              <text x={sx} y={sy - 13} className="role-label" fill={color}>
                {ROLE_LABELS[role] ?? role}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
