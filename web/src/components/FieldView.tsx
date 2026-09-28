import { useMemo } from "react";
import { playerColor, playerLabel } from "../playerColors";
import type { Play } from "../types";

interface Props {
  play: Play;
  frame: number;
  showTrails: boolean;
}

const X_MIN = -12;
const X_MAX = 30;
const Y_HALF = 28;
const WIDTH = 900;
const HEIGHT = (WIDTH * (X_MAX - X_MIN)) / (Y_HALF * 2);

/**
 * Bird's-eye view. Downfield runs up the screen, the way a coverage diagram
 * is drawn on a board.
 */
export function FieldView({ play, frame, showTrails }: Props) {
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
      lines.push({
        y: sy,
        label: x === 0 ? "LOS" : `${x > 0 ? "+" : ""}${x}`,
        major: x % 10 === 0,
      });
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
      <rect x={0} y={0} width={WIDTH} height={HEIGHT} fill="#c4a574" rx={16} />
      <rect x={10} y={10} width={WIDTH - 20} height={HEIGHT - 20} fill="#d7b68a" rx={12} />

      {yardLines.map((line) => (
        <g key={line.label}>
          <line
            x1={18}
            x2={WIDTH - 18}
            y1={line.y}
            y2={line.y}
            stroke={line.label === "LOS" ? "#1c2430" : "#fff"}
            strokeOpacity={line.label === "LOS" ? 0.85 : line.major ? 0.55 : 0.28}
            strokeWidth={line.label === "LOS" ? 2.5 : 1.2}
            strokeDasharray={line.label === "LOS" ? "8 6" : undefined}
          />
          <text x={24} y={line.y - 6} className="yard-label">
            {line.label}
          </text>
        </g>
      ))}

      {[-1, 1].map((sign) => {
        const hashY = play.situation.league === "nfl" ? 3.08 : 6.67;
        const [sx] = toScreen(0, sign * hashY - play.situation.ball_y_from_center);
        return (
          <line
            key={sign}
            x1={sx}
            x2={sx}
            y1={16}
            y2={HEIGHT - 16}
            stroke="#ffffff"
            strokeOpacity={0.35}
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
          const color = playerColor(player.track_id, player.side, player.role);
          return (
            <polyline
              key={`trail-${player.track_id}`}
              points={points.join(" ")}
              fill="none"
              stroke={color}
              strokeOpacity={0.45}
              strokeWidth={3}
            />
          );
        })}

      {play.players.map((player) => {
        const px = player.x[frame];
        const py = player.y[frame];
        if (px === null || py === null) return null;
        const [sx, sy] = toScreen(px, py);
        const color = playerColor(player.track_id, player.side, player.role);
        const isDefense = player.side === "defense";
        const isRusher = player.role === "blitz";
        const label = playerLabel(player.jersey, player.side, player.role);
        const radius = isRusher ? 12 : isDefense ? 11 : 9;
        return (
          <g key={player.track_id}>
            {isRusher ? (
              <rect
                x={sx - radius}
                y={sy - radius}
                width={radius * 2}
                height={radius * 2}
                rx={3}
                fill={color}
                stroke="#1c2430"
                strokeOpacity={0.35}
                strokeWidth={1.5}
              />
            ) : (
              <circle
                cx={sx}
                cy={sy}
                r={radius}
                fill={color}
                stroke="#1c2430"
                strokeOpacity={0.35}
                strokeWidth={1.5}
              />
            )}
            {label && (
              <text
                x={sx}
                y={sy + 4}
                textAnchor="middle"
                fill="#fff"
                fontSize={isRusher ? 8 : 9}
                fontWeight={700}
              >
                {label}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
