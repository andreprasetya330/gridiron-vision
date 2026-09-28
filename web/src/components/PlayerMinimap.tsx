import type { VisionDetection, VisionPlayer, Play } from "../types";

interface Props {
  play: Play;
  frame: number;
}

const DEFAULT_WIDTH = 1200;
const DEFAULT_HEIGHT = 533;
const ENDZONE = 100;

function visionAt(play: Play, frame: number): { boxes: VisionDetection[]; players: VisionPlayer[] } {
  const frames = play.vision_frames ?? [];
  const current = frames[frame] ?? frames[0];
  return {
    boxes: current?.boxes ?? [],
    players: current?.players ?? [],
  };
}

/**
 * Bird's-eye layout of the Roboflow player workflow: same 120-yard canvas
 * the backend paints, with offense / defense dots at projected feet.
 */
export function PlayerMinimap({ play, frame }: Props) {
  const width = play.minimap_width ?? DEFAULT_WIDTH;
  const height = play.minimap_height ?? DEFAULT_HEIGHT;
  const { players } = visionAt(play, frame);
  const yardLines: number[] = [];
  for (let x = ENDZONE; x <= width - ENDZONE; x += 50) {
    yardLines.push(x);
  }

  return (
    <svg
      className="player-minimap"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Roboflow player minimap"
    >
      <rect x={0} y={0} width={width} height={height} fill="#2a782a" rx={12} />
      <rect x={1} y={1} width={width - 2} height={height - 2} fill="none" stroke="#fff" strokeWidth={4} />
      <line x1={ENDZONE} y1={0} x2={ENDZONE} y2={height} stroke="#fff" strokeWidth={4} />
      <line
        x1={width - ENDZONE}
        y1={0}
        x2={width - ENDZONE}
        y2={height}
        stroke="#fff"
        strokeWidth={4}
      />
      {yardLines.map((x) => (
        <line
          key={x}
          x1={x}
          y1={0}
          x2={x}
          y2={height}
          stroke="#dcebd9"
          strokeWidth={(x - ENDZONE) % 100 === 0 ? 3 : 1}
        />
      ))}
      <text x={16} y={28} className="minimap-legend offense">
        OFFENSE
      </text>
      <text x={16} y={54} className="minimap-legend defense">
        DEFENSE
      </text>
      {players.map((player, index) => {
        const color =
          player.side === "offense"
            ? "#2850ff"
            : player.side === "defense"
              ? "#ff3c28"
              : "#f5f5f5";
        return (
          <g key={player.track_id || `${player.class_name}-${index}`}>
            <circle
              cx={player.minimap_x}
              cy={player.minimap_y}
              r={13}
              fill="#141414"
            />
            <circle cx={player.minimap_x} cy={player.minimap_y} r={11} fill={color} />
          </g>
        );
      })}
    </svg>
  );
}
