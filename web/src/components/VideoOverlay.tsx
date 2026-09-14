import { useEffect, useRef } from "react";
import type { Play } from "../types";

interface Props {
  play: Play;
  frame: number;
  showRoles: boolean;
}

const ROLE_COLORS: Record<string, string> = {
  man: "#f97316",
  deep_zone: "#38bdf8",
  underneath_zone: "#a78bfa",
  blitz: "#ef4444",
};

function invert3(m: number[][]): number[][] | null {
  const a = m[0][0],
    b = m[0][1],
    c = m[0][2],
    d = m[1][0],
    e = m[1][1],
    f = m[1][2],
    g = m[2][0],
    h = m[2][1],
    i = m[2][2];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-12) return null;
  const invDet = 1 / det;
  return [
    [(e * i - f * h) * invDet, (c * h - b * i) * invDet, (b * f - c * e) * invDet],
    [(f * g - d * i) * invDet, (a * i - c * g) * invDet, (c * d - a * f) * invDet],
    [(d * h - e * g) * invDet, (b * g - a * h) * invDet, (a * e - b * d) * invDet],
  ];
}

function project(H: number[][], x: number, y: number): [number, number] {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w];
}

function fieldToImage(play: Play, xNorm: number, yNorm: number): [number, number] | null {
  if (!play.homography || play.origin_x == null || play.origin_y == null) return null;
  const toImage = invert3(play.homography);
  if (!toImage) return null;
  const sign = play.play_direction === "right" ? 1 : -1;
  const fx = play.origin_x + sign * xNorm;
  const fy = play.origin_y + sign * yNorm;
  return project(toImage, fx, fy);
}

/**
 * Video with a canvas overlay locked to the scrubber.
 *
 * When the play carries a homography, boxes are projected through it so they
 * land on the players rather than near them. Without one - synthetic tracks,
 * or any import that skipped vision - this still draws, but in a fallback
 * affine that is only good enough to review, not to coach from.
 */
export function VideoOverlay({ play, frame, showRoles }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const fps = play.video_fps ?? 30;
  const snapFrame = play.snap_frame_in_video ?? 0;
  const timeOffset = play.time_grid[frame] ?? 0;
  const videoTime = Math.max(0, snapFrame / fps + timeOffset);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (Math.abs(video.currentTime - videoTime) > 0.02) {
      video.currentTime = videoTime;
    }
  }, [videoTime]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = video.videoWidth || play.video_width || video.clientWidth || 960;
    const height = video.videoHeight || play.video_height || video.clientHeight || 540;
    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    ctx.font = "600 13px ui-sans-serif, system-ui";
    ctx.textAlign = "center";

    for (const player of play.players) {
      const x = player.x[frame];
      const y = player.y[frame];
      if (x === null || y === null) continue;

      const projected = fieldToImage(play, x, y);
      const sx = projected ? projected[0] : width * (0.5 + y / 60);
      const sy = projected ? projected[1] : height * (0.85 - x / 60);

      const role = play.prediction?.roles[player.track_id]?.role ?? player.role;
      const color =
        player.side === "offense"
          ? "#e2e8f0"
          : (showRoles && role && ROLE_COLORS[role]) || "#f87171";

      const boxH = projected ? Math.max(28, height * 0.05) : 34;
      const boxW = boxH * 0.7;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(sx - boxW / 2, sy - boxH, boxW, boxH);

      if (player.side === "defense" && showRoles && role) {
        ctx.fillStyle = color;
        ctx.fillText(role.replaceAll("_", " "), sx, sy - boxH - 6);
      }
    }
  }, [play, frame, showRoles]);

  if (!play.video_path) {
    return null;
  }

  return (
    <div className="video-wrap">
      <video
        ref={videoRef}
        src={`/api/plays/${encodeURIComponent(play.play_id)}/video`}
        preload="auto"
        muted
        playsInline
      />
      <canvas ref={canvasRef} className="video-canvas" />
    </div>
  );
}
