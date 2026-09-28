import { useEffect, useRef, useState } from "react";
import { OFFENSE_COLOR, playerColor, playerLabel } from "../playerColors";
import type { Play, VisionDetection } from "../types";

interface Props {
  play: Play;
  frame: number;
}

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

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function visionBoxes(play: Play, frame: number): VisionDetection[] {
  const frames = play.vision_frames ?? [];
  return frames[frame]?.boxes ?? frames[0]?.boxes ?? [];
}

function detectionColor(det: VisionDetection, index: number): string {
  if (det.side === "offense" || det.class_name.includes("offense")) return OFFENSE_COLOR;
  if (!det.side && det.class_name.includes("official")) return "#e2e8f0";
  return playerColor(`rf-${det.class_name}-${index}`, "defense", null);
}

/**
 * Boxes on real film from the player-tracking workflow. Falls back to
 * projecting field tracks through the homography when those boxes are absent.
 */
export function VideoOverlay({ play, frame }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [ready, setReady] = useState(false);
  const isStill =
    play.media_kind === "image" ||
    /\.(jpg|jpeg|png|webp|bmp)$/i.test(play.video_path ?? "");

  const fps = play.video_fps ?? 30;
  const snapFrame = play.snap_frame_in_video ?? 0;
  const timeOffset = play.time_grid[frame] ?? 0;
  const videoTime = Math.max(0, snapFrame / fps + timeOffset);
  const mediaSrc = `/api/plays/${encodeURIComponent(play.play_id)}/video`;

  useEffect(() => {
    setReady(false);
  }, [play.play_id]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || isStill) return;
    if (Math.abs(video.currentTime - videoTime) > 0.02) {
      video.currentTime = videoTime;
    }
  }, [videoTime, isStill]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const image = imageRef.current;
    if (!canvas || !ready) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width =
      (isStill ? image?.naturalWidth : video?.videoWidth) || play.video_width || 1280;
    const height =
      (isStill ? image?.naturalHeight : video?.videoHeight) || play.video_height || 720;
    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    const srcW = play.video_width || width;
    const srcH = play.video_height || height;
    const scaleX = width / srcW;
    const scaleY = height / srcH;
    const boxes = visionBoxes(play, frame);

    if (boxes.length) {
      for (let i = 0; i < boxes.length; i += 1) {
        const det = boxes[i];
        if (!det.box) continue;
        const [x1, y1, x2, y2] = det.box;
        const left = x1 * scaleX;
        const top = y1 * scaleY;
        const boxW = Math.max(8, (x2 - x1) * scaleX);
        const boxH = Math.max(12, (y2 - y1) * scaleY);
        const color = detectionColor(det, i);
        const label =
          det.side === "offense"
            ? "O"
            : det.side === "defense"
              ? "D"
              : det.class_name === "official"
                ? "REF"
                : "";

        ctx.fillStyle = color + "40";
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(2, boxH * 0.05);
        ctx.strokeRect(left, top, boxW, boxH);
        ctx.fillRect(left, top, boxW, boxH);
        if (label) {
          ctx.font = `700 ${Math.max(11, boxH * 0.22)}px "DM Sans", system-ui`;
          ctx.textAlign = "center";
          ctx.lineWidth = 3;
          ctx.strokeStyle = "rgba(15, 23, 42, 0.7)";
          ctx.strokeText(label, left + boxW / 2, Math.max(14, top - 6));
          ctx.fillStyle = "#ffffff";
          ctx.fillText(label, left + boxW / 2, Math.max(14, top - 6));
        }
      }
    } else {
      const ordered = [...play.players].sort((a, b) => {
        const ax = a.x[frame] ?? -99;
        const bx = b.x[frame] ?? -99;
        return ax - bx;
      });

      for (const player of ordered) {
        const x = player.x[frame];
        const y = player.y[frame];
        if (x === null || y === null) continue;

        const projected = fieldToImage(play, x, y);
        const sx = projected ? projected[0] : width * (0.5 + y / 60);
        const sy = projected ? projected[1] : height * (0.85 - x / 60);
        const boxH = projected ? Math.max(36, height * 0.07) : 40;
        const boxW = boxH * 0.48;
        const color = playerColor(player.track_id, player.side, player.role);
        const label = playerLabel(player.jersey, player.side, player.role);

        ctx.fillStyle = color + "55";
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(2, boxH * 0.06);
        ctx.strokeRect(sx - boxW / 2, sy - boxH, boxW, boxH);
        ctx.fillRect(sx - boxW / 2, sy - boxH, boxW, boxH);
        if (label) {
          ctx.font = `700 ${Math.max(12, boxH * 0.2)}px "DM Sans", system-ui`;
          ctx.textAlign = "center";
          ctx.lineWidth = 3;
          ctx.strokeStyle = "rgba(15, 23, 42, 0.65)";
          ctx.strokeText(label, sx, sy - boxH - 6);
          ctx.fillStyle = "#ffffff";
          ctx.fillText(label, sx, sy - boxH - 6);
        }
      }
    }

    const pred = play.prediction;
    const title = pred?.coverage ?? play.coverage ?? "Unscored";
    const conf = pred ? `${Math.round(pred.confidence * 100)}%` : "";
    const look = pred?.presnap?.coverage ?? pred?.disguise?.showed;
    ctx.font = '700 28px "DM Sans", system-ui';
    const text = conf ? `${title}  ${conf}` : title;
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = "rgba(15, 23, 42, 0.78)";
    roundRect(ctx, 20, 20, Math.min(width - 32, tw + 48), look ? 76 : 52, 14);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.textAlign = "left";
    ctx.fillText(text, 36, 54);
    if (look) {
      ctx.font = '500 16px "DM Sans", system-ui';
      ctx.fillStyle = "rgba(255,255,255,0.8)";
      ctx.fillText(`Showed ${look}`, 36, 78);
    }
  }, [play, frame, ready, isStill]);

  if (!play.video_path) return null;

  return (
    <div className="film-stage">
      {isStill ? (
        <img
          key={play.play_id}
          ref={imageRef}
          src={mediaSrc}
          alt="Uploaded film still"
          onLoad={() => setReady(true)}
        />
      ) : (
        <video
          key={play.play_id}
          ref={videoRef}
          src={mediaSrc}
          preload="auto"
          muted
          playsInline
          onLoadedMetadata={() => setReady(true)}
        />
      )}
      <canvas ref={canvasRef} className="film-canvas" />
      <div className="film-badge">
        {play.vision_frames?.length ? "Player tracking" : "Vision overlay"}
      </div>
    </div>
  );
}
