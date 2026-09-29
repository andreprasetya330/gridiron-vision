import type { Play, VisionFrame } from "./types";

export function visionTime(play: Play, index: number, frame: VisionFrame): number {
  if (typeof frame.t === "number") return frame.t;
  const fps = play.video_fps || 10;
  if (typeof frame.video_frame === "number") return frame.video_frame / fps;
  const snap = (play.snap_frame_in_video ?? 0) / fps;
  const grid = play.time_grid[index];
  if (typeof grid === "number") return Math.max(0, snap + grid);
  return index / fps;
}

export function visionIndexAt(play: Play, videoTime: number): number {
  const frames = play.vision_frames ?? [];
  if (!frames.length) return 0;
  let best = 0;
  let bestDist = Number.POSITIVE_INFINITY;
  for (let i = 0; i < frames.length; i += 1) {
    const dist = Math.abs(visionTime(play, i, frames[i]) - videoTime);
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  }
  return best;
}

export function visionSampleAt(play: Play, videoTime: number): VisionFrame | undefined {
  const frames = play.vision_frames ?? [];
  if (!frames.length) return undefined;
  return frames[visionIndexAt(play, videoTime)];
}
