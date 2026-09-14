export type Side = "offense" | "defense";

export interface Situation {
  down: number | null;
  distance: number | null;
  yardline: number | null;
  quarter: number | null;
  score_margin: number | null;
  offense_personnel: string | null;
  ball_y_from_center: number;
  hash_side: string | null;
  league: string;
}

export interface Quality {
  defenders_detected: number;
  offense_detected: number;
  mean_detection_conf: number;
  registration_error_yd: number;
  frames_with_full_defense: number;
  notes: string[];
  score: number;
  usable: boolean;
}

export interface PlayerTrack {
  track_id: string;
  side: Side;
  jersey: number | null;
  position: string | null;
  role: string | null;
  is_ball_carrier: boolean;
  /** null marks a frame where the player was never observed, not a zero. */
  x: (number | null)[];
  y: (number | null)[];
}

export interface RolePrediction {
  role: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface Prediction {
  play_id: string;
  coverage: string;
  confidence: number;
  probabilities: Record<string, number>;
  runner_up: string;
  roles: Record<string, RolePrediction>;
  quality_score: number;
  usable: boolean;
}

export interface Play {
  play_id: string;
  source: string;
  season: number | null;
  week: number | null;
  defense_team: string | null;
  offense_team: string | null;
  coverage: string | null;
  coverage_source: string | null;
  video_path: string | null;
  snap_frame_in_video: number | null;
  video_fps: number | null;
  homography: number[][] | null;
  origin_x: number | null;
  origin_y: number | null;
  play_direction: string;
  video_width: number | null;
  video_height: number | null;
  situation: Situation;
  quality: Quality;
  time_grid: number[];
  players: PlayerTrack[];
  prediction: Prediction | null;
}

export interface PlaySummary {
  play_id: string;
  season: number | null;
  week: number | null;
  defense_team: string | null;
  offense_team: string | null;
  coverage_truth: string | null;
  coverage_predicted: string | null;
  confidence: number | null;
  usable: boolean;
  source: string;
  has_video: boolean;
  situation: Partial<Situation>;
}

export interface Health {
  status: string;
  plays: number;
  predictions: number;
  coverages: string[];
  roles: string[];
  time_grid: number[];
}
