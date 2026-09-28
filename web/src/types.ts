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

export interface Disguise {
  showed: string | null;
  ran: string;
  showed_shell: string | null;
  ran_shell: string | null;
  kind: string;
  disguised: boolean;
  family_mismatch: boolean;
  shell_mismatch: boolean;
  showed_confidence: number | null;
  ran_confidence: number | null;
}

export interface LookPrediction {
  coverage: string;
  confidence: number;
  probabilities: Record<string, number>;
  runner_up?: string;
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
  presnap?: LookPrediction | null;
  disguise?: Disguise | null;
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
  coverage_family?: string | null;
  coverage_shell?: string | null;
  minimap_url?: string | null;
  media_kind?: "video" | "image" | null;
  vision_model?: string | null;
  vision_frames?: VisionFrame[] | null;
  minimap_width?: number | null;
  minimap_height?: number | null;
}

export interface VisionDetection {
  class_name: string;
  side: Side | null;
  confidence: number;
  box: [number, number, number, number] | null;
}

export interface VisionPlayer {
  track_id?: string | null;
  side: Side | null;
  class_name: string;
  minimap_x: number;
  minimap_y: number;
}

export interface VisionFrame {
  boxes: VisionDetection[];
  players: VisionPlayer[];
}

export interface PlaySummary {
  play_id: string;
  season: number | null;
  week: number | null;
  defense_team: string | null;
  offense_team: string | null;
  coverage_truth: string | null;
  coverage_predicted: string | null;
  coverage_presnap?: string | null;
  confidence: number | null;
  usable: boolean;
  source: string;
  has_video: boolean;
  disguised?: boolean;
  disguise_kind?: string | null;
  coverage_family?: string | null;
  coverage_shell?: string | null;
  situation: Partial<Situation>;
}

export interface Health {
  status: string;
  plays: number;
  predictions: number;
  corpus?: string;
  sources?: Record<string, number>;
  coverages: string[];
  roles: string[];
  time_grid: number[];
}
