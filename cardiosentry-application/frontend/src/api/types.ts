/**
 * TypeScript types mirroring the CardioSentry backend API contract.
 * The frontend holds NO model knowledge: labels, display names, thresholds and
 * input size all come from GET /api/config (or the response payloads).
 */

/** Normalized [0,1] point. The API represents quads as [[x,y] x4] (TL,TR,BR,BL). */
export type Point = [number, number];

// ---------------------------------------------------------------- health
export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  active_model: string | null;
  device: string;
  torch_version: string;
  uptime_s: number;
  free_space_gb: number;
  free_space_warn: boolean;
}

/** Model input geometry as [height, width] — torch order, the same order the
 *  descriptor's `preprocess.input_size` uses. Square models report [768, 768]. */
export type ModelInputSize = [number, number];

/** What capture.json and the rectify response record about the input chain. */
export interface PreprocessRecord {
  input_size: ModelInputSize;
  tensor_shape: number[];
  tensor_dtype: string;
  tensor_sha256: string;
  preprocess_ms: number;
  [key: string]: unknown;
}

// ----------------------------------------------------------------- config
export interface ActiveModelConfig {
  id: string;
  display_name: string;
  version: string;
  labels: string[];
  display_names: Record<string, string>;
  suppressed_labels: string[];
  input_size: ModelInputSize;
  views: ModelInputSize[];
  checkpoint_sha256_short: string;
  thresholds: Record<string, number>;
  threshold_source: string;
}

export interface AppConfig {
  app: { version: string; git_sha: string | null };
  active_model: ActiveModelConfig;
  quality: {
    block_on_poor: boolean;
    warn_blur_varlap: number;
    warn_brightness_lo: number;
    warn_brightness_hi: number;
    warn_contrast_p5p95: number;
    warn_px_per_mm: number;
    warn_area_fraction: number;
    warn_skew_deg: number;
  };
  detection: { confidence_warn: number; expected_aspect: number | null };
  rectify: {
    canvas_px: [number, number];
    margin_px: number;
    geometry_modes: string[];
    default_geometry_mode: string;
  };
  capture: { max_upload_bytes: number; blind_mode_available: boolean };
  explain: ExplainConfig;
  notice: string;
}

// ----------------------------------------------------------------- explain
export type ExplainMethod = "gradcam" | "integrated_gradients" | "jacobian";
export type IgBaseline = "white" | "black" | "grey" | "blur";

/** Whole model passes one run will cost, from GET /api/config. The UI quotes
 *  this BEFORE starting a run, because Integrated Gradients is minutes. */
export interface ExplainCost {
  forward: number;
  backward_full: number;
  backward_partial: number;
  note?: string;
}

export interface ExplainConfig {
  enabled: boolean;
  methods: ExplainMethod[];
  default_ig_steps: number;
  max_ig_steps: number;
  default_ig_baseline: IgBaseline;
  ig_baselines: IgBaseline[];
  default_colormap: string;
  default_overlay_alpha: number;
  /** The printed ID/Age/Sex block as [x0, y0, x1, y1] fractions. */
  header_box: [number, number, number, number] | null;
  header_redacted_in_training: boolean;
  cost: Record<ExplainMethod, ExplainCost>;
}

/** Attribution mass inside the header box. `lift` is that fraction over the
 *  box's share of the sheet AREA: 1.0 is its fair share, 4.0 is a shortcut. */
export interface HeaderCheck {
  box: [number, number, number, number];
  box_area_frac: number;
  frac: number;
  lift: number | null;
  peak_inside: boolean;
}

export interface MapSummary {
  empty: boolean;
  centroid_xy: [number, number] | null;
  peak_xy: [number, number] | null;
  peak_value: number | null;
  /** Smallest fraction of the sheet holding half the attribution mass. */
  mass_half_area_frac: number | null;
  positive_frac: number | null;
}

export interface LabelAttribution {
  label: string;
  display_name: string;
  /** Integrated Gradients is signed (red up / blue down); the others are not. */
  signed: boolean;
  score: number | null;
  threshold: number | null;
  positive: boolean | null;
  stats: Record<string, number | boolean | null>;
  summary: MapSummary;
  header: HeaderCheck | null;
  normalization: Record<string, unknown>;
  per_view: Record<string, number[]>;
  files: { heatmap?: string; overlay?: string };
}

export interface ExplainResponse {
  explain_id: string;
  capture_id: string;
  method: ExplainMethod;
  model: ModelRunInfo;
  run_id: string | null;
  labels: string[];
  attributions: LabelAttribution[];
  meta: Record<string, unknown>;
  /** Jacobian only: `coupling[a][b]` = cosine between the two gradient rows. */
  extras: { coupling?: Record<string, Record<string, number>>; [k: string]: unknown };
  timing_ms: Record<string, number>;
  created_utc: string;
  model_input_sha256: string;
  caveats: string[];
}

export interface ExplainSummary {
  explain_id: string;
  method: ExplainMethod;
  model_id: string;
  created_utc: string;
  labels: string[];
  run_id: string | null;
  /** The capture was re-rectified: this map explains pixels it no longer has. */
  stale: boolean;
  max_header_frac: number | null;
  thumbnail_url: string | null;
}

export interface ExplainListResponse {
  capture_id: string;
  explanations: ExplainSummary[];
  total: number;
}

export interface ExplainRequest {
  method: ExplainMethod;
  model_id?: string | null;
  labels?: string[] | null;
  target_layer?: string | null;
  steps?: number | null;
  baseline?: IgBaseline | null;
  colormap?: string | null;
  overlay_alpha?: number | null;
  run_id?: string | null;
}

// ----------------------------------------------------------------- models
export interface ModelListItem {
  id: string;
  display_name: string;
  version: string;
  notes: string;
  labels: string[];
  model_labels: string[];
  suppressed_labels: string[];
  views: ModelInputSize[];
  call_style: string;
  input_size: ModelInputSize;
  checkpoint_sha256_short: string;
  weights: string;
  active: boolean;
  loaded: boolean;
}

export interface ModelsResponse {
  models: ModelListItem[];
  active_id: string;
}

// ----------------------------------------------------------------- quality
export interface QualityMetrics {
  level: "ok" | "warn" | "poor";
  blur_varlap: number | null;
  brightness: number | null;
  contrast_p5p95: number | null;
  clipped_high_frac: number | null;
  clipped_low_frac: number | null;
  glare_blob_frac: number | null;
  est_px_per_mm: number | null;
  area_fraction: number | null;
  skew_deg: number | null;
  grid_period_px: number | null;
  grid_detected: boolean | null;
  flags: string[];
}

// ---------------------------------------------------------------- detection
export interface DetectionInfo {
  attempted?: boolean;
  success: boolean;
  method: string;
  confidence: number;
  quad_auto_px: Point[] | null;
  quad_norm: Point[] | null;
  area_fraction: number | null;
  aspect_ratio: number | null;
  params: Record<string, unknown>;
  score_breakdown: Record<string, unknown>;
}

// ----------------------------------------------------------------- original
export interface OriginalInfo {
  width: number;
  height: number;
  oriented_width: number;
  oriented_height: number;
  sha256: string;
  uploaded_sha256?: string;
  bytes: number;
  format: string;
  exif_orientation?: number;
  orientation_applied?: boolean;
  exif_gps_present?: boolean;
  gps_stripped?: boolean;
  [key: string]: unknown;
}

// --------------------------------------------------------------- upload 201
export interface CaptureUploadResponse {
  capture_id: string;
  original: OriginalInfo;
  detection: DetectionInfo;
  quality: QualityMetrics;
  files: { preview: string; overlay: string };
}

// ----------------------------------------------------------------- rectify
export interface RectifyResponse {
  capture_id: string;
  corrected_px: [number, number];
  geometry_mode: string;
  mean_corner_shift_px: number | null;
  quality: QualityMetrics;
  preprocess: PreprocessRecord;
  files: { corrected: string; model_input: string };
}

// ---------------------------------------------------------------- prediction
export interface ModelRunInfo {
  id: string;
  display_name: string;
  version: string;
  checkpoint_sha256: string;
  checkpoint_sha256_short: string;
  protocol_fingerprint: string | null;
  epoch: number | null;
  views: ModelInputSize[];
  input_size: ModelInputSize;
  device: string;
  dtype: string;
  amp: boolean;
  torch_num_threads: number | null;
}

export interface PredictionResponse {
  run_id: string;
  capture_id: string;
  model: ModelRunInfo;
  scores: Record<string, number>;
  logits: Record<string, number>;
  thresholds: Record<string, number>;
  threshold_source: string;
  positive: string[];
  timing_ms: { preprocess: number; forward: number; total: number };
  created_utc: string;
  model_input_sha256: string;
}

export interface PredictionRecord extends PredictionResponse {
  schema_version: number;
  descriptor_snapshot?: {
    labels?: string[];
    display_names?: Record<string, string>;
    [key: string]: unknown;
  };
  env: Record<string, unknown>;
}

// ---------------------------------------------------------------- annotation
export interface AnnotationRecord {
  reference_labels: Record<string, boolean> | null;
  labels_entered_utc: string | null;
  labels_entered_before_prediction: boolean | null;
  truth_source: string | null;
  notes: string;
  device_vendor: string;
  paper_size: string;
}

export interface AnnotationRequest {
  reference_labels?: Record<string, boolean> | null;
  labels_entered_before_prediction?: boolean | null;
  truth_source?: string | null;
  notes?: string | null;
  sheet_id?: string | null;
  device_vendor?: string | null;
  paper_size?: string | null;
}

// ----------------------------------------------------------------- correction
export interface CorrectionInfo {
  quad_final_px: Point[] | null;
  quad_auto_px: Point[] | null;
  corners_manually_adjusted: boolean;
  mean_corner_shift_px: number | null;
  source: string;
  geometry_mode: string;
  homography: number[][] | null;
  warp_interpolation: string;
  canvas_px: number[] | null;
  margin_px: number | null;
  paper_fill_rgb: number[] | null;
  corrected_px: number[] | null;
  est_px_per_mm_original: number | null;
  scale_factor_to_input?: number | null;
}

// ------------------------------------------------------------- capture record
export interface CaptureRecord {
  schema_version: number;
  capture_id: string;
  created_utc: string;
  app: Record<string, unknown>;
  session: {
    sheet_id?: string | null;
    retake_of?: string | null;
    blind_mode?: boolean;
    notes?: string;
    [key: string]: unknown;
  };
  source: {
    mode?: string;
    declared_filename?: string;
    client_ua?: string;
    [key: string]: unknown;
  };
  original: OriginalInfo;
  quality_original: QualityMetrics | null;
  detection: DetectionInfo | null;
  correction: CorrectionInfo | null;
  preprocess: PreprocessRecord | null;
  quality_model_input: QualityMetrics | null;
  annotation: AnnotationRecord;
  predictions: string[];
  explanations?: string[];
  prediction_runs?: PredictionRecord[];
}

// ------------------------------------------------------------------ history
export interface CaptureSummary {
  capture_id: string;
  created_utc: string;
  sheet_id: string | null;
  thumbnail_url: string | null;
  has_correction: boolean;
  latest_model_id: string | null;
  latest_positive: string[];
  quality_level: string;
}

export interface CaptureListResponse {
  captures: CaptureSummary[];
  page: number;
  total: number;
  has_more: boolean;
}

// ---------------------------------------------------------------- requests
export interface RectifyRequest {
  quad_norm: Point[];
  source: "auto" | "manual";
  geometry_mode: string;
}

export interface PredictRequest {
  model_id?: string | null;
  thresholds?: Record<string, number> | null;
}

export interface DeleteResponse {
  capture_id: string;
  trashed: string;
}
