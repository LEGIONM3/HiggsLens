export interface DatasetStatus {
  exists: boolean;
  filepath: string | null;
  file_size_bytes: number;
  record_id: number;
  doi: string;
  last_modified: string | null;
  validation_report: any | null;
}

export interface ModelInfo {
  id: string;
  display_name: string;
  status: 'available' | 'unavailable' | 'incompatible';
  required: boolean;
  supports_missing: boolean;
  preprocessing_pipeline: string;
  hyperparameters_schema: Record<string, any>;
}

export interface ModelMetrics {
  model_id: string;
  feature_set: string;
  mode: string;
  seeds_evaluated: number[];
  roc_auc_mean: number;
  roc_auc_std: number;
  pr_auc_mean: number;
  pr_auc_std: number;
  log_loss_mean: number;
  log_loss_std: number;
  balanced_accuracy_mean: number;
  balanced_accuracy_std: number;
  f1_mean: number;
  f1_std: number;
  brier_score_mean: number;
  brier_score_std: number;
  optimal_threshold: number;
  ams_score: number;
  ams_default_threshold_score: number;
  training_duration_seconds: number;
  model_card_path?: string;
  stability_status: string;
  calibration_status: string;
}

export interface RecommendationExplanation {
  recommended_model_id: string;
  validation_objective_used: string;
  feature_set_used: string;
  mean_roc_auc: number;
  variance_across_seeds: number;
  calibration_quality: string;
  training_duration_seconds: number;
  selected_threshold: number;
  weighted_ams: number;
  warnings: string[];
  reasoning: string;
}

export interface ExperimentRun {
  run_id: string;
  timestamp: string;
  git_commit: string | null;
  dataset_fingerprint: string;
  split_strategy: string;
  feature_set: string;
  mode: string;
  seeds: number[];
  metrics_by_model: Record<string, ModelMetrics>;
  recommendation?: RecommendationExplanation;
}

export interface JobStatus {
  job_id: string;
  state: 'queued' | 'validating_data' | 'preprocessing' | 'training' | 'evaluating' | 'saving_artifacts' | 'completed' | 'failed';
  current_model: string | null;
  completed_models: number;
  total_models: number;
  progress_message: string;
  started_timestamp: string;
  updated_timestamp: string;
  error_details: string | null;
  run_id: string | null;
}

export interface ReconstructedObject {
  object_type: string;
  label: string;
  pt: number;
  eta: number;
  phi: number;
  present: boolean;
}

export interface PredictionResponse {
  event_id: number | null;
  objects: ReconstructedObject[];
  missing_transverse_energy: {
    magnitude: number;
    phi: number;
    sumet: number | null;
  };
  jet_summary: {
    count: number;
    total_pt: number;
  };
  prediction: {
    model_id: string;
    model_version: string;
    feature_set: string;
    signal_probability: number;
    background_probability: number;
    predicted_class: string;
    decision_threshold: number;
    distance_from_threshold: number;
    validation_status: string;
  };
  missing_adjusted_fields: string[];
}
