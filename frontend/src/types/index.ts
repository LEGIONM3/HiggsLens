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

export interface ReconstructedObject {
  object_type: 'tau' | 'lepton' | 'jet_leading' | 'jet_subleading' | string;
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
    predicted_class: 'signal' | 'background';
    decision_threshold: number;
    distance_from_threshold: number;
    validation_status: string;
  };
  missing_adjusted_fields: string[];
}

export interface LabDatasetManifest {
  dataset_id: string;
  filename: string;
  row_count: number;
  column_count: number;
  feature_columns: string[];
  label_column: string;
  weight_column?: string | null;
  created_at: string;
  content_hash: string;
}

export interface LabExperimentSummary {
  experiment_id: string;
  dataset_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  model_ids: string[];
  created_at: string;
  completed_at?: string | null;
  error_message?: string | null;
}

export interface LabExperimentDetail {
  summary: LabExperimentSummary;
  dataset_manifest?: LabDatasetManifest | null;
  split_config: Record<string, number>;
  seed: number;
  per_model_results: Record<string, {
    test_metrics: ModelMetrics;
    validation_optimal_threshold: number;
    validation_ams?: number | null;
    is_weighted: boolean;
    subsample_size?: number | null;
  }>;
}
