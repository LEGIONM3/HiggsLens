export interface ReportGalleryCategory {
  category: 'signal' | 'background' | 'interesting' | null;
  rank: number | null;
  selection_method: string | null;
}

export interface ReportEventSummary {
  event_id: number;
  features: Record<string, number>;
  source_split: string;
  gallery: ReportGalleryCategory | null;
}

export interface ReportClassification {
  model_id: 'xgboost';
  signal_probability: number;
  predicted_label: 'signal' | 'background';
  threshold: number;
}

export interface FeatureAttribution {
  feature: string;
  value: number;
  contribution: number;
}

export interface ObjectGroupAttribution {
  group: string;
  total_abs_contribution: number;
  signed_contribution: number;
}

export interface ReportExplanation {
  base_value: number;
  margin: number;
  attributions: FeatureAttribution[];
  object_groups: ObjectGroupAttribution[];
}

export interface ReproducibilityModelArtifact {
  model_id: string;
  feature_schema_version: string;
  device: string;
  training_run_origin: string;
  subsample_notes: string;
}

export interface ReproducibilityDataset {
  record: string;
  doi: string;
  content_hash: string;
}

export interface ReproducibilityInferenceContract {
  feature_count: number;
  sentinel_value: number;
  prediction_path: string;
  explanation_path: string;
}

export interface ReproducibilityManifest {
  report_contract_version: string;
  dataset: ReproducibilityDataset;
  inference_contract: ReproducibilityInferenceContract;
  certified_models: ReproducibilityModelArtifact[];
  frozen_leaderboard_status: string;
  server_capabilities: {
    supported_report_formats: string[];
    retraining_supported: boolean;
    public_holdout_access: boolean;
    champion_model_id: string;
  };
}

export interface EventAnalysisReport {
  report_version: string;
  generated_at: string;
  event: ReportEventSummary;
  classification: ReportClassification;
  explanation: ReportExplanation;
  reproducibility: ReproducibilityManifest;
  provenance: {
    statement: string;
  };
}
