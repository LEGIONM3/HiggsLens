import { DatasetStatus, ModelInfo, JobStatus, ExperimentRun, PredictionResponse } from '../types';

const API_BASE = 'http://localhost:8000/api/v1';

export async function fetchDatasetStatus(): Promise<DatasetStatus> {
  const res = await fetch(`${API_BASE}/dataset/status`);
  if (!res.ok) throw new Error('Failed to fetch dataset status');
  return res.json();
}

export async function triggerDatasetDownload(force = false): Promise<{ success: boolean; message: string }> {
  const res = await fetch(`${API_BASE}/dataset/download`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ force, dry_run: false }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Failed to trigger download');
  return data;
}

export async function triggerDatasetValidate(): Promise<any> {
  const res = await fetch(`${API_BASE}/dataset/validate`, { method: 'POST' });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Validation failed');
  return data;
}

export async function fetchModelRegistry(): Promise<Record<string, ModelInfo>> {
  const res = await fetch(`${API_BASE}/models/registry`);
  if (!res.ok) throw new Error('Failed to fetch model registry');
  const data = await res.json();
  return data.models;
}

export async function startTrainingJob(mode: 'fast' | 'research', featureSet: string, models?: string[]): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/experiments/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, feature_set: featureSet, models }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to start training');
  }
  return res.json();
}

export async function fetchJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/experiments/jobs/${jobId}`);
  if (!res.ok) throw new Error('Failed to fetch job status');
  return res.json();
}

export async function fetchExperimentRuns(): Promise<ExperimentRun[]> {
  const res = await fetch(`${API_BASE}/experiments`);
  if (!res.ok) throw new Error('Failed to fetch experiment runs');
  return res.json();
}

export async function runLivePrediction(modelId: string, eventId?: number, features?: Record<string, number>): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_id: modelId,
      feature_set: 'all_physics',
      event_id: eventId,
      features: features,
    }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to run prediction');
  }
  return res.json();
}
