import { LabDatasetManifest, LabExperimentDetail, LabExperimentSummary } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api/v1';

export async function uploadLabDataset(
  file: File,
  featureColumns: string[],
  labelColumn: string,
  weightColumn?: string
): Promise<LabDatasetManifest> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('feature_columns', JSON.stringify(featureColumns));
  formData.append('label_column', labelColumn);
  if (weightColumn) {
    formData.append('weight_column', weightColumn);
  }

  const response = await fetch(`${API_BASE}/lab/datasets`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(errData.detail || 'Failed to upload dataset.');
  }

  return response.json();
}

export async function fetchLabDatasets(): Promise<LabDatasetManifest[]> {
  const response = await fetch(`${API_BASE}/lab/datasets`);
  if (!response.ok) {
    throw new Error(`Failed to fetch lab datasets: ${response.statusText}`);
  }
  const data = await response.json();
  return data.datasets || [];
}

export async function createLabExperiment(
  datasetId: string,
  modelIds: string[],
  splitConfig: Record<string, number> = { train: 0.7, validation: 0.15, test: 0.15 }
): Promise<LabExperimentSummary> {
  const response = await fetch(`${API_BASE}/lab/experiments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_id: datasetId,
      model_ids: modelIds,
      split_config: splitConfig,
      seed: 42,
      sentinel_strategy: 'keep-as-value',
    }),
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(errData.detail || 'Failed to create experiment.');
  }

  return response.json();
}

export async function fetchLabExperiments(): Promise<LabExperimentSummary[]> {
  const response = await fetch(`${API_BASE}/lab/experiments`);
  if (!response.ok) {
    throw new Error(`Failed to fetch lab experiments: ${response.statusText}`);
  }
  const data = await response.json();
  return data.experiments || [];
}

export async function fetchLabExperimentDetail(experimentId: string): Promise<LabExperimentDetail> {
  const response = await fetch(`${API_BASE}/lab/experiments/${experimentId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch lab experiment detail: ${response.statusText}`);
  }
  return response.json();
}
