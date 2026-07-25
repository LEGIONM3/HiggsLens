import React, { useState, useEffect } from 'react';
import { LabDatasetManifest, LabExperimentDetail, LabExperimentSummary, ModelInfo } from '../types';
import { createLabExperiment, fetchLabDatasets, fetchLabExperimentDetail, fetchLabExperiments, uploadLabDataset } from '../services/labApi';
import { AlertTriangle, Upload, Play, RefreshCw, CheckCircle2, XCircle, Clock, Database, FlaskConical, ShieldAlert, Cpu } from 'lucide-react';

interface LabComponentProps {
  models: Record<string, ModelInfo>;
}

export const LabComponent: React.FC<LabComponentProps> = ({ models }) => {
  // Datasets state
  const [datasets, setDatasets] = useState<LabDatasetManifest[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  
  // Upload form state
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [featureColsInput, setFeatureColsInput] = useState<string>('PRI_tau_pt, PRI_tau_eta, PRI_lep_pt, PRI_met');
  const [labelColInput, setLabelColInput] = useState<string>('Label');
  const [weightColInput, setWeightColInput] = useState<string>('Weight');
  const [isUploading, setIsUploading] = useState<boolean>(false);

  // Model selection state (default random_forest & logistic_regression)
  const [selectedModels, setSelectedModels] = useState<string[]>(['random_forest', 'logistic_regression']);

  // Experiments state
  const [experiments, setExperiments] = useState<LabExperimentSummary[]>([]);
  const [activeExperimentId, setActiveExperimentId] = useState<string | null>(null);
  const [experimentDetail, setExperimentDetail] = useState<LabExperimentDetail | null>(null);
  const [isSubmittingJob, setIsSubmittingJob] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Available model candidates list
  const modelCandidates = [
    { id: 'dummy_prior', name: 'Dummy Prior Baseline', family: 'baseline', requires: null },
    { id: 'logistic_regression', name: 'Logistic Regression', family: 'linear', requires: null },
    { id: 'random_forest', name: 'Random Forest', family: 'tree_ensemble', requires: null },
    { id: 'histogram_gradient_boosting', name: 'Hist. Gradient Boosting', family: 'boosting', requires: null },
    { id: 'mlp', name: 'Multi-Layer Perceptron (MLP)', family: 'neural_network', requires: null },
    { id: 'xgboost', name: 'XGBoost Classifier', family: 'boosting', requires: 'xgboost' },
    { id: 'lightgbm', name: 'LightGBM Classifier', family: 'boosting', requires: 'lightgbm' },
    { id: 'svm_rbf', name: 'SVM (RBF Kernel)', family: 'support_vector', requires: null },
    { id: 'calibrated_ensemble', name: 'Calibrated Voting Ensemble', family: 'ensemble', requires: null },
    { id: 'quantum_kernel_svm', name: 'Quantum Kernel SVM (Stub)', family: 'quantum_ml', requires: 'qiskit_machine_learning' },
    { id: 'variational_quantum_classifier', name: 'Variational Quantum Classifier (Stub)', family: 'quantum_ml', requires: 'pennylane' },
  ];

  const loadLabData = async () => {
    try {
      const [dsList, expList] = await Promise.all([
        fetchLabDatasets().catch(() => []),
        fetchLabExperiments().catch(() => []),
      ]);
      setDatasets(dsList);
      if (dsList.length > 0 && !selectedDatasetId) {
        setSelectedDatasetId(dsList[0].dataset_id);
      }
      setExperiments(expList);
      if (expList.length > 0 && !activeExperimentId) {
        setActiveExperimentId(expList[0].experiment_id);
      }
    } catch (err) {
      console.error('Error loading lab data:', err);
    }
  };

  useEffect(() => {
    loadLabData();
  }, []);

  // Poll experiment detail when active job is queued/running
  useEffect(() => {
    if (!activeExperimentId) return;

    let isSubscribed = true;
    const fetchDetail = async () => {
      try {
        const detail = await fetchLabExperimentDetail(activeExperimentId);
        if (isSubscribed) {
          setExperimentDetail(detail);
        }
      } catch (err) {
        console.error('Error polling experiment detail:', err);
      }
    };

    fetchDetail();
    const interval = setInterval(() => {
      if (experimentDetail?.summary.status === 'queued' || experimentDetail?.summary.status === 'running') {
        fetchDetail();
      }
    }, 3000);

    return () => {
      isSubscribed = false;
      clearInterval(interval);
    };
  }, [activeExperimentId, experimentDetail?.summary.status]);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) {
      setErrorMsg('Please select a CSV file to upload.');
      return;
    }

    setIsUploading(true);
    setErrorMsg(null);
    try {
      const cols = featureColsInput.split(',').map((c) => c.trim()).filter(Boolean);
      const manifest = await uploadLabDataset(uploadFile, cols, labelColInput.trim(), weightColInput.trim() || undefined);
      setDatasets((prev) => [manifest, ...prev]);
      setSelectedDatasetId(manifest.dataset_id);
      setUploadFile(null);
    } catch (err: any) {
      setErrorMsg(err.message || 'Upload failed.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleToggleModel = (mId: string) => {
    if (selectedModels.includes(mId)) {
      if (selectedModels.length > 1) {
        setSelectedModels(selectedModels.filter((id) => id !== mId));
      }
    } else {
      if (selectedModels.length < 5) {
        setSelectedModels([...selectedModels, mId]);
      }
    }
  };

  const handleStartExperiment = async () => {
    if (!selectedDatasetId) {
      setErrorMsg('Please upload or select a dataset first.');
      return;
    }
    if (selectedModels.length === 0) {
      setErrorMsg('Please select at least 1 candidate model.');
      return;
    }

    setIsSubmittingJob(true);
    setErrorMsg(null);
    try {
      const summary = await createLabExperiment(selectedDatasetId, selectedModels);
      setExperiments((prev) => [summary, ...prev]);
      setActiveExperimentId(summary.experiment_id);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to submit experiment job.');
    } finally {
      setIsSubmittingJob(false);
    }
  };

  return (
    <div className="flex flex-col gap-6 animate-fadeIn">
      {/* Experimental Disclaimer Banner */}
      <div className="glass-panel p-4 border-l-4 border-l-amber-500 bg-amber-950/20 flex items-start gap-3">
        <ShieldAlert className="w-6 h-6 text-amber-400 shrink-0 mt-0.5" />
        <div className="flex flex-col gap-1">
          <h3 className="text-sm font-bold text-amber-200 uppercase tracking-wide">
            Experimental Sandboxed Zone — User-Supplied Data
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            Models trained in HiggsLens Lab use user-uploaded datasets and certified pre-trained weights, benchmarked on CERN/ATLAS open data remain strictly untouched under <code className="text-cyan-400">models/artifacts/</code>. Lab artifacts persist isolated in <code className="text-amber-400">models/lab_artifacts/</code>.
          </p>
        </div>
      </div>

      {errorMsg && (
        <div className="p-3 rounded-lg bg-red-950/50 border border-red-800 text-red-300 text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-red-400" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Grid Layout: Left Column (Dataset & Models), Right Column (Jobs & Leaderboard) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Upload & Configure */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          {/* Dataset Upload Card */}
          <div className="glass-panel p-5 flex flex-col gap-4">
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <Upload className="w-4 h-4 text-cyan-400" />
              1. Custom Dataset Upload (Max 200 MB / 500k Rows)
            </h4>

            <form onSubmit={handleUpload} className="flex flex-col gap-3 text-xs">
              <div>
                <label className="block text-slate-400 mb-1">Select CSV File</label>
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  className="w-full text-slate-300 bg-slate-900/80 border border-slate-800 rounded-lg p-2 file:mr-3 file:py-1 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-cyan-500 file:text-slate-950 hover:file:bg-cyan-400 cursor-pointer"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Feature Columns (Comma-separated)</label>
                <input
                  type="text"
                  value={featureColsInput}
                  onChange={(e) => setFeatureColsInput(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-slate-200 mono"
                  placeholder="e.g. PRI_tau_pt, PRI_met, ..."
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-slate-400 mb-1">Binary Label Column</label>
                  <input
                    type="text"
                    value={labelColInput}
                    onChange={(e) => setLabelColInput(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-slate-200 mono"
                    placeholder="e.g. Label"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">Event Weight Column (Opt)</label>
                  <input
                    type="text"
                    value={weightColInput}
                    onChange={(e) => setWeightColInput(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-slate-200 mono"
                    placeholder="e.g. Weight"
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={isUploading || !uploadFile}
                className="mt-2 w-full py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-slate-950 font-bold transition-all flex items-center justify-center gap-2"
              >
                {isUploading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                Upload Lab Dataset
              </button>
            </form>

            {/* Uploaded Datasets Selector */}
            {datasets.length > 0 && (
              <div className="pt-3 border-t border-slate-800/80 flex flex-col gap-2">
                <label className="text-xs text-slate-400 flex items-center gap-1">
                  <Database className="w-3.5 h-3.5 text-cyan-400" />
                  Select Active Dataset ({datasets.length} Available)
                </label>
                <select
                  value={selectedDatasetId}
                  onChange={(e) => setSelectedDatasetId(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 mono"
                >
                  {datasets.map((d) => (
                    <option key={d.dataset_id} value={d.dataset_id}>
                      {d.filename} ({d.row_count.toLocaleString()} rows, {d.feature_columns.length} features)
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          {/* Model Selection Card */}
          <div className="glass-panel p-5 flex flex-col gap-4">
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <FlaskConical className="w-4 h-4 text-purple-400" />
              2. Select Candidate Models (Max 5)
            </h4>

            <div className="flex flex-col gap-2 max-h-72 overflow-y-auto pr-1">
              {modelCandidates.map((m) => {
                const isSelected = selectedModels.includes(m.id);
                return (
                  <label
                    key={m.id}
                    className={`p-2.5 rounded-lg border text-xs flex items-center justify-between cursor-pointer transition-all ${
                      isSelected
                        ? 'bg-purple-950/40 border-purple-500/80 text-white'
                        : 'bg-slate-900/60 border-slate-800/80 text-slate-400 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => handleToggleModel(m.id)}
                        className="rounded border-slate-700 text-purple-600 focus:ring-purple-500"
                      />
                      <div>
                        <span className="font-semibold text-slate-200">{m.name}</span>
                        <span className="text-[10px] text-slate-500 block">{m.family}</span>
                      </div>
                    </div>
                  </label>
                );
              })}
            </div>

            <button
              onClick={handleStartExperiment}
              disabled={isSubmittingJob || !selectedDatasetId || selectedModels.length === 0}
              className="py-2.5 rounded-lg bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-400 hover:to-purple-500 disabled:opacity-50 text-white font-bold transition-all flex items-center justify-center gap-2 shadow-lg shadow-purple-500/20"
            >
              {isSubmittingJob ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
              Launch Experiment Job
            </button>
          </div>
        </div>

        {/* Right Column: Experiment Leaderboard & Status */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          {/* Experiment Selection & Status Bar */}
          <div className="glass-panel p-5 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-bold text-white flex items-center gap-2">
                <Cpu className="w-4 h-4 text-emerald-400" />
                3. Experiment Job Status & Leaderboard
              </h4>
              {experiments.length > 0 && (
                <select
                  value={activeExperimentId || ''}
                  onChange={(e) => setActiveExperimentId(e.target.value)}
                  className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-300 mono"
                >
                  {experiments.map((exp) => (
                    <option key={exp.experiment_id} value={exp.experiment_id}>
                      Job {exp.experiment_id.slice(0, 8)} ({exp.status})
                    </option>
                  ))}
                </select>
              )}
            </div>

            {experimentDetail ? (
              <div className="flex flex-col gap-4">
                {/* Status Badge */}
                <div className="flex items-center justify-between bg-slate-900/90 p-3 rounded-lg border border-slate-800">
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-slate-400">Status:</span>
                    <span className={`px-2 py-0.5 rounded text-[11px] font-bold uppercase ${
                      experimentDetail.summary.status === 'completed'
                        ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                        : experimentDetail.summary.status === 'failed'
                        ? 'bg-red-950 text-red-400 border border-red-800'
                        : 'bg-amber-950 text-amber-400 border border-amber-800 animate-pulse'
                    }`}>
                      {experimentDetail.summary.status}
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-400 mono">
                    Created: {new Date(experimentDetail.summary.created_at).toLocaleTimeString()}
                  </div>
                </div>

                {experimentDetail.summary.error_message && (
                  <div className="p-3 rounded-lg bg-red-950/60 border border-red-800 text-red-300 text-xs">
                    Error: {experimentDetail.summary.error_message}
                  </div>
                )}

                {/* Per-Model Results Table */}
                {Object.keys(experimentDetail.per_model_results).length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="border-b border-slate-800 text-slate-400 font-semibold bg-slate-900/60">
                          <th className="p-2.5">Model ID</th>
                          <th className="p-2.5">Test ROC-AUC</th>
                          <th className="p-2.5">Test F1</th>
                          <th className="p-2.5">Opt. Thresh</th>
                          <th className="p-2.5">Test AMS</th>
                          <th className="p-2.5">Fit Time</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 text-slate-200 mono">
                        {Object.entries(experimentDetail.per_model_results).map(([mId, res]) => {
                          const tm = res.test_metrics;
                          return (
                            <tr key={mId} className="hover:bg-slate-900/40">
                              <td className="p-2.5 font-bold text-purple-300">{mId}</td>
                              <td className="p-2.5 text-emerald-400 font-bold">{tm.roc_auc_mean?.toFixed(4) ?? 'N/A'}</td>
                              <td className="p-2.5">{tm.f1_mean?.toFixed(4) ?? 'N/A'}</td>
                              <td className="p-2.5 text-cyan-400">{res.validation_optimal_threshold?.toFixed(4)}</td>
                              <td className="p-2.5 font-bold text-amber-400">
                                {res.is_weighted && tm.ams_score ? tm.ams_score.toFixed(4) : 'Unweighted'}
                              </td>
                              <td className="p-2.5 text-slate-400">{tm.training_duration_seconds?.toFixed(2)}s</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="p-6 text-center text-xs text-slate-500">
                    {experimentDetail.summary.status === 'running' || experimentDetail.summary.status === 'queued'
                      ? 'Training in progress... Status updates automatically.'
                      : 'No training results recorded for this experiment.'}
                  </div>
                )}
              </div>
            ) : (
              <div className="p-8 text-center text-xs text-slate-500">
                No active experiment selected. Upload a dataset and launch an experiment job.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
