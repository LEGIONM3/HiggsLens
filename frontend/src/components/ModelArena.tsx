import React, { useState, useEffect } from 'react';
import { ModelInfo, JobStatus, ExperimentRun } from '../types';
import { startTrainingJob, fetchJobStatus, fetchExperimentRuns } from '../services/api';
import { Cpu, Play, Award, CheckCircle2, AlertCircle, Clock, BarChart3, TrendingUp, ShieldAlert } from 'lucide-react';

interface Props {
  models: Record<string, ModelInfo>;
  onRefresh: () => void;
}

export const ModelArenaComponent: React.FC<Props> = ({ models, onRefresh }) => {
  const [mode, setMode] = useState<'fast' | 'research'>('fast');
  const [featureSet, setFeatureSet] = useState<string>('all_physics');
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(true);

  const loadRuns = async () => {
    try {
      const data = await fetchExperimentRuns();
      setRuns(data.reverse()); // Latest first
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingRuns(false);
    }
  };

  useEffect(() => {
    loadRuns();
  }, []);

  useEffect(() => {
    if (!activeJobId) return;
    const interval = setInterval(async () => {
      try {
        const status = await fetchJobStatus(activeJobId);
        setJobStatus(status);
        if (status.state === 'completed' || status.state === 'failed') {
          setActiveJobId(null);
          loadRuns();
          onRefresh();
        }
      } catch (err) {
        console.error(err);
      }
    }, 1500);
    return () => clearInterval(interval);
  }, [activeJobId]);

  const handleStartTraining = async () => {
    try {
      const job = await startTrainingJob(mode, featureSet);
      setActiveJobId(job.job_id);
      setJobStatus(job);
    } catch (err: any) {
      alert(`Error launching training job: ${err.message}`);
    }
  };

  const latestRun = runs[0];

  return (
    <div className="glass-panel p-6 flex flex-col gap-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-slate-800 pb-4 gap-4">
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-lg bg-purple-500/10 border border-purple-500/30 text-purple-400">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              Reproducible Model Arena
              <span className="badge badge-purple">Multi-Seed Training</span>
            </h2>
            <p className="text-sm text-slate-400">
              Evaluate required baselines & optional plugin architectures with strict zero test-set leakage
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as any)}
            disabled={!!activeJobId}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500"
          >
            <option value="fast">⚡ Fast Mode (Bounded Stratified Sample, Seed 42)</option>
            <option value="research">🔬 Research Mode (Full Dataset, Seeds [42, 123, 2026])</option>
          </select>

          <select
            value={featureSet}
            onChange={(e) => setFeatureSet(e.target.value)}
            disabled={!!activeJobId}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500"
          >
            <option value="all_physics">Physics Features: All (Primary + Derived)</option>
            <option value="primary_only">Physics Features: Primary Measurements Only</option>
            <option value="derived_only">Physics Features: Derived Angles & Invariant Masses Only</option>
          </select>

          <button
            onClick={handleStartTraining}
            disabled={!!activeJobId}
            className="glow-btn text-sm"
          >
            <Play className="w-4 h-4 fill-current" />
            {activeJobId ? 'Training in Progress...' : 'Launch Training Arena'}
          </button>
        </div>
      </div>

      {/* Model Candidate Status Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {Object.entries(models).map(([id, info]) => (
          <div
            key={id}
            className={`p-3 rounded-lg border flex flex-col justify-between gap-2 transition-all ${
              info.status === 'available'
                ? 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                : 'bg-slate-950/40 border-slate-900 opacity-60'
            }`}
          >
            <div>
              <div className="flex items-center justify-between gap-1">
                <span className="text-xs font-bold text-white truncate" title={info.display_name}>
                  {info.display_name.split(' (')[0]}
                </span>
                {info.required ? (
                  <span className="badge badge-cyan text-[10px] px-1.5 py-0.2">Required</span>
                ) : (
                  <span className="badge badge-amber text-[10px] px-1.5 py-0.2">Plugin</span>
                )}
              </div>
              <span className="text-[11px] text-slate-400 mono mt-1 block">
                Missing: {info.supports_missing ? 'Native' : 'Impute'}
              </span>
            </div>
            <div className="flex items-center justify-between pt-2 border-t border-slate-800/60 text-xs">
              <span className="text-slate-500">Status</span>
              <span
                className={`font-semibold capitalize flex items-center gap-1 ${
                  info.status === 'available' ? 'text-emerald-400' : 'text-slate-500'
                }`}
              >
                {info.status === 'available' ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
                {info.status}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Live Job Progress Banner */}
      {jobStatus && jobStatus.state !== 'completed' && (
        <div className="p-4 rounded-xl bg-gradient-to-r from-purple-950/40 via-cyan-950/40 to-slate-900 border border-cyan-500/30 flex flex-col gap-3 animate-pulse">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-cyan-300 font-semibold text-sm">
              <Clock className="w-4 h-4 animate-spin" />
              <span>Job Status: {jobStatus.state.toUpperCase().replace('_', ' ')}</span>
            </div>
            <span className="mono text-xs text-slate-400">
              Completed {jobStatus.completed_models} / {jobStatus.total_models} models
            </span>
          </div>
          <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
            <div
              className="bg-gradient-to-r from-cyan-400 to-purple-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${(jobStatus.completed_models / Math.max(1, jobStatus.total_models)) * 100}%` }}
            />
          </div>
          <p className="text-xs text-slate-300 mono">{jobStatus.progress_message}</p>
        </div>
      )}

      {/* Latest Run Results & Recommendation Engine */}
      {latestRun ? (
        <div className="flex flex-col gap-4 mt-2">
          {latestRun.recommendation && (
            <div className="p-5 rounded-xl bg-gradient-to-br from-cyan-950/50 to-purple-950/40 border border-cyan-500/40 flex flex-col gap-3 shadow-lg">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-cyan-300 font-bold text-base">
                  <Award className="w-5 h-5 text-amber-400" />
                  <span>Recommendation Engine Winner: {models[latestRun.recommendation.recommended_model_id]?.display_name || latestRun.recommendation.recommended_model_id}</span>
                </div>
                <span className="badge badge-emerald">Optimal Validation Selection</span>
              </div>
              <p className="text-sm text-slate-200 leading-relaxed">
                {latestRun.recommendation.reasoning}
              </p>
              {latestRun.recommendation.warnings.length > 0 && (
                <div className="flex flex-col gap-1 pt-2 border-t border-cyan-500/20">
                  {latestRun.recommendation.warnings.map((w, idx) => (
                    <div key={idx} className="flex items-center gap-2 text-xs text-amber-300">
                      <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0" />
                      <span>{w}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="overflow-x-auto rounded-lg border border-slate-800">
            <table className="w-full text-left border-collapse text-sm">
              <thead>
                <tr className="bg-slate-900/90 text-slate-400 text-xs uppercase tracking-wider border-b border-slate-800">
                  <th className="p-3 font-semibold">Candidate Model</th>
                  <th className="p-3 font-semibold">ROC-AUC (Val)</th>
                  <th className="p-3 font-semibold">Weighted AMS</th>
                  <th className="p-3 font-semibold">Optimal Thresh</th>
                  <th className="p-3 font-semibold">Brier (Calib)</th>
                  <th className="p-3 font-semibold">Stability</th>
                  <th className="p-3 font-semibold text-right">Duration</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
                {Object.values(latestRun.metrics_by_model).map((m) => {
                  const isRec = latestRun.recommendation?.recommended_model_id === m.model_id;
                  return (
                    <tr
                      key={m.model_id}
                      className={`hover:bg-slate-900/50 transition-colors ${
                        isRec ? 'bg-cyan-500/5 font-medium' : ''
                      }`}
                    >
                      <td className="p-3 flex items-center gap-2">
                        {isRec && <Award className="w-4 h-4 text-amber-400 shrink-0" />}
                        <span className="text-white">
                          {models[m.model_id]?.display_name || m.model_id}
                        </span>
                      </td>
                      <td className="p-3 mono text-cyan-300">
                        {m.roc_auc_mean.toFixed(4)}
                        {m.roc_auc_std > 0 && <span className="text-slate-500 text-xs ml-1">±{m.roc_auc_std.toFixed(3)}</span>}
                      </td>
                      <td className="p-3 mono text-purple-300 font-semibold">
                        {m.ams_score.toFixed(4)}
                      </td>
                      <td className="p-3 mono text-slate-300">
                        {m.optimal_threshold.toFixed(2)}
                      </td>
                      <td className="p-3 mono">
                        <span className={m.brier_score_mean < 0.18 ? 'text-emerald-400' : 'text-amber-400'}>
                          {m.brier_score_mean.toFixed(4)}
                        </span>
                      </td>
                      <td className="p-3">
                        <span
                          className={`badge text-[10px] ${
                            m.stability_status === 'stable' ? 'badge-emerald' : 'badge-amber'
                          }`}
                        >
                          {m.stability_status}
                        </span>
                      </td>
                      <td className="p-3 mono text-right text-slate-400">
                        {m.training_duration_seconds.toFixed(2)}s
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="text-center py-8 text-slate-500 flex flex-col items-center gap-2">
          <BarChart3 className="w-8 h-8 text-slate-600" />
          <span>No experiment runs recorded yet. Launch the Training Arena above to evaluate model candidates!</span>
        </div>
      )}
    </div>
  );
};
