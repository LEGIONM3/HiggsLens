import React, { useState } from 'react';
import { ReproducibilityManifest } from '../../types/report';
import { LoadingSkeleton } from '../common/LoadingSkeleton';

export const ReproducibilitySection: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [manifest, setManifest] = useState<ReproducibilityManifest | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggleSection = async () => {
    if (!isOpen && !manifest) {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch('/api/v1/reproducibility');
        if (!res.ok) {
          throw new Error(`Failed to load reproducibility manifest (${res.status})`);
        }
        const data: ReproducibilityManifest = await res.json();
        setManifest(data);
      } catch (err: any) {
        setError(err.message || 'Error fetching manifest');
      } finally {
        setLoading(false);
      }
    }
    setIsOpen(!isOpen);
  };

  return (
    <div className="mt-8 border border-slate-700/80 rounded-xl bg-slate-900/60 p-5 shadow-lg">
      <button
        onClick={toggleSection}
        className="w-full flex items-center justify-between text-left font-semibold text-sky-400 hover:text-sky-300 transition-colors focus:outline-none focus:ring-2 focus:ring-sky-500 rounded-md p-1"
        aria-expanded={isOpen}
        aria-label="Toggle Reproducibility Details Section"
      >
        <span className="flex items-center gap-2 text-base">
          🛡️ Reproducibility & Artifact Manifest
        </span>
        <span className="text-sm text-slate-400">
          {isOpen ? '▲ Hide Details' : '▼ View Manifest'}
        </span>
      </button>

      {isOpen && (
        <div className="mt-4 pt-4 border-t border-slate-800 space-y-4 text-sm text-slate-300">
          {loading && <LoadingSkeleton height="h-32" />}

          {error && (
            <div className="p-3 bg-red-950/60 border border-red-800/60 rounded-lg text-red-300 text-xs">
              ⚠️ {error}
            </div>
          )}

          {manifest && !loading && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
                  <div className="text-xs uppercase text-slate-400 font-semibold mb-1">
                    CERN Open Data Provenance
                  </div>
                  <div>Record ID: <strong className="text-slate-200">{manifest.dataset.record}</strong></div>
                  <div>DOI: <a href={`https://doi.org/${manifest.dataset.doi}`} target="_blank" rel="noreferrer" className="text-sky-400 underline">{manifest.dataset.doi}</a></div>
                  <div className="text-xs text-slate-500 truncate mt-1">Hash: {manifest.dataset.content_hash}</div>
                </div>

                <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
                  <div className="text-xs uppercase text-slate-400 font-semibold mb-1">
                    Inference & Schema Contract
                  </div>
                  <div>Features: <strong className="text-slate-200">{manifest.inference_contract.feature_count} canonical features</strong></div>
                  <div>Sentinel Value: <code className="text-amber-400">{manifest.inference_contract.sentinel_value}</code></div>
                  <div>Status: <span className="text-emerald-400 font-semibold">{manifest.frozen_leaderboard_status}</span></div>
                </div>
              </div>

              <div>
                <div className="text-xs uppercase text-slate-400 font-semibold mb-2">
                  Certified Pre-trained Models ({manifest.certified_models.length})
                </div>
                <div className="max-h-48 overflow-y-auto border border-slate-800 rounded-lg divide-y divide-slate-800/60 bg-slate-950/80">
                  {manifest.certified_models.map((m) => (
                    <div key={m.model_id} className="p-2.5 flex items-center justify-between text-xs">
                      <div>
                        <span className="font-mono text-sky-300 font-semibold">{m.model_id}</span>
                        <span className="text-slate-500 ml-2">({m.training_run_origin})</span>
                      </div>
                      <div className="text-slate-400 text-right">
                        <span className="px-2 py-0.5 bg-slate-800 rounded text-slate-300">{m.device}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="p-3 bg-sky-950/30 border border-sky-800/40 rounded-lg text-xs text-sky-200">
                🔒 <strong>Holdout Isolation:</strong> Holdout dataset split (<code>KaggleSet == "u"</code>, 18,238 events) is strictly unexposed via public API endpoints. Zero ML model retraining is executed on the public serving path.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
