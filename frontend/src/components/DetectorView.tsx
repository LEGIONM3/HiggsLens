import React, { useState } from 'react';
import { ModelInfo, PredictionResponse } from '../types';
import { runLivePrediction } from '../services/api';
import { Eye, Zap, Compass, Activity, Sliders, CheckCircle, ShieldAlert } from 'lucide-react';

interface Props {
  models: Record<string, ModelInfo>;
}

export const DetectorViewComponent: React.FC<Props> = ({ models }) => {
  const [selectedModel, setSelectedModel] = useState<string>('histogram_gradient_boosting');
  const [eventId, setEventId] = useState<string>('100000');
  const [useCustomSliders, setUseCustomSliders] = useState<boolean>(false);
  const [customFeatures, setCustomFeatures] = useState({
    PRI_tau_pt: 45.2,
    PRI_lep_pt: 52.8,
    PRI_met: 65.0,
    PRI_jet_num: 1,
    PRI_jet_leading_pt: 85.0,
  });
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const eid = useCustomSliders ? undefined : parseInt(eventId, 10);
      const feats = useCustomSliders ? customFeatures : undefined;
      const res = await runLivePrediction(selectedModel, eid, feats);
      setPrediction(res);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const availableModels = Object.entries(models).filter(([_, info]) => info.status === 'available');

  return (
    <div className="glass-panel p-6 flex flex-col gap-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-slate-800 pb-4 gap-4">
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
            <Eye className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              Event Kinematics & Detector View
              <span className="badge badge-emerald">Interactive Prediction</span>
            </h2>
            <p className="text-sm text-slate-400">
              Visualize $H \to \tau\tau$ decay geometry ($p_T, \eta, \phi$), missing transverse energy (E_T^miss), and classification scores
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500"
          >
            {availableModels.map(([id, info]) => (
              <option key={id} value={id}>
                {info.display_name}
              </option>
            ))}
          </select>

          <div className="flex items-center bg-slate-900 rounded-lg border border-slate-800 p-1">
            <button
              onClick={() => setUseCustomSliders(false)}
              className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                !useCustomSliders ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-white'
              }`}
            >
              CERN Event ID
            </button>
            <button
              onClick={() => setUseCustomSliders(true)}
              className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                useCustomSliders ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-white'
              }`}
            >
              Kinematic Sliders
            </button>
          </div>

          {!useCustomSliders && (
            <input
              type="number"
              value={eventId}
              onChange={(e) => setEventId(e.target.value)}
              placeholder="EventId (e.g. 100000)"
              className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500 w-36 mono"
            />
          )}

          <button onClick={handlePredict} disabled={loading} className="glow-btn text-sm">
            <Zap className="w-4 h-4 fill-current" />
            {loading ? 'Analyzing Event...' : 'Classify Event'}
          </button>
        </div>
      </div>

      {useCustomSliders && (
        <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-slate-400 font-medium">Hadronic Tau $p_T$ (GeV)</label>
            <input
              type="range"
              min="20"
              max="200"
              value={customFeatures.PRI_tau_pt}
              onChange={(e) => setCustomFeatures({ ...customFeatures, PRI_tau_pt: parseFloat(e.target.value) })}
              className="accent-cyan-400"
            />
            <span className="mono text-xs text-cyan-300 font-bold">{customFeatures.PRI_tau_pt} GeV</span>
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-xs text-slate-400 font-medium">Lepton $p_T$ (GeV)</label>
            <input
              type="range"
              min="20"
              max="200"
              value={customFeatures.PRI_lep_pt}
              onChange={(e) => setCustomFeatures({ ...customFeatures, PRI_lep_pt: parseFloat(e.target.value) })}
              className="accent-purple-400"
            />
            <span className="mono text-xs text-purple-300 font-bold">{customFeatures.PRI_lep_pt} GeV</span>
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-xs text-slate-400 font-medium">Missing E_T (E_T^miss)</label>
            <input
              type="range"
              min="10"
              max="300"
              value={customFeatures.PRI_met}
              onChange={(e) => setCustomFeatures({ ...customFeatures, PRI_met: parseFloat(e.target.value) })}
              className="accent-amber-400"
            />
            <span className="mono text-xs text-amber-300 font-bold">{customFeatures.PRI_met} GeV</span>
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-xs text-slate-400 font-medium">Jet Multiplicity (N_jets)</label>
            <input
              type="range"
              min="0"
              max="3"
              step="1"
              value={customFeatures.PRI_jet_num}
              onChange={(e) => setCustomFeatures({ ...customFeatures, PRI_jet_num: parseInt(e.target.value, 10) })}
              className="accent-emerald-400"
            />
            <span className="mono text-xs text-emerald-300 font-bold">{customFeatures.PRI_jet_num} jets</span>
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-xs text-slate-400 font-medium">Leading Jet $p_T$ (GeV)</label>
            <input
              type="range"
              min="30"
              max="300"
              disabled={customFeatures.PRI_jet_num === 0}
              value={customFeatures.PRI_jet_leading_pt}
              onChange={(e) => setCustomFeatures({ ...customFeatures, PRI_jet_leading_pt: parseFloat(e.target.value) })}
              className="accent-rose-400"
            />
            <span className="mono text-xs text-rose-300 font-bold">
              {customFeatures.PRI_jet_num === 0 ? '-999.0 (undefined)' : `${customFeatures.PRI_jet_leading_pt} GeV`}
            </span>
          </div>
        </div>
      )}

      {error && (
        <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-500/30 text-rose-300 text-sm flex items-center gap-2">
          <ShieldAlert className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {prediction ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Classification Result & Confidence Panel */}
          <div className="p-5 rounded-xl bg-gradient-to-b from-slate-900 to-slate-950 border border-slate-800 flex flex-col justify-between gap-4">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs uppercase font-semibold text-slate-400 tracking-wider">Classification Score</span>
                <span className={`badge ${prediction.prediction.predicted_class === 'signal' ? 'badge-cyan' : 'badge-amber'}`}>
                  {prediction.prediction.predicted_class === 'signal' ? 'Signal Candidate (H → ττ)' : 'Background Process'}
                </span>
              </div>
              <div className="mt-4 flex items-baseline gap-3">
                <span className="text-4xl font-extrabold text-white mono">
                  {(prediction.prediction.signal_probability * 100).toFixed(1)}%
                </span>
                <span className="text-sm text-slate-400">Signal Probability</span>
              </div>

              {/* Gauge */}
              <div className="w-full bg-slate-950 rounded-full h-3 mt-3 overflow-hidden border border-slate-800 flex">
                <div
                  className="bg-gradient-to-r from-amber-500 via-cyan-400 to-purple-500 transition-all duration-500"
                  style={{ width: `${prediction.prediction.signal_probability * 100}%` }}
                />
              </div>

              <div className="flex justify-between text-xs text-slate-500 mono mt-1">
                <span>0.0 (Background)</span>
                <span>Thresh: {prediction.prediction.decision_threshold}</span>
                <span>1.0 (Signal)</span>
              </div>
            </div>

            <div className="pt-3 border-t border-slate-800/80 flex flex-col gap-1 text-xs text-slate-400">
              <div className="flex justify-between">
                <span>Model Candidate:</span>
                <span className="font-semibold text-white">{models[prediction.prediction.model_id]?.display_name || prediction.prediction.model_id}</span>
              </div>
              <div className="flex justify-between">
                <span>Partition Status:</span>
                <span className="mono text-cyan-300">{prediction.prediction.validation_status}</span>
              </div>
              {prediction.missing_adjusted_fields.length > 0 && (
                <div className="flex justify-between text-amber-400">
                  <span>Sentinels Handled:</span>
                  <span className="mono">{prediction.missing_adjusted_fields.length} fields (-999.0)</span>
                </div>
              )}
            </div>
          </div>

          {/* Reconstructed Objects & Kinematics Table */}
          <div className="lg:col-span-2 p-5 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col justify-between gap-4">
            <div>
              <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2 mb-3">
                <Activity className="w-4 h-4 text-cyan-400" />
                Reconstructed Collider Objects & Transverse Energy ($p_T, \eta, \phi$)
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider">
                      <th className="pb-2 font-semibold">Object Type</th>
                      <th className="pb-2 font-semibold">Status</th>
                      <th className="pb-2 font-semibold">$p_T$ (GeV)</th>
                      <th className="pb-2 font-semibold">$\eta$ (Pseudorapidity)</th>
                      <th className="pb-2 font-semibold">$\phi$ (Azimuthal)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50">
                    {prediction.objects.map((obj, i) => (
                      <tr key={i} className="hover:bg-slate-900/40">
                        <td className="py-2.5 font-medium text-white flex items-center gap-2">
                          <span className={`w-2 h-2 rounded-full ${obj.present ? 'bg-cyan-400' : 'bg-slate-600'}`} />
                          {obj.label}
                        </td>
                        <td className="py-2.5">
                          {obj.present ? (
                            <span className="text-emerald-400 font-semibold">Detected</span>
                          ) : (
                            <span className="text-slate-500 mono">-999.0 (N/A)</span>
                          )}
                        </td>
                        <td className="py-2.5 mono text-cyan-300">{obj.present ? obj.pt.toFixed(2) : '—'}</td>
                        <td className="py-2.5 mono text-slate-300">{obj.present ? obj.eta.toFixed(3) : '—'}</td>
                        <td className="py-2.5 mono text-slate-300">{obj.present ? obj.phi.toFixed(3) : '—'}</td>
                      </tr>
                    ))}
                    {/* MET Row */}
                    <tr className="bg-slate-950/40 font-semibold">
                      <td className="py-2.5 text-amber-300 flex items-center gap-2">
                        <Compass className="w-3.5 h-3.5 text-amber-400" />
                        Missing E_T (E_T^miss)
                      </td>
                      <td className="py-2.5 text-amber-300">Kinematic Imbalance</td>
                      <td className="py-2.5 mono text-amber-300">{prediction.missing_transverse_energy.magnitude.toFixed(2)}</td>
                      <td className="py-2.5 mono text-slate-500">—</td>
                      <td className="py-2.5 mono text-amber-300">{prediction.missing_transverse_energy.phi.toFixed(3)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div className="flex items-center justify-between text-xs text-slate-500 pt-2 border-t border-slate-800/80">
              <span>Total Jet Multiplicity: {prediction.jet_summary.count}</span>
              <span>Scalar $p_T$ Sum ($H_T$): {prediction.jet_summary.total_pt.toFixed(2)} GeV</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-10 text-slate-500 flex flex-col items-center gap-2">
          <Zap className="w-8 h-8 text-slate-600" />
          <span>Select or input kinematic features and click 'Classify Event' to inspect ATLAS full-detector predictions.</span>
        </div>
      )}
    </div>
  );
};
