import React, { useState, useEffect, useCallback } from 'react';

export interface EventEditorProps {
  initialPriFeatures: Record<str, number>;
  baseEventId?: number;
  onDeriveComplete: (
    fullFeatures: Record<string, number>,
    prediction: {
      model_id: string;
      probability: number;
      predicted_label: string;
      threshold: number;
    },
    mmcPolicy: string,
    notes: string[]
  ) => void;
}

export const EventEditor: React.FC<EventEditorProps> = ({
  initialPriFeatures,
  baseEventId,
  onDeriveComplete,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [priFeatures, setPriFeatures] = useState<Record<string, number>>(initialPriFeatures);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [mmcPolicy, setMmcPolicy] = useState<string>('sentinel');
  const [responseNotes, setResponseNotes] = useState<string[]>([]);

  // Sync initial features when base event changes
  useEffect(() => {
    if (initialPriFeatures && Object.keys(initialPriFeatures).length > 0) {
      setPriFeatures(initialPriFeatures);
    }
  }, [initialPriFeatures]);

  // Jet Multiplicity Auto-Snap logic (Amendment 4)
  const handleJetNumChange = (newJetNum: number) => {
    const updated = { ...priFeatures, PRI_jet_num: newJetNum };

    if (newJetNum === 0) {
      // Snap leading & subleading jet features to -999.0
      updated.PRI_jet_leading_pt = -999.0;
      updated.PRI_jet_leading_eta = -999.0;
      updated.PRI_jet_leading_phi = -999.0;
      updated.PRI_jet_subleading_pt = -999.0;
      updated.PRI_jet_subleading_eta = -999.0;
      updated.PRI_jet_subleading_phi = -999.0;
      updated.PRI_jet_all_pt = 0.0;
    } else if (newJetNum === 1) {
      // Enable leading jet, snap subleading jet to -999.0
      if (updated.PRI_jet_leading_pt === -999.0) updated.PRI_jet_leading_pt = 50.0;
      if (updated.PRI_jet_leading_eta === -999.0) updated.PRI_jet_leading_eta = 0.0;
      if (updated.PRI_jet_leading_phi === -999.0) updated.PRI_jet_leading_phi = 0.0;

      updated.PRI_jet_subleading_pt = -999.0;
      updated.PRI_jet_subleading_eta = -999.0;
      updated.PRI_jet_subleading_phi = -999.0;
      updated.PRI_jet_all_pt = updated.PRI_jet_leading_pt;
    } else if (newJetNum >= 2) {
      // Enable both leading and subleading jets
      if (updated.PRI_jet_leading_pt === -999.0) updated.PRI_jet_leading_pt = 80.0;
      if (updated.PRI_jet_leading_eta === -999.0) updated.PRI_jet_leading_eta = 0.5;
      if (updated.PRI_jet_leading_phi === -999.0) updated.PRI_jet_leading_phi = 1.0;

      if (updated.PRI_jet_subleading_pt === -999.0) updated.PRI_jet_subleading_pt = 45.0;
      if (updated.PRI_jet_subleading_eta === -999.0) updated.PRI_jet_subleading_eta = -0.8;
      if (updated.PRI_jet_subleading_phi === -999.0) updated.PRI_jet_subleading_phi = -2.0;

      updated.PRI_jet_all_pt = updated.PRI_jet_leading_pt + updated.PRI_jet_subleading_pt;
    }

    setPriFeatures(updated);
  };

  const handleFeatureChange = (key: string, value: number) => {
    setPriFeatures((prev) => ({ ...prev, [key]: value }));
  };

  // Debounced API call to POST /api/v1/events/derive
  const executeDerive = useCallback(async () => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const resp = await fetch('/api/v1/events/derive', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: priFeatures,
          base_event_id: baseEventId,
          model_id: 'xgboost',
        }),
      });

      if (!resp.ok) {
        const errData = await resp.json();
        throw new Error(errData.detail || `Server returned status ${resp.status}`);
      }

      const data = await resp.json();
      setMmcPolicy(data.mmc_policy);
      setResponseNotes(data.notes || []);

      onDeriveComplete(data.features, data.prediction, data.mmc_policy, data.notes || []);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown derivation error';
      setErrorMsg(msg);
    } finally {
      setIsLoading(false);
    }
  }, [priFeatures, baseEventId, onDeriveComplete]);

  // Debounce derive call on feature changes
  useEffect(() => {
    if (!isOpen) return;
    const timer = setTimeout(() => {
      executeDerive();
    }, 300);
    return () => clearTimeout(timer);
  }, [priFeatures, isOpen, executeDerive]);

  const jetNum = priFeatures.PRI_jet_num ?? 0;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl overflow-hidden backdrop-blur-md shadow-xl text-slate-100">
      {/* Toggle Header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-5 py-3.5 bg-slate-950/60 hover:bg-slate-800/60 transition-colors flex items-center justify-between text-left border-b border-slate-800"
      >
        <div className="flex items-center gap-2.5">
          <span className="text-cyan-400 font-semibold text-sm">Advanced — Event Kinematics Editor</span>
          <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
            17 PRI Parameters
          </span>
        </div>
        <span className="text-slate-400 text-xs flex items-center gap-1 font-mono">
          {isOpen ? 'Collapse ▲' : 'Expand ▼'}
        </span>
      </button>

      {/* Editor Content */}
      {isOpen && (
        <div className="p-5 space-y-5">
          {errorMsg && (
            <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/50 text-rose-300 text-xs">
              <strong>Validation Error (422):</strong> {errorMsg}
            </div>
          )}

          {/* Jet Multiplicity Selector */}
          <div className="bg-slate-950/60 p-3.5 rounded-lg border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <label htmlFor="jet-multiplicity-select" className="text-xs font-semibold text-cyan-300">Jet Multiplicity (PRI_jet_num)</label>
              <span className="text-[10px] text-slate-400">Auto-snaps jet sentinels</span>
            </div>
            <div className="flex gap-2">
              {[0, 1, 2, 3].map((num) => (
                <button
                  key={num}
                  id={`jet-num-btn-${num}`}
                  type="button"
                  onClick={() => handleJetNumChange(num)}
                  className={`flex-1 py-1.5 text-xs font-mono rounded border transition-colors ${
                    jetNum === num
                      ? 'bg-cyan-600 border-cyan-400 text-white font-bold'
                      : 'bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  {num} {num === 1 ? 'Jet' : 'Jets'}
                </button>
              ))}
            </div>
          </div>

          {/* Key Primary Sliders Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            {/* Tau pT */}
            <div className="space-y-1">
              <div className="flex justify-between">
                <label htmlFor="tau-pt-slider" className="text-slate-300">Tau pT (GeV)</label>
                <span className="font-mono text-cyan-300">{priFeatures.PRI_tau_pt?.toFixed(1)}</span>
              </div>
              <input
                id="tau-pt-slider"
                type="range"
                min={20.0}
                max={150.0}
                step={0.5}
                value={priFeatures.PRI_tau_pt || 20.0}
                onChange={(e) => handleFeatureChange('PRI_tau_pt', parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-500"
              />
            </div>

            {/* Lepton pT */}
            <div className="space-y-1">
              <div className="flex justify-between">
                <label htmlFor="lep-pt-slider" className="text-slate-300">Lepton pT (GeV)</label>
                <span className="font-mono text-cyan-300">{priFeatures.PRI_lep_pt?.toFixed(1)}</span>
              </div>
              <input
                id="lep-pt-slider"
                type="range"
                min={20.0}
                max={150.0}
                step={0.5}
                value={priFeatures.PRI_lep_pt || 20.0}
                onChange={(e) => handleFeatureChange('PRI_lep_pt', parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-500"
              />
            </div>

            {/* MET */}
            <div className="space-y-1">
              <div className="flex justify-between">
                <label htmlFor="met-slider" className="text-slate-300">MET (GeV)</label>
                <span className="font-mono text-cyan-300">{priFeatures.PRI_met?.toFixed(1)}</span>
              </div>
              <input
                id="met-slider"
                type="range"
                min={0.0}
                max={150.0}
                step={0.5}
                value={priFeatures.PRI_met || 0.0}
                onChange={(e) => handleFeatureChange('PRI_met', parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-500"
              />
            </div>

            {/* Tau Eta */}
            <div className="space-y-1">
              <div className="flex justify-between">
                <label htmlFor="tau-eta-slider" className="text-slate-300">Tau &eta;</label>
                <span className="font-mono text-cyan-300">{priFeatures.PRI_tau_eta?.toFixed(2)}</span>
              </div>
              <input
                id="tau-eta-slider"
                type="range"
                min={-2.5}
                max={2.5}
                step={0.05}
                value={priFeatures.PRI_tau_eta || 0.0}
                onChange={(e) => handleFeatureChange('PRI_tau_eta', parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-500"
              />
            </div>

            {/* Lepton Eta */}
            <div className="space-y-1">
              <div className="flex justify-between">
                <label htmlFor="lep-eta-slider" className="text-slate-300">Lepton &eta;</label>
                <span className="font-mono text-cyan-300">{priFeatures.PRI_lep_eta?.toFixed(2)}</span>
              </div>
              <input
                id="lep-eta-slider"
                type="range"
                min={-2.5}
                max={2.5}
                step={0.05}
                value={priFeatures.PRI_lep_eta || 0.0}
                onChange={(e) => handleFeatureChange('PRI_lep_eta', parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-500"
              />
            </div>

            {/* MET Phi */}
            <div className="space-y-1">
              <div className="flex justify-between">
                <label htmlFor="met-phi-slider" className="text-slate-300">MET &phi;</label>
                <span className="font-mono text-cyan-300">{priFeatures.PRI_met_phi?.toFixed(2)}</span>
              </div>
              <input
                id="met-phi-slider"
                type="range"
                min={-3.14}
                max={3.14}
                step={0.05}
                value={priFeatures.PRI_met_phi || 0.0}
                onChange={(e) => handleFeatureChange('PRI_met_phi', parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-500"
              />
            </div>
          </div>

          {/* Mandatory MMC Sentinel Caveat Notice */}
          <div className="p-3 bg-amber-950/30 border border-amber-800/40 rounded-lg text-xs text-amber-200/90 leading-relaxed flex items-start gap-2">
            <svg className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div>
              <span className="font-semibold">MMC Policy: {mmcPolicy.toUpperCase()}</span> &mdash; DER_mass_MMC cannot be recomputed from edited kinematics; the model receives &apos;not available&apos; (-999), matching the dataset convention.
            </div>
          </div>

          {/* Response Notes */}
          {responseNotes.length > 0 && (
            <div className="text-[11px] text-slate-400 space-y-1 font-mono bg-slate-950/50 p-2.5 rounded border border-slate-800/60">
              <div className="font-sans text-xs text-slate-300 font-medium">Re-derivation Log:</div>
              {responseNotes.map((note, idx) => (
                <div key={idx}>&bull; {note}</div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
