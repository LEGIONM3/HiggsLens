import React, { useState } from 'react';
import { Beaker, Upload, Play, CheckCircle2, AlertCircle } from 'lucide-react';
import { ModelInfo } from '../../../types';
import { uploadLabDataset, createLabExperiment } from '../../../services/labApi';

interface LabViewProps {
  models?: Record<string, ModelInfo>;
}

export const LabView: React.FC<LabViewProps> = ({ models = {} }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [uploading, setUploading] = useState<boolean>(false);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setSelectedFile(file);
    setUploading(true);
    setStatusMessage('Uploading custom dataset to segregated lab zone...');
    try {
      const manifest = await uploadLabDataset(file, ['DER_mass_MMC', 'PRI_tau_pt', 'PRI_lep_pt'], 'Label');
      setDatasetId(manifest.dataset_id);
      setStatusMessage(`Dataset uploaded successfully! Manifest ID: ${manifest.dataset_id}`);
    } catch (err: any) {
      setStatusMessage(`Upload failed: ${err.message || 'Lab API offline'}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6 animate-fadeIn">
      <div className="bg-[#090d16] border border-slate-800 rounded-2xl p-6 flex flex-col gap-6 shadow-2xl">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-400">
              <Beaker className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                Segregated Experimental Lab
                <span className="px-2 py-0.5 rounded bg-amber-950 text-amber-300 text-[10px] font-mono font-semibold">
                  Experimental Zone
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Upload custom datasets, configure candidate models, and run isolated training &amp; evaluation experiments
              </p>
            </div>
          </div>
        </div>

        {/* Custom Dataset Upload Dropzone */}
        <div className="p-6 rounded-xl bg-[#05070c] border border-dashed border-slate-700 flex flex-col items-center justify-center gap-3 text-center">
          <Upload className="w-8 h-8 text-cyan-400" />
          <div>
            <h4 className="text-sm font-bold text-white">Upload Custom Dataset (CSV)</h4>
            <p className="text-xs text-slate-400 mt-1">
              Supports custom feature columns, target labels, and sample weights.
            </p>
          </div>

          <label className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs cursor-pointer transition-colors">
            Select CSV File
            <input type="file" accept=".csv" onChange={handleFileUpload} className="hidden" />
          </label>

          {statusMessage && (
            <div className="mt-2 text-xs font-mono text-cyan-300 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>{statusMessage}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
