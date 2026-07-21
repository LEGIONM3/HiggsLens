import React, { useState } from 'react';
import { DatasetStatus } from '../types';
import { triggerDatasetDownload, triggerDatasetValidate } from '../services/api';
import { Database, Download, CheckCircle, AlertTriangle, ShieldCheck, RefreshCw, Layers } from 'lucide-react';

interface Props {
  status: DatasetStatus | null;
  onRefresh: () => void;
}

export const DatasetCardComponent: React.FC<Props> = ({ status, onRefresh }) => {
  const [downloading, setDownloading] = useState(false);
  const [validating, setValidating] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const handleDownload = async () => {
    setDownloading(true);
    setMessage('Streaming canonical CERN dataset (Record 328)...');
    try {
      const res = await triggerDatasetDownload(false);
      setMessage(res.message);
      onRefresh();
    } catch (err: any) {
      setMessage(`Download error: ${err.message}`);
    } finally {
      setDownloading(false);
    }
  };

  const handleValidate = async () => {
    setValidating(true);
    setMessage('Running scientific schema & sentinel validation checks...');
    try {
      await triggerDatasetValidate();
      setMessage('Validation passed! Artifacts & verification reports generated.');
      onRefresh();
    } catch (err: any) {
      setMessage(`Validation failed: ${err.message}`);
    } finally {
      setValidating(false);
    }
  };

  const formatSize = (bytes: number) => {
    if (!bytes) return '0 B';
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(2)} MiB`;
  };

  return (
    <div className="glass-panel p-6 flex flex-col gap-6">
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <Database className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              CERN Open Data Pipeline
              <span className="badge badge-cyan">Record 328</span>
            </h2>
            <p className="text-sm text-slate-400">
              ATLAS $H \to \tau\tau$ Machine Learning Challenge 2014 (`10.7483/OPENDATA.ATLAS.ZBP2.M5T8`)
            </p>
          </div>
        </div>
        <button
          onClick={onRefresh}
          className="p-2 rounded-lg bg-slate-900/80 border border-slate-800 hover:border-slate-700 text-slate-400 hover:text-white transition-all"
          title="Refresh Dataset Status"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {status ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-lg bg-slate-900/60 border border-slate-800/80 flex flex-col gap-1">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Cache Status</span>
            <div className="flex items-center gap-2 mt-1">
              {status.exists ? (
                <>
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <span className="text-base font-semibold text-emerald-300">Cached Locally</span>
                </>
              ) : (
                <>
                  <AlertTriangle className="w-5 h-5 text-amber-400" />
                  <span className="text-base font-semibold text-amber-300">Not Cached</span>
                </>
              )}
            </div>
            <span className="text-xs text-slate-400 mono mt-1">
              Size: {formatSize(status.file_size_bytes)}
            </span>
          </div>

          <div className="p-4 rounded-lg bg-slate-900/60 border border-slate-800/80 flex flex-col gap-1">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Validation State</span>
            <div className="flex items-center gap-2 mt-1">
              {status.validation_report?.is_valid ? (
                <>
                  <ShieldCheck className="w-5 h-5 text-cyan-400" />
                  <span className="text-base font-semibold text-cyan-300">Verified & Valid</span>
                </>
              ) : (
                <>
                  <AlertTriangle className="w-5 h-5 text-slate-500" />
                  <span className="text-base font-semibold text-slate-400">Unverified</span>
                </>
              )}
            </div>
            <span className="text-xs text-slate-400 mono mt-1 truncate">
              {status.validation_report ? `SHA256: ${status.validation_report.data_fingerprint.slice(0, 16)}...` : 'Run validation to inspect'}
            </span>
          </div>

          <div className="p-4 rounded-lg bg-slate-900/60 border border-slate-800/80 flex flex-col gap-1">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Partitions (`KaggleSet`)</span>
            <div className="flex items-center gap-2 mt-1">
              <Layers className="w-5 h-5 text-purple-400" />
              <span className="text-base font-semibold text-purple-300">Official Mapping</span>
            </div>
            <span className="text-xs text-slate-400 mono mt-1">
              t: Train | b: Val | v: Test | u: Holdout
            </span>
          </div>
        </div>
      ) : (
        <div className="text-center py-8 text-slate-400">Loading dataset status...</div>
      )}

      {message && (
        <div className="p-3 rounded-lg bg-cyan-950/40 border border-cyan-500/30 text-cyan-300 text-sm mono flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          {message}
        </div>
      )}

      <div className="flex flex-wrap items-center justify-end gap-3 pt-2 border-t border-slate-800/60">
        <button
          onClick={handleDownload}
          disabled={downloading || validating}
          className="glow-btn text-sm"
        >
          <Download className="w-4 h-4" />
          {downloading ? 'Downloading...' : status?.exists ? 'Re-download Dataset' : 'Download CERN Dataset'}
        </button>
        {status?.exists && (
          <button
            onClick={handleValidate}
            disabled={validating || downloading}
            className="px-4 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-white font-medium text-sm transition-all border border-slate-700 flex items-center gap-2"
          >
            <ShieldCheck className="w-4 h-4 text-cyan-400" />
            {validating ? 'Validating...' : 'Run Scientific Validation'}
          </button>
        )}
      </div>
    </div>
  );
};
