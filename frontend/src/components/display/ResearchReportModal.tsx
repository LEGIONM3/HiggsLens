import React, { useEffect, useState } from 'react';
import { EventAnalysisReport } from '../../types/report';
import { LoadingSkeleton } from '../common/LoadingSkeleton';
import { ReproducibilitySection } from '../leaderboard/ReproducibilitySection';

interface ResearchReportModalProps {
  eventId: number;
  isOpen: boolean;
  onClose: () => void;
}

export const ResearchReportModal: React.FC<ResearchReportModalProps> = ({
  eventId,
  isOpen,
  onClose,
}) => {
  const [report, setReport] = useState<EventAnalysisReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen && eventId) {
      setLoading(true);
      setError(null);
      fetch(`/api/v1/events/${eventId}/report?format=json`)
        .then((res) => {
          if (!res.ok) {
            throw new Error(`Failed to load research report (${res.status})`);
          }
          return res.json();
        })
        .then((data: EventAnalysisReport) => {
          setReport(data);
          setLoading(false);
        })
        .catch((err: any) => {
          setError(err.message || 'Error fetching report');
          setLoading(false);
        });
    }
  }, [isOpen, eventId]);

  if (!isOpen) return null;

  const handleDownloadJson = () => {
    if (!report) return;
    const jsonStr = JSON.stringify(report, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `higgslens-event-${eventId}-report.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handlePrintHtml = () => {
    // Amendment 5: Open canonical HTML directly on user gesture
    window.open(`/api/v1/events/${eventId}/report?format=html`, '_blank');
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4 overflow-y-auto"
      role="dialog"
      aria-modal="true"
      aria-labelledby="report-modal-title"
    >
      <div className="bg-slate-900 border border-slate-700/80 rounded-2xl max-w-4xl w-full p-6 shadow-2xl space-y-6 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div>
            <h2 id="report-modal-title" className="text-xl font-bold text-slate-100 flex items-center gap-2">
              📄 Research Report — Event #{eventId}
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Canonical analysis payload for certified model XGBoost
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 text-2xl font-bold p-1 rounded-lg hover:bg-slate-800 transition-colors"
            aria-label="Close Research Report Modal"
          >
            &times;
          </button>
        </div>

        {loading && <LoadingSkeleton height="h-64" />}

        {error && (
          <div className="p-4 bg-red-950/60 border border-red-800 rounded-xl text-red-300 text-sm">
            ⚠️ {error}
          </div>
        )}

        {report && !loading && (
          <div className="space-y-6">
            {/* Header Action Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 p-4 bg-slate-950/60 border border-slate-800 rounded-xl">
              <div className="text-xs text-slate-400">
                Report Generated At: <strong className="text-slate-200">{new Date(report.generated_at).toUTCString()}</strong>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleDownloadJson}
                  className="px-4 py-2 bg-sky-600 hover:bg-sky-500 text-white font-semibold text-xs rounded-lg shadow transition-colors flex items-center gap-1.5 focus:ring-2 focus:ring-sky-400"
                  aria-label="Download JSON Report"
                >
                  ⬇️ Download JSON
                </button>
                <button
                  onClick={handlePrintHtml}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs rounded-lg shadow transition-colors flex items-center gap-1.5 focus:ring-2 focus:ring-emerald-400"
                  aria-label="Print or Save PDF in Browser"
                >
                  🖨️ Print / Save as PDF in browser
                </button>
              </div>
            </div>

            {/* Classification Card */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
                <div className="text-xs uppercase text-slate-400 font-semibold">Predicted Label</div>
                <div className={`text-lg font-bold mt-1 ${report.classification.predicted_label === 'signal' ? 'text-emerald-400' : 'text-red-400'}`}>
                  {report.classification.predicted_label.toUpperCase()}
                </div>
              </div>
              <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
                <div className="text-xs uppercase text-slate-400 font-semibold">Signal Probability</div>
                <div className="text-lg font-bold text-slate-100 mt-1">
                  {report.classification.signal_probability.toFixed(4)}
                </div>
              </div>
              <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
                <div className="text-xs uppercase text-slate-400 font-semibold">Decision Threshold</div>
                <div className="text-lg font-bold text-slate-100 mt-1">
                  {report.classification.threshold.toFixed(4)}
                </div>
              </div>
              <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
                <div className="text-xs uppercase text-slate-400 font-semibold">TreeSHAP Margin</div>
                <div className="text-lg font-bold text-slate-100 mt-1">
                  {report.explanation.margin >= 0 ? '+' : ''}{report.explanation.margin.toFixed(4)}
                </div>
              </div>
            </div>

            {/* Top 10 Attributions Table */}
            <div>
              <h3 className="text-sm font-semibold text-sky-400 mb-2">Top 10 Feature Attributions (TreeSHAP)</h3>
              <div className="border border-slate-800 rounded-lg overflow-x-auto bg-slate-950/60">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-800/60 text-slate-400 uppercase font-semibold">
                    <tr>
                      <th className="p-2.5">Feature</th>
                      <th className="p-2.5">Value</th>
                      <th className="p-2.5">TreeSHAP Contribution</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {[...report.explanation.attributions]
                      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
                      .slice(0, 10)
                      .map((attr) => (
                        <tr key={attr.feature}>
                          <td className="p-2.5 font-mono text-sky-300">{attr.feature}</td>
                          <td className="p-2.5 text-slate-200">{attr.value.toFixed(4)}</td>
                          <td className={`p-2.5 font-semibold ${attr.contribution >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {attr.contribution >= 0 ? '+' : ''}{attr.contribution.toFixed(6)}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Honesty Note */}
            <div className="p-3 bg-sky-950/30 border-l-4 border-sky-500 rounded-r-lg text-xs text-sky-200">
              ℹ️ <strong>Attribution Honesty Note:</strong> Feature attributions describe how the model reached its score. They are not statements of physical causation.
            </div>

            {/* Provenance Statement */}
            <div className="p-3 bg-slate-950/80 border border-slate-800 rounded-lg text-xs text-slate-400">
              📌 {report.provenance.statement}
            </div>

            {/* Reproducibility Panel */}
            <ReproducibilitySection />
          </div>
        )}
      </div>
    </div>
  );
};
