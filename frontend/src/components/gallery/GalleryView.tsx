import React, { useEffect, useState } from 'react';
import { LoadingSkeleton } from '../common/LoadingSkeleton';

export interface GalleryEvent {
  event_id: number;
  features: Record<string, number>;
  signal_probability: number;
  predicted_label: 'signal' | 'background';
  threshold: number;
  gallery_category: 'signal' | 'background' | 'interesting';
  gallery_rank: number;
  selection_method: string;
}

export interface GalleryData {
  events: GalleryEvent[];
  total_count: number;
  categories: {
    signal: number;
    background: number;
    interesting: number;
  };
  selection_method: string;
}

interface GalleryViewProps {
  onSelectEventForDisplay: (eventId: number) => void;
}

export const GalleryView: React.FC<GalleryViewProps> = ({ onSelectEventForDisplay }) => {
  const [data, setData] = useState<GalleryData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'all' | 'signal' | 'background' | 'interesting'>('all');

  useEffect(() => {
    let isMounted = true;
    fetch('/api/v1/events/gallery?model_id=xgboost')
      .then((res) => {
        if (!res.ok) throw new Error(`Gallery API returned HTTP ${res.status}`);
        return res.json();
      })
      .then((json) => {
        if (isMounted) {
          setData(json);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message);
          setLoading(false);
        }
      });
    return () => {
      isMounted = false;
    };
  }, []);

  if (loading) {
    return (
      <div className="p-8 max-w-7xl mx-auto space-y-6">
        <LoadingSkeleton height="h-32" />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <LoadingSkeleton key={i} height="h-48" />
          ))}
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-8 max-w-xl mx-auto text-center bg-red-950/40 border border-red-800/60 rounded-2xl text-red-200 mt-12">
        <span className="text-3xl block mb-2">⚠️</span>
        <h3 className="font-semibold text-lg text-red-300 mb-1">Failed to Load Gallery</h3>
        <p className="text-xs text-red-300/80">{error || "Gallery data unavailable"}</p>
      </div>
    );
  }

  const filteredEvents = activeTab === 'all'
    ? data.events
    : data.events.filter((e) => e.gallery_category === activeTab);

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header Banner */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-cyan-400 text-xs font-semibold uppercase tracking-wider mb-1">
            <span>🖼️ Curated Event Gallery</span>
            <span className="px-2 py-0.5 bg-cyan-950 border border-cyan-800 text-cyan-300 rounded-full font-mono text-[10px]">
              {data.total_count} Events Scored
            </span>
          </div>
          <h1 className="text-xl font-bold text-slate-100">ATLAS Collision Event Gallery</h1>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Explore curated high-confidence signal events ($H \to \tau\tau$), background events ($Z \to \tau\tau$, $t\bar&#123;t&#125;$), and near-threshold interesting cases from the test split.
          </p>
        </div>

        {/* Category Filters */}
        <div className="flex bg-slate-950 p-1 rounded-xl border border-slate-800">
          {(['all', 'signal', 'background', 'interesting'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all capitalize ${
                activeTab === tab
                  ? 'bg-cyan-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {tab} ({tab === 'all' ? data.total_count : data.categories[tab] || 0})
            </button>
          ))}
        </div>
      </div>

      {/* Visually Hidden Accessibility Table Fallback */}
      <div className="sr-only">
        <table aria-label="Curated ATLAS Event Gallery Data Table">
          <caption>Scored ATLAS Collision Events by Category</caption>
          <thead>
            <tr>
              <th scope="col">Event ID</th>
              <th scope="col">Category</th>
              <th scope="col">Signal Probability</th>
              <th scope="col">Predicted Label</th>
            </tr>
          </thead>
          <tbody>
            {data.events.map((e) => (
              <tr key={e.event_id}>
                <td>{e.event_id}</td>
                <td>{e.gallery_category}</td>
                <td>{(e.signal_probability * 100).toFixed(1)}%</td>
                <td>{e.predicted_label}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Event Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredEvents.map((ev) => {
          const isSignal = ev.gallery_category === 'signal';
          const isInteresting = ev.gallery_category === 'interesting';
          const cardBorder = isSignal
            ? 'border-cyan-800/60 bg-cyan-950/20'
            : isInteresting
            ? 'border-amber-800/60 bg-amber-950/20'
            : 'border-orange-900/40 bg-slate-900/60';

          const badgeColor = isSignal
            ? 'bg-cyan-950 text-cyan-300 border-cyan-800'
            : isInteresting
            ? 'bg-amber-950 text-amber-300 border-amber-800'
            : 'bg-orange-950 text-orange-300 border-orange-800';

          return (
            <div
              key={ev.event_id}
              className={`border rounded-2xl p-5 shadow-lg flex flex-col justify-between transition-all hover:scale-[1.01] hover:border-cyan-500/50 ${cardBorder}`}
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="font-mono font-bold text-slate-200 text-sm">
                    Event #{ev.event_id}
                  </span>
                  <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-semibold uppercase border ${badgeColor}`}>
                    {ev.gallery_category} #{ev.gallery_rank}
                  </span>
                </div>

                {/* Probability Gauge & SVG Chart with aria-label */}
                <div className="my-3 space-y-1.5">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-400">Signal Probability:</span>
                    <span className={`font-mono font-bold ${isSignal ? 'text-cyan-300' : 'text-orange-300'}`}>
                      {(ev.signal_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div
                    role="img"
                    aria-label={`Probability gauge for Event ${ev.event_id}: ${(ev.signal_probability * 100).toFixed(1)}%`}
                    className="w-full bg-slate-950 h-3 rounded-full overflow-hidden p-0.5 border border-slate-800"
                  >
                    <div
                      className={`h-full rounded-full transition-all ${
                        isSignal ? 'bg-cyan-400' : isInteresting ? 'bg-amber-400' : 'bg-orange-500'
                      }`}
                      style={{ width: `${Math.min(100, ev.signal_probability * 100)}%` }}
                    />
                  </div>
                </div>

                {/* Key Physics Features Summary */}
                <div className="grid grid-cols-2 gap-2 my-4 text-[11px] font-mono bg-slate-950/80 p-2.5 rounded-xl border border-slate-800/80">
                  <div>
                    <span className="text-slate-500 block text-[10px]">DER_mass_MMC</span>
                    <span className="text-slate-300 font-semibold">
                      {ev.features.DER_mass_MMC !== -999.0 ? `${ev.features.DER_mass_MMC.toFixed(1)} GeV` : 'sentinel'}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block text-[10px]">DER_met_pt</span>
                    <span className="text-slate-300 font-semibold">
                      {ev.features.DER_met_pt.toFixed(1)} GeV
                    </span>
                  </div>
                </div>
              </div>

              {/* View in 3D Action Button */}
              <button
                onClick={() => onSelectEventForDisplay(ev.event_id)}
                className="w-full py-2.5 mt-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl font-medium text-xs shadow-md transition-all flex items-center justify-center gap-2 group"
              >
                <span>View Event in 3D</span>
                <span className="group-hover:translate-x-0.5 transition-transform">→</span>
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};
