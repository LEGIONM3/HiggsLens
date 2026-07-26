import React, { useMemo, useState } from "react";
import { findNearestScanPoint, formatMetricValue, type ThresholdScanPoint } from "../../lib/thresholdExplorer";

export interface ThresholdExplorerProps {
  modelId: string;
  officialThreshold: number;
  officialAms: number;
  scanPoints: ThresholdScanPoint[];
  onThresholdSelect?: (point: ThresholdScanPoint) => void;
}

export const ThresholdExplorer: React.FC<ThresholdExplorerProps> = ({
  modelId,
  officialThreshold,
  officialAms,
  scanPoints,
  onThresholdSelect,
}) => {
  const sortedPoints = useMemo(() => {
    return [...scanPoints].sort((a, b) => a.threshold - b.threshold);
  }, [scanPoints]);

  const [selectedThreshold, setSelectedThreshold] = useState<number>(officialThreshold);

  const { nearestPoint } = useMemo(() => {
    if (sortedPoints.length === 0) {
      return {
        nearestPoint: { threshold: officialThreshold, ams: officialAms },
        nearestIndex: 0,
      };
    }
    return findNearestScanPoint(sortedPoints, selectedThreshold);
  }, [sortedPoints, selectedThreshold, officialThreshold, officialAms]);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setSelectedThreshold(val);
    if (onThresholdSelect && sortedPoints.length > 0) {
      const { nearestPoint: point } = findNearestScanPoint(sortedPoints, val);
      onThresholdSelect(point);
    }
  };

  // SVG Chart bounds
  const width = 600;
  const height = 240;
  const padding = 40;

  const minX = 0.0;
  const maxX = 1.0;
  const maxAMS = Math.max(...sortedPoints.map((p) => p.ams), officialAms, 1.0) * 1.1;

  const getSvgCoords = (th: number, ams: number) => {
    const x = padding + ((th - minX) / (maxX - minX)) * (width - 2 * padding);
    const y = height - padding - (ams / maxAMS) * (height - 2 * padding);
    return { x, y };
  };

  const polylinePoints = useMemo(() => {
    return sortedPoints.map((p) => {
      const { x, y } = getSvgCoords(p.threshold, p.ams);
      return `${x},${y}`;
    }).join(" ");
  }, [sortedPoints, maxAMS]);

  const selectedCoords = getSvgCoords(nearestPoint.threshold, nearestPoint.ams);
  const officialCoords = getSvgCoords(officialThreshold, officialAms);

  return (
    <div className="p-5 bg-slate-950/90 border border-slate-800 rounded-xl text-slate-100 space-y-4">
      {/* Title */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div>
          <span className="text-xs uppercase tracking-wider text-cyan-400 font-mono font-semibold">
            Interactive Threshold / AMS Explorer
          </span>
          <h3 className="text-base font-semibold text-slate-200">
            Model Scan: <span className="text-white font-mono">{modelId}</span>
          </h3>
        </div>
        <div className="text-right">
          <div className="text-xs text-slate-400">Current Selected Point</div>
          <div className="text-sm font-bold font-mono text-cyan-300">
            Th: {formatMetricValue(nearestPoint.threshold)} | AMS: {formatMetricValue(nearestPoint.ams)}
          </div>
        </div>
      </div>

      {/* SVG Curve Chart */}
      <div className="w-full overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto bg-slate-900/60 rounded-lg border border-slate-800/80">
          {/* Grid lines */}
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#334155" strokeWidth="1" />
          <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#334155" strokeWidth="1" />

          {/* Stored scan points line */}
          <polyline fill="none" stroke="#06b6d4" strokeWidth="2" points={polylinePoints} />

          {/* Official Frozen Threshold Marker */}
          <g>
            <line
              x1={officialCoords.x}
              y1={padding}
              x2={officialCoords.x}
              y2={height - padding}
              stroke="#f59e0b"
              strokeWidth="1.5"
              strokeDasharray="4 4"
            />
            <circle cx={officialCoords.x} cy={officialCoords.y} r="5" fill="#f59e0b" stroke="#0f172a" strokeWidth="2" />
            <text x={officialCoords.x + 8} y={officialCoords.y - 8} fill="#f59e0b" fontSize="10" className="font-mono font-semibold">
              official frozen threshold (from certified artifact)
            </text>
          </g>

          {/* Draggable/Snapped Cursor Marker */}
          <circle cx={selectedCoords.x} cy={selectedCoords.y} r="7" fill="#38bdf8" stroke="#ffffff" strokeWidth="2" />
        </svg>
      </div>

      {/* Interactive Slider */}
      <div className="space-y-1.5 pt-2">
        <div className="flex justify-between text-xs text-slate-400 font-mono">
          <span>0.0</span>
          <span>Target Threshold Slider (Snaps to Nearest Stored Scan Point): <strong>{selectedThreshold.toFixed(4)}</strong></span>
          <span>1.0</span>
        </div>
        <input
          type="range"
          min="0.0"
          max="1.0"
          step="0.005"
          value={selectedThreshold}
          onChange={handleSliderChange}
          className="w-full accent-cyan-500 bg-slate-800 h-2 rounded-lg cursor-pointer"
        />
      </div>

      {/* Note */}
      <div className="text-[11px] text-slate-400 italic">
        * Explorer operates strictly on stored artifact scan points from metrics.json. No metric recomputation or dataset access occurs on this path.
      </div>
    </div>
  );
};
