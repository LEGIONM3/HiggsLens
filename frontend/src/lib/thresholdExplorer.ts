/**
 * Pure TypeScript utility for nearest-point threshold snapping and metric normalization on stored scan data.
 * Driven exclusively by stored artifact points; zero client-side metric calculation.
 */

export interface ThresholdScanPoint {
  threshold: number;
  ams: number;
  tpr?: number;
  fpr?: number;
  precision?: number;
  recall?: number;
}

export interface ThresholdExplorerState {
  selectedThreshold: number;
  nearestPoint: ThresholdScanPoint;
  nearestIndex: number;
}

/**
 * Finds the nearest stored scan point to a requested threshold value.
 */
export function findNearestScanPoint(
  points: ThresholdScanPoint[],
  targetThreshold: number
): { nearestPoint: ThresholdScanPoint; nearestIndex: number } {
  if (!points || points.length === 0) {
    throw new Error("Threshold scan points array cannot be empty.");
  }

  let minDiff = Infinity;
  let nearestIdx = 0;

  for (let i = 0; i < points.length; i++) {
    const diff = Math.abs(points[i].threshold - targetThreshold);
    if (diff < minDiff) {
      minDiff = diff;
      nearestIdx = i;
    }
  }

  return {
    nearestPoint: points[nearestIdx],
    nearestIndex: nearestIdx,
  };
}

/**
 * Formats a metric value to 4 decimal places safely.
 */
export function formatMetricValue(val: number | undefined | null, fallback = "N/A"): string {
  if (val === undefined || val === null || isNaN(val)) {
    return fallback;
  }
  return val.toFixed(4);
}
