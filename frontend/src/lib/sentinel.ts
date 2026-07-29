export function isSentinelValue(val: number | null | undefined): boolean {
  if (val === null || val === undefined) return true;
  return Math.abs(val - (-999.0)) < 1e-3 || isNaN(val);
}

export function formatPhysicsValue(val: number | null | undefined, unit = '', decimals = 2): string {
  if (isSentinelValue(val)) {
    return 'N/A (Unmeasured / Multiplicity < Threshold)';
  }
  return `${val!.toFixed(decimals)} ${unit}`.trim();
}

export function countMissingSentinels(features: Record<string, number>): {
  missingCount: number;
  totalFeatures: number;
  missingnessPercentage: number;
} {
  const keys = Object.keys(features);
  if (keys.length === 0) {
    return { missingCount: 0, totalFeatures: 0, missingnessPercentage: 0 };
  }
  let missing = 0;
  keys.forEach((key) => {
    if (isSentinelValue(features[key])) {
      missing++;
    }
  });
  return {
    missingCount: missing,
    totalFeatures: keys.length,
    missingnessPercentage: Math.round((missing / keys.length) * 100),
  };
}
