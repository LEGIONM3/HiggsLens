import { describe, expect, it } from "vitest";
import { findNearestScanPoint, formatMetricValue, type ThresholdScanPoint } from "./thresholdExplorer";

describe("thresholdExplorer pure logic", () => {
  const samplePoints: ThresholdScanPoint[] = [
    { threshold: 0.1, ams: 1.2 },
    { threshold: 0.3, ams: 2.1 },
    { threshold: 0.5, ams: 3.4 },
    { threshold: 0.8118, ams: 3.5754 },
    { threshold: 0.9, ams: 2.8 },
  ];

  it("snaps to exact target when present", () => {
    const { nearestPoint, nearestIndex } = findNearestScanPoint(samplePoints, 0.5);
    expect(nearestPoint.threshold).toBe(0.5);
    expect(nearestPoint.ams).toBe(3.4);
    expect(nearestIndex).toBe(2);
  });

  it("snaps to nearest stored point when target is between points", () => {
    const { nearestPoint } = findNearestScanPoint(samplePoints, 0.79);
    expect(nearestPoint.threshold).toBe(0.8118);
    expect(nearestPoint.ams).toBe(3.5754);
  });

  it("handles boundary thresholds cleanly", () => {
    const { nearestPoint: minPoint } = findNearestScanPoint(samplePoints, 0.0);
    expect(minPoint.threshold).toBe(0.1);

    const { nearestPoint: maxPoint } = findNearestScanPoint(samplePoints, 1.0);
    expect(maxPoint.threshold).toBe(0.9);
  });

  it("formats metric values cleanly", () => {
    expect(formatMetricValue(0.912345)).toBe("0.9123");
    expect(formatMetricValue(null)).toBe("N/A");
  });
});
