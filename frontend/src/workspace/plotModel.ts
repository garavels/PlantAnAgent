interface PlotPoint {
  x: number;
  y: number;
}

/** Step values take effect at their timestamp; later duplicate points win. */
export function plotValueAt(
  points: PlotPoint[],
  x: number,
  step = false,
): PlotPoint | null {
  if (!points.length || !Number.isFinite(x)) return null;
  if (step) {
    return points.reduce<PlotPoint | null>((best, point) => (
      point.x <= x && (!best || point.x >= best.x) ? point : best
    ), null);
  }
  return points.reduce((best, point) => (
    Math.abs(point.x - x) < Math.abs(best.x - x) ? point : best
  ));
}
