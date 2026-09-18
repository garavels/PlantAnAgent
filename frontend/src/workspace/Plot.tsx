import { useEffect, useId, useRef, useState } from "react";
import { n } from "./format";
import { plotValueAt } from "./plotModel";
export interface Series {
  label: string;
  color: string;
  points: { x: number; y: number }[];
  step?: boolean;
  dashed?: boolean;
}
export function Plot({
  title,
  unit,
  series,
  domain,
  yDomain,
  valueDigits,
  showPoints = false,
  cursor,
  onSelect,
  band,
  markers = [],
  markerLabel = "Delivery settlement",
  compact = false,
  xLabel = "Elapsed hours",
}: {
  title: string;
  unit: string;
  series: Series[];
  domain?: [number, number];
  yDomain?: [number, number];
  valueDigits?: number;
  showPoints?: boolean;
  cursor?: number;
  onSelect?: (x: number) => void;
  band?: { x: number; low: number; high: number }[];
  markers?: { x: number; label: string; color?: string }[];
  markerLabel?: string;
  compact?: boolean;
  xLabel?: string;
}) {
  const box = useRef<HTMLElement>(null),
    id = useId();
  const [width, setWidth] = useState(600),
    [hover, setHover] = useState<number | null>(null);
  useEffect(() => {
    if (!box.current) return;
    const o = new ResizeObserver((e) =>
      setWidth(Math.max(240, e[0].contentRect.width)),
    );
    o.observe(box.current);
    return () => o.disconnect();
  }, []);
  const pts = series.flatMap((s) => s.points);
  const start = domain?.[0] ?? Math.min(0, ...pts.map((p) => p.x));
  const end = Math.max(
    start + 0.25,
    domain?.[1] ?? Math.max(1, ...pts.map((p) => p.x)),
  );
  const validYDomain =
    yDomain &&
    Number.isFinite(yDomain[0]) &&
    Number.isFinite(yDomain[1]) &&
    yDomain[0] < yDomain[1];
  const lo = validYDomain
    ? yDomain[0]
    : Math.min(0, ...pts.map((p) => p.y), ...(band?.map((p) => p.low) ?? []));
  const hi = validYDomain
    ? yDomain[1]
    : Math.max(1, ...pts.map((p) => p.y), ...(band?.map((p) => p.high) ?? []));
  const bottom = compact ? 115 : 175,
    right = width - 15,
    left = 62;
  const x = (v: number) =>
    left + ((v - start) / (end - start)) * (right - left);
  const y = (v: number) => bottom - ((v - lo) / (hi - lo)) * (bottom - 25);
  const valueLabel = (value: number, defaultDigits: number) =>
    valueDigits !== undefined &&
    value !== 0 &&
    Math.abs(value) < 0.5 * 10 ** -valueDigits
      ? value.toExponential(1)
      : n(value, valueDigits ?? defaultDigits);
  const path = (p: Series["points"], step = false) =>
    p
      .map((v, i) =>
        i === 0
          ? `M${x(v.x)},${y(v.y)}`
          : step
            ? `H${x(v.x)}V${y(v.y)}`
            : `L${x(v.x)},${y(v.y)}`,
      )
      .join(" ");
  const current = hover ?? cursor;
  const values = series.map((s) => ({
    ...s,
    value: plotValueAt(s.points, current ?? start, s.step),
  }));
  return (
    <figure ref={box} className="w-plot">
      <figcaption>
        <strong>{title}</strong>
        <span>{unit}</span>
      </figcaption>
      <svg
        viewBox={`0 0 ${width} ${bottom + 48}`}
        role={markers.length > 0 && onSelect ? "group" : "img"}
        aria-label={`${title}. ${unit}; ${xLabel}.`}
        onPointerMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          setHover(
            Math.max(
              start,
              Math.min(
                end,
                start +
                  ((((e.clientX - r.left) / r.width) * width - left) /
                    (right - left)) *
                    (end - start),
              ),
            ),
          );
        }}
        onPointerLeave={() => setHover(null)}
        onClick={() => {
          if (hover !== null) onSelect?.(hover);
        }}
      >
        <defs>
          <clipPath id={id}>
            <rect x={left} y={15} width={right - left} height={bottom - 14} />
          </clipPath>
        </defs>
        {[lo, (lo + hi) / 2, hi].map((v, i) => (
          <g key={i}>
            <line
              x1={left}
              x2={right}
              y1={y(v)}
              y2={y(v)}
              className="w-gridline"
            />
            <text x={left - 9} y={y(v) + 4} textAnchor="end">
              {valueLabel(v, 0)}
            </text>
          </g>
        ))}
        <g clipPath={`url(#${id})`}>
          {markers.map((marker) => (
            <g key={`${marker.x}:${marker.label}`}>
              <line
                x1={x(marker.x)}
                x2={x(marker.x)}
                y1={25}
                y2={bottom}
                stroke={marker.color ?? "var(--ch4)"}
                opacity=".3"
                strokeDasharray="3 4"
              />
            </g>
          ))}
          {band?.length ? (
            <path
              d={`${path(band.map((p) => ({ x: p.x, y: p.high })))} ${band
                .slice()
                .reverse()
                .map((p) => `L${x(p.x)},${y(p.low)}`)
                .join(" ")} Z`}
              fill="var(--co2)"
              opacity=".16"
            />
          ) : null}
          {series.map((s) => (
            <g key={s.label}>
              <path
                d={path(s.points, s.step)}
                fill="none"
                stroke={s.color}
                strokeWidth="2"
                strokeDasharray={s.dashed ? "5 4" : undefined}
              />
              {(showPoints || s.points.length === 1) &&
                s.points.map((point, index) => (
                  <circle
                    key={index}
                    cx={x(point.x)}
                    cy={y(point.y)}
                    r={showPoints ? 3 : 4}
                    fill={s.color}
                  />
                ))}
            </g>
          ))}
          {current !== undefined && current !== null && (
            <line
              x1={x(current)}
              x2={x(current)}
              y1={15}
              y2={bottom}
              stroke="var(--text)"
              opacity=".6"
              strokeDasharray="3 3"
            />
          )}
        </g>
        {markers.map((marker) => (
          <g
            key={`${marker.x}:${marker.label}`}
            className="w-plot-marker"
            role={onSelect ? "button" : undefined}
            tabIndex={onSelect ? 0 : undefined}
            aria-label={marker.label}
            style={{ cursor: onSelect ? "pointer" : undefined }}
            onClick={(event) => {
              event.stopPropagation();
              onSelect?.(marker.x);
            }}
            onKeyDown={(event) => {
              if (onSelect && (event.key === "Enter" || event.key === " ")) {
                event.preventDefault();
                event.stopPropagation();
                onSelect(marker.x);
              }
            }}
          >
            <title>{marker.label}</title>
            <rect
              x={x(marker.x) - 12}
              y={3}
              width={24}
              height={24}
              fill="transparent"
            />
            <circle
              cx={x(marker.x)}
              cy={15}
              r={4}
              fill={marker.color ?? "var(--ch4)"}
            />
          </g>
        ))}
        {[start, (start + end) / 2, end].map((v, i) => (
          <text
            key={i}
            x={x(v)}
            y={bottom + 23}
            textAnchor={i === 0 ? "start" : i === 2 ? "end" : "middle"}
          >
            {n(v, end - start <= 1 ? 2 : 1)}
          </text>
        ))}
        <text x={right} y={bottom + 43} textAnchor="end">
          {xLabel}
        </text>
      </svg>
      <div className="w-legend">
        {values.map((s) => (
          <span key={s.label}>
            <i style={{ background: s.color }} />
            {s.label}
            {hover !== null && s.value
              ? ` ${valueLabel(s.value.y, 1)} ${unit}`
              : ""}
          </span>
        ))}
        {band && <span>p10–p90 range</span>}
        {markers.length > 0 && (
          <span>
            <i style={{ background: "var(--ch4)" }} />
            {markerLabel}
          </span>
        )}
      </div>
    </figure>
  );
}
