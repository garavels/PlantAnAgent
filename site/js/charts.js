const NS = "http://www.w3.org/2000/svg";

export function el(tag, attrs = {}, children = []) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== undefined && value !== null) node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child == null) continue;
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

export const scale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);

function path(values, x, y, step = 1) {
  let d = "";
  values.forEach((v, i) => {
    if (v == null) return;
    d += `${d ? "L" : "M"}${x(i * step).toFixed(1)},${y(v).toFixed(1)}`;
  });
  return d;
}

/** Week-long time-series chart. Series values are sampled every `step` hours. */
export function weekChart({ series, yMax, yLabel, step = 0.25, height = 220, bands = [], ticks = [], format = (v) => v }) {
  const W = 680, H = height, m = { l: 44, r: 10, t: 10, b: 24 };
  const x = scale(0, 168, m.l, W - m.r);
  const y = scale(0, yMax, H - m.b, m.t);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": yLabel });
  for (const band of bands) {
    svg.append(el("rect", { x: x(band.from), y: m.t, width: x(band.to) - x(band.from), height: H - m.b - m.t, fill: band.color ?? "rgb(224 122 95 / .12)" }));
    if (band.label) svg.append(el("text", { x: x(band.from) + 4, y: m.t + 12, style: "fill: var(--bad)" }, band.label));
  }
  for (let d = 0; d <= 7; d++) {
    svg.append(el("line", { x1: x(d * 24), x2: x(d * 24), y1: m.t, y2: H - m.b, stroke: "rgb(236 235 226 / .08)" }));
    if (d < 7) svg.append(el("text", { x: x(d * 24 + 12), y: H - 8, "text-anchor": "middle" }, `D${d + 1}`));
  }
  for (const t of [0, 0.5, 1]) {
    svg.append(el("line", { x1: m.l, x2: W - m.r, y1: y(yMax * t), y2: y(yMax * t), stroke: "rgb(236 235 226 / .08)" }));
    svg.append(el("text", { x: m.l - 6, y: y(yMax * t) + 3, "text-anchor": "end" }, format(yMax * t)));
  }
  for (const tick of ticks) {
    svg.append(el("line", { x1: x(tick.t), x2: x(tick.t), y1: H - m.b, y2: H - m.b - 6, stroke: tick.color, "stroke-width": 2 }));
  }
  for (const s of series) {
    if (s.area) svg.append(el("path", { d: `${path(s.values, x, y, step)}L${x((s.values.length - 1) * step)},${y(0)}L${x(0)},${y(0)}Z`, fill: s.color, opacity: s.opacity ?? 0.25 }));
    else svg.append(el("path", { d: path(s.values, x, y, step), fill: "none", stroke: s.color, "stroke-width": s.width ?? 1.6, "stroke-dasharray": s.dash, opacity: s.opacity }));
  }
  return svg;
}

/** Grouped horizontal-axis bar chart: groups along x, one bar per series. */
export function groupedBars({ groups, series, yMax, height = 240, format = (v) => v, yLabel }) {
  const W = 680, H = height, m = { l: 48, r: 10, t: 12, b: 30 };
  const y = scale(0, yMax, H - m.b, m.t);
  const gw = (W - m.l - m.r) / groups.length;
  const bw = Math.min(series.length === 1 ? 44 : 16, (gw - 14) / series.length);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": yLabel });
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(el("line", { x1: m.l, x2: W - m.r, y1: y(yMax * t), y2: y(yMax * t), stroke: "rgb(236 235 226 / .08)" }));
    svg.append(el("text", { x: m.l - 6, y: y(yMax * t) + 3, "text-anchor": "end" }, format(yMax * t)));
  }
  groups.forEach((group, gi) => {
    const x0 = m.l + gi * gw + (gw - bw * series.length) / 2;
    svg.append(el("text", { x: m.l + gi * gw + gw / 2, y: H - 10, "text-anchor": "middle" }, group.label));
    series.forEach((s, si) => {
      const v = s.values[gi] ?? 0;
      svg.append(el("rect", { x: x0 + si * bw, y: y(Math.min(v, yMax)), width: bw - 2, height: Math.max(0.5, y(0) - y(Math.min(v, yMax))), fill: s.color, rx: 1 },
        el("title", {}, `${s.label} · ${group.label}: ${format(v)}`)));
    });
  });
  return svg;
}

/** Horizontal interval plot: one row per item, point estimate with a CI whisker. */
export function intervals({ rows, min, max, height, format = (v) => v, zeroLabel = "0" }) {
  const W = 680, rowH = 34, H = height ?? rows.length * rowH + 34, m = { l: 210, r: 20, t: 8, b: 24 };
  const x = scale(min, max, m.l, W - m.r);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  svg.append(el("line", { x1: x(0), x2: x(0), y1: m.t, y2: H - m.b, stroke: "rgb(236 235 226 / .35)" }));
  svg.append(el("text", { x: x(0), y: H - 8, "text-anchor": "middle" }, zeroLabel));
  for (const t of [min, max]) svg.append(el("text", { x: x(t), y: H - 8, "text-anchor": t === min ? "start" : "end" }, format(t)));
  rows.forEach((row, i) => {
    const cy = m.t + i * rowH + rowH / 2;
    svg.append(el("text", { x: 0, y: cy + 4, style: row.strong ? "fill: var(--ink)" : "" }, row.label));
    if (row.interval) svg.append(el("line", { x1: x(row.interval[0]), x2: x(row.interval[1]), y1: cy, y2: cy, stroke: row.color, "stroke-width": 2 }));
    svg.append(el("circle", { cx: x(row.value), cy, r: 5, fill: row.color }));
    svg.append(el("text", { x: x(row.value), y: cy - 9, "text-anchor": "middle", style: "fill: var(--ink)" }, format(row.value)));
  });
  return svg;
}
