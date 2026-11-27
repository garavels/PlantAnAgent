export const fmt = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });
export const fmt1 = new Intl.NumberFormat("en-GB", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
export const eur = (v) => `${v < 0 ? "−" : ""}€${fmt.format(Math.abs(v))}`;
export const signedEur = (v) => `${v > 0 ? "+" : v < 0 ? "−" : ""}€${fmt.format(Math.abs(v))}`;
export const kg = (v, digits = 0) => `${(digits ? fmt1 : fmt).format(v)} kg`;
export const pct = (v) => `${fmt1.format(v * 100)}%`;
export const REPO = "https://github.com/garavels/PlantAnAgent";
export const PAPER = "./paper/plantanagent.pdf";

export const CONDITIONS = {
  ordinary: { label: "Ordinary week", short: "Ordinary", note: "variable weather and random air-capture faults" },
  control: { label: "No faults", short: "No faults", note: "same weather, no equipment faults" },
  pv6: { label: "Solar out 6 h", short: "Solar −6 h", outage: "solar" },
  pv18: { label: "Solar out 18 h", short: "Solar −18 h", outage: "solar" },
  pv36: { label: "Solar out 36 h", short: "Solar −36 h", outage: "solar" },
  meth24: { label: "Methanator down 24 h", short: "Meth −24 h", outage: "methanator" },
  meth48: { label: "Methanator down 48 h", short: "Meth −48 h", outage: "methanator" },
};
export const STRESS = ["pv6", "pv18", "pv36", "meth24", "meth48"];
export const MIXTURE = { ordinary: 0.5, pv6: 0.1, pv18: 0.1, pv36: 0.1, meth24: 0.1, meth48: 0.1 };

// Names follow Table 1 of the paper.
export const CONTROLLERS = [
  { id: "original-daylight", label: "Daylight", color: "var(--solar)" },
  { id: "original-storage", label: "Storage rule", color: "var(--rule-ctl)" },
  { id: "storage-m48-b60-r3", label: "Tuned storage", color: "var(--ink)" },
  { id: "imitation-parent-903", label: "Imitated neural", color: "var(--co2)" },
  { id: "refined", label: "Refined neural", color: "var(--learned)" },
];
export const WEEK_LABELS = {
  "cem-ridge-bias-g16-5901": "Refined neural",
  "storage-m48-b60-r3": "Tuned storage",
  "original-daylight": "Daylight",
};
export const WEEK_COLORS = {
  "cem-ridge-bias-g16-5901": "var(--learned)",
  "storage-m48-b60-r3": "var(--ink)",
  "original-daylight": "var(--solar)",
};

const cache = new Map();
export function load(path) {
  if (!cache.has(path)) cache.set(path, fetch(path).then((r) => (r.ok ? r.json() : null)).catch(() => null));
  return cache.get(path);
}
export const evaluation = () => load("./data/evaluation.json");
export const plant = () => load("./data/plant.json");
export const week = (condition) => load(`./data/weeks/${condition}.json`);

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const worstQuarter = (xs) => mean([...xs].sort((a, b) => b - a).slice(0, Math.ceil(xs.length / 4)));

/** Table-1 style summary. The refined row is the mean of its three separately operated policies. */
export function tabulate(data) {
  const byController = {};
  for (const row of data.rows) (byController[row.controller] ??= []).push(row);
  const summarise = (rows) => {
    const at = (condition) => rows.filter((r) => r.condition === condition);
    const stress = rows.filter((r) => STRESS.includes(r.condition));
    const conditions = {};
    for (const c of Object.keys(CONDITIONS)) {
      const xs = at(c);
      if (xs.length) conditions[c] = { unmet: mean(xs.map((r) => r.unmet_kg)), contribution: mean(xs.map((r) => r.contribution_eur)), allMet: mean(xs.map((r) => +r.all_deliveries)), missed: mean(xs.map((r) => r.missed)) };
    }
    return {
      conditions,
      ordinary: conditions.ordinary.contribution,
      mixture: Object.entries(MIXTURE).reduce((sum, [c, w]) => sum + w * conditions[c].contribution, 0),
      stressUnmet: mean(STRESS.map((c) => conditions[c].unmet)),
      stressSuccess: mean(STRESS.map((c) => conditions[c].allMet)),
      stressMissed: mean(stress.map((r) => r.missed)),
      stressWorstQuarter: worstQuarter(stress.map((r) => r.unmet_kg)),
    };
  };
  const out = {};
  for (const c of CONTROLLERS) {
    if (c.id !== "refined") { out[c.id] = summarise(byController[c.id]); continue; }
    const members = data.primary_claim.members.map((m) => summarise(byController[m]));
    const avg = (f) => mean(members.map(f));
    out.refined = {
      members,
      conditions: Object.fromEntries(Object.keys(members[0].conditions).map((k) => [k, Object.fromEntries(
        Object.keys(members[0].conditions[k]).map((f) => [f, avg((m) => m.conditions[k][f])]))])),
      ...Object.fromEntries(["ordinary", "mixture", "stressUnmet", "stressSuccess", "stressMissed", "stressWorstQuarter"].map((f) => [f, avg((m) => m[f])])),
    };
  }
  return out;
}

export const html = (strings, ...values) => strings.reduce((acc, s, i) => acc + s + (values[i] ?? ""), "");
export function block(markup) {
  const div = document.createElement("div");
  div.className = "prose";
  div.innerHTML = markup;
  return div;
}
export function figure(svg, caption, legend) {
  const fig = document.createElement("figure");
  fig.className = "fig";
  if (legend) {
    const l = document.createElement("div");
    l.className = "legend";
    l.innerHTML = legend.map((x) => `<span style="--c:${x.color}"><i></i>${x.label}</span>`).join("");
    fig.append(l);
  }
  fig.append(svg);
  if (caption) {
    const c = document.createElement("figcaption");
    c.innerHTML = caption;
    fig.append(c);
  }
  return fig;
}
