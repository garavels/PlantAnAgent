import { el, scale } from "./charts.js";
import { startRouter } from "./sheets.js";

const fmt = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });
const HOURS = 168;
const START_HOUR = 6;

const week = await fetch("./data/featured-week.json").then((r) => r.json());
const s = week.series;
const cap = week.capacities;

const clockLabel = (t) => {
  const h = t + START_HOUR;
  const minutes = Math.floor((h % 24) * 60 + 1e-6);
  return `Day ${Math.min(7, Math.floor(h / 24) + 1)} · ${String(Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}`;
};

function sample(key, t) {
  const f = Math.min(t * 4, s[key].length - 1);
  const i = Math.floor(f);
  const a = s[key][i], b = s[key][Math.min(i + 1, s[key].length - 1)];
  return a + (b - a) * (f - i);
}

const meters = [
  { key: "solar_potential_kw", name: "Sunlight", max: cap.solar_peak_kw, unit: "kW", color: "var(--solar)" },
  { key: "electrolyser_power_kw", name: "Electrolyser", max: cap.electrolyser_kw, unit: "kW", color: "var(--h2)" },
  { key: "dac_power_kw", name: "Air capture", max: cap.dac_kw, unit: "kW", color: "var(--co2)" },
  { key: "methanator_power_kw", name: "Methanator", max: cap.methanator_kw, unit: "kW", color: "var(--ch4)" },
  { sep: "Storage" },
  { key: "battery_soc", name: "Battery", max: 1, unit: "%", color: "var(--stored)", scale: 100 },
  { key: "h2_level_kg", name: "Hydrogen", max: cap.h2_kg, unit: "kg", color: "var(--h2)" },
  { key: "co2_level_kg", name: "CO₂", max: cap.co2_kg, unit: "kg", color: "var(--co2)" },
  { key: "ch4_level_kg", name: "Methane", max: cap.ch4_kg, unit: "kg", color: "var(--ch4)" },
];

const $ = (selector) => document.querySelector(selector);
const meterList = $("[data-meters]");
const meterRefs = meters.map((m) => {
  const li = document.createElement("li");
  if (m.sep) {
    li.className = "sep";
    li.innerHTML = `<p class="kicker">${m.sep}</p>`;
    meterList.append(li);
    return null;
  }
  li.style.setProperty("--c", m.color);
  li.innerHTML = `<span class="name">${m.name}</span><span class="bar"><span></span></span><span class="val"></span>`;
  meterList.append(li);
  return { ...m, bar: li.querySelector(".bar > span"), val: li.querySelector(".val") };
}).filter(Boolean);

const trucks = $("[data-trucks]");
const truckRefs = week.deliveries.map((d) => {
  const span = document.createElement("span");
  span.title = `${clockLabel(d.due_h)} · ${d.requested_kg} kg`;
  trucks.append(span);
  return span;
});

// Flow diagram -------------------------------------------------------------
const flow = $("[data-flow]");
const node = (x, y, w, h, label, sub) => el("g", { class: "node" }, [
  el("rect", { x, y, width: w, height: h, rx: 6 }),
  el("text", { x: x + 8, y: y + 15 }, label),
  sub ? el("text", { x: x + 8, y: y + h - 8, style: "fill: var(--ink-3)" }, sub) : null,
]);
const tank = (x, y, label, color) => {
  const level = el("rect", { class: "level", x: x + 1, y: y + 59, width: 46, height: 0, style: `fill: ${color}`, opacity: 0.85, rx: 3 });
  flow.append(el("g", { class: "node" }, [el("rect", { x, y, width: 48, height: 60, rx: 6 }), level, el("text", { x: x + 24, y: y - 5, "text-anchor": "middle" }, label)]));
  return { level, y, h: 58 };
};
const pipes = {};
const pipe = (id, d, color) => {
  pipes[id] = el("path", { class: "pipe", d, stroke: color });
  flow.append(pipes[id]);
};
pipe("sun-el", "M96,36 C130,36 130,36 158,36", "var(--solar)");
pipe("sun-dac", "M96,44 C128,60 128,160 158,170", "var(--solar)");
pipe("sun-bat", "M48,58 L48,146", "var(--solar)");
pipe("bat-el", "M96,166 C128,150 128,56 158,46", "var(--stored)");
pipe("bat-dac", "M96,176 L158,176", "var(--stored)");
pipe("el-h2", "M256,36 L290,36", "var(--h2)");
pipe("dac-co2", "M256,176 L290,176", "var(--co2)");
pipe("h2-me", "M338,36 C360,36 356,92 374,100", "var(--h2)");
pipe("co2-me", "M338,176 C360,176 356,120 374,112", "var(--co2)");
pipe("me-ch4", "M450,106 L462,106", "var(--ch4)");
pipe("ch4-truck", "M486,140 L486,176", "var(--ch4)");
flow.append(node(10, 14, 86, 44, "SUN", "5 MWp PV"));
flow.append(node(10, 146, 86, 48, "BATTERY", "8 MWh"));
flow.append(node(158, 14, 98, 44, "ELECTROLYSER", "H₂O → H₂"));
flow.append(node(158, 154, 98, 44, "AIR CAPTURE", "air → CO₂"));
flow.append(node(374, 84, 76, 44, "METHANATOR", "CO₂+4H₂"));
const tanks = {
  h2: tank(290, 16, "H₂", "var(--h2)"),
  co2: tank(290, 146, "CO₂", "var(--co2)"),
  ch4: tank(462, 78, "CH₄", "var(--ch4)"),
};
flow.append(el("text", { x: 486, y: 192, "text-anchor": "middle" }, "TRUCK"));
const truckLabel = el("text", { x: 486, y: 204, "text-anchor": "middle", style: "fill: var(--ch4)" }, "");
flow.append(truckLabel);

// Timeline -------------------------------------------------------------------
const timeline = $("[data-timeline]");
const TW = 520, TH = 62;
const tx = scale(0, HOURS, 0, TW);
const tsvg = el("svg", { viewBox: `0 0 ${TW} ${TH}`, preserveAspectRatio: "none", "aria-hidden": "true" });
const area = (key, max, color, opacity) => {
  let d = `M0,${TH - 12}`;
  s[key].forEach((v, i) => { d += `L${tx(i / 4).toFixed(1)},${(TH - 12 - (v / max) * (TH - 18)).toFixed(1)}`; });
  return el("path", { d: `${d}L${TW},${TH - 12}Z`, fill: color, opacity });
};
tsvg.append(area("solar_potential_kw", cap.solar_peak_kw, "var(--solar)", 0.22));
let ch4Line = "";
s.ch4_level_kg.forEach((v, i) => { ch4Line += `${i ? "L" : "M"}${tx(i / 4).toFixed(1)},${(TH - 12 - (v / 1000) * (TH - 18)).toFixed(1)}`; });
tsvg.append(el("path", { d: ch4Line, fill: "none", stroke: "var(--ch4)", "stroke-width": 1.4, "vector-effect": "non-scaling-stroke" }));
for (let d = 1; d < 7; d++) tsvg.append(el("line", { x1: tx(d * 24 - START_HOUR), x2: tx(d * 24 - START_HOUR), y1: 0, y2: TH - 12, stroke: "rgb(236 235 226 / .12)", "vector-effect": "non-scaling-stroke" }));
const tickRefs = week.deliveries.map((d) => {
  const tick = el("rect", { x: tx(d.due_h) - 1, y: TH - 8, width: 2, height: 8, fill: "rgb(236 235 226 / .25)" });
  tsvg.append(tick);
  return tick;
});
const head = el("line", { x1: 0, x2: 0, y1: 0, y2: TH, stroke: "var(--ink)", "stroke-width": 1.2, "vector-effect": "non-scaling-stroke" });
tsvg.append(head);
timeline.append(tsvg);
timeline.setAttribute("role", "slider");
timeline.setAttribute("aria-label", "Replay position");
timeline.tabIndex = 0;

// Playback ---------------------------------------------------------------------
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
let t = 0, speed = 3, playing = !reduced, last = performance.now(), dash = 0, holdUntil = 0;
const clock = $("[data-clock]"), mode = $("[data-mode]"), eur = $("[data-eur]"), count = $("[data-delivery-count]");
const img = document.querySelector(".art img");
const playButton = $("[data-play]");

function render() {
  const i = Math.min(Math.round(t * 4), s.t_h.length - 1);
  for (const m of meterRefs) {
    const v = sample(m.key, t);
    m.bar.style.width = `${Math.max(0, Math.min(1, v / m.max)) * 100}%`;
    m.val.textContent = `${fmt.format(v * (m.scale ?? 1))} ${m.unit}`;
  }
  clock.textContent = clockLabel(t);
  mode.textContent = s.mode[i].replace("_", " ");
  mode.dataset.state = s.mode[i];
  eur.textContent = `${s.net_value_eur[i] < 0 ? "−" : ""}€${fmt.format(Math.abs(s.net_value_eur[i]))}`;
  let met = 0;
  week.deliveries.forEach((d, k) => {
    const done = t >= d.due_h;
    const ok = done && d.status === "met";
    truckRefs[k].className = done ? (ok ? "met" : "missed") : "";
    tickRefs[k].setAttribute("fill", done ? (ok ? "var(--ch4)" : "var(--bad)") : "rgb(236 235 226 / .25)");
    if (ok) met++;
  });
  count.textContent = `${met} / ${week.deliveries.length}`;
  const sun = sample("solar_potential_kw", t) / cap.solar_peak_kw;
  img.style.filter = `saturate(${0.55 + 0.4 * sun}) brightness(${0.5 + 0.42 * Math.sqrt(sun)})`;
  head.setAttribute("x1", tx(t));
  head.setAttribute("x2", tx(t));
  timeline.setAttribute("aria-valuetext", clockLabel(t));
  for (const [id, key, cap_] of [["h2", "h2_level_kg", cap.h2_kg / 2], ["co2", "co2_level_kg", cap.co2_kg / 2], ["ch4", "ch4_level_kg", 1000]]) {
    const h = Math.min(1, sample(key, t) / cap_) * tanks[id].h;
    tanks[id].level.setAttribute("height", h.toFixed(1));
    tanks[id].level.setAttribute("y", (tanks[id].y + 59 - h).toFixed(1));
  }
  const recent = week.deliveries.find((d) => t >= d.due_h && t < d.due_h + 1);
  truckLabel.textContent = recent ? `−${recent.delivered_kg} kg` : "";
  return recent;
}

function animatePipes(dt) {
  const battery = sample("battery_power_kw", t);
  const charge = Math.max(0, battery), discharge = Math.max(0, -battery);
  const el_ = sample("electrolyser_power_kw", t), dac = sample("dac_power_kw", t), me = sample("methanator_power_kw", t);
  const fromSun = Math.max(0, sample("solar_available_kw", t) - charge);
  const flows = {
    "sun-el": fromSun > 50 ? el_ : 0, "sun-dac": fromSun > 50 ? dac : 0, "sun-bat": charge,
    "bat-el": discharge > 1 ? Math.min(discharge, el_) : 0, "bat-dac": discharge > 1 ? Math.min(discharge, dac) : 0,
    "el-h2": sample("h2_production_kg_h", t) * 50, "dac-co2": sample("co2_capture_kg_h", t) * 8,
    "h2-me": me * 5, "co2-me": me * 5, "me-ch4": sample("ch4_production_kg_h", t) * 20,
    "ch4-truck": render.recent ? 2000 : 0,
  };
  dash -= dt * 0.03;
  for (const [id, value] of Object.entries(flows)) {
    const on = value > 1;
    pipes[id].classList.toggle("on", on);
    if (on) pipes[id].style.strokeDashoffset = (dash * (0.6 + Math.min(1.6, value / 1200))).toFixed(1);
  }
}

function frame(now) {
  const dt = Math.min(100, now - last);
  last = now;
  if (playing) {
    if (t < HOURS) t = Math.min(HOURS, t + (dt / 1000) * speed);
    else if (!holdUntil) holdUntil = now + 2500;
    else if (now > holdUntil) { t = 0; holdUntil = 0; }
  }
  render.recent = render();
  if (!reduced) animatePipes(dt);
  requestAnimationFrame(frame);
}

function seek(event) {
  const box = timeline.getBoundingClientRect();
  holdUntil = 0;
  t = Math.max(0, Math.min(HOURS, ((event.clientX - box.left) / box.width) * HOURS));
}
timeline.addEventListener("pointerdown", (event) => {
  timeline.setPointerCapture(event.pointerId);
  seek(event);
  timeline.addEventListener("pointermove", seek);
});
timeline.addEventListener("pointerup", () => timeline.removeEventListener("pointermove", seek));
timeline.addEventListener("keydown", (event) => {
  const delta = { ArrowLeft: -1, ArrowRight: 1, PageDown: -24, PageUp: 24 }[event.key];
  if (delta) { t = Math.max(0, Math.min(HOURS, t + delta)); event.preventDefault(); }
});
document.querySelectorAll("[data-speed]").forEach((button) => button.addEventListener("click", () => {
  speed = Number(button.dataset.speed);
  document.querySelectorAll("[data-speed]").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
  if (!playing) playButton.click();
}));
const syncPlay = () => { playButton.textContent = playing ? "pause" : "play"; };
playButton.addEventListener("click", () => { playing = !playing; syncPlay(); });
syncPlay();
requestAnimationFrame(frame);

startRouter(week);
