import { el, scale } from "./charts.js";
import { fmt, plant, week as loadWeek, CONDITIONS, WEEK_LABELS, WEEK_COLORS } from "./common.js";
import { startRouter } from "./sheets.js";

const HOURS = 168;
const START_HOUR = 6;
const { capacities: cap, conditions, week_controllers: controllers } = await plant();
const params = new URLSearchParams(location.search);
const view = {
  condition: conditions.includes(params.get("week")) ? params.get("week") : "pv36",
  controller: controllers[0],
};
let week = await loadWeek(view.condition);

const clockLabel = (t) => {
  const h = t + START_HOUR;
  const minutes = Math.floor((h % 24) * 60 + 1e-6);
  return `Day ${Math.min(7, Math.floor(h / 24) + 1)} · ${String(Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}`;
};

const series = () => week.controllers[view.controller].series;
function at(values, t) {
  const f = Math.min(t * 4, values.length - 1);
  const i = Math.floor(f);
  const a = values[i], b = values[Math.min(i + 1, values.length - 1)];
  return a + (b - a) * (f - i);
}
const event = () => (week.event.onset_h != null && week.event.end_h > week.event.onset_h ? week.event : null);
const outage = (t) => {
  const e = event();
  return e && t >= e.onset_h && t < e.end_h ? CONDITIONS[week.condition].outage : null;
};
function state(t) {
  const s = series();
  const v = (k) => at(s[k], t);
  return {
    sun: outage(t) === "solar" ? 0 : at(week.solar_potential_kw, t),
    potential: at(week.solar_potential_kw, t),
    electrolyser: v("solar_to_electrolyser_kw") + v("battery_to_electrolyser_kw"),
    dac: v("solar_to_dac_kw") + v("battery_to_dac_kw"),
    methanator: v("solar_to_methanator_kw") + v("battery_to_methanator_kw"),
    soc: v("battery_soc") / 1000,
    h2: v("h2_level_kg"), co2: v("co2_level_kg"), ch4: v("ch4_level_kg"),
    flows: {
      "sun-el": v("solar_to_electrolyser_kw"), "sun-dac": v("solar_to_dac_kw"), "sun-bat": v("solar_to_battery_kw"),
      "sun-me": v("solar_to_methanator_kw"), "bat-el": v("battery_to_electrolyser_kw"), "bat-dac": v("battery_to_dac_kw"),
      "bat-me": v("battery_to_methanator_kw"),
    },
  };
}

const $ = (selector) => document.querySelector(selector);
const meters = [
  { key: "sun", name: "Sunlight", max: cap.solar_peak_kw, unit: "kW", color: "var(--solar)" },
  { key: "electrolyser", name: "Electrolyser", max: cap.electrolyser_rated_kw, unit: "kW", color: "var(--h2)" },
  { key: "dac", name: "Air capture", max: cap.dac_rated_kw, unit: "kW", color: "var(--co2)" },
  { key: "methanator", name: "Methanator", max: cap.methanator_rated_kw, unit: "kW", color: "var(--ch4)" },
  { sep: "Storage" },
  { key: "soc", name: "Battery", max: 1, unit: "%", color: "var(--stored)", scale: 100 },
  { key: "h2", name: "Hydrogen", max: cap.h2_capacity_kg, unit: "kg", color: "var(--h2)" },
  { key: "co2", name: "CO₂", max: cap.co2_capacity_kg, unit: "kg", color: "var(--co2)" },
  { key: "ch4", name: "Methane", max: cap.ch4_capacity_kg, unit: "kg", color: "var(--ch4)" },
];
const meterRefs = meters.map((m) => {
  const li = document.createElement("li");
  $("[data-meters]").append(li);
  if (m.sep) {
    li.className = "sep";
    li.innerHTML = `<p class="kicker">${m.sep}</p>`;
    return null;
  }
  li.style.setProperty("--c", m.color);
  li.innerHTML = `<span class="name">${m.name}</span><span class="bar"><span></span></span><span class="val"></span>`;
  return { ...m, bar: li.querySelector(".bar > span"), val: li.querySelector(".val") };
}).filter(Boolean);

const truckRefs = Array.from({ length: 28 }, () => $("[data-trucks]").appendChild(document.createElement("span")));

// Flow diagram -------------------------------------------------------------------
const flow = $("[data-flow]");
const node = (x, y, w, h, label, sub) => {
  const subText = sub ? el("text", { x: x + 8, y: y + h - 8, class: "sub" }, sub) : null;
  const g = el("g", { class: "node" }, [el("rect", { x, y, width: w, height: h, rx: 6 }), el("text", { x: x + 8, y: y + 15 }, label), subText]);
  flow.append(g);
  return { g, subText, sub };
};
const tank = (x, y, label, color) => {
  const level = el("rect", { class: "level", x: x + 1, y: y + 59, width: 46, height: 0, style: `fill: ${color}`, opacity: 0.85, rx: 3 });
  flow.append(el("g", { class: "node" }, [el("rect", { x, y, width: 48, height: 60, rx: 6 }), level, el("text", { x: x + 24, y: y - 5, "text-anchor": "middle" }, label)]));
  return { level, y, h: 58 };
};
const pipes = {};
const pipe = (id, d, color) => flow.append(pipes[id] = el("path", { class: "pipe", d, stroke: color }));
pipe("sun-el", "M96,36 L158,36", "var(--solar)");
pipe("sun-dac", "M96,44 C128,60 128,160 158,170", "var(--solar)");
pipe("sun-bat", "M48,58 L48,146", "var(--solar)");
pipe("sun-me", "M60,14 C60,0 412,0 412,84", "var(--solar)");
pipe("bat-el", "M96,166 C128,150 128,56 158,46", "var(--stored)");
pipe("bat-dac", "M96,176 L158,176", "var(--stored)");
pipe("bat-me", "M60,194 C60,214 412,214 412,128", "var(--stored)");
pipe("el-h2", "M256,36 L290,36", "var(--h2)");
pipe("dac-co2", "M256,176 L290,176", "var(--co2)");
pipe("h2-me", "M338,36 C360,36 356,92 374,100", "var(--h2)");
pipe("co2-me", "M338,176 C360,176 356,120 374,112", "var(--co2)");
pipe("me-ch4", "M450,106 L462,106", "var(--ch4)");
pipe("ch4-truck", "M486,140 L486,170", "var(--ch4)");
const sunNode = node(10, 14, 86, 44, "SUN", "5 MWp PV");
node(10, 146, 86, 48, "BATTERY", "8 MWh");
node(158, 14, 98, 44, "ELECTROLYSER", "H₂O → H₂");
node(158, 154, 98, 44, "AIR CAPTURE", "air → CO₂");
const methNode = node(374, 84, 76, 44, "METHANATOR", "CO₂+4H₂");
const tanks = { h2: tank(290, 16, "H₂", "var(--h2)"), co2: tank(290, 146, "CO₂", "var(--co2)"), ch4: tank(462, 78, "CH₄", "var(--ch4)") };
flow.append(el("text", { x: 486, y: 182, "text-anchor": "middle" }, "TRUCK"));
const truckLabel = flow.appendChild(el("text", { x: 486, y: 196, "text-anchor": "middle", class: "truck-label" }, ""));

// Condition picker and controller race ------------------------------------------------
const picker = $("[data-conditions]");
for (const c of conditions) {
  const b = picker.appendChild(document.createElement("button"));
  b.type = "button";
  b.dataset.condition = c;
  b.textContent = CONDITIONS[c].short;
  b.title = CONDITIONS[c].label;
  b.addEventListener("click", () => selectWeek(c));
}
const race = $("[data-race]");
const raceRefs = controllers.map((id) => {
  const li = race.appendChild(document.createElement("li"));
  li.style.setProperty("--c", WEEK_COLORS[id]);
  li.innerHTML = `<button type="button"><span class="name">${WEEK_LABELS[id]}</span><span class="bar"><span></span></span><span class="val"></span></button>`;
  li.querySelector("button").addEventListener("click", () => { view.controller = id; rebuild(); });
  return { id, li, bar: li.querySelector(".bar > span"), val: li.querySelector(".val") };
});

// Timeline -------------------------------------------------------------------------
const timeline = $("[data-timeline]");
const TW = 520, TH = 62, BASE = TH - 12;
const tx = scale(0, HOURS, 0, TW);
const tsvg = timeline.appendChild(el("svg", { viewBox: `0 0 ${TW} ${TH}`, preserveAspectRatio: "none", "aria-hidden": "true" }));
const layer = tsvg.appendChild(el("g"));
const head = tsvg.appendChild(el("line", { x1: 0, x2: 0, y1: 0, y2: TH, stroke: "var(--ink)", "stroke-width": 1.2, "vector-effect": "non-scaling-stroke" }));
let tickRefs = [];
timeline.setAttribute("role", "slider");
timeline.setAttribute("aria-label", "Replay position");
timeline.setAttribute("aria-valuemin", "0");
timeline.setAttribute("aria-valuemax", String(HOURS));
timeline.tabIndex = 0;
const line = (values, max) => values.map((v, i) => `${i ? "L" : "M"}${tx(i / 4).toFixed(1)},${(BASE - (v / max) * (BASE - 6)).toFixed(1)}`).join("");

function rebuild() {
  const s = series();
  const e = event();
  const children = [];
  if (e) children.push(el("rect", { x: tx(e.onset_h), width: tx(e.end_h) - tx(e.onset_h), y: 0, height: BASE, class: "outage" }));
  children.push(el("path", { d: `${line(week.solar_potential_kw, cap.solar_peak_kw)}L${TW},${BASE}L0,${BASE}Z`, fill: "var(--solar)", opacity: 0.2 }));
  for (let d = 1; d < 7; d++) children.push(el("line", { x1: tx(d * 24 - START_HOUR), x2: tx(d * 24 - START_HOUR), y1: 0, y2: BASE, stroke: "rgb(236 235 226 / .12)", "vector-effect": "non-scaling-stroke" }));
  children.push(el("path", { d: line(s.ch4_level_kg, 1200), fill: "none", stroke: "var(--ch4)", "stroke-width": 1.4, "vector-effect": "non-scaling-stroke" }));
  const deliveries = week.controllers[view.controller].deliveries;
  tickRefs = deliveries.map((d) => el("rect", { x: tx(d.due_h) - 1, y: TH - 8, width: 2, height: 8 }));
  layer.replaceChildren(...children, ...tickRefs);
  deliveries.forEach((d, k) => { truckRefs[k].title = `${clockLabel(d.due_h)} · ${d.delivered_kg} of ${d.requested_kg} kg`; });
  picker.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.condition === week.condition)));
  raceRefs.forEach((r) => r.li.classList.toggle("on", r.id === view.controller));
  $("[data-week-title]").textContent = `Test week 1 · ${CONDITIONS[week.condition].short}`;
  sunNode.g.classList.remove("down");
  methNode.g.classList.remove("down");
}

async function selectWeek(condition) {
  week = await loadWeek(condition);
  view.condition = condition;
  const url = new URL(location.href);
  url.searchParams.set("week", condition);
  history.replaceState(history.state, "", url);
  t = 0;
  holdUntil = 0;
  rebuild();
}

// Playback ---------------------------------------------------------------------------
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
let t = 0, speed = 3, playing = !reduced, last = performance.now(), dash = 0, holdUntil = 0;
const clock = $("[data-clock]"), mode = $("[data-mode]"), unmet = $("[data-unmet]"), count = $("[data-delivery-count]");
const img = $(".art img");
const playButton = $("[data-play]");
const unmetBy = (id, time) => week.controllers[id].deliveries.reduce((sum, d) => sum + (time >= d.due_h ? d.unmet_kg : 0), 0);
const raceMax = () => Math.max(200, ...controllers.map((id) => week.controllers[id].unmet_kg));

function render() {
  const now = state(t);
  for (const m of meterRefs) {
    const v = now[m.key];
    m.bar.style.width = `${Math.max(0, Math.min(1, v / m.max)) * 100}%`;
    m.val.textContent = `${fmt.format(v * (m.scale ?? 1))} ${m.unit}`;
  }
  clock.textContent = clockLabel(t);
  const down = outage(t);
  mode.textContent = down ? `${down} down` : now.potential > 1 ? "sun" : "night";
  mode.dataset.state = down ? "down" : now.potential > 1 ? "sun" : "night";
  sunNode.g.classList.toggle("down", down === "solar");
  methNode.g.classList.toggle("down", down === "methanator");
  sunNode.subText.textContent = down === "solar" ? "OFFLINE" : sunNode.sub;
  methNode.subText.textContent = down === "methanator" ? "IN REPAIR" : methNode.sub;

  const deliveries = week.controllers[view.controller].deliveries;
  let met = 0;
  deliveries.forEach((d, k) => {
    const done = t >= d.due_h;
    const ok = d.unmet_kg <= 1e-6;
    truckRefs[k].className = done ? (ok ? "met" : "missed") : "";
    tickRefs[k].setAttribute("fill", done ? (ok ? "var(--ch4)" : "var(--bad)") : "rgb(236 235 226 / .25)");
    if (done && ok) met++;
  });
  count.textContent = `${met} / ${deliveries.length}`;
  unmet.textContent = `${fmt.format(unmetBy(view.controller, t))} kg`;
  const max = raceMax();
  for (const r of raceRefs) {
    const v = unmetBy(r.id, t);
    r.bar.style.width = `${Math.min(1, v / max) * 100}%`;
    r.val.textContent = `${fmt.format(v)} kg`;
  }

  const sun = now.potential / cap.solar_peak_kw;
  img.style.filter = `saturate(${0.55 + 0.4 * sun}) brightness(${0.5 + 0.42 * Math.sqrt(sun)})`;
  head.setAttribute("x1", tx(t));
  head.setAttribute("x2", tx(t));
  timeline.setAttribute("aria-valuenow", t.toFixed(1));
  timeline.setAttribute("aria-valuetext", clockLabel(t));
  for (const [id, cap_] of [["h2", cap.h2_capacity_kg / 2], ["co2", cap.co2_capacity_kg / 2], ["ch4", 1200]]) {
    const h = Math.min(1, now[id] / cap_) * tanks[id].h;
    tanks[id].level.setAttribute("height", h.toFixed(1));
    tanks[id].level.setAttribute("y", (tanks[id].y + 59 - h).toFixed(1));
  }
  const recent = deliveries.find((d) => t >= d.due_h && t < d.due_h + 1);
  truckLabel.textContent = recent ? (recent.unmet_kg > 1e-6 ? `short ${fmt.format(recent.unmet_kg)} kg` : `−${fmt.format(recent.delivered_kg)} kg`) : "";
  truckLabel.classList.toggle("bad", Boolean(recent && recent.unmet_kg > 1e-6));
  return { now, recent };
}

function animatePipes(dt, { now, recent }) {
  const flows = {
    ...now.flows,
    "el-h2": now.electrolyser, "dac-co2": now.dac, "h2-me": now.methanator > 30 ? now.methanator * 5 : 0,
    "co2-me": now.methanator > 30 ? now.methanator * 5 : 0, "me-ch4": now.methanator > 30 ? now.methanator * 5 : 0,
    "ch4-truck": recent && recent.delivered_kg > 0 ? 2000 : 0,
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
  const frameState = render();
  if (!reduced) animatePipes(dt, frameState);
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
  if (delta) { t = Math.max(0, Math.min(HOURS, t + delta)); holdUntil = 0; event.preventDefault(); }
});
document.querySelectorAll("[data-speed]").forEach((button) => button.addEventListener("click", () => {
  speed = Number(button.dataset.speed);
  document.querySelectorAll("[data-speed]").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
  if (!playing) playButton.click();
}));
const syncPlay = () => { playButton.textContent = playing ? "pause" : "play"; };
playButton.addEventListener("click", () => { playing = !playing; syncPlay(); });
syncPlay();
rebuild();
requestAnimationFrame(frame);

startRouter({ current: () => week, select: selectWeek });
