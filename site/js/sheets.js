import { weekChart, groupedBars, intervals } from "./charts.js";
import { findings } from "./findings.js";
import {
  fmt, fmt1, eur, signedEur, kg, pct, REPO, PAPER, CONDITIONS, STRESS, CONTROLLERS, WEEK_LABELS, WEEK_COLORS,
  evaluation, plant, week as loadWeek, tabulate, html, block, figure,
} from "./common.js";

let hero;
const clock = (h) => `Day ${Math.floor((h + 6) / 24) + 1}, ${String(Math.floor((h + 6) % 24)).padStart(2, "0")}:00`;

// This week -------------------------------------------------------------------------------
async function weekSheet() {
  const week = hero.current();
  const { conditions, week_seed: seed } = await plant();
  const all = await Promise.all(conditions.map(loadWeek));
  const ids = Object.keys(week.controllers);
  const c = CONDITIONS[week.condition];
  const e = week.event.end_h > week.event.onset_h ? week.event : null;
  const what = c.outage === "solar" ? `the solar field is offline from ${clock(e.onset_h)} to ${clock(e.end_h)}`
    : c.outage === "methanator" ? `the methanator is under repair from ${clock(e.onset_h)} until ${clock(e.end_h)}, four hours before a delivery` : c.note;
  const missed = (id) => week.controllers[id].deliveries.filter((d) => d.unmet_kg > 1e-6).length;
  return [
    block(html`
      <p class="kicker">Replay on the front page · ${c.label}</p>
      <h3>One test week, three controllers, the same weather</h3>
      <p>Every controller faces identical sunlight and opening stocks; ${what}. The trucks expect <strong>180 kg of methane every six hours</strong>, 28 times.
      Pick a controller in the race panel to replay its decisions.</p>
      <div class="lead-stat">${ids.map((id) => `<div style="--c:${WEEK_COLORS[id]}"><b>${kg(week.controllers[id].unmet_kg)}</b><span>${WEEK_LABELS[id]}: unmet, ${missed(id)} of 28 deliveries short, ${eur(week.controllers[id].contribution_eur)} contribution</span></div>`).join("")}</div>`),
    figure(weekChart({
      yMax: 1500, yLabel: "Methane stored (kg)",
      bands: e ? [{ from: e.onset_h, to: e.end_h, label: c.outage === "solar" ? "solar offline" : "methanator in repair" }] : [],
      series: [
        { values: week.solar_potential_kw.map((v) => (v / 5000) * 1500), color: "var(--solar)", area: true, opacity: 0.14 },
        ...ids.map((id) => ({ values: week.controllers[id].series.ch4_level_kg, color: WEEK_COLORS[id], width: 1.8 })),
      ],
      ticks: week.controllers[ids[0]].deliveries.map((d) => ({ t: d.due_h, color: "var(--ink-3)" })),
      format: (v) => fmt.format(v),
    }), "Methane in the tank. Each tick is a 180 kg collection; an empty tank at a tick means a short delivery. Shaded: available sunlight.",
    [...ids.map((id) => ({ label: WEEK_LABELS[id], color: WEEK_COLORS[id] })), { label: "Sunlight", color: "var(--solar)" }]),
    block(html`
      <h4>The same week under every condition</h4>
      <table class="data"><thead><tr><th>Condition</th>${ids.map((id) => `<th>${WEEK_LABELS[id]}</th>`).join("")}</tr></thead><tbody>
      ${all.map((w) => `<tr class="${w.condition === week.condition ? "hl" : ""}"><td style="font-family:var(--font)"><a href="?week=${w.condition}" data-week="${w.condition}">${CONDITIONS[w.condition].label}</a></td>${ids.map((id) => `<td>${kg(w.controllers[id].unmet_kg)}</td>`).join("")}</tr>`).join("")}
      </tbody></table>
      <p>Unmet methane over the week. This is the first of the 32 final test weeks (seed ${fmt.format(seed).replaceAll(",", "")}), chosen for replay before any result was known; the refined policy shown is the first of the three searches.
      A single week is an anecdote. The averages over all 32 weeks are in <a href="#/results">Results</a>.</p>`),
  ];
}

// Plant -------------------------------------------------------------------------------------
async function plantSheet() {
  const c = (await plant()).capacities;
  return [block(html`
    <p class="kicker">The digital twin</p>
    <h3>A methane plant with no grid connection</h3>
    <p>An electrolyser uses electricity to make hydrogen, direct air capture (DAC) pulls carbon dioxide from the air, and a methanator combines the two into methane.
    Separate tanks store each gas, and a battery stores electricity for when the sun is not enough. Every component tracks its own state and limits, coupled through shared electrical and material flows.</p>
    <div class="chain"><span style="border-color:var(--solar)">solar PV</span><em>→</em><span style="border-color:var(--stored)">battery</span><em>→</em>
      <span style="border-color:var(--h2)">electrolysis → H₂</span><em>+</em><span style="border-color:var(--co2)">DAC → CO₂</span><em>→</em>
      <span style="border-color:var(--ch4)">CO₂ + 4 H₂ → CH₄ + 2 H₂O</span><em>→</em><span style="border-color:var(--ch4)">scheduled deliveries</span></div>
    <table class="data">
      <thead><tr><th>Unit</th><th>Size</th><th>Modelled behaviour</th></tr></thead>
      <tbody>
        <tr><td>Solar field</td><td>${fmt.format(c.solar_peak_kw / 1000)} MW</td><td style="text-align:left">Variable weather; outages in the stress tests</td></tr>
        <tr><td>Battery</td><td>${fmt.format(c.battery_capacity_kwh / 1000)} MWh</td><td style="text-align:left">5% minimum charge, shared 2.5 MW rating, throughput wear</td></tr>
        <tr><td>Electrolyser</td><td>${fmt.format(c.electrolyser_rated_kw / 1000)} MW</td><td style="text-align:left">52 kWh per kg of hydrogen, part-load efficiency, life wear</td></tr>
        <tr><td>Air capture</td><td>${fmt1.format(c.dac_rated_kw / 1000)} MW</td><td style="text-align:left">2 kWh per kg of CO₂; loses readiness when idle; random faults</td></tr>
        <tr><td>Methanator</td><td>${fmt.format(c.methanator_rated_kw)} kW</td><td style="text-align:left">1.2 kWh per kg of methane; must be warmed up before producing</td></tr>
        <tr><td>H₂ / CO₂ / CH₄ tanks</td><td>${fmt.format(c.h2_capacity_kg)} / ${fmt.format(c.co2_capacity_kg)} / ${fmt.format(c.ch4_capacity_kg)} kg</td><td style="text-align:left">Production stops when full; deliveries fail when empty</td></tr>
      </tbody>
    </table>
    <p>Each kilogram of methane needs 0.5 kg of hydrogen and 2.75 kg of CO₂: <strong>32.7 kWh before losses</strong>. The mission is 180 kg every six hours for seven days, 5,040 kg in total, or about 23.5 MWh of process electricity a day.</p>
    <h4>Why a week can have enough sun and still fail</h4>
    <ul>
      <li><strong>Battery or gas?</strong> Charging the battery keeps every option open but loses energy and wears the cells. Making gas now uses today's sun but ties it to one process and one tank.</li>
      <li><strong>Warm or cold?</strong> Keeping DAC and the methanator ready costs power; shutting them down saves it but delays the next start.</li>
      <li><strong>Deliveries do not wait for the sun.</strong> Methane, feedstock or warm equipment can be missing at the moment a truck arrives.</li>
    </ul>
    <h4>What the controller sees and does</h4>
    <p>Every 15 minutes it observes plant conditions, stored resources, delivery requirements and uncertain weather forecasts (149 numbers). It sets four solar allocations (battery, electrolysis, carbon capture, methanation) and three battery-power requests, one per process. The plant then advances in one-minute steps.</p>
    <h4>How it is scored</h4>
    <p>Delivered methane earns €4/kg and every undelivered kilogram costs €20. Modelled battery and electrolyser wear are charged, and so is any depletion of the opening stocks, so they are not free. The result is the <strong>operating contribution</strong>. It excludes plant-wide capital recovery and DAC and methanator wear, so it is not a profit figure.</p>
    <p>The repository also contains an interactive 3D control room for the same plant (<code>make dev</code>).</p>`)];
}

// Results -----------------------------------------------------------------------------------
async function resultsSheet() {
  const data = await evaluation();
  if (!data) {
    return [block(html`<p class="kicker">Learned-control study</p><h3>Results are being regenerated</h3>
      <p>This page reads <code>site/data/evaluation.json</code>, written by <code>tools/build_showcase_data.py</code> from a completed <code>tools/reproduce.py evaluate</code> run.</p>`)];
  }
  const t = tabulate(data);
  const pc = data.primary_claim;
  const seeds = new Set(data.rows.map((r) => r.seed)).size;
  const mix = pc.comparisons.mixture.metrics.net_operating_value_eur;
  const checkLabels = {
    three_independent_optimizer_seeds: "Three independently searched policies",
    all_cases_complete_and_audited: "Every case completed and passed the minute-level audits",
    mixture_contribution_gain_at_least_500_eur: "Mixture contribution at least €500/week above the tuned rule",
    mixture_contribution_975ci_positive: "… with a 97.5% interval above zero",
    stress_unmet_reduction_at_least_20_percent: "Unmet methane under stress at least 20% lower",
    stress_unmet_delta_975ci_negative: "… with a 97.5% interval below zero",
    stress_all_delivery_success_not_lower: "No fewer stress weeks with every delivery met",
    ordinary_contribution_loss_at_most_500_eur: "Ordinary weeks no more than €500 worse",
  };
  const passed = Object.values(pc.checks).filter(Boolean).length;
  const groups = Object.keys(CONDITIONS).filter((k) => t.refined.conditions[k]);
  const shown = CONTROLLERS.filter((c) => ["original-daylight", "storage-m48-b60-r3", "refined"].includes(c.id));
  const span = Math.max(600, ...pc.individual_policies.map((i) => Math.abs(i.mixture_contribution_delta)), ...mix.interval.map(Math.abs)) * 1.15;
  return [
    block(html`
      <p class="kicker">Final test · ${fmt.format(data.case_count)} audited simulated weeks</p>
      <h3>Close to a tuned rule, with a measurable trade-off against a naive one</h3>
      <p>Each controller ran the same ${seeds} new weather and opening-stock scenarios under ${groups.length} conditions. Contribution is in € per week; the stress columns average the five disruptions.
      The refined row averages three separately searched policies.</p>
      <table class="data"><thead><tr><th>Controller</th><th>Ordinary</th><th>Mixture</th><th>Stress unmet</th><th>Stress success</th></tr></thead><tbody>
      ${CONTROLLERS.map((c) => `<tr class="${c.id === "refined" ? "hl" : ""}"><td style="font-family:var(--font)">${c.label}</td><td>${eur(t[c.id].ordinary)}</td><td>${eur(t[c.id].mixture)}</td><td>${kg(t[c.id].stressUnmet, 1)}</td><td>${pct(t[c.id].stressSuccess)}</td></tr>`).join("")}
      </tbody></table>
      <p>Mixture: half ordinary weeks, half split equally across the five disruptions. Stress success: share of disrupted weeks in which all 28 deliveries were met.</p>`),
    figure(groupedBars({
      groups: groups.map((k) => ({ label: CONDITIONS[k].short })),
      series: shown.map((c) => ({ label: c.label, color: c.color, values: groups.map((k) => t[c.id].conditions[k].unmet) })),
      yMax: Math.ceil(Math.max(...shown.flatMap((c) => groups.map((k) => t[c.id].conditions[k].unmet))) / 100) * 100 || 100,
      format: (v) => fmt.format(v), yLabel: "Mean unmet methane per week (kg)",
    }), `Mean unmet methane per week over the ${seeds} test scenarios. Solar outages start on the third morning; methanator repairs end after dark, four hours before a delivery.`,
    shown.map((c) => ({ label: c.label, color: c.color }))),
    block(html`<h4>Against the tuned storage rule</h4>
      <p>The success bar was fixed before the final test. ${passed} of ${Object.keys(pc.checks).length} checks passed${pc.strong_headline_bar_met ? ", so the strong claim holds" : "; the strong claim is not supported"}.</p>
      <ul class="checks">${Object.entries(pc.checks).map(([k, ok]) => `<li class="${ok ? "ok" : ""}">${checkLabels[k] ?? k}</li>`).join("")}</ul>`),
    figure(intervals({
      rows: [
        { label: "Refined neural (3 policies)", value: mix.mean, interval: mix.interval, color: "var(--learned)", strong: true },
        ...pc.individual_policies.map((i) => ({ label: `Search seed ${i.optimizer_seed}`, value: i.mixture_contribution_delta, color: i.both_directions_improved ? "var(--learned)" : "var(--bad)" })),
      ],
      min: -span, max: span, format: signedEur, zeroLabel: "tuned storage",
    }), `Mixture contribution minus the tuned storage rule (€/week). Whisker: paired bootstrap 97.5% interval over the ${seeds} scenarios, conditional on these trained policies. Red: this policy did not improve both contribution and stress unmet methane.`),
    block(html`<p style="font-size:13px;color:var(--ink-3)">${pc.limitation}. ${data.distribution_caveat}.</p>`),
  ];
}

// Method --------------------------------------------------------------------------------------
const methodSimple = [
  { name: "The question", body: `<p class="kicker">In plain words</p><h3>Can software learn to run the plant better than a good rulebook?</h3>
    <p>The plant must hand over 180 kg of methane every six hours for a week, on solar power alone. We asked whether a controller trained on simulated weeks keeps that schedule better, and earns more, than hand-written rules, especially when equipment fails.</p>` },
  { name: "The rules", body: `<p class="kicker">The opponents</p><h3>Two simple strategies, one of them tuned</h3>
    <p>The <strong>daylight rule</strong> follows the available sunlight. The <strong>storage rule</strong> builds gas reserves and uses the battery to support production. Before the comparison we tuned the storage rule's reserve targets and replenishment rates (18 combinations) on separate trial weeks.</p>` },
  { name: "The learner", body: `<p class="kicker">The challenger</p><h3>Copy the rule, then search for better</h3>
    <p>A neural network first learns to reproduce the storage rule's decisions from plant measurements (imitation learning). The cross-entropy method then tries variations of 14 settings that govern whether and how strongly the seven power commands are used, keeping the best performers on full simulated weeks. Three searches start from the same network. Once trained, the network never consults the rule.</p>` },
  { name: "The test", body: `<p class="kicker">The exam</p><h3>New scenarios, the same for everyone</h3>
    <p>After the controllers were chosen, each one ran 32 new weather and opening-stock scenarios, each replayed under ordinary conditions, five disruptions (solar outages of 6, 18 and 36 hours; methanator repairs of 24 and 48 hours) and a no-fault reference. Every controller saw identical conditions, and every simulated minute was audited for energy and material balance.</p>` },
];

function methodFull() {
  return block(html`
    <p class="kicker">Full method</p>
    <h3>Fixed plant, adaptive development, fresh final test</h3>
    <h4>Environment</h4>
    <p>Seven-day episodes of 672 decisions, each held for 15 minutes over one-minute physics. Policies observe a 149-dimensional causal vector of measurements and forecasts and output seven actions: four solar shares and three battery-power shares against the shared 2,500 kW battery rating. There is no action shield, reserve floor or fallback controller.</p>
    <h4>Conditions</h4>
    <p><strong>Ordinary</strong>: variable weather and random DAC faults. <strong>No faults</strong>: the same weather without equipment faults (reference only). <strong>Solar −6/18/36 h</strong>: the solar field goes offline on the third morning (hour 48). <strong>Meth −24/48 h</strong>: the methanator is under repair until hour 86, after dark and four hours before a delivery. Disruption weeks keep weather variation and isolate the outage. The <strong>mixture</strong> weights ordinary weeks by 0.5 and each disruption by 0.1; this is a declared objective, not a probability of failure.</p>
    <h4>Baselines</h4>
    <p>Daylight-only and storage-aware rules. The storage rule was tuned over reserve target and replenishment time (18 combinations) on development weeks; the best setting is the reference for the headline comparison.</p>
    <h4>Learned controller</h4>
    <ol>
      <li>Demonstrations from the storage rule, then Gaussian behaviour cloning.</li>
      <li>DAgger-style relabelling of states the learner visits on its own.</li>
      <li>A hybrid actor: learned OFF gates plus censored Gaussian ON actions, with a ridge refit of the solar ON-mean row.</li>
      <li>Cross-entropy search over 14 actor biases, each candidate scored on weekly contribution across all conditions (16 candidates per generation, 4 elites). No gradient updates. Three search seeds; the generation-15 checkpoints were fixed before the final test.</li>
    </ol>
    <p>PPO fine-tuning of the whole network was also tried and was less competitive within the compute budget. Different starting networks and budgets mean this does not show that one learning method is inherently better.</p>
    <h4>Headline test</h4>
    <p>The three policies are averaged within each scenario and condition, then compared with the tuned storage rule on the 32 paired scenarios. The pre-registered strong claim needs at least €500/week more mixture contribution with a 97.5% interval above zero, at least 20% less stress unmet methane with an interval below zero, no lower all-delivery rate under stress, and an ordinary-week loss of at most €500. Intervals: 10,000-replicate paired bootstrap over scenarios, Bonferroni-adjusted for two endpoints.</p>
    <h4>Limitations</h4>
    <ul>
      <li>An assumed plant and a simplified simulator; no field data or physical calibration of start-up and degradation.</li>
      <li>Disruption schedules are known and fixed. Transfer to other outage timings, demand patterns and initial inventories is untested.</li>
      <li>Intervals cover new scenarios for these trained policies, not the variability of retraining.</li>
      <li>Contribution excludes plant-wide capital recovery and DAC and methanator wear. Economic inputs are assumptions, not quotes.</li>
    </ul>`);
}

function methodSheet(view) {
  if (view === "full") return { nodes: [methodFull()] };
  if (view === "simple") return { slides: methodSimple };
  const cards = document.createElement("div");
  cards.className = "cards";
  cards.innerHTML = `<button type="button" data-go="#/method/simple"><p class="kicker">4 cards · 2 minutes</p><h4>The simple version</h4><p>The question, the opponents, the learner and the exam.</p></button>
    <button type="button" data-go="#/method/full"><p class="kicker">Full protocol</p><h4>The full method</h4><p>Environment, conditions, baselines, training recipe, statistics and limitations.</p></button>`;
  return { nodes: [block(html`<p class="kicker">Method</p><h3>How the study was run</h3><p>Four short cards for anyone, or the full protocol. The <a href="${PAPER}">paper</a> has the complete account.</p>`), cards] };
}

// Reproduce & about ---------------------------------------------------------------------------
function reproduceSheet() {
  return [block(html`
    <p class="kicker">Reproduce</p>
    <h3>Every number here can be regenerated</h3>
    <p>Python 3.12, <a href="https://docs.astral.sh/uv/">uv</a> and Node 22. The three refined policies and the imitated starting network are in <code>experiments/checkpoints/</code>.</p>
    <span class="cmd">git clone ${REPO}
cd PlantAnAgent
uv sync --frozen --python 3.12 --group train --group optimisation</span>
    <h4>Quick check (minutes)</h4>
    <span class="cmd">uv run python tools/reproduce.py smoke --output out/smoke</span>
    <h4>Final test (${fmt.format(32 * 7 * 7)} weeks, a few hours on 8 cores)</h4>
    <span class="cmd">uv run python tools/reproduce.py evaluate --output out/eval --workers 8
uv run python tools/build_showcase_data.py out/eval</span>
    <h4>Retrain from scratch</h4>
    <span class="cmd">uv run python tools/reproduce.py train --output out/train --workers 8
uv run python tools/reproduce.py evaluate --training-root out/train --output out/eval-retrained --workers 8</span>
    <p>Re-running the evaluation repeats the published comparison; it is not new held-out evidence. Retraining reproduces the selected pipeline, not every exploratory candidate.</p>
    <h4>Run the 3D control room</h4>
    <span class="cmd">make setup &amp;&amp; make dev</span>`)];
}

function aboutSheet() {
  return [block(html`
    <p class="kicker">About</p>
    <h3>PlantAnAgent: a digital twin for learning to operate a methane plant</h3>
    <p>By <strong>Filippo Gombac</strong> and <strong>Vittorio Garavelli</strong>. Submitted to SoTA Commission II: Intermittent Abundance.</p>
    <p>PlantAnAgent is a modular digital twin and reinforcement-learning environment for an off-grid, solar-powered methane plant. It links electricity generation, battery storage, hydrogen production, carbon capture and methanation to scheduled deliveries.
    Learned controllers trained by imitation and cross-entropy search reduce unmet demand under disruptions relative to daylight-only operation, at some cost in ordinary weeks, and stay close to a tuned storage rule without a clear advantage. The environment and trained policies are a basis for studying these trade-offs.</p>
    <p><a href="${PAPER}">Read the paper (PDF)</a> · <a href="${REPO}">Code, checkpoints and evidence on GitHub</a></p>
    <p style="font-size:13px;color:var(--ink-3)">Everything shown is simulation of an assumed plant. Nothing here is a deployment qualification, a vendor quote or a business case.</p>`)];
}

// Router ----------------------------------------------------------------------------------------
const sheet = document.querySelector("[data-sheet]");
const title = document.querySelector("[data-sheet-title]");
const body = document.querySelector("[data-sheet-body]");
const progress = document.querySelector("[data-progress]");
const counter = document.querySelector("[data-count]");
const foot = document.querySelector("[data-sheet-foot]");
const prev = document.querySelector("[data-prev]");
const next = document.querySelector("[data-next]");
const slideName = document.querySelector("[data-slide-name]");
const stage = document.querySelector("#stage");
let slides = null, slideIndex = 0, opener = null;

function showSlide(i) {
  slideIndex = Math.max(0, Math.min(slides.length - 1, i));
  body.replaceChildren(...[].concat(slides[slideIndex].render ? slides[slideIndex].render() : block(slides[slideIndex].body)));
  progress.innerHTML = slides.map((_, k) => `<span class="${k <= slideIndex ? "on" : ""}" data-slide="${k}" title="${slides[k].name}"></span>`).join("");
  counter.textContent = `${slideIndex + 1}/${slides.length}`;
  slideName.textContent = slides[slideIndex].name;
  prev.disabled = slideIndex === 0;
  next.textContent = slideIndex === slides.length - 1 ? "Done" : "Next ›";
  body.scrollTop = 0;
}

async function open(route) {
  const [name, view] = route.split("/");
  const titles = { plant: "The plant", week: "This week", results: "Results", findings: "Findings", method: "Method", reproduce: "Reproduce", about: "About" };
  if (!titles[name]) return close(false);
  opener ??= document.activeElement;
  document.querySelectorAll(".topnav a").forEach((a) => a.toggleAttribute("aria-current", a.getAttribute("href") === `#/${name}`));
  stage.inert = true;
  title.textContent = titles[name];
  sheet.hidden = false;
  slides = null;
  progress.replaceChildren();
  counter.textContent = "";
  foot.hidden = true;
  let nodes;
  if (name === "findings") slides = await findings();
  else if (name === "method") ({ nodes, slides } = methodSheet(view));
  else nodes = await { plant: plantSheet, week: weekSheet, results: resultsSheet, reproduce: reproduceSheet, about: aboutSheet }[name]();
  if (slides) {
    foot.hidden = false;
    showSlide(0);
  } else {
    body.replaceChildren(...nodes);
    body.scrollTop = 0;
  }
  body.focus({ preventScroll: true });
}

function close(updateHash = true) {
  sheet.hidden = true;
  stage.inert = false;
  document.querySelectorAll(".topnav a").forEach((a) => a.removeAttribute("aria-current"));
  if (updateHash && location.hash) history.pushState(null, "", location.pathname + location.search);
  opener?.focus?.();
  opener = null;
}

export function startRouter(heroWeek) {
  hero = heroWeek;
  const route = () => open(location.hash.replace(/^#\/?/, ""));
  addEventListener("hashchange", route);
  route();
  document.querySelector("[data-close]").addEventListener("click", () => close());
  sheet.addEventListener("click", (event) => {
    if (event.target === sheet) close();
    const go = event.target.closest("[data-go]");
    if (go) location.hash = go.dataset.go;
    const dot = event.target.closest("[data-slide]");
    if (dot && slides) showSlide(Number(dot.dataset.slide));
    const pick = event.target.closest("[data-week]");
    if (pick) {
      event.preventDefault();
      hero.select(pick.dataset.week).then(() => open("week"));
    }
  });
  prev.addEventListener("click", () => slides && showSlide(slideIndex - 1));
  next.addEventListener("click", () => (slides && slideIndex < slides.length - 1 ? showSlide(slideIndex + 1) : close()));
  addEventListener("keydown", (event) => {
    if (sheet.hidden) return;
    if (event.key === "Escape") close();
    if (slides && event.key === "ArrowRight") showSlide(slideIndex + 1);
    if (slides && event.key === "ArrowLeft") showSlide(slideIndex - 1);
  });
}
