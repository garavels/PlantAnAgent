import { el, weekChart, groupedBars, intervals } from "./charts.js";
import { findings } from "./findings.js";

const fmt = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });
const fmt1 = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 1 });
const eur = (v) => `${v < 0 ? "−" : ""}€${fmt.format(Math.abs(v))}`;
const signedEur = (v) => `${v > 0 ? "+" : v < 0 ? "−" : ""}€${fmt.format(Math.abs(v))}`;
const kg = (v) => `${fmt.format(v)} kg`;
const REPO = "https://github.com/garavels/PlantAnAgent";
let week;
const STRESS = ["pv6", "pv18", "pv36", "meth24", "meth48"];
const MIXTURE = { ordinary: 0.5, pv6: 0.1, pv18: 0.1, pv36: 0.1, meth24: 0.1, meth48: 0.1 };
const CONDITION_LABELS = { ordinary: "Ordinary", control: "Control", pv6: "PV 6h", pv18: "PV 18h", pv36: "PV 36h", meth24: "Meth 24h", meth48: "Meth 48h" };

export const CONTROLLERS = [
  { id: "recipe", label: "Learned policy (3 seeds)", color: "var(--learned)" },
  { id: "storage-m48-b60-r3", label: "Tuned storage rule", color: "var(--ink)" },
  { id: "original-storage", label: "Storage rule", color: "var(--rule-ctl)" },
  { id: "original-daylight", label: "Daylight rule", color: "var(--solar)" },
  { id: "imitation-parent-903", label: "Imitation start", color: "var(--co2)" },
];

let evaluationPromise;
export function evaluation() {
  evaluationPromise ??= fetch("./data/evaluation.json").then((r) => (r.ok ? r.json() : null)).catch(() => null);
  return evaluationPromise;
}

/** Per-controller, per-condition means over seeds; the recipe averages its three policies within each seed first. */
export function tabulate(data) {
  const members = data.primary_claim.members;
  const cell = {};
  for (const row of data.rows) {
    const id = members.includes(row.controller) ? "recipe" : row.controller;
    const key = `${id}|${row.condition}|${row.seed}`;
    (cell[key] ??= []).push(row);
  }
  const seeds = [...new Set(data.rows.map((r) => r.seed))];
  const conditions = [...new Set(data.rows.map((r) => r.condition))];
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const out = {};
  for (const c of CONTROLLERS) {
    out[c.id] = {};
    for (const condition of conditions) {
      const perSeed = seeds.map((seed) => cell[`${c.id}|${condition}|${seed}`]).filter(Boolean);
      if (!perSeed.length) continue;
      out[c.id][condition] = {
        unmet: mean(perSeed.map((rows) => mean(rows.map((r) => r.unmet_kg)))),
        contribution: mean(perSeed.map((rows) => mean(rows.map((r) => r.contribution_eur)))),
        allMet: mean(perSeed.map((rows) => mean(rows.map((r) => (r.all_deliveries ? 1 : 0))))),
        missed: mean(perSeed.map((rows) => mean(rows.map((r) => r.missed)))),
      };
    }
    const at = out[c.id];
    out[c.id].panels = {
      mixture: Object.entries(MIXTURE).reduce((sum, [k, w]) => sum + w * (at[k]?.contribution ?? NaN), 0),
      stressUnmet: mean(STRESS.map((k) => at[k]?.unmet ?? NaN)),
      stressAllMet: mean(STRESS.map((k) => at[k]?.allMet ?? NaN)),
    };
  }
  return out;
}

const html = (strings, ...values) => strings.reduce((acc, s, i) => acc + s + (values[i] ?? ""), "");
function block(markup) {
  const div = document.createElement("div");
  div.className = "prose";
  div.innerHTML = markup;
  return div;
}
function figure(svg, caption, legend) {
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

// Featured week ------------------------------------------------------------------
function weekSheet() {
  const m = week.totals, p = week.provenance;
  const s = week.series;
  const kwh = (v) => `${fmt1.format(v / 1000)} MWh`;
  return [
    block(html`
      <p class="kicker">Replay on the front page</p>
      <h3>One week, twenty-eight deliveries, no misses</h3>
      <p>The front page replays a single simulated week driven by a trained PPO policy (optimizer seed ${p.optimizer_seed}). Every fifteen minutes the
      policy splits the solar output and battery between the electrolyser, air capture and methanator. A truck collects <strong>150 kg of methane every six hours</strong>, sun or no sun.</p>
      <div class="lead-stat">
        <div style="--c: var(--ch4)"><b>${p.mission.deliveries_met} / ${p.mission.deliveries_total}</b><span>deliveries met</span></div>
        <div style="--c: var(--ch4)"><b>${kg(m.delivered_kg)}</b><span>delivered, ${kg(m.ch4_produced_kg)} produced</span></div>
        <div style="--c: var(--solar)"><b>${kwh(m.curtailed_kwh)}</b><span>of sunlight left unused</span></div>
        <div style="--c: var(--stored)"><b>${eur(week.economics.net_value_eur)}</b><span>modelled operating contribution</span></div>
      </div>`),
    figure(weekChart({
      yMax: 1000, yLabel: "Methane in storage",
      series: [
        { values: s.solar_potential_kw.map((v) => v / 5), color: "var(--solar)", area: true, opacity: 0.18 },
        { values: s.ch4_level_kg, color: "var(--ch4)", width: 2 },
        { values: s.battery_soc.map((v) => v * 1000), color: "var(--stored)", dash: "4 3" },
      ],
      ticks: week.deliveries.map((d) => ({ t: d.due_h, color: "var(--ch4)" })),
      format: (v) => fmt.format(v),
    }), "Methane in the tank (kg) falls by 150 kg at each truck tick and is rebuilt in daylight. Night is the hard part: the tank and battery must carry the plant to morning.",
    [{ label: "Methane stored (kg)", color: "var(--ch4)" }, { label: "Battery charge (full = 1,000 line)", color: "var(--stored)" }, { label: "Sunlight (scaled)", color: "var(--solar)" }]),
    block(html`
      <h4>How to read it</h4>
      <p>This week was <strong>selected because it succeeded</strong>. In its study the same policy passed ${p.evaluation.passed} of ${p.evaluation.total} held-out weeks, and the hand-written rule it started from passed just as many.
      Treat the replay as an illustration of how the plant runs, not as proof that learning wins. That question is answered in <a href="#/results">Results</a>.</p>
      <p>Scenario seed ${p.scenario_seed} · ${week.totals.decisions} decisions · physics every minute · hard safety checks ${p.mission.hard_safety_passed ? "passed" : "failed"}.
      Source: <a href="${REPO}/blob/main/${week.source}"><code>${week.source}</code></a>.</p>`),
  ];
}

// Plant ------------------------------------------------------------------------------
function plantSheet() {
  const c = week.capacities;
  return [block(html`
    <p class="kicker">The digital twin</p>
    <h3>A methane plant with no grid connection</h3>
    <p>Everything runs on a 5 MW solar field and one shared battery. Water is split into hydrogen, CO₂ is pulled from the air, and the two are combined into methane.
    Each step has its own tank, start-up time and power limit, so the controller is always trading one bottleneck against another.</p>
    <div class="chain"><span style="border-color:var(--solar)">sunlight</span><em>→</em><span style="border-color:var(--stored)">battery</span><em>→</em>
      <span style="border-color:var(--h2)">2 H₂O → 2 H₂ + O₂</span><em>+</em><span style="border-color:var(--co2)">air → CO₂</span><em>→</em>
      <span style="border-color:var(--ch4)">CO₂ + 4 H₂ → CH₄ + 2 H₂O</span><em>→</em><span style="border-color:var(--ch4)">truck</span></div>
    <table class="data">
      <thead><tr><th>Unit</th><th>Size</th><th>What limits it</th></tr></thead>
      <tbody>
        <tr><td>Solar field</td><td>${fmt.format(c.solar_peak_kw)} kWp</td><td style="text-align:left">Weather, night, outages</td></tr>
        <tr><td>Battery</td><td>${fmt.format(c.battery_kwh)} kWh</td><td style="text-align:left">2,500 kW shared power rating, throughput wear</td></tr>
        <tr><td>Electrolyser</td><td>${fmt.format(c.electrolyser_kw)} kW</td><td style="text-align:left">Efficiency curve, temperature, life wear</td></tr>
        <tr><td>Direct air capture</td><td>${fmt.format(c.dac_kw)} kW</td><td style="text-align:left">Start-up readiness, heat loss, random faults</td></tr>
        <tr><td>Methanator</td><td>${fmt.format(c.methanator_kw)} kW</td><td style="text-align:left">Warm-up energy, repair outages</td></tr>
        <tr><td>H₂ / CO₂ / CH₄ tanks</td><td>${fmt.format(c.h2_kg)} / ${fmt.format(c.co2_kg)} / ${fmt.format(c.ch4_kg)} kg</td><td style="text-align:left">Fill level; methane must be on hand when the truck arrives</td></tr>
      </tbody>
    </table>
    <h4>Why it is a hard control problem</h4>
    <ul>
      <li><strong>The promise does not wait for the sun.</strong> Deliveries are due every six hours, including at night.</li>
      <li><strong>Warm units are expensive to keep, cold units are slow to start.</strong> Readiness is modelled minute by minute.</li>
      <li><strong>Everything shares one battery.</strong> Charging for the night competes with making hydrogen now.</li>
      <li><strong>The controller sees forecasts, not the future.</strong> It gets causal solar forecasts with uncertainty bands, never the true weather.</li>
    </ul>
    <p>The interactive control room in the repository (<code>make dev</code>) shows the same plant in 3D, with scenarios, site forecasts and an economics view.</p>`)];
}

// Results ------------------------------------------------------------------------------
async function resultsSheet() {
  const data = await evaluation();
  if (!data) {
    return [block(html`<p class="kicker">Learned-control study</p><h3>Results are being regenerated</h3>
      <p>This page reads <code>site/data/evaluation.json</code>, produced by <code>tools/reproduce.py evaluate</code> and <code>tools/build_showcase_data.py</code>.</p>`)];
  }
  const t = tabulate(data);
  const pc = data.primary_claim;
  const mix = pc.comparisons.mixture.metrics.net_operating_value_eur;
  const unmet = pc.comparisons.stress.metrics.unmet_kg;
  const ordinary = pc.comparisons.ordinary.metrics.net_operating_value_eur;
  const success = pc.comparisons.stress.metrics.all_delivery_success;
  const checkLabels = {
    three_independent_optimizer_seeds: "Three independently trained policies",
    all_cases_complete_and_audited: "Every case ran to completion and passed the minute-level audits",
    mixture_contribution_gain_at_least_500_eur: "Mixture contribution at least €500/week higher",
    mixture_contribution_975ci_positive: "…with a 97.5% interval above zero",
    stress_unmet_reduction_at_least_20_percent: "Stressed unmet methane at least 20% lower",
    stress_unmet_delta_975ci_negative: "…with a 97.5% interval below zero",
    stress_all_delivery_success_not_lower: "No drop in stressed weeks with all 28 deliveries",
    ordinary_contribution_loss_at_most_500_eur: "Ordinary weeks no more than €500 worse",
  };
  const rows = CONTROLLERS.filter((c) => t[c.id].ordinary);
  const nodes = [block(html`
    <p class="kicker">Learned-control study · ${data.case_count} audited weeks</p>
    <h3>${findings.resultsHeadline(pc)}</h3>
    <p>${findings.resultsLede(pc, t)}</p>
    <div class="lead-stat">
      <div style="--c: var(--learned)"><b>${signedEur(mix.mean)}</b><span>per week, mixture contribution vs tuned rule<br />97.5% CI ${signedEur(mix.interval[0])} to ${signedEur(mix.interval[1])}</span></div>
      <div style="--c: var(--learned)"><b>${fmt.format(Math.abs(pc.relative_stress_unmet_reduction * 100))}% ${pc.relative_stress_unmet_reduction >= 0 ? "less" : "more"}</b><span>methane left undelivered under stress<br />${kg(unmet.reference_mean)} → ${kg(unmet.treatment_mean)} per week</span></div>
      <div style="--c: var(--ink)"><b>${signedEur(ordinary.mean)}</b><span>per week in ordinary weeks</span></div>
      <div style="--c: var(--ch4)"><b>${fmt.format(success.treatment_mean * 100)}% vs ${fmt.format(success.reference_mean * 100)}%</b><span>stressed weeks with all 28 deliveries met</span></div>
    </div>
    <h4>Pre-registered success bar</h4>
    <ul class="checks">${Object.entries(pc.checks).map(([k, ok]) => `<li class="${ok ? "ok" : ""}">${checkLabels[k] ?? k}</li>`).join("")}</ul>`)];
  nodes.push(figure(groupedBars({
    groups: Object.keys(CONDITION_LABELS).filter((k) => t.recipe[k]).map((k) => ({ label: CONDITION_LABELS[k] })),
    series: rows.map((c) => ({ label: c.label, color: c.color, values: Object.keys(CONDITION_LABELS).filter((k) => t.recipe[k]).map((k) => t[c.id][k].unmet) })),
    yMax: Math.ceil(Math.max(...rows.flatMap((c) => Object.keys(CONDITION_LABELS).map((k) => t[c.id][k]?.unmet ?? 0))) / 100) * 100 || 100,
    format: (v) => fmt.format(v), yLabel: "Mean unmet methane per week",
  }), `Mean methane owed but not delivered per week (kg), over ${new Set(data.rows.map((r) => r.seed)).size} fresh test weeks. PV = solar field offline from hour 48; Meth = methanator under repair until hour 86. Control has no disruption.`,
  rows.map((c) => ({ label: c.label, color: c.color }))));
  nodes.push(block(html`<h4>Every controller, every panel</h4>
    <table class="data"><thead><tr><th>Controller</th><th>Ordinary €/wk</th><th>Mixture €/wk</th><th>Stress unmet</th><th>Stress all-28</th></tr></thead><tbody>
    ${rows.map((c) => `<tr class="${c.id === "recipe" ? "hl" : ""}"><td style="font-family:var(--font)">${c.label}</td><td>${eur(t[c.id].ordinary.contribution)}</td><td>${eur(t[c.id].panels.mixture)}</td><td>${kg(t[c.id].panels.stressUnmet)}</td><td>${fmt.format(t[c.id].panels.stressAllMet * 100)}%</td></tr>`).join("")}
    </tbody></table>
    <h4>Each trained policy on its own</h4>`));
  const individual = pc.individual_policies;
  const span = Math.max(600, ...individual.map((i) => Math.abs(i.mixture_contribution_delta)), Math.abs(mix.interval[0]), Math.abs(mix.interval[1])) * 1.15;
  nodes.push(figure(intervals({
    rows: [
      { label: "Recipe (3 policies averaged)", value: mix.mean, interval: mix.interval, color: "var(--learned)", strong: true },
      ...individual.map((i) => ({ label: `Policy, search seed ${i.optimizer_seed}`, value: i.mixture_contribution_delta, color: i.both_directions_improved ? "var(--learned)" : "var(--bad)" })),
    ],
    min: -span, max: span, format: signedEur, zeroLabel: "same as tuned rule",
  }), "Mixture contribution difference against the tuned storage rule (€/week). Whisker: paired bootstrap 97.5% interval over the 32 test weeks, conditional on these trained policies. Red = this policy did not improve both contribution and stressed unmet methane."));
  const trace = data.traces.pv36;
  if (trace) {
    const ids = [["cem-ridge-bias-g16-5901", "Learned policy (seed 5901)", "var(--learned)"], ["storage-m48-b60-r3", "Tuned storage rule", "var(--ink)"], ["original-daylight", "Daylight rule", "var(--solar)"]].filter(([id]) => trace[id]);
    const ev = trace[ids[0][0]].event;
    nodes.push(block(html`<h4>Inside one stressed week</h4>`));
    nodes.push(figure(weekChart({
      yMax: 1500, yLabel: "Methane stored during a 36-hour solar outage",
      bands: [{ from: ev.onset_h, to: ev.end_h, label: "solar field offline" }],
      series: ids.map(([id, , color]) => ({ values: trace[id].ch4_level_kg, color, width: 1.8 })),
      format: (v) => fmt.format(v),
    }), `Methane in storage (kg) in the first test week (seed ${data.trace_seed}) with the solar field offline for 36 hours. This panel was fixed before any result was seen.`,
    ids.map(([, label, color]) => ({ label, color }))));
  }
  nodes.push(block(html`<p style="font-size:13px;color:var(--ink-3)">${pc.limitation}. ${data.distribution_caveat}. Contribution is a simulated operating margin that excludes solar capital and fixed costs; it is not business profit.</p>`));
  return nodes;
}

// Method ------------------------------------------------------------------------------------
const methodSimple = [
  { name: "The question", body: `<p class="kicker">In plain words</p><h3>Can software learn to run the plant better than a good rulebook?</h3>
    <p>We built a detailed simulator of a solar methane plant and promised a truck 180 kg of methane every six hours for a week. Then we asked: can a controller trained on simulated weeks keep that promise better, and more cheaply, than carefully tuned hand-written rules, especially when things break?</p>` },
  { name: "The rules", body: `<p class="kicker">The opponents</p><h3>Hand-written rules set a high bar</h3>
    <p>The <strong>daylight rule</strong> runs everything when the sun shines. The <strong>storage rule</strong> watches the tanks and battery and refills what will run out first. We gave the storage rule a fair tuning budget (18 settings) on separate practice weeks, and the winner became the champion to beat.</p>` },
  { name: "The learner", body: `<p class="kicker">The challenger</p><h3>Start by copying, then search for something better</h3>
    <p>The learned policy is a small neural network. It first imitates the storage rule, including in situations its own mistakes lead it into. A search then nudges its decision thresholds, scoring every candidate on full simulated weeks of profit and penalties. Three independent searches were run so that one lucky result could not carry the conclusion.</p>` },
  { name: "The test", body: `<p class="kicker">The exam</p><h3>Fresh weeks, opened once</h3>
    <p>All choices were frozen before the final 32 test weeks were simulated. Each week was replayed seven ways: normal, healthy equipment, solar outages of 6, 18 and 36 hours, and methanator repairs of 24 and 48 hours. Every minute of every run was checked for conservation of energy and material. The pass mark was written down in advance.</p>` },
];

function methodFull() {
  return block(html`
    <p class="kicker">Full method</p>
    <h3>Fixed plant, adaptive development, fresh final test</h3>
    <h4>Environment</h4>
    <p>Seven-day episodes of 672 decisions, each held for 15 minutes over a 60-second physics step. The plant (shared 8 MWh battery, dynamic DAC and methanator readiness, random DAC faults in ordinary weeks) and its economics are unchanged from the parent study. Policies observe a 149-dimensional causal vector of measurements and forecasts and output 7 actions: four solar shares and three battery-power shares against the shared 2,500 kW rating. There is no action shield, reserve floor or fallback controller.</p>
    <h4>Conditions</h4>
    <p><code>ordinary</code> (stochastic weather and DAC faults), <code>control</code> (healthy equipment), <code>pv6/18/36</code> (solar field offline from elapsed hour 48) and <code>meth24/48</code> (methanator outage ending at hour 86). Timings are fixed, so success concerns these known families, not surprise events.</p>
    <h4>Objective</h4>
    <p>Weekly operating contribution: delivered methane value minus unmet-delivery penalties, battery throughput and electrolyser life reserves, and changes in ending methane and battery inventory. The training and selection mixture is 50% ordinary and 10% for each of the five stress conditions. These weights are a design choice, not estimated hazard rates.</p>
    <h4>Seeds</h4>
    <table class="data"><tbody>
      <tr><td>Training</td><td>50,000,000 – 50,999,999</td></tr>
      <tr><td>Development</td><td>51,000,000 – 51,000,015 (16 weeks)</td></tr>
      <tr><td>Final test</td><td>52,000,000 – 52,000,031 (32 weeks, opened once)</td></tr>
    </tbody></table>
    <h4>Baselines</h4>
    <p>Original daylight and storage rules, and the storage rule tuned over methane buffer (48/72/96 h), battery target (0.35/0.60/0.85) and replenishment time (3/6 h) on development weeks. Champion: <code>storage-m48-b60-r3</code>.</p>
    <h4>Learned recipe</h4>
    <ol>
      <li>Storage-rule demonstrations on training seeds; Gaussian behaviour cloning.</li>
      <li>Dataset aggregation (DAgger): states visited by the cloned policy are relabelled by the teacher.</li>
      <li>Hybrid actor: a learned OFF gate per action plus a censored-Gaussian ON value.</li>
      <li>Parent-centred ridge refit of the solar ON-mean row.</li>
      <li>Cross-entropy search over 14 actor biases, each candidate scored on exact weekly contribution across all conditions (16 candidates per generation, 4 elites). No gradient updates; three search seeds (5901–5903). The generation-15 checkpoint was fixed before the final test.</li>
    </ol>
    <h4>Pre-registered headline</h4>
    <p>The three policies are averaged within each week and condition, then compared to the champion on the 32 paired weeks. A strong claim requires ≥ €500/week mixture gain with a 97.5% interval above zero, ≥ 20% less stressed unmet methane with an interval below zero, no lower all-28-deliveries rate under stress, and an ordinary-week loss no worse than €500. Intervals: 10,000-replicate paired bootstrap over weeks, Bonferroni-adjusted for two endpoints.</p>
    <h4>Audits</h4>
    <p>Every simulated minute is checked for energy, material, readiness and fault accounting, and the economic ledger is reconciled independently of the reward. Paired runs share identical solar forcing and opening stocks. Full traces are kept for the first test week in every controller–condition cell, chosen before outcomes were known.</p>
    <h4>Limitations</h4>
    <ul>
      <li>A simplified simulator, not a qualified plant model; no field data.</li>
      <li>Economic figures are literature-anchored assumptions, not vendor quotes or a bankable model; solar capital and fixed costs are excluded.</li>
      <li>Intervals cover fresh weather and stock draws for these policies, not the variability of retraining.</li>
      <li>Disruption timings are fixed and could be learned; they are not a test of surprise events.</li>
    </ul>`);
}

function methodSheet(view) {
  if (view === "full") return { nodes: [methodFull()] };
  if (view === "simple") return { slides: methodSimple };
  const nodes = [block(html`<p class="kicker">Method</p><h3>How the study was run</h3>
    <p>Two ways in: four short cards for anyone, or the full protocol with every number that the result depends on.</p>`)];
  const cards = document.createElement("div");
  cards.className = "cards";
  cards.innerHTML = `<button type="button" data-go="#/method/simple"><p class="kicker">4 cards · 2 minutes</p><h4>The simple version</h4><p>The question, the opponents, the learner, and the exam.</p></button>
    <button type="button" data-go="#/method/full"><p class="kicker">Full protocol</p><h4>The full method</h4><p>Environment, objective, seeds, baselines, training recipe, statistics, audits and limitations.</p></button>`;
  nodes.push(cards);
  return { nodes };
}

// Reproduce & about ----------------------------------------------------------------------------------
function reproduceSheet() {
  return [block(html`
    <p class="kicker">Reproduce</p>
    <h3>Every number here can be regenerated</h3>
    <p>Python 3.12, <a href="https://docs.astral.sh/uv/">uv</a> and Node 22. Checkpoints for the three learned policies and the imitation start are in <code>experiments/checkpoints/</code>.</p>
    <span class="cmd">git clone ${REPO}
cd PlantAnAgent
uv sync --frozen --python 3.12 --group train --group optimisation</span>
    <h4>Quick check (minutes)</h4>
    <span class="cmd">uv run python tools/reproduce.py smoke --output out/smoke</span>
    <h4>Full final test (${fmt.format(32 * 7 * 7)} weeks, a few hours on 8 cores)</h4>
    <span class="cmd">uv run python tools/reproduce.py evaluate --output out/eval --workers 8
uv run python tools/build_showcase_data.py --evaluation out/eval</span>
    <h4>Retrain the recipe from scratch</h4>
    <span class="cmd">uv run python tools/reproduce.py train --output out/train --workers 8
uv run python tools/reproduce.py evaluate --training-root out/train --output out/eval-retrained --workers 8</span>
    <p>A re-run of the evaluation repeats the published comparison; it is not new held-out evidence. Retraining reproduces the selected pipeline, not every exploratory candidate from development.</p>
    <h4>Run the control room</h4>
    <span class="cmd">make setup &amp;&amp; make dev</span>`)];
}

function aboutSheet() {
  return [block(html`
    <p class="kicker">About</p>
    <h3>Plant an Agent</h3>
    <p>By <strong>Filippo Gombac</strong> and <strong>Vittorio Garavelli</strong>. A digital twin of an off-grid solar methane plant, an interactive control room, and a pre-registered study of learned control against tuned rules.</p>
    <p>Everything shown is simulation. Nothing here is a deployment qualification, a vendor quote or a business case.</p>
    <p><a href="${REPO}">Source code and evidence on GitHub</a></p>`)];
}

// Router -----------------------------------------------------------------------------------------------
const sheet = document.querySelector("[data-sheet]");
const title = document.querySelector("[data-sheet-title]");
const body = document.querySelector("[data-sheet-body]");
const progress = document.querySelector("[data-progress]");
const counter = document.querySelector("[data-count]");
const foot = document.querySelector("[data-sheet-foot]");
const prev = document.querySelector("[data-prev]");
const next = document.querySelector("[data-next]");
const slideName = document.querySelector("[data-slide-name]");
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
  const titles = { plant: "The plant", week: "Featured week", results: "Results", findings: "Findings", method: "Method", reproduce: "Reproduce", about: "About" };
  if (!titles[name]) return close(false);
  opener ??= document.activeElement;
  document.querySelectorAll(".topnav a").forEach((a) => a.toggleAttribute("aria-current", a.getAttribute("href") === `#/${name}`));
  if (a11yHidden()) a11yHidden().inert = true;
  title.textContent = titles[name];
  sheet.hidden = false;
  slides = null;
  progress.replaceChildren();
  counter.textContent = "";
  foot.hidden = true;
  let nodes;
  if (name === "findings") slides = await findings.slides();
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

const a11yHidden = () => document.querySelector("#stage");

function close(updateHash = true) {
  sheet.hidden = true;
  if (a11yHidden()) a11yHidden().inert = false;
  document.querySelectorAll(".topnav a").forEach((a) => a.removeAttribute("aria-current"));
  if (updateHash && location.hash) history.pushState(null, "", location.pathname + location.search);
  opener?.focus?.();
  opener = null;
}

export function startRouter(featuredWeek) {
  week = featuredWeek;
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

export { block, figure, eur, signedEur, kg, fmt, CONDITION_LABELS, STRESS };
