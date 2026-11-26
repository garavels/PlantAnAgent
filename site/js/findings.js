import { groupedBars } from "./charts.js";
import { fmt, fmt1, signedEur, kg, pct, CONDITIONS, STRESS, CONTROLLERS, evaluation, tabulate, block, figure } from "./common.js";

const color = (id) => CONTROLLERS.find((c) => c.id === id).color;
const label = (id) => CONTROLLERS.find((c) => c.id === id).label;
const reduction = (from, to) => (1 - to / from) * 100;

function stressBars(t, ids, yLabel, field = "unmet") {
  const values = ids.flatMap((id) => STRESS.map((k) => t[id].conditions[k][field]));
  return figure(groupedBars({
    groups: STRESS.map((k) => ({ label: CONDITIONS[k].short })),
    series: ids.map((id) => ({ label: label(id), color: color(id), values: STRESS.map((k) => t[id].conditions[k][field]) })),
    yMax: Math.ceil(Math.max(...values) / 100) * 100 || 100, format: (v) => fmt.format(v), yLabel,
  }), null, ids.map((id) => ({ label: label(id), color: color(id) })));
}

/** Findings follow Section 4 of the paper; every figure is computed from evaluation.json. */
export async function findings() {
  const data = await evaluation();
  if (!data) return [{ name: "Pending", body: `<p class="kicker">Findings</p><h3>Results are being regenerated</h3><p>The findings are computed from <code>site/data/evaluation.json</code>.</p>` }];
  const t = tabulate(data);
  const r = t.refined, im = t["imitation-parent-903"], day = t["original-daylight"], tuned = t["storage-m48-b60-r3"];
  const perSeed = data.primary_claim.individual_policies;
  return [
    {
      name: "Refinement helps",
      render: () => [
        block(`<p class="kicker">Finding 1 · imitation, then search</p><h3>Search improves on the imitated rule: ${signedEur(r.mixture - im.mixture)} a week</h3>
          <p>The network trained to copy the storage rule is a useful start but slightly worse than the rule itself. Cross-entropy search on the copied network raises mixture contribution by <strong>${signedEur(r.mixture - im.mixture)}/week</strong> and cuts unmet methane under stress from <strong>${kg(im.stressUnmet, 1)}</strong> to <strong>${kg(r.stressUnmet, 1)}</strong> a week.</p>`),
        stressBars(t, ["imitation-parent-903", "refined"], "Mean unmet methane per week (kg)"),
      ],
    },
    {
      name: "A real trade-off",
      render: () => [
        block(`<p class="kicker">Finding 2 · against daylight-only operation</p><h3>${fmt1.format(reduction(day.stressUnmet, r.stressUnmet))}% less unmet demand under disruption, for less in ordinary weeks</h3>
          <p>Running everything when the sun shines earns the most in an ordinary week. The refined policies give up <strong>${signedEur(r.ordinary - day.ordinary).replace("−", "")}/week</strong> of that, hold more in reserve, and in return gain <strong>${signedEur(r.mixture - day.mixture)}/week</strong> across the mixture and leave ${fmt1.format(reduction(day.stressUnmet, r.stressUnmet))}% less methane undelivered when something breaks.</p>`),
        stressBars(t, ["original-daylight", "refined"], "Mean unmet methane per week (kg)"),
      ],
    },
    {
      name: "A tuned rule holds",
      render: () => [
        block(`<p class="kicker">Finding 3 · against the tuned storage rule</p><h3>No consistent advantage over a well-tuned rule</h3>
          <p>Across the 32 scenarios the refined policies average <strong>${signedEur(r.mixture - tuned.mixture)}/week</strong> and <strong>${kg(tuned.stressUnmet - r.stressUnmet, 1)}</strong> less unmet methane under stress than the tuned storage rule. Differences this small, with the three searches disagreeing, do not establish an advantage.</p>
          <table class="data"><thead><tr><th>Search seed</th><th>Mixture vs tuned</th><th>Stress unmet vs tuned</th><th>Better on both</th></tr></thead><tbody>
          ${perSeed.map((i) => `<tr><td>${i.optimizer_seed}</td><td>${signedEur(i.mixture_contribution_delta)}</td><td>${i.stress_unmet_delta > 0 ? "+" : "−"}${kg(Math.abs(i.stress_unmet_delta), 1)}</td><td>${i.both_directions_improved ? "yes" : "no"}</td></tr>`).join("")}
          </tbody></table>`),
        stressBars(t, ["storage-m48-b60-r3", "refined"], "Mean unmet methane per week (kg)"),
      ],
    },
    {
      name: "Reliability depends on the measure",
      render: () => [
        block(`<p class="kicker">Finding 4 · what counts as reliable</p><h3>More perfect weeks, but more missed trucks</h3>
          <p>Under stress the refined policies meet every delivery in slightly more weeks (${pct(r.stressSuccess)} against ${pct(tuned.stressSuccess)}). Yet they miss more individual deliveries on average (<strong>${fmt1.format(r.stressMissed)}</strong> against <strong>${fmt1.format(tuned.stressMissed)}</strong> a week), and in the worst quarter of stress weeks their average shortfall is higher: <strong>${kg(r.stressWorstQuarter, 1)}</strong> against <strong>${kg(tuned.stressWorstQuarter, 1)}</strong>.</p>
          <div class="lead-stat">
            <div style="--c:var(--learned)"><b>${pct(r.stressSuccess)}</b><span>stress weeks with all 28 deliveries, refined</span></div>
            <div style="--c:var(--ink)"><b>${pct(tuned.stressSuccess)}</b><span>tuned storage</span></div>
            <div style="--c:var(--learned)"><b>${kg(r.stressWorstQuarter)}</b><span>worst-quarter shortfall, refined</span></div>
            <div style="--c:var(--ink)"><b>${kg(tuned.stressWorstQuarter)}</b><span>tuned storage</span></div>
          </div>
          <p>A larger and more varied evaluation, with outage timings the controllers have not seen, is the next step. These results concern an assumed plant and known disruption schedules.</p>`),
      ],
    },
  ];
}
