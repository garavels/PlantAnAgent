import { el, groupedBars } from "./charts.js";
import { evaluation, tabulate, block, figure, signedEur, kg, fmt, CONTROLLERS, CONDITION_LABELS, STRESS } from "./sheets.js";

function pctLess(reference, treatment) {
  return reference > 0 ? Math.round((1 - treatment / reference) * 100) : 0;
}

export const findings = {
  resultsHeadline(pc) {
    if (pc.strong_headline_bar_met) return "The learned policy cleared the pre-registered bar";
    return "The learned policy did not clear the full pre-registered bar";
  },
  resultsLede(pc) {
    const passed = Object.values(pc.checks).filter(Boolean).length;
    return `Three independently searched policies were compared with the best hand-tuned storage rule on 32 fresh test weeks, each replayed under seven conditions. ${passed} of ${Object.keys(pc.checks).length} pre-registered checks passed.`;
  },
  async slides() {
    const data = await evaluation();
    const slides = [
      {
        name: "A good rule is hard to beat",
        render: () => [
          block(`<p class="kicker">Finding 1 · earlier study</p><h3>A trained policy matched a good rule. It did not beat it.</h3>
            <p>In the first study (25 kg/h delivered as 150 kg every six hours), all three PPO policies passed <strong>14 of 16 held-out weeks</strong>. So did the coverage rule they were built around. A strict model-predictive controller passed only 6.</p>`),
          figure(groupedBars({
            groups: [{ label: "PPO (3 seeds)" }, { label: "Coverage rule" }, { label: "Strict MPC" }],
            series: [{ label: "Held-out weeks passed", color: "var(--ch4)", values: [14, 14, 6] }],
            yMax: 16, height: 200, format: (v) => fmt.format(v), yLabel: "Held-out weeks passed of 16",
          }), "Full held-out missions passed, out of 16. The two weeks the learned policies lost both fail on day 1, on methane. That result motivated the second study: train for disruptions on purpose."),
        ],
      },
    ];
    if (data) {
      const t = tabulate(data);
      const pc = data.primary_claim;
      const reference = t["storage-m48-b60-r3"], learned = t.recipe;
      slides.push({
        name: "Under stress",
        render: () => [
          block(`<p class="kicker">Finding 2 · learned-control study</p><h3>Under disruptions, the learned policy left ${pctLess(reference.panels.stressUnmet, learned.panels.stressUnmet)}% less methane undelivered</h3>
            <p>Averaged over the five stress conditions: ${kg(learned.panels.stressUnmet)} per week against ${kg(reference.panels.stressUnmet)} for the tuned storage rule.</p>`),
          figure(groupedBars({
            groups: STRESS.map((k) => ({ label: CONDITION_LABELS[k] })),
            series: CONTROLLERS.slice(0, 2).map((c) => ({ label: c.label, color: c.color, values: STRESS.map((k) => t[c.id][k].unmet) })),
            yMax: Math.ceil(Math.max(...STRESS.flatMap((k) => [learned[k].unmet, reference[k].unmet])) / 50) * 50 || 50,
            format: (v) => fmt.format(v), yLabel: "Mean unmet methane per stressed week",
          }), "Mean unmet methane per week (kg).", CONTROLLERS.slice(0, 2)),
        ],
      });
      slides.push({
        name: "Seed by seed",
        render: () => [block(`<p class="kicker">Finding 3 · consistency</p><h3>${pc.consistent_across_optimizer_seeds ? "All three searches improved on the rule" : "Not every search improved on the rule"}</h3>
          <table class="data"><thead><tr><th>Search seed</th><th>Mixture €/wk vs rule</th><th>Stress unmet vs rule</th><th>Both better</th></tr></thead><tbody>
          ${pc.individual_policies.map((i) => `<tr><td>${i.optimizer_seed}</td><td>${signedEur(i.mixture_contribution_delta)}</td><td>${i.stress_unmet_delta > 0 ? "+" : "−"}${kg(Math.abs(i.stress_unmet_delta))}</td><td>${i.both_directions_improved ? "yes" : "no"}</td></tr>`).join("")}
          </tbody></table>`)],
      });
    }
    return slides;
  },
};
