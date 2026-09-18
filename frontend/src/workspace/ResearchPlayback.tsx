import type { ResearchPlaybackProvenance } from "../types/v2";

export function ResearchMissionSummary({ provenance: p, onInspectResult }: {
  provenance: ResearchPlaybackProvenance;
  onInspectResult?: () => void;
}) {
  return (
    <section className="w-research-mission" aria-label="Verified recorded mission">
      <div className="w-research-mission-heading">
        <div>
          <span className="w-eyebrow">Recorded simulation · PPO {p.optimizer_seed} · Scenario {p.scenario_seed}</span>
          <h2>Successful trained PPO week</h2>
        </div>
        {onInspectResult && <button className="w-button" onClick={onInspectResult}>Inspect final result</button>}
      </div>
      <div className="w-research-outcomes">
        <span><strong>{p.mission.deliveries_met}/{p.mission.deliveries_total}</strong> deliveries met</span>
        <span><strong>{p.mission.daily_returns_met}/{p.mission.daily_returns_total}</strong> daily returns met</span>
        <span><strong>{p.mission.hard_safety_passed ? "Passed" : "Failed"}</strong> safety checks</span>
        <span><strong>{p.mission.delivery_rate_kg_h} kg CH₄/h</strong> delivery mission</span>
      </div>
      <p>Across untouched scenarios: <strong>{p.evaluation.passed}/{p.evaluation.total} full missions passed.</strong> Dedicated off-grid PV · €{p.electricity.marginal_eur_per_mwh}/MWh marginal electricity charge · fixed costs excluded.</p>
    </section>
  );
}
