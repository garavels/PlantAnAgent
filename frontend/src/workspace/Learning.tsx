import { useEffect, useId, useState } from "react";
import type { ComponentProps, KeyboardEvent } from "react";
import { Plot } from "./Plot";
import { eur, n } from "./format";
import { RegisteredLearning } from "./RegisteredLearning";
import { compareSummary, mean, parseResearchResponse, RESOURCES } from "./adaptiveLearning";
import type { ResearchActor, ResearchCase, ResearchComparison, ResearchEvidence, ResearchPairCase, ResearchPolicy } from "./adaptiveLearning";
import "./learning.css";
const TABS = ["Training", "Evaluation", "History"] as const;
type Tab = typeof TABS[number];
const money = (value: number | null) => value === null ? "Unavailable" : eur(value);
const deltaMoney = (value: number | null) => value === null ? "Unavailable" : `${value > 0 ? "+" : ""}${eur(value)}`;
const quantity = (value: number | null, unit = "") => value === null ? "Unavailable" : `${n(value, 2)}${unit ? ` ${unit}` : ""}`;
const deltaQuantity = (value: number | null, unit = "") => {
  if (value === null) return "Unavailable";
  if (value !== 0 && Math.abs(value) < 0.005) {
    return `${value > 0 ? "+" : "−"}${Math.abs(value).toExponential(2)}${unit ? ` ${unit}` : ""}`;
  }
  return `${value > 0 ? "+" : ""}${quantity(value, unit)}`;
};
const origin = (actor: ResearchActor) => actor.evidence_origin === "fresh_batch009" ? "Batch 009" : "Reused from batch 008";
const count = (value: number | null, total: number) => value === null ? "Unavailable" : `${value} / ${total}`;
const passed = (value: boolean | null | undefined) => value === null || value === undefined ? "Unavailable" : value ? "Passed" : "Failed";

export function Learning() {
  const [data, setData] = useState<ResearchEvidence | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const [selected, setSelected] = useState("");
  const [comparatorId, setComparatorId] = useState("OperatingMPC");
  const [tab, setTab] = useState<Tab>("Training");
  const [seed, setSeed] = useState(9700300);
  const id = useId();
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    void (async () => {
      try {
        const response = await fetch("/api/v2/ppo/research", { cache: "no-store", signal: controller.signal });
        if (!response.ok) throw new Error(`Research evidence unavailable (${response.status}). Refresh to retry.`);
        const parsed = parseResearchResponse(await response.json());
        if (!controller.signal.aborted) setData(parsed);
      } catch (reason) {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Research evidence could not be loaded. Refresh to retry.");
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    })();
    return () => controller.abort();
  }, [refresh]);
  const policy = data?.policies.find((p) => p.id === selected) ?? data?.policies[0];
  const actor = data?.actors.find((a) => a.id === policy?.id);
  const comparisons = data?.comparisons.filter((c) => c.candidate === policy?.id) ?? [];
  const comparison = comparisons.find((c) => c.comparator === comparatorId) ?? comparisons.find((c) => c.comparator === "OperatingMPC") ?? comparisons[0];
  const comparator = data?.actors.find((a) => a.id === comparison?.comparator);
  const changeTab = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let next: number;
    if (event.key === "ArrowRight") next = (index + 1) % TABS.length;
    else if (event.key === "ArrowLeft") next = (index + TABS.length - 1) % TABS.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = TABS.length - 1;
    else return;
    event.preventDefault();
    setTab(TABS[next]);
    document.getElementById(`${id}-tab-${next}`)?.focus();
  };
  return <>
    <div className="w-page-heading">
      <h1>Learning</h1>
      <button className="w-button" disabled={loading} onClick={() => setRefresh((value) => value + 1)}>{loading ? "Loading evidence…" : "Refresh evidence"}</button>
    </div>
    {error && <p className="w-banner" role="alert">{error}{data ? " Showing the previously loaded evidence; the refresh did not succeed." : ""}</p>}
    {loading && data && <p className="w-learning-updating" role="status">Refreshing evidence…</p>}
    {!data ? <section className="w-section" aria-busy={loading}>
      <h2>{loading ? "Loading research evidence…" : error ? "Research evidence unavailable" : "Research evidence pending"}</h2>
      {!loading && !error && <p className="w-note">No published adaptive training record is available.</p>}
    </section> : <>
      <div className="w-learning-context">
        <span className="w-tag">Experimental</span>
        <p>{data.policies.length} trained policies · {data.task.development_seeds.length} development scenarios · {data.task.hours} hours per scenario</p>
        <p>Held-out confirmation pending</p>
      </div>
      <div className="w-learning-tabs" role="tablist" aria-label="Learning views">
        {TABS.map((name, index) => <button
          key={name}
          id={`${id}-tab-${index}`}
          role="tab"
          aria-selected={tab === name}
          aria-controls={`${id}-panel`}
          tabIndex={tab === name ? 0 : -1}
          onClick={() => setTab(name)}
          onKeyDown={(event) => changeTab(event, index)}
        >{name}</button>)}
      </div>
      <div
        className="w-learning-panel"
        id={`${id}-panel`}
        role="tabpanel"
        aria-labelledby={`${id}-tab-${TABS.indexOf(tab)}`}
        tabIndex={0}
      >
        {tab !== "History" && policy && <div className="w-learning-toolbar">
          <label>
            Policy
            <select aria-label="Research policy" value={policy.id} onChange={(event) => setSelected(event.target.value)}>
              {data.policies.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
            </select>
          </label>
          {tab === "Evaluation" && <label>
            Compare with
            <select
              aria-label="Evaluation comparator"
              value={comparison?.comparator ?? ""}
              disabled={!comparisons.length}
              onChange={(event) => setComparatorId(event.target.value)}
            >
              {comparisons.map((c) => <option key={c.comparator} value={c.comparator}>{data.actors.find((a) => a.id === c.comparator)?.label ?? c.comparator}</option>)}
            </select>
          </label>}
          <div className="w-learning-context">
            <span>Optimizer seed {policy.optimizer_seed}</span>
            <span>γ {policy.reward_gamma}</span>
          </div>
        </div>}
        {tab === "Training" && policy && <Training data={data} policy={policy} onSelect={setSelected} />}
        {tab === "Evaluation" && actor && <Evaluation
          data={data}
          actor={actor}
          comparator={comparator}
          comparison={comparison}
          seed={seed}
          onSeed={setSeed}
        />}
        {tab === "History" && <History data={data} />}
      </div>
    </>}
  </>;
}


function LearningPlot(props: ComponentProps<typeof Plot>) {
  const values = props.series.flatMap((series) => series.points.map((point) => point.y)).filter(Number.isFinite);
  const low = Math.min(0, ...values);
  const high = Math.max(0, ...values);
  const range = high - low;
  const digits = range >= 10 ? 1 : range >= 1 ? 2 : range >= 0.01 ? 4 : 6;
  return <Plot {...props} yDomain={range > 0 ? [low, high] : [0, 1]} valueDigits={digits} showPoints />;
}

function Training({ data, policy, onSelect }: {
  data: ResearchEvidence;
  policy: ResearchPolicy;
  onSelect: (id: string) => void
}) {
  const rows = policy.rollouts;
  const series = (key: "reward_sum" | "value_loss" | "explained_variance" | "approx_kl" | "clip_fraction", label: string, color = "var(--accent)") => [{
    label,
    color,
    points: rows.filter((row) => row[key] !== null).map((row) => ({ x: row.transitions, y: row[key]! }))
  }];
  const resourceSeries = (key: "mean_episode_cost" | "multipliers") => RESOURCES.map((resource, index) => ({
    label: resource.label,
    color: resource.color,
    points: rows.filter((row) => row[key] !== null).map((row) => ({ x: row.transitions, y: row[key]![index] }))
  }));
  return <>
    <section className="w-section">
      <div className="w-panel-heading">
        <h2>Policies</h2>
        <span className="w-tag">Training closed</span>
      </div>
      <div className="w-table">
        <table>
          <thead>
            <tr>
              <th>Policy</th>
              <th>Training</th>
              <th>Transitions</th>
              <th>Updates</th>
              <th>Evaluation</th>
              <th>Operational status</th>
            </tr>
          </thead>
          <tbody>
            {data.policies.map((p) => <tr key={p.id} className={p.id === policy.id ? "w-learning-selected" : undefined}>
              <th>
                <button className="w-text-button" aria-pressed={p.id === policy.id} onClick={() => onSelect(p.id)}>{p.label}</button>
                <span className="w-learning-state">Batch {p.campaign_id} · seed {p.optimizer_seed}{data.history.find((h) => h.id === p.campaign_id)?.status === "aborted" ? " · Campaign aborted" : ""}</span>
              </th>
              <td>Completed</td>
              <td>{n(p.transitions, 0)}</td>
              <td>{n(p.minibatch_updates, 0)}</td>
              <td>8 scenarios</td>
              <td>Experimental</td>
            </tr>)}
          </tbody>
        </table>
      </div>
    </section>
    <section className="w-section">
      <div className="w-panel-heading">
        <h2>Training history</h2>
        <span className="w-tag">{rows.length} recorded points</span>
      </div>
      <p className="w-note">Measured at rollout boundaries: 8,192 transitions per point. Reward is scaled by {policy.reward_scale}; it is not weekly operating value.</p>
      <div className="w-plan-top">
        <LearningPlot
          title="Rollout reward"
          unit="scaled reward sum"
          xLabel="Training transitions"
          series={series("reward_sum", "Reward")}
        />
        {rows.some((row) => row.value_loss !== null) ? <LearningPlot
          title="Critic MSE"
          unit="scaled reward²"
          xLabel="Training transitions"
          series={series("value_loss", "Critic MSE", "var(--solar)")}
        /> : <div>
          <h3>Critic MSE</h3>
          <p className="w-note">Not recorded in this training source.</p>
        </div>}
      </div>
      <details className="w-learning-detail">
        <summary>Optimization diagnostics</summary>
        <div className="w-plan-top">
          {rows.some((row) => row.explained_variance !== null) && <LearningPlot
            title="Explained variance"
            unit="ratio"
            xLabel="Training transitions"
            series={series("explained_variance", "Explained variance")}
          />}
          {rows.some((row) => row.approx_kl !== null) && <LearningPlot
            title="Approximate KL"
            unit="divergence"
            xLabel="Training transitions"
            series={series("approx_kl", "Approximate KL", "var(--co2)")}
          />}
          {rows.some((row) => row.mean_episode_cost !== null) && <LearningPlot
            title="Mean episode resource cost"
            unit="normalized cost"
            xLabel="Training transitions"
            series={resourceSeries("mean_episode_cost")}
          />}
          {rows.some((row) => row.multipliers !== null) && <LearningPlot
            title="Constraint multipliers"
            unit="λ"
            xLabel="Training transitions"
            series={resourceSeries("multipliers")}
          />}
        </div>
        <div className="w-table">
          <table>
            <caption>Recorded rollout summaries; critic diagnostics precede each update. Missing diagnostics are shown as —.</caption>
            <thead>
              <tr>
                <th>Transitions</th>
                <th>Reward sum</th>
                <th>Mean reward</th>
                <th>Critic MSE</th>
                <th>Explained variance</th>
                <th>Approx. KL</th>
                <th>Clip fraction</th>
                <th>Episodes</th>
                <th>Cycles</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => <tr key={r.transitions}>
                <th>{n(r.transitions, 0)}</th>
                <td>{n(r.reward_sum, 4)}</td>
                {(["training_mean_reward", "value_loss", "explained_variance", "approx_kl", "clip_fraction"] as const).map((key) => <td key={key}>{r[key] === null ? "—" : n(r[key], 6)}</td>)}
                <td>{r.completed_episodes}</td>
                <td>{r.complete_cycles}</td>
              </tr>)}
            </tbody>
          </table>
        </div>
      </details>
      <dl className="w-evidence-grid">
        <div><dt>Algorithm</dt><dd>{policy.algorithm}</dd></div>
        <div>
          <dt>Training parameters</dt>
          <dd>Reward γ {policy.reward_gamma} · GAE λ {policy.gae_lambda} · {n(policy.transitions, 0)} transitions</dd>
        </div>
      </dl>
      {policy.limitations.length > 0 && <ul className="w-learning-issues">
        {policy.limitations.map((limit) => <li key={limit}>{limit}</li>)}
      </ul>}
    </section>
    <EvidenceIdentity data={data} policy={policy} />
  </>;
}

function Evaluation({ data, actor, comparator, comparison, seed, onSeed }: {
  data: ResearchEvidence;
  actor: ResearchActor;
  comparator?: ResearchActor;
  comparison?: ResearchComparison;
  seed: number;
  onSeed: (seed: number) => void
}) {
  const summary = compareSummary(comparison);
  const actorCase = actor.cases.find((row) => row.seed === seed)!;
  const comparatorCase = comparator?.cases.find((row) => row.seed === seed);
  const pairCase = comparison?.cases.find((row) => row.seed === seed);
  const meanValue = mean(actor.cases.map((row) => row.net_operating_value_eur));
  const safetyCount = actor.cases.filter((row) => row.hard_safety_passed === true).length;
  const returnCount = actor.cases.filter((row) => row.all_daily_returns_passed === true).length;
  return <>
    <section className="w-section">
      <div className="w-panel-heading">
        <h2>Development evaluation</h2>
        <span className="w-tag">Experimental</span>
      </div>
      <p className="w-note">{origin(actor)}{comparator ? ` · Comparator: ${origin(comparator).toLowerCase()}` : ""}. {data.task.requested_methane_kg.toLocaleString("en-GB")} kg contracted across {data.task.deliveries} deliveries per scenario.</p>
      <dl className="w-learning-summary">
        <div>
          <dt>Mean operating value</dt>
          <dd>{money(meanValue)}<small>Per 168-hour scenario</small></dd>
        </div>
        <div>
          <dt>Mean change{comparator ? ` vs ${comparator.label}` : ""}</dt>
          <dd>
            {deltaMoney(summary.meanValue)}
            <small>{summary.available ? `${summary.wins} / 8 scenarios improved` : "Full-week comparison unavailable"}</small>
          </dd>
        </div>
        <div>
          <dt>Hard safety</dt>
          <dd>{safetyCount} / 8<small>Complete scenarios passed</small></dd>
        </div>
        <div>
          <dt>All daily returns</dt>
          <dd>{returnCount} / 8<small>Complete scenarios passed</small></dd>
        </div>
      </dl>
      <div className="w-table">
        <table>
          <thead>
            <tr><th>Acceptance check</th><th>Result</th><th>Observed outcome</th></tr>
          </thead>
          <tbody>
            <tr>
              <th>Hard safety</th>
              <td>{safetyCount === 8 ? "Passed" : "Failed"}</td>
              <td>{safetyCount} / 8 scenarios</td>
            </tr>
            <tr>
              <th>Every individual delivery preserved</th>
              <td>{summary.deliveryRegressions === null ? "Unavailable" : summary.deliveryRegressions === 0 ? "Passed" : "Failed"}</td>
              <td>{summary.deliveryRegressions === null ? "Comparison unavailable" : `${count(summary.deliveryRegressions, 224)} worsened`}</td>
            </tr>
            <tr>
              <th>Every daily resource return preserved</th>
              <td>{summary.resourceRegressions === null ? "Unavailable" : summary.resourceRegressions === 0 ? "Passed" : "Failed"}</td>
              <td>{summary.resourceRegressions === null ? "Comparison unavailable" : `${count(summary.resourceRegressions, 224)} resource boundaries worsened`}</td>
            </tr>
            <tr>
              <th>Worst scenario value change</th>
              <td colSpan={2}>{deltaMoney(summary.worstValue)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="w-note">Daily returns cover battery, H₂, CO₂ and CH₄ at seven boundaries. Regression counts use exact recorded signs. Positive resource differences mean larger deficits.</p>
      {!summary.available && <p className="w-banner">{comparison ? "The comparator has no complete weekly results. Stopped cases remain in the scenario record." : "This paired comparison was not published."}</p>}
      {comparator && <AbsoluteComparison actor={actor} comparator={comparator} />}
      {comparator && <PlanModes actor={comparator} />}
    </section>
    <section className="w-section">
      <h2>Scenarios</h2>
      <div className="w-table">
        <table>
          <thead>
            <tr>
              <th>Seed</th>
              <th>Operating value</th>
              <th>Value change</th>
              <th>Delivery change</th>
              <th>Worse deliveries</th>
              <th>Worse returns</th>
            </tr>
          </thead>
          <tbody>
            {actor.cases.map((row) => {
              const pair = comparison?.cases.find((p) => p.seed === row.seed);
              return <tr key={row.seed} className={row.seed === seed ? "w-learning-selected" : undefined}>
                <th>
                  <button
                    className="w-text-button"
                    aria-pressed={row.seed === seed}
                    aria-label={`Inspect scenario ${row.seed}`}
                    onClick={() => onSeed(row.seed)}
                  >{row.seed}</button>
                </th>
                <td>{money(row.net_operating_value_eur)}</td>
                <td>{deltaMoney(pair?.delta_net_operating_value_eur ?? null)}</td>
                <td>{deltaQuantity(pair?.delta_delivered_kg ?? null, "kg")}</td>
                <td>{pair?.available ? `${pair.delivery_regressions.length} / 28` : "Unavailable"}</td>
                <td>{pair?.available ? `${pair.resource_regressions.length} / 28` : "Unavailable"}</td>
              </tr>;
            })}
          </tbody>
        </table>
      </div>
      <Scenario
        actor={actor}
        candidate={actorCase}
        comparator={comparator}
        reference={comparatorCase}
        pair={pairCase}
      />
    </section>
    <details className="w-section">
      <summary>All controllers</summary>
      <ActorTable actors={data.actors} />
      <p className="w-note">Stopped prefixes are excluded from full-week metrics and complete-episode safety. Recovery executes an optimized MPC plan with inventory or forecast-delivery deficits; Fallback uses the heuristic after plan rejection.</p>
    </details>
    <details className="w-section">
      <summary>Evaluation limits</summary>
      <ul className="w-learning-issues">
        {data.limitations.map((limit) => <li key={limit}>{limit}</li>)}
      </ul>
    </details>
  </>;
}

function PlanModes({ actor }: { actor: ResearchActor }) {
  if (actor.cases.some((row) => row.mpc_modes === null)) return null;
  const modes = actor.cases.reduce((total, row) => ({
    scheduled: total.scheduled + row.mpc_modes!.scheduled,
    recovery: total.recovery + row.mpc_modes!.recovery,
    fallback: total.fallback + row.mpc_modes!.fallback
  }), { scheduled: 0, recovery: 0, fallback: 0 });
  return <p className="w-note">{actor.label}: {n(modes.scheduled + modes.recovery, 0)} optimized MPC plans; {n(modes.fallback, 0)} heuristic fallbacks. Recovery is an optimized plan with inventory or forecast-delivery deficits.</p>;
}

function AbsoluteComparison({ actor, comparator }: {
  actor: ResearchActor;
  comparator: ResearchActor
}) {
  const metrics: {
    label: string;
    value: (actor: ResearchActor) => number | null;
    format: (value: number | null) => string
  }[] = [
      {
        label: "Operating value",
        value: (a) => mean(a.cases.map((c) => c.net_operating_value_eur)),
        format: money
      },
      {
        label: "Delivered methane",
        value: (a) => mean(a.cases.map((c) => c.delivered_kg)),
        format: (v) => quantity(v, "kg")
      },
      {
        label: "Delivery fill",
        value: (a) => {
          const v = mean(a.cases.map((c) => c.delivered_kg));
          return v === null ? null : v / 6720 * 100;
        },
        format: (v) => quantity(v, "%")
      },
      {
        label: "Battery throughput",
        value: (a) => mean(a.cases.map((c) => c.battery_throughput_kwh)),
        format: (v) => quantity(v, "kWh")
      },
    ];
  return <div className="w-table">
    <table>
      <caption>Absolute means across the eight complete weekly scenarios</caption>
      <thead>
        <tr><th>Metric</th><th>{actor.label}</th><th>{comparator.label}</th></tr>
      </thead>
      <tbody>
        {metrics.map((metric) => <tr key={metric.label}>
          <th>{metric.label}</th>
          <td>{metric.format(metric.value(actor))}</td>
          <td>{metric.format(metric.value(comparator))}</td>
        </tr>)}
      </tbody>
    </table>
  </div>;
}

function Scenario({ actor, candidate, comparator, reference, pair }: {
  actor: ResearchActor;
  candidate: ResearchCase;
  comparator?: ResearchActor;
  reference?: ResearchCase;
  pair?: ResearchPairCase
}) {
  return <div className="w-learning-scenario" key={candidate.seed}>
    <div className="w-panel-heading">
      <h3>Scenario {candidate.seed}</h3>
      <span className="w-tag">{candidate.status === "completed" ? "Completed" : "Stopped"}</span>
    </div>
    <div className="w-table">
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>{actor.label}</th>
            <th>{comparator?.label ?? "Comparator"}</th>
            <th>Change</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>Operating value</th>
            <td>{money(candidate.net_operating_value_eur)}</td>
            <td>{money(reference?.net_operating_value_eur ?? null)}</td>
            <td>{deltaMoney(pair?.delta_net_operating_value_eur ?? null)}</td>
          </tr>
          <tr>
            <th>Delivered methane</th>
            <td>{quantity(candidate.delivered_kg, "kg")}</td>
            <td>{quantity(reference?.delivered_kg ?? null, "kg")}</td>
            <td>{deltaQuantity(pair?.delta_delivered_kg ?? null, "kg")}</td>
          </tr>
          <tr>
            <th>Battery throughput</th>
            <td>{quantity(candidate.battery_throughput_kwh, "kWh")}</td>
            <td>{quantity(reference?.battery_throughput_kwh ?? null, "kWh")}</td>
            <td>{deltaQuantity(pair?.delta_battery_throughput_kwh ?? null, "kWh")}</td>
          </tr>
          <tr>
            <th>Hard safety</th>
            <td>{passed(candidate.hard_safety_passed)}</td>
            <td>{passed(reference?.hard_safety_passed)}</td>
            <td>—</td>
          </tr>
          <tr>
            <th>All daily returns</th>
            <td>{passed(candidate.all_daily_returns_passed)}</td>
            <td>{passed(reference?.all_daily_returns_passed)}</td>
            <td>—</td>
          </tr>
        </tbody>
      </table>
    </div>
    {reference?.status === "stopped" && <p className="w-note">{comparator?.label} stopped after {reference.decisions} decisions. Complete-week outcomes are unavailable.</p>}
    {(candidate.failures.length > 0 || (reference?.failures.length ?? 0) > 0) && <ul className="w-learning-issues">
      {candidate.failures.map((reason) => <li key={`candidate-${reason}`}>{actor.label}: {reason}</li>)}
      {reference?.failures.map((reason) => <li key={`comparator-${reason}`}>{comparator?.label}: {reason}</li>)}
    </ul>}
    <details className="w-learning-detail">
      <summary>Deliveries · {pair?.available ? `${pair.delivery_regressions.length} worsened` : "comparison unavailable"}</summary>
      <div className="w-table">
        <table>
          <thead>
            <tr>
              <th>Delivery</th>
              <th>Required</th>
              <th>{actor.label}</th>
              <th>{comparator?.label ?? "Comparator"}</th>
              <th>Change</th>
            </tr>
          </thead>
          <tbody>
            {candidate.deliveries.map((delivery) => {
              const ref = reference?.deliveries.find((row) => row.ordinal === delivery.ordinal);
              const difference = pair?.available && ref ? delivery.delivered_kg - ref.delivered_kg : null;
              return <tr key={delivery.ordinal}>
                <th>{delivery.ordinal}</th>
                <td>{quantity(delivery.requested_kg, "kg")}</td>
                <td>{quantity(delivery.delivered_kg, "kg")}</td>
                <td>{quantity(ref?.delivered_kg ?? null, "kg")}</td>
                <td className={difference !== null && difference < 0 ? "w-amber" : undefined}>
                  {deltaQuantity(difference, "kg")}
                  {difference !== null && difference < 0 && <span className="w-learning-state">Worsened</span>}
                </td>
              </tr>;
            })}
          </tbody>
        </table>
      </div>
    </details>
    <details className="w-learning-detail">
      <summary>Daily resource deficits · {pair?.available ? `${pair.resource_regressions.length} worsened` : "comparison unavailable"}</summary>
      <div className="w-table">
        <table>
          <thead>
            <tr>
              <th>Cycle</th>
              <th>Resource</th>
              <th>{actor.label}</th>
              <th>{comparator?.label ?? "Comparator"}</th>
              <th>Change</th>
            </tr>
          </thead>
          <tbody>
            {candidate.daily_resources.flatMap((cycle) => RESOURCES.map((resource) => {
              const ref = reference?.daily_resources.find((row) => row.cycle === cycle.cycle);
              const difference = pair?.available && ref ? cycle.deficits[resource.key] - ref.deficits[resource.key] : null;
              return <tr key={`${cycle.cycle}-${resource.key}`}>
                <th>{cycle.cycle}</th>
                <th>{resource.label}</th>
                <td>{quantity(cycle.deficits[resource.key], resource.unit)}</td>
                <td>{quantity(ref?.deficits[resource.key] ?? null, resource.unit)}</td>
                <td className={difference !== null && difference > 0 ? "w-amber" : undefined}>
                  {deltaQuantity(difference, resource.unit)}
                  {difference !== null && difference > 0 && <span className="w-learning-state">Worsened</span>}
                </td>
              </tr>;
            }))}
          </tbody>
        </table>
      </div>
    </details>
  </div>;
}

function ActorTable({ actors }: { actors: ResearchActor[] }) {
  return <div className="w-table">
    <table>
      <thead>
        <tr>
          <th>Controller</th>
          <th>Evidence</th>
          <th>Complete</th>
          <th>Hard safety</th>
          <th>All daily returns</th>
          <th>Mean fill</th>
          <th>Mean operating value</th>
        </tr>
      </thead>
      <tbody>
        {actors.map((actor) => {
          const complete = actor.cases.filter((row) => row.status === "completed").length;
          const full = complete === 8;
          const delivered = mean(actor.cases.map((row) => row.delivered_kg));
          return <tr key={actor.id}>
            <th>
              {actor.label}
              {!full && <span className="w-learning-state">Stopped</span>}
            </th>
            <td>{origin(actor)}</td>
            <td>{complete} / 8</td>
            <td>{full ? count(actor.cases.filter((row) => row.hard_safety_passed).length, 8) : "Unavailable"}</td>
            <td>{full ? count(actor.cases.filter((row) => row.all_daily_returns_passed).length, 8) : "Unavailable"}</td>
            <td>{delivered === null ? "Unavailable" : `${n(delivered / 6720 * 100, 2)}%`}</td>
            <td>{money(mean(actor.cases.map((row) => row.net_operating_value_eur)))}</td>
          </tr>;
        })}
      </tbody>
    </table>
  </div>;
}

function History({ data }: { data: ResearchEvidence }) {
  const [registeredOpen, setRegisteredOpen] = useState(false);
  return <>
    <section className="w-section">
      <h2>Campaign history</h2>
      <div className="w-table">
        <table>
          <thead><tr><th>Campaign</th><th>Status</th><th>Outcome</th></tr></thead>
          <tbody>
            {data.history.map((history) => <tr key={history.id}>
              <th>{history.label}</th>
              <td>{history.status[0].toUpperCase() + history.status.slice(1)}</td>
              <td>
                {history.summary}
                {history.limitations.length > 0 && <details className="w-learning-detail">
                  <summary>Recorded limits</summary>
                  <ul className="w-learning-issues">
                    {history.limitations.map((limit) => <li key={limit}>{limit}</li>)}
                  </ul>
                </details>}
              </td>
            </tr>)}
          </tbody>
        </table>
      </div>
    </section>
    <details className="w-section" onToggle={(event) => setRegisteredOpen(event.currentTarget.open)}>
      <summary>Registered study · Pending</summary>
      <p className="w-note">3 families · 5 optimizer seeds · 15 policies. Protected seeds remain unopened.</p>
      {registeredOpen && <RegisteredLearning />}
    </details>
    <EvidenceIdentity data={data} />
  </>;
}

function EvidenceIdentity({ data, policy }: {
  data: ResearchEvidence;
  policy?: ResearchPolicy
}) {
  const identities = [
    ["Source", data.source.git_head], ["Source tree", data.source.git_tree], ["Public evidence", data.sha256],
    ...(policy ? [["Checkpoint", policy.checkpoint.sha256], ["Training source", policy.training_source.sha256]] : []),
  ];
  return <details className="w-section">
    <summary>Evidence identity</summary>
    <p className="w-note">Closed research records · independently verified · development evidence</p>
    <dl className="w-evidence-grid">
      {identities.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
    </dl>
    <details className="w-learning-detail">
      <summary>Source artifacts</summary>
      <dl className="w-evidence-grid">
        {data.provenance.artifacts.map((artifact) => <div key={artifact.id}>
          <dt>{artifact.id} · {n(artifact.bytes, 0)} bytes</dt>
          <dd>{artifact.sha256}</dd>
        </div>)}
      </dl>
    </details>
  </details>;
}
