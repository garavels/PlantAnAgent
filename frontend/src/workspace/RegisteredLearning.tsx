import { useEffect, useState } from "react";
import { Plot } from "./Plot";
import { n } from "./format";
import { LearningEvaluation } from "./LearningEvaluation";

interface Rollout {
  index: number;
  transitions: number;
  objective_sum: number;
  completed_episodes: number;
  complete_cycles: number;
  raw_cost: number[];
  training_cost: number[];
  mean_episode_cost: number[] | null;
  multipliers: number[];
  cost_value_losses: number[];
}
interface Slot {
  algorithm: string;
  optimizer_seed: number;
  training_completed: boolean;
  checkpoint_admitted: boolean;
  failure: string | null;
  checkpoint_evidence_sha256: string | null;
  rollouts: Rollout[];
}
interface Training {
  schema_version: "plant-public-ppo-training-v1";
  source_commit: string;
  training_report_sha256: string;
  runtime_provenance_sha256: string;
  campaign_status: "completed" | "failed" | "aborted";
  slots: Slot[];
  sha256: string;
}
const FAMILIES: Record<string, string> = {
  "plant-beta-ppo-pi-lagrangian-v1": "PI-Lagrangian",
  "plant-beta-ppo-i-lagrangian-v1": "I-Lagrangian",
  "plant-beta-sb3-ppo-scalar-v1": "Scalar PPO",
};
const RESOURCES = ["Battery", "Hydrogen", "Carbon dioxide", "Methane"];
const COLORS = ["var(--stored)", "var(--h2)", "var(--co2)", "var(--ch4)"];

export function RegisteredLearning() {
  const [data, setData] = useState<Training | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refresh, setRefresh] = useState(0);
  const [selected, setSelected] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    setData(null);
    void (async () => {
      try {
        const response = await fetch("/api/v2/ppo/training", {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok)
          throw new Error(
            `Training evidence unavailable (${response.status}).`,
          );
        const body = (await response.json()) as {
          status: string;
          evidence: Training | null;
        };
        if (body.status === "pending" && body.evidence === null) return;
        if (
          body.status !== "available" ||
          body.evidence?.schema_version !== "plant-public-ppo-training-v1" ||
          body.evidence.slots.length !== 15
        ) {
          throw new Error("Unsupported training evidence.");
        }
        setData(body.evidence);
      } catch (reason) {
        if (!controller.signal.aborted)
          setError(reason instanceof Error ? reason.message : String(reason));
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    })();
    return () => controller.abort();
  }, [refresh]);
  const slot = data?.slots[selected];
  const rows = slot?.rollouts ?? [];
  const resourceSeries = (
    key: "raw_cost" | "multipliers" | "cost_value_losses",
  ) =>
    RESOURCES.map((label, index) => ({
      label,
      color: COLORS[index],
      points: rows.map((r) => ({ x: r.transitions, y: r[key][index] })),
    }));
  return (
    <>
      <div className="w-page-heading">
        <h2>Registered study</h2>
        <button
          className="w-button"
          disabled={loading}
          onClick={() => setRefresh(refresh + 1)}
        >
          Refresh evidence
        </button>
      </div>
      {error ? (
        <p className="w-banner" role="alert">
          {error}
        </p>
      ) : !data ? (
        <section className="w-section">
          <h2>
            {loading ? "Loading evidence…" : "Registered training pending"}
          </h2>
          <p className="w-note">PI-Lagrangian · I-Lagrangian · Scalar PPO</p>
          <p>5 optimizer seeds per family · 1,048,576 transitions per policy</p>
          <p className="w-note">
            Results become available after campaign closure and custody
            admission.
          </p>
        </section>
      ) : (
        <>
          <section className="w-section">
            <div className="w-panel-heading">
              <h2>PPO training</h2>
              <span className="w-tag">
                {data.campaign_status} ·{" "}
                {data.slots.filter((s) => s.checkpoint_admitted).length} / 15
                checkpoints admitted
              </span>
            </div>
            <p className="w-note">
              Custody admitted at export · file integrity checked · training
              diagnostics only
            </p>
            <div className="w-table">
              <table>
                <thead>
                  <tr>
                    <th>Family</th>
                    <th>Optimizer seed</th>
                    <th>Checkpoint</th>
                    <th>Transitions</th>
                    <th>Failure</th>
                  </tr>
                </thead>
                <tbody>
                  {data.slots.map((s, index) => (
                    <tr key={`${s.algorithm}-${s.optimizer_seed}`}>
                      <th>
                        <button
                          className="w-text-button"
                          aria-pressed={selected === index}
                          onClick={() => setSelected(index)}
                        >
                          {FAMILIES[s.algorithm]}
                        </button>
                      </th>
                      <td>{s.optimizer_seed}</td>
                      <td>
                        {s.checkpoint_admitted
                          ? "Admitted"
                          : s.training_completed
                            ? "Rejected"
                            : "Unavailable"}
                      </td>
                      <td>
                        {s.rollouts.length
                          ? n(s.rollouts.at(-1)!.transitions, 0)
                          : "—"}
                      </td>
                      <td>{s.failure ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <section className="w-section">
            <div className="w-panel-heading">
              <h2>Training curves</h2>
              <label>
                Policy{" "}
                <select
                  aria-label="Training policy"
                  value={selected}
                  onChange={(e) => setSelected(Number(e.target.value))}
                >
                  {data.slots.map((s, index) => (
                    <option key={index} value={index}>
                      {FAMILIES[s.algorithm]} · seed {s.optimizer_seed}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {rows.length ? (
              <>
                <p className="w-note">
                  Each point covers 8,192 transitions across four environments.
                  Objectives follow the frozen family reward; they are not
                  episode returns or directly comparable profit.
                </p>
                <div className="w-plan-top">
                  <Plot
                    title="Rollout objective"
                    unit="sum"
                    xLabel="Training transitions"
                    series={[
                      {
                        label: "Objective",
                        color: "var(--accent)",
                        points: rows.map((r) => ({
                          x: r.transitions,
                          y: r.objective_sum,
                        })),
                      },
                    ]}
                  />
                  <Plot
                    title="Resource deficits"
                    unit="sum of normalized cycle costs"
                    xLabel="Training transitions"
                    series={resourceSeries("raw_cost")}
                  />
                  <Plot
                    title="Constraint multipliers"
                    unit="λ"
                    xLabel="Training transitions"
                    series={resourceSeries("multipliers")}
                  />
                  <Plot
                    title="Cost value loss"
                    unit="loss"
                    xLabel="Training transitions"
                    series={resourceSeries("cost_value_losses")}
                  />
                </div>
                <p className="w-note">
                  {n(
                    rows.reduce((sum, r) => sum + r.completed_episodes, 0),
                    0,
                  )}{" "}
                  completed episodes ·{" "}
                  {n(
                    rows.reduce((sum, r) => sum + r.complete_cycles, 0),
                    0,
                  )}{" "}
                  complete cycles. Scalar PPO has no learned cost critic or
                  multiplier updates.
                </p>
              </>
            ) : (
              <p>No admitted curves for this slot. {slot?.failure}</p>
            )}
          </section>
          <details className="w-section">
            <summary>Evidence identity</summary>
            <dl className="w-evidence-grid">
              {Object.entries({
                Source: data.source_commit,
                "Training report": data.training_report_sha256,
                Runtime: data.runtime_provenance_sha256,
                Checkpoint: slot?.checkpoint_evidence_sha256 ?? "Unavailable",
                Export: data.sha256,
              }).map(([key, value]) => (
                <div key={key}>
                  <dt>{key}</dt>
                  <dd>{value}</dd>
                </div>
              ))}
            </dl>
          </details>
        </>
      )}
      <LearningEvaluation refresh={refresh} />
    </>
  );
}
