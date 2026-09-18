import { useEffect, useState } from "react";
import { eur, words } from "./format";

interface Actor {
  id: string;
  arm: string;
  algorithm: string | null;
  optimizer_seed: number | null;
  states: Record<string, number>;
  failures: Record<string, number>;
  execution_complete: number;
  hard_safety_passed: number;
  cycle_feasible: number;
  eligible_pairs: number | null;
  iqm_delta_eur: number | null;
  cvar20_delta_eur: number | null;
  frozen_champion: boolean;
}
interface Stage {
  schema_version: "plant-public-ppo-evaluation-v1";
  stage: "development" | "heldout";
  campaign_status: string;
  actors: Actor[];
  gate_checks: Record<string, boolean>;
  gate_passed: boolean;
  result_claim: string | null;
  stage_report_sha256: string;
}
const family: Record<string, string> = {
  "plant-beta-ppo-pi-lagrangian-v1": "PI-Lagrangian",
  "plant-beta-ppo-i-lagrangian-v1": "I-Lagrangian",
  "plant-beta-sb3-ppo-scalar-v1": "Scalar PPO",
};
export function LearningEvaluation({ refresh }: { refresh: number }) {
  const [data, setData] = useState<{
    development: Stage | null;
    heldout: Stage | null;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    setError(null);
    setData(null);
    void (async () => {
      try {
        const response = await fetch("/api/v2/ppo/evaluation", {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok)
          throw new Error(
            `Evaluation evidence unavailable (${response.status}).`,
          );
        const body = (await response.json()) as {
          development: Stage | null;
          heldout: Stage | null;
        };
        if (
          !Object.hasOwn(body, "development") ||
          !Object.hasOwn(body, "heldout") ||
          [body.development, body.heldout].some(
            (s) =>
              s &&
              (s.schema_version !== "plant-public-ppo-evaluation-v1" ||
                s.actors.length !== 18),
          )
        )
          throw new Error("Unsupported evaluation evidence.");
        setData(body);
      } catch (reason) {
        if (!controller.signal.aborted)
          setError(reason instanceof Error ? reason.message : String(reason));
      }
    })();
    return () => controller.abort();
  }, [refresh]);
  return (
    <section className="w-section">
      <h2>Evaluation</h2>
      {error ? (
        <p role="alert">{error}</p>
      ) : !data ? (
        <p>Loading evidence…</p>
      ) : !data.development ? (
        <p>Development pending · held-out sealed</p>
      ) : (
        <>
          {[data.development, data.heldout]
            .filter((s): s is Stage => s !== null)
            .map((stage) => (
              <div key={stage.stage}>
                <div className="w-panel-heading">
                  <h3>
                    {stage.stage === "development" ? "Development" : "Held-out"}
                  </h3>
                  <span className="w-tag">
                    {stage.campaign_status} · 576 retained slots
                  </span>
                </div>
                <p className="w-note">
                  Reconstructed at export · file integrity checked. Paired value
                  compares PPO with research E-beta and requires all 32 eligible
                  pairs.
                </p>
                <details className="w-section">
                  <summary>Execution, safety & resource returns</summary>
                  <div className="w-table">
                    <table>
                      <thead>
                        <tr>
                          <th>Controller</th>
                          <th>Complete episodes</th>
                          <th>Hard safety passed</th>
                          <th>All 7 returns feasible</th>
                        </tr>
                      </thead>
                      <tbody>
                        {stage.actors.map((a) => (
                          <tr key={a.id}>
                            <th>
                              {a.algorithm
                                ? `${family[a.algorithm]} · ${a.optimizer_seed}`
                                : a.arm}
                            </th>
                            <td>{a.execution_complete} / 32</td>
                            <td>{a.hard_safety_passed} / 32</td>
                            <td>{a.cycle_feasible} / 32</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
                <div className="w-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Controller</th>
                        <th>Verified cells</th>
                        <th>Eligible pairs</th>
                        <th>IQM Δ</th>
                        <th>Lower 20% CVaR Δ</th>
                        <th>Selection</th>
                        <th>Failures / missing</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stage.actors.map((a) => (
                        <tr key={a.id}>
                          <th>
                            {a.algorithm
                              ? `${family[a.algorithm]} · ${a.optimizer_seed}`
                              : a.arm}
                          </th>
                          <td>{a.states.verified ?? 0} / 32</td>
                          <td>
                            {a.eligible_pairs === null
                              ? "—"
                              : `${a.eligible_pairs} / 32`}
                          </td>
                          <td>
                            {a.iqm_delta_eur === null
                              ? "—"
                              : eur(a.iqm_delta_eur)}
                          </td>
                          <td>
                            {a.cvar20_delta_eur === null
                              ? "—"
                              : eur(a.cvar20_delta_eur)}
                          </td>
                          <td>{a.frozen_champion ? "Frozen champion" : "—"}</td>
                          <td>
                            {Object.entries(a.states)
                              .filter(
                                ([key, count]) =>
                                  key !== "verified" && count > 0,
                              )
                              .map(([key, count]) => `${words(key)}: ${count}`)
                              .join(" · ") || "—"}
                            {Object.keys(a.failures).length > 0 && (
                              <details>
                                <summary>Reasons</summary>
                                {Object.entries(a.failures).map(
                                  ([key, count]) => (
                                    <p key={key}>
                                      {words(key)}: {count}
                                    </p>
                                  ),
                                )}
                              </details>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <details className="w-section">
                  <summary>
                    {stage.stage === "development"
                      ? "Held-out eligibility"
                      : "Held-out success criteria"}{" "}
                    · {stage.gate_passed ? "Passed" : "Not passed"}
                  </summary>
                  <dl className="w-evidence-grid">
                    {Object.entries(stage.gate_checks).map(([key, value]) => (
                      <div key={key}>
                        <dt>{words(key)}</dt>
                        <dd>{value ? "Passed" : "Not passed"}</dd>
                      </div>
                    ))}
                    <div>
                      <dt>Stage report</dt>
                      <dd>{stage.stage_report_sha256}</dd>
                    </div>
                  </dl>
                </details>
                {stage.result_claim && <p>{stage.result_claim}</p>}
              </div>
            ))}
          {!data.heldout && (
            <p>
              {data.development.gate_passed
                ? "Held-out eligible · separate execution authorization required"
                : "Held-out sealed · development gate not passed"}
            </p>
          )}
        </>
      )}
    </section>
  );
}
