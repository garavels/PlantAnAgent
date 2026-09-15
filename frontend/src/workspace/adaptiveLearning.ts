export type Resource = "battery" | "h2" | "co2" | "ch4";
export type ResourceValues = Record<Resource, number>;

export interface Artifact {
  id: string;
  sha256: string;
  bytes: number
}

export interface ResearchRollout {
  index: number;
  transitions: number;
  reward_sum: number;
  completed_episodes: number;
  complete_cycles: number;
  training_mean_reward: number | null;
  explained_variance: number | null;
  value_loss: number | null;
  approx_kl: number | null;
  clip_fraction: number | null;
  mean_episode_cost: number[] | null;
  multipliers: number[] | null;
}

export interface ResearchPolicy {
  id: string;
  label: string;
  campaign_id: string;
  algorithm: string;
  optimizer_seed: number;
  training_status: "completed";
  evaluation_status: "evaluated";
  operational_status: "experimental";
  transitions: number;
  minibatch_updates: number;
  reward_gamma: number;
  gae_lambda: number;
  reward_scale: number;
  checkpoint: Artifact;
  training_source: Artifact;
  rollouts: ResearchRollout[];
  limitations: string[];
}

export interface ResearchCase {
  seed: number;
  status: "completed" | "stopped";
  decisions: number;
  failures: string[];
  net_operating_value_eur: number | null;
  delivered_kg: number | null;
  battery_throughput_kwh: number | null;
  hard_safety_passed: boolean | null;
  prefix_hard_safety_passed: boolean;
  all_daily_returns_passed: boolean | null;
  mpc_modes: {
    scheduled: number;
    recovery: number;
    fallback: number
  } | null;
  deliveries: {
    ordinal: number;
    delivered_kg: number;
    requested_kg: number
  }[];
  daily_resources: {
    cycle: number;
    deficits: ResourceValues
  }[];
}

export interface ResearchActor {
  id: string;
  label: string;
  kind: "heuristic" | "mpc" | "ppo" | "initial";
  evidence_origin: "reused_batch008" | "fresh_batch009";
  cases: ResearchCase[];
}

export interface ResearchPairCase {
  seed: number;
  available: boolean;
  candidate_failures: string[];
  comparator_failures: string[];
  delta_net_operating_value_eur: number | null;
  delta_delivered_kg: number | null;
  delta_battery_throughput_kwh: number | null;
  delivery_regressions: {
    ordinal: number;
    difference_kg: number
  }[];
  resource_regressions: {
    cycle: number;
    resource: Resource;
    difference: number
  }[];
}

export interface ResearchComparison {
  candidate: string;
  comparator: string;
  evidence_batch: "008" | "009";
  cases: ResearchPairCase[];
}

export interface ResearchHistory {
  id: string;
  label: string;
  status: "completed" | "aborted" | "failed" | "incomplete";
  summary: string;
  policy_ids: string[];
  source: Artifact;
  limitations: string[];
}

export interface ResearchEvidence {
  schema_version: "plant-public-adaptive-learning-v1";
  source: {
    git_head: string;
    git_tree: string
  };
  provenance: {
    exported_from_closed_evidence: true;
    independent_verification_passed: true;
    development_only: true;
    artifacts: Artifact[]
  };
  task: {
    hours: 168;
    decisions: 672;
    deliveries: 28;
    requested_methane_kg: 6720;
    development_seeds: number[]
  };
  policies: ResearchPolicy[];
  actors: ResearchActor[];
  comparisons: ResearchComparison[];
  history: ResearchHistory[];
  registered_study: {
    status: "pending";
    families: 3;
    optimizer_seeds: 5;
    policies: 15;
    protected_seeds_opened: false
  };
  limitations: string[];
  sha256: string;
}
export const RESOURCES: {
  key: Resource;
  label: string;
  unit: string;
  color: string
}[] = [
    { key: "battery", label: "Battery", unit: "kWh", color: "var(--stored)" },
    { key: "h2", label: "Hydrogen", unit: "kg", color: "var(--h2)" },
    { key: "co2", label: "Carbon dioxide", unit: "kg", color: "var(--co2)" },
    { key: "ch4", label: "Methane", unit: "kg", color: "var(--ch4)" },
  ];

function requireValue(condition: unknown): asserts condition {
  if (!condition) throw new Error("Research evidence is incomplete or unsupported. Refresh after the evidence has been republished.");
}

function record(value: unknown): Record<string, unknown> {
  requireValue(value !== null && typeof value === "object" && !Array.isArray(value));
  return value as Record<string, unknown>;
}

function text(value: unknown) { requireValue(typeof value === "string" && value.length > 0 && value.length <= 4000); }

function finite(value: unknown) { requireValue(typeof value === "number" && Number.isFinite(value)); }

function integer(value: unknown) {
  finite(value);
  requireValue(Number.isSafeInteger(value) && (value as number) >= 0);
}

function nullableNumber(value: unknown) { if (value !== null) finite(value); }

function nullableBoolean(value: unknown) { requireValue(value === null || typeof value === "boolean"); }

function hash(value: unknown, length = 64) {
  requireValue(typeof value === "string" && new RegExp(`^[a-f0-9]{${length}}$`).test(value));
}

function array(value: unknown, max: number): unknown[] {
  requireValue(Array.isArray(value) && value.length <= max);
  return value;
}

function strings(value: unknown, max = 100) { array(value, max).forEach(text); }

function artifact(value: unknown) {
  const v = record(value);
  text(v.id);
  hash(v.sha256);
  integer(v.bytes);
}

function seeds(value: unknown, expected: number[]) {
  const rows = array(value, 8);
  requireValue(rows.length === expected.length);
  rows.forEach((row, index) => requireValue(record(row).seed === expected[index]));
  return rows;
}
/** Validate the complete rendered surface before accepting any API data. */

export function parseResearchResponse(input: unknown): ResearchEvidence | null {
  const body = record(input);
  if (body.status === "pending" && body.evidence === null) return null;
  requireValue(body.status === "available");
  const e = record(body.evidence);
  requireValue(e.schema_version === "plant-public-adaptive-learning-v1");
  hash(e.sha256);
  const source = record(e.source);
  hash(source.git_head, 40);
  hash(source.git_tree, 40);
  const provenance = record(e.provenance);
  requireValue(provenance.exported_from_closed_evidence === true && provenance.independent_verification_passed === true && provenance.development_only === true);
  array(provenance.artifacts, 200).forEach(artifact);
  const task = record(e.task);
  requireValue(task.hours === 168 && task.decisions === 672 && task.deliveries === 28 && task.requested_methane_kg === 6720);
  const expectedSeeds = Array.from({ length: 8 }, (_, i) => 9700300 + i);
  const taskSeeds = array(task.development_seeds, 8);
  requireValue(taskSeeds.length === 8 && taskSeeds.every((seed, i) => seed === expectedSeeds[i]));
  const policyIds = new Set<string>();
  const policies = array(e.policies, 4);
  requireValue(policies.length === 4);
  policies.forEach((policy) => {
    const p = record(policy);
    [p.id, p.label, p.campaign_id, p.algorithm].forEach(text);
    requireValue(!policyIds.has(p.id as string));
    policyIds.add(p.id as string);
    requireValue(p.training_status === "completed" && p.evaluation_status === "evaluated" && p.operational_status === "experimental");
    [p.optimizer_seed, p.transitions, p.minibatch_updates].forEach(integer);
    [p.reward_gamma, p.gae_lambda, p.reward_scale].forEach(finite);
    requireValue((p.reward_gamma as number) > 0 && (p.reward_gamma as number) <= 1);
    artifact(p.checkpoint);
    artifact(p.training_source);
    strings(p.limitations);
    const rollouts = array(p.rollouts, 8);
    requireValue(rollouts.length === 2 || rollouts.length === 8);
    rollouts.forEach((rollout, index) => {
      const r = record(rollout);
      [r.index, r.transitions, r.completed_episodes, r.complete_cycles].forEach(integer);
      requireValue(r.transitions === (index + 1) * 8192);
      finite(r.reward_sum);
      [r.training_mean_reward, r.explained_variance, r.value_loss, r.approx_kl, r.clip_fraction].forEach(nullableNumber);
      [r.mean_episode_cost, r.multipliers].forEach((cost) => {
        if (cost !== null) {
          const costs = array(cost, 4);
          requireValue(costs.length === 4);
          costs.forEach(finite);
        }
      });
    });
    requireValue(record(rollouts[rollouts.length - 1]).transitions === p.transitions);
  });
  const actors = array(e.actors, 9);
  requireValue(actors.length === 9);
  const actorIds = new Set<string>();
  actors.forEach((actor) => {
    const a = record(actor);
    text(a.id);
    text(a.label);
    requireValue(!actorIds.has(a.id as string));
    actorIds.add(a.id as string);
    requireValue(["heuristic", "mpc", "ppo", "initial"].includes(a.kind as string));
    requireValue(["reused_batch008", "fresh_batch009"].includes(a.evidence_origin as string));
    seeds(a.cases, expectedSeeds).forEach((row) => {
      const c = record(row);
      requireValue(c.status === "completed" || c.status === "stopped");
      integer(c.decisions);
      requireValue((c.decisions as number) <= 672);
      strings(c.failures);
      [c.net_operating_value_eur, c.delivered_kg, c.battery_throughput_kwh].forEach(nullableNumber);
      [c.hard_safety_passed, c.all_daily_returns_passed].forEach(nullableBoolean);
      requireValue(typeof c.prefix_hard_safety_passed === "boolean");
      if (c.mpc_modes !== null) {
        const modes = record(c.mpc_modes);
        [modes.scheduled, modes.recovery, modes.fallback].forEach(integer);
        requireValue((modes.scheduled as number) + (modes.recovery as number) + (modes.fallback as number) === c.decisions);
      }
      if (c.status === "completed") {
        requireValue(c.decisions === 672);
        [c.net_operating_value_eur, c.delivered_kg, c.battery_throughput_kwh].forEach(finite);
        requireValue(typeof c.hard_safety_passed === "boolean" && typeof c.all_daily_returns_passed === "boolean");
      } else {
        requireValue(c.net_operating_value_eur === null && c.delivered_kg === null && c.battery_throughput_kwh === null && c.hard_safety_passed === null && c.all_daily_returns_passed === null);
      }
      const deliveries = array(c.deliveries, 28);
      const cycles = array(c.daily_resources, 7);
      requireValue(c.status !== "completed" || (deliveries.length === 28 && cycles.length === 7));
      deliveries.forEach((delivery) => {
        const d = record(delivery);
        integer(d.ordinal);
        finite(d.delivered_kg);
        finite(d.requested_kg);
      });
      cycles.forEach((cycle) => {
        const d = record(cycle);
        integer(d.cycle);
        const deficits = record(d.deficits);
        RESOURCES.forEach(({ key }) => finite(deficits[key]));
      });
    });
  });
  policyIds.forEach((id) => requireValue(actorIds.has(id)));
  const pairs = new Set<string>();
  array(e.comparisons, 80).forEach((comparison) => {
    const c = record(comparison);
    requireValue(policyIds.has(c.candidate as string) && actorIds.has(c.comparator as string) && c.candidate !== c.comparator);
    const key = `${c.candidate}/${c.comparator}`;
    requireValue(!pairs.has(key));
    pairs.add(key);
    requireValue(c.evidence_batch === "008" || c.evidence_batch === "009");
    seeds(c.cases, expectedSeeds).forEach((row) => {
      const p = record(row);
      requireValue(typeof p.available === "boolean");
      strings(p.candidate_failures);
      strings(p.comparator_failures);
      [p.delta_net_operating_value_eur, p.delta_delivered_kg, p.delta_battery_throughput_kwh].forEach((v) => p.available ? finite(v) : requireValue(v === null));
      const deliveries = array(p.delivery_regressions, 28);
      const resources = array(p.resource_regressions, 28);
      requireValue(p.available || (!deliveries.length && !resources.length));
      deliveries.forEach((row) => {
        const d = record(row);
        integer(d.ordinal);
        finite(d.difference_kg);
        requireValue((d.difference_kg as number) < 0);
      });
      resources.forEach((row) => {
        const d = record(row);
        integer(d.cycle);
        finite(d.difference);
        requireValue((d.difference as number) > 0 && RESOURCES.some(({ key }) => key === d.resource));
      });
    });
  });
  array(e.history, 30).forEach((history) => {
    const h = record(history);
    [h.id, h.label, h.summary].forEach(text);
    requireValue(["completed", "aborted", "failed", "incomplete"].includes(h.status as string));
    strings(h.policy_ids, 30);
    artifact(h.source);
    strings(h.limitations);
  });
  const study = record(e.registered_study);
  requireValue(study.status === "pending" && study.families === 3 && study.optimizer_seeds === 5 && study.policies === 15 && study.protected_seeds_opened === false);
  strings(e.limitations);
  return e as unknown as ResearchEvidence;
}

export function mean(values: (number | null)[]): number | null {
  if (!values.length || values.some((value) => value === null)) return null;
  return (values as number[]).reduce((sum, value) => sum + value, 0) / values.length;
}

export function compareSummary(comparison: ResearchComparison | undefined) {
  const cases = comparison?.cases ?? [];
  const available = cases.length === 8 && cases.every((row) => row.available);
  return {
    available,
    meanValue: available ? mean(cases.map((row) => row.delta_net_operating_value_eur)) : null,
    worstValue: available ? Math.min(...cases.map((row) => row.delta_net_operating_value_eur!)) : null,
    wins: available ? cases.filter((row) => row.delta_net_operating_value_eur! > 0).length : null,
    deliveryRegressions: available ? cases.reduce((sum, row) => sum + row.delivery_regressions.length, 0) : null,
    resourceRegressions: available ? cases.reduce((sum, row) => sum + row.resource_regressions.length, 0) : null,
  };
}
