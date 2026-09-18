import type { PublicRecording, RecordingSummary, V2DemoSnapshot } from "../types/v2";

/** The server verifies custody and hashes; the client also checks the playback contract. */
export function checkRecording(record: PublicRecording) {
  const snapshot = record?.snapshot;
  if (!snapshot || !Array.isArray(record.frames) || !record.frames.length ||
    record.frames.length > (record.schema === "plant-site-recording-v1" ? 961 : 673) || record.frames.some((frame, index) =>
      frame?.clock?.decision_index !== index || frame.clock.elapsed_hours !== index / 4) ||
    snapshot.frame.clock.decision_index !== record.frames.length - 1) {
    throw new Error("This recording has an unsupported playback timeline.");
  }
  if (record.schema === "plant-site-recording-v1") {
    const site = snapshot.site_forecast;
    const parameters = snapshot.site_parameters;
    const scenario = parameters?.scenario;
    const horizon = scenario ? scenario.duration_days * 24 : 240;
    if (snapshot.mode !== "site_forecast" || !site || !parameters ||
      !/^[a-f0-9]{64}$/.test(site.id) || !Number.isFinite(Date.parse(site.start_time)) ||
      snapshot.environment.id !== "plant-site-forecast-v1" ||
      snapshot.environment.fixture_seed !== null || snapshot.environment.horizon_hours !== horizon ||
      !Number.isInteger(horizon / 24) || horizon < 24 || horizon > 240 || record.frames.length > horizon * 4 + 1 ||
      (scenario && (scenario.version !== "plant-scenario-v1" || scenario.unsupported.length > 0)) ||
      !Array.isArray(site.hours) || site.hours.length < 264 ||
      !["H", "MPC"].includes(snapshot.controller.id) ||
      record.frames.some((frame) => {
        const elapsed = frame.clock.elapsed_hours;
        const hour = site.hours[Math.floor(Math.max(0, elapsed * 3600 - 60) / 3600)];
        const sample = frame.weather?.site_sample;
        const cloudFields = ["cloud_cover_pct", "cloud_cover_low_pct", "cloud_cover_mid_pct", "cloud_cover_high_pct"] as const;
        return frame.clock.decision_count !== horizon * 4 || frame.weather?.reference_kind !== "forecast_pv" ||
          !sample || Date.parse(sample.time) !== Date.parse(site.start_time) + elapsed * 3_600_000 ||
          sample.interval_start !== hour?.time || sample.shortwave_w_m2 !== hour?.shortwave_w_m2 ||
          cloudFields.some((key) => (sample[key] ?? null) !== (hour?.[key] ?? null)) ||
          (sample.conditions_time !== undefined && Date.parse(sample.conditions_time) !== Date.parse(hour.time) + 3_600_000) ||
          frame.forecasts.forecast_id !== site.id;
      })) {
      throw new Error("This recording does not match its retained site forecast.");
    }
    return record;
  }
  if (record.schema === "plant-public-recording-v1") {
    if (snapshot.mode !== "interactive_fixture" || snapshot.environment.fixture_seed !== 9_000_034) {
      throw new Error("This is not a supported public fixture recording.");
    }
    return record;
  }
  if (record.schema === "plant-research-playback-v1") {
    const p = record.provenance;
    const final = record.frames.at(-1)!;
    const sha = (value: unknown) => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
    if (snapshot.mode === "recorded_research" && snapshot.controller.id === "PPO" &&
      snapshot.status === "complete" && !snapshot.error && record.frames.length === 673 &&
      snapshot.environment.fixture_seed === 9_930_000 && p?.scenario_seed === 9_930_000 &&
      p.optimizer_seed === 101 && sha(record.sha256) && sha(p.source_recording_sha256) &&
      sha(p.benchmark_contract_sha256) && sha(p.checkpoint_sha256) &&
      p.mission?.delivery_rate_kg_h === 25 && p.mission.horizon_h === 168 &&
      p.mission.deliveries_met === 28 && p.mission.deliveries_total === 28 &&
      p.mission.daily_returns_met === 7 && p.mission.daily_returns_total === 7 &&
      p.mission.mission_passed === true && p.mission.hard_safety_passed === true &&
      p.evaluation?.split === "held_out" && p.evaluation.passed === 14 && p.evaluation.total === 16 &&
      p.electricity?.mode === "dedicated_pv_no_export" && p.electricity.marginal_eur_per_mwh === 0 &&
      p.electricity.fixed_costs_included === false && final.delivery.unmet_kg === 0 &&
      Math.abs(final.delivery.delivered_kg - 4200) < 1e-6 &&
      final.delivery.commitments.length === 28 &&
      final.delivery.commitments.every((c) => c.status === "met" && c.unmet_kg === 0) &&
      final.cycle_settlements.length === 7 && final.cycle_settlements.every((c) => c.feasible)) {
      return record;
    }
    throw new Error("This recording does not match the verified trained PPO mission.");
  }
  throw new Error("This recording format is not supported.");
}

export function featuredRecording(catalog: RecordingSummary[]) {
  return catalog.find((item) => item.available && item.featured && item.mode === "recorded_research");
}

export function missionDescription(snapshot?: V2DemoSnapshot | null, record?: PublicRecording | null) {
  const scenario = snapshot?.site_parameters?.scenario;
  if (scenario) {
    const total = scenario.deliveries.reduce((sum, item) => sum + item.kg, 0);
    return `${scenario.name} · ${scenario.duration_days} days · ${scenario.deliveries.length} deliveries · ${total.toLocaleString()} kg CH₄`;
  }
  if (record?.schema === "plant-research-playback-v1") {
    return `${record.provenance.mission.delivery_rate_kg_h} kg CH₄/h · 150 kg every 6 hours · 7-day mission`;
  }
  const delivery = snapshot?.frame.delivery.commitments[0];
  return delivery
    ? `${delivery.requested_kg} kg every ${delivery.due_hours} hours · ${snapshot!.environment.horizon_hours / 24}-day mission`
    : "240 kg every 6 hours · 7-day mission";
}
