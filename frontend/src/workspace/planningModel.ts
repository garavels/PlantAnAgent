import type { OperatingOutlook, V2DemoFrame } from "../types/v2";

type RetainedPlan = V2DemoFrame["plan"];

export interface LatestPlanDelivery {
  delivery: OperatingOutlook["deliveries"][number];
  issuedAtHours: number;
}

/** Use the retained plan's projection for this deadline, keeping its original age. */
export function latestPlanDelivery(frame: V2DemoFrame): LatestPlanDelivery | null {
  const { plan, decision, delivery, clock } = frame;
  const operation = decision?.operation;
  const due = delivery.next_due_hours;
  const timeTolerance = 1e-8;
  const quantityTolerance = 1e-6;
  if (
    !plan?.points.length || !operation || decision?.controller_id !== "MPC"
    || (operation.mode !== "scheduled" && operation.mode !== "recovery")
    || operation.forecast_statistic !== "issued_mean"
    || operation.restoration_policy !== "operating_target"
    || operation.fallback_controller !== null || operation.solver_failure !== null
    || due === null || !Number.isFinite(due)
    || !Number.isFinite(clock.elapsed_hours) || clock.elapsed_hours < 0
    || due <= clock.elapsed_hours + timeTolerance
    || !Number.isFinite(delivery.next_requested_kg) || delivery.next_requested_kg <= 0
    || !Number.isFinite(plan.issued_at_hours) || plan.issued_at_hours < 0
    || !Number.isFinite(operation.issued_at_hours)
    || Math.abs(operation.issued_at_hours - plan.issued_at_hours) > timeTolerance
    || plan.issued_at_hours > clock.elapsed_hours + timeTolerance
    || !Number.isFinite(operation.horizon_hours) || operation.horizon_hours <= 0
    || due > operation.issued_at_hours + operation.horizon_hours + timeTolerance
  ) return null;

  const matching = operation.deliveries.filter((row) => (
    Number.isFinite(row.due_hours) && Math.abs(row.due_hours - due) <= timeTolerance
  ));
  if (matching.length !== 1) return null;
  const projected = matching[0];
  if (
    ![projected.requested_kg, projected.predicted_kg, projected.shortfall_kg]
      .every((value) => Number.isFinite(value) && value >= 0)
    || Math.abs(projected.requested_kg - delivery.next_requested_kg) > quantityTolerance
    || projected.predicted_kg > projected.requested_kg + quantityTolerance
    || Math.abs(projected.shortfall_kg - Math.max(0, projected.requested_kg - projected.predicted_kg)) > quantityTolerance
  ) return null;

  return { delivery: projected, issuedAtHours: operation.issued_at_hours };
}

export interface RetainedPlanTiming {
  intervalHours: number;
  startHours: number;
  endHours: number;
  durationHours: number;
  domain: [number, number];
}

/** A plan row requests power at its start and predicts inventory at its end. */
export function retainedPlanTiming(
  plan: RetainedPlan,
  decisionMinutes: number,
): RetainedPlanTiming | null {
  if (!plan?.points.length || !Number.isFinite(decisionMinutes) || decisionMinutes <= 0) {
    return null;
  }
  const intervalHours = decisionMinutes / 60;
  const startHours = plan.issued_at_hours + Math.min(...plan.points.map((point) => point.offset_hours));
  const endHours = plan.issued_at_hours + Math.max(...plan.points.map((point) => point.offset_hours)) + intervalHours;
  if (!Number.isFinite(startHours) || !Number.isFinite(endHours)) return null;
  return {
    intervalHours,
    startHours,
    endHours,
    durationHours: endHours - startHours,
    domain: [startHours, endHours],
  };
}

/** Match the completed boundary, including when a site keeps an older plan. */
export function plannedInventoryAtFrame(
  frame: Pick<V2DemoFrame, "plan" | "clock">,
  decisionMinutes: number,
) {
  const plan = frame.plan;
  const timing = retainedPlanTiming(plan, decisionMinutes);
  if (!timing || !plan) return null;
  return plan.points.find((point) => Math.abs(
    plan.issued_at_hours + point.offset_hours + timing.intervalHours
      - frame.clock.elapsed_hours,
  ) < 1e-8) ?? null;
}
