import type { V2DemoFrame } from "../types/v2";

export const AGENT_ACTION_DWELL_MS = 2600;

export type AgentAction = {
  unit: "electrolyser" | "battery" | "dac" | "methanator";
  label: string;
  title: string;
  explanation: string;
  requestedKw: number;
  appliedKw: number;
  requestLabel: string;
  appliedLabel: string;
  detail?: string;
};

// This is a tour around the plant. These targets were issued together.
const EQUIPMENT: {
  unit: AgentAction["unit"];
  label: string;
  title: string;
  process: string;
  guards: RegExp;
}[] = [
  {
    unit: "electrolyser",
    label: "Electrolyser",
    title: "Set electrolyser power",
    process: "the electrolyser",
    guards: /^(Electrolyser\b|Battery (is offline: battery support dropped|at its |support clamped ))/,
  },
  {
    unit: "battery",
    label: "Battery",
    title: "Set the battery charge ceiling",
    process: "battery charging",
    guards: /^(Battery\b|Electrolyser is offline: battery support dropped$)/,
  },
  {
    unit: "dac",
    label: "Direct air capture",
    title: "Set carbon-capture power",
    process: "direct air capture",
    guards: /^DAC\b/,
  },
  {
    unit: "methanator",
    label: "Methanator",
    title: "Set methanator power",
    process: "the methanator",
    guards: /^Methanator\b/,
  },
];

const validPower = (value: number) => Number.isFinite(value) && value >= 0;

// Preserve a real nonzero command, including the small PPO residual targets.
const power = (value: number) => value.toLocaleString("en-GB", value > 0 && value < 1
  ? { maximumSignificantDigits: 3 }
  : { maximumFractionDigits: 1 });

/** Describe recorded commands and outcomes without inferring policy motivation. */
export function deriveAgentActions(frame: V2DemoFrame): AgentAction[] {
  const decision = frame.decision;
  if (!decision) return [];

  return EQUIPMENT.flatMap((equipment): AgentAction[] => {
    const target = decision.targets.find((candidate) => candidate.id === equipment.unit);
    if (!target || !validPower(target.requested_kw) || !validPower(target.applied_mean_kw)) return [];

    const requested = power(target.requested_kw);
    const applied = power(target.applied_mean_kw);
    const battery = equipment.unit === "battery";
    const details: string[] = [];
    if (battery) {
      details.push("Charging can exceed the ceiling when the plant recovers unused solar power.");
      const supportRecorded = decision.minutes.length > 0 &&
        decision.minutes.every((minute) => validPower(minute.battery_support_kw));
      if (supportRecorded) {
        const supportMean = decision.minutes.reduce((sum, minute) => sum + minute.battery_support_kw, 0) /
          decision.minutes.length;
        details.push(`Automatic battery support to the electrolyser averaged ${power(supportMean)} kW; it is separate from the charge command.`);
      } else {
        details.push("Battery support to the electrolyser is dispatched automatically, separately from the charge command.");
      }
    }

    // Only retained execution notes with a known equipment prefix are attributed.
    // A requested/applied difference by itself is not evidence of a guard action.
    const adjustments = [...new Set(decision.minutes.flatMap((minute) => minute.adjustments ?? []))]
      .filter((note) => equipment.guards.test(note));
    if (adjustments.length) details.push(`Recorded adjustments: ${adjustments.join("; ")}`);
    if (decision.minutes.some((minute) => minute.adjustments === null)) {
      details.push("Detailed guard adjustments were not recorded for this interval.");
    }

    return [{
      unit: equipment.unit,
      label: equipment.label,
      title: equipment.title,
      explanation: battery
        ? `The controller set a ${requested} kW direct-charge ceiling. Battery charging averaged ${applied} kW during this interval.`
        : `The controller requested ${requested} kW for ${equipment.process}. Applied power averaged ${applied} kW during this interval.`,
      requestedKw: target.requested_kw,
      appliedKw: target.applied_mean_kw,
      requestLabel: battery ? "Charge ceiling" : "Requested power",
      appliedLabel: battery ? "Charging mean" : "Applied mean",
      ...(details.length ? { detail: details.join(" ") } : {}),
    }];
  });
}
