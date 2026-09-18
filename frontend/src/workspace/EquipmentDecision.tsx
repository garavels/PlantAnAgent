import type { UnitId } from "../components/plant-3d/selection";
import type { V2DemoFrame } from "../types/v2";
import { n, words } from "./format";
import { equipmentTime } from "./equipmentTime";

// Prefixes emitted by safety/validation.py. Battery support adjustments apply
// to both the battery and the electrolyser receiving its power.
const GUARD_NOTES_BY_UNIT: Partial<Record<UnitId, RegExp>> = {
  electrolyser: /^(Electrolyser\b|Battery (is offline: battery support dropped|at its |support clamped ))/,
  battery: /^(Battery\b|Electrolyser is offline: battery support dropped$)/,
  dac: /^DAC\b/,
  methanator: /^Methanator\b/,
  hub: /^(Electrolyser\b|Battery\b|DAC\b|Methanator\b)/,
};

/** Commands belong to the completed interval ending at this frame. */
export function EquipmentDecision({
  frame,
  unit,
}: {
  frame: V2DemoFrame;
  unit: UnitId;
}) {
  if (unit === "solar" || unit === "h2Tank" || unit === "co2Tank" || unit === "ch4Tank") {
    return null;
  }

  const decision = frame.decision;
  const displayTime = (hours: number) => equipmentTime(frame, hours);
  const heading = unit === "hub" ? "Dispatch comparison" : "Controller command";
  if (!decision) {
    return (
      <section className="w-equipment-decision">
        <h3>{heading}</h3>
        <p className="w-equipment-decision-note">
          No controller command has executed yet.
        </p>
      </section>
    );
  }

  const target = decision.targets.find((item) => item.id === unit);
  const showsBattery = unit === "battery" || unit === "hub";
  const support = decision.minutes.length
    ? decision.minutes.reduce((total, minute) => total + minute.battery_support_kw, 0) /
      decision.minutes.length
    : null;
  const supportRow = support !== null && (
    <div>
      <dt>Automatic support mean</dt>
      <dd>{n(support)} kW</dd>
    </div>
  );
  const guardPattern = GUARD_NOTES_BY_UNIT[unit];
  const adjustments = guardPattern
    ? [...new Set(decision.minutes.flatMap((minute) => minute.adjustments ?? []))]
        .filter((note) => guardPattern.test(note))
    : [];

  return (
    <section className="w-equipment-decision">
      <h3>{heading}</h3>
      <p className="w-equipment-decision-time">
        <span>Previous {n((frame.clock.elapsed_hours - decision.at_hours) * 60)}-minute interval · {decision.controller_id}</span>
        <span>{displayTime(decision.at_hours)} → {displayTime(frame.clock.elapsed_hours)}</span>
      </p>
      {unit === "hub" ? (
        <>
          <table className="w-equipment-decision-table">
            <caption>Power in kW · battery request is a charge ceiling</caption>
            <thead>
              <tr>
                <th scope="col">Component</th>
                <th scope="col">Requested</th>
                <th scope="col">Applied mean</th>
              </tr>
            </thead>
            <tbody>
              {decision.targets.map((item) => (
                <tr key={item.id}>
                  <th scope="row">{item.id === "battery" ? "Battery charging" : item.label}</th>
                  <td>{n(item.requested_kw)}</td>
                  <td>{n(item.applied_mean_kw)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {supportRow && <dl className="w-equipment-decision-outcome">{supportRow}</dl>}
        </>
      ) : target ? (
        <div className="w-equipment-decision-target">
          <dl>
            <div>
              <dt>{unit === "battery" ? "Charge ceiling" : "Requested power"}</dt>
              <dd>{n(target.requested_kw)} kW</dd>
            </div>
            <div>
              <dt>{unit === "battery" ? "Charging mean" : "Applied mean"}</dt>
              <dd>{n(target.applied_mean_kw)} kW</dd>
            </div>
            {unit === "battery" && supportRow}
          </dl>
        </div>
      ) : (
        <p className="w-equipment-decision-note">
          No power target was recorded for this interval.
        </p>
      )}
      {showsBattery && (
        <details className="w-equipment-decision-about">
          <summary>About battery charging</summary>
          <p className="w-equipment-decision-note">
            The charge ceiling excludes recovered solar spill, so charging can
            exceed it. Battery support to the electrolyser is dispatched automatically.
          </p>
        </details>
      )}
      {adjustments.length > 0 && (
        <div className="w-equipment-decision-execution">
          <h4>Execution adjustments</h4>
          <p className="w-equipment-decision-note">Recorded during the previous interval.</p>
          <ul className="w-equipment-decision-adjustments">
            {adjustments.map((note) => <li key={note}>{words(note)}</li>)}
          </ul>
        </div>
      )}
      {decision.minutes.some((minute) => minute.adjustments === null) && (
        <p className="w-equipment-decision-note">Detailed guard adjustments were not recorded for this interval.</p>
      )}
    </section>
  );
}
