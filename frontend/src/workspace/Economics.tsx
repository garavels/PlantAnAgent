import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import type { V2DemoFrame } from "../types/v2";
import { readFrame } from "./api";
import { useWorkspace } from "./context";
import { duration, eur, n } from "./format";
import {
  economicSummary,
  inventoryChangeSinceOpening,
  ledgerValue,
  periodStartIndex,
} from "./economicsModel";
import type { EconomicPeriod } from "./economicsModel";
import { Plot } from "./Plot";
import "./economics.css";

type Summary = ReturnType<typeof economicSummary>;
const signed = (value: number) => `${value > 0.004 ? "+" : ""}${eur(value)}`;
const quantity = (value: number | null, unit: string, digits = 1) =>
  value === null ? "Not recorded" : `${n(value, digits)} ${unit}`;
const LABELS: Record<string, string> = {
  delivered_service_value: "Methane delivered",
  unmet_delivery_penalty: "Missed deliveries",
  solar_opportunity_cost: "Electricity used",
  battery_throughput_reserve: "Battery wear",
  electrolyser_life_reserve: "Electrolyser wear",
  methane_endowment_liability_change: "Gas inventory adjustment",
  battery_endowment_liability_change: "Battery inventory adjustment",
};
const GROUPS = [
  {
    title: "Delivery service",
    ids: ["delivered_service_value", "unmet_delivery_penalty"],
  },
  {
    title: "Energy & equipment",
    ids: [
      "solar_opportunity_cost",
      "battery_throughput_reserve",
      "electrolyser_life_reserve",
    ],
  },
  {
    title: "Inventory adjustments",
    ids: [
      "methane_endowment_liability_change",
      "battery_endowment_liability_change",
    ],
  },
];

// Read historical boundaries without moving the workspace cursor. A keyed result
// prevents a late response from showing figures from another period or run.
function useBoundary(index: number) {
  const w = useWorkspace();
  const run = w.snapshot!.run_id;
  const key = `${run}:${index}`;
  const direct =
    w.recording?.frames[index] ?? (index === w.index ? w.frame : null);
  const [result, setResult] = useState<{
    key: string;
    frame?: V2DemoFrame;
    error?: string;
  } | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (direct) return;
    let active = true;
    void readFrame(run, index).then(
      (frame) => {
        if (active) setResult({ key, frame });
      },
      () => {
        if (active)
          setResult({ key, error: "The opening reading could not be loaded." });
      },
    );
    return () => {
      active = false;
    };
  }, [direct, index, key, run, attempt]);
  return {
    frame: direct ?? (result?.key === key ? result.frame : null),
    error: !direct && result?.key === key ? result.error : null,
    retry: () => {
      setResult(null);
      setAttempt((value) => value + 1);
    },
  };
}

function assumption(frame: V2DemoFrame, key: string) {
  const value = frame.assumptions[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
function rate(frame: V2DemoFrame, key: string, unit: string) {
  const value = assumption(frame, key);
  return value === null ? "Rate not recorded" : `€${n(value, 4)}/${unit}`;
}
function explanation(id: string, summary: Summary, frame: V2DemoFrame) {
  switch (id) {
    case "delivered_service_value":
      return `${n(summary.deliveredKg)} kg delivered × ${rate(frame, "delivered_service_value_eur_per_kg", "kg")}`;
    case "unmet_delivery_penalty":
      return `${n(summary.unmetKg)} kg unfulfilled × ${rate(frame, "unmet_delivery_penalty_eur_per_kg", "kg")} at settlement`;
    case "solar_opportunity_cost": {
      const cost = summary.entries.find((entry) => entry.id === id)?.value;
      const price =
        cost !== undefined &&
        summary.acceptedSolarKwh !== null &&
        summary.acceptedSolarKwh > 0
          ? ` · €${n((-cost / summary.acceptedSolarKwh) * 1000, 2)}/MWh weighted price`
          : "";
      return `${quantity(summary.acceptedSolarKwh, "kWh solar accepted")}${price}`;
    }
    case "battery_throughput_reserve":
      return `${quantity(summary.batteryThroughputKwh, "kWh charged + discharged")} × ${rate(frame, "battery_throughput_eur_per_kwh", "kWh")}`;
    case "electrolyser_life_reserve":
      return `${quantity(summary.operatingHours, "h operating", 2)} · ${quantity(summary.coldStarts, summary.coldStarts === 1 ? "cold start" : "cold starts")} · ${quantity(summary.hotStarts, summary.hotStarts === 1 ? "hot start" : "hot starts")}`;
    case "methane_endowment_liability_change":
      return "Change in depletion of methane and convertible H₂ / CO₂, valued against opening stock.";
    case "battery_endowment_liability_change":
      return "Change in stored-energy depletion. Replenishment reverses earlier charges.";
    default:
      return "Recorded contribution to the modelled operating value.";
  }
}

function ValueBreakdown({
  summary,
  frame,
}: {
  summary: Summary;
  frame: V2DemoFrame;
}) {
  const max = Math.max(
    1,
    ...summary.entries.map((entry) => Math.abs(entry.value)),
  );
  const known = GROUPS.flatMap((group) => group.ids);
  const other = summary.entries.filter((entry) => !known.includes(entry.id));
  const groups = other.length
    ? [
        ...GROUPS,
        {
          title: "Other recorded contributions",
          ids: other.map((entry) => entry.id),
        },
      ]
    : GROUPS;
  return (
    <section className="ec-ledger" aria-labelledby="ec-drivers-title">
      <h2 id="ec-drivers-title">What drove the result</h2>
      <p className="ec-muted">Signed contributions in the selected period.</p>
      {groups.map((group) => (
        <div className="ec-ledger-group" key={group.title}>
          <h3>{group.title}</h3>
          {group.ids.map((id) => {
            const entry = summary.entries.find((item) => item.id === id);
            return (
              <div className="ec-ledger-row" key={id}>
                <div>
                  <strong>{LABELS[id] ?? entry?.label ?? id}</strong>
                  <p>{explanation(id, summary, frame)}</p>
                </div>
                <div className="ec-ledger-amount">
                  <strong
                    className={
                      entry && entry.value > 0.004 ? "ec-credit" : undefined
                    }
                  >
                    {entry ? signed(entry.value) : "Not recorded"}
                  </strong>
                  <span className="ec-bar" aria-hidden="true">
                    <i
                      style={{
                        width: `${entry ? (Math.abs(entry.value) / max) * 100 : 0}%`,
                        background:
                          entry && entry.value > 0
                            ? "var(--stored)"
                            : "var(--solar)",
                      }}
                    />
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      ))}
      <div className="ec-ledger-total">
        <span>Modelled operating value</span>
        <strong>{signed(summary.netValue)}</strong>
      </div>
    </section>
  );
}

function Inventory({ frame, openingFrame }: { frame: V2DemoFrame; openingFrame?: V2DemoFrame | null }) {
  const w = useWorkspace();
  const methane = ledgerValue(frame, "methane_endowment_liability_change");
  const battery = ledgerValue(frame, "battery_endowment_liability_change");
  const provision =
    frame.clock.decision_index === 0
      ? 0
      : methane === null || battery === null
        ? null
        : -(methane + battery);
  return (
    <section className="ec-inventory" aria-labelledby="ec-inventory-title">
      <h2 id="ec-inventory-title">Resources at this time</h2>
      <p className="ec-muted">
        {w.time(frame.clock.elapsed_hours)} · compared with run opening
      </p>
      <table>
        <thead>
          <tr>
            <th scope="col">Resource</th>
            <th scope="col">Stored</th>
            <th scope="col">Change</th>
          </tr>
        </thead>
        <tbody>
          {frame.resources.map((resource) => {
            const change = inventoryChangeSinceOpening(frame, resource, openingFrame);
            return (
              <tr key={resource.id}>
                <th scope="row">
                  {resource.label}
                  <small>{resource.unit}</small>
                </th>
                <td>{n(resource.current)}</td>
                <td
                  className={
                    change !== null && change < -0.05 ? "ec-caution" : undefined
                  }
                >
                  {change === null
                    ? "—"
                    : `${change > 0.05 ? "+" : ""}${n(change)}`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="ec-provision">
        <span>Outstanding depletion charge</span>
        <strong>{provision === null ? "Not recorded" : eur(provision)}</strong>
      </div>
      <p className="ec-muted">
        Already included in the run’s value. Using opening stock increases this
        charge; replenishment reduces it. Individual gas changes are combined as
        methane-equivalent inventory.
      </p>
      <Link className="ec-link" to="/deliveries">
        Inspect deliveries & resources
      </Link>
    </section>
  );
}

const ASSUMPTIONS = [
  ["delivered_service_value_eur_per_kg", "Methane delivery value", "€/kg"],
  ["unmet_delivery_penalty_eur_per_kg", "Missed-delivery penalty", "€/kg"],
  ["battery_throughput_eur_per_kwh", "Battery wear rate", "€/kWh throughput"],
  [
    "battery_replacement_eur_per_kwh",
    "Battery replacement cost",
    "€/kWh capacity",
  ],
  ["battery_full_cycle_life", "Battery life", "full cycles"],
  ["electrolyser_capital_eur_per_kw", "Electrolyser capital cost", "€/kW"],
  ["electrolyser_life_h", "Electrolyser life", "hours"],
  ["electrolyser_low_load_wear_multiplier", "Low-load wear multiplier", "×"],
  ["cold_start_equivalent_operating_h", "Cold-start wear", "equivalent hours"],
  ["hot_start_equivalent_operating_h", "Hot-start wear", "equivalent hours"],
  [
    "aborted_start_equivalent_operating_h",
    "Aborted-start wear",
    "equivalent hours",
  ],
  [
    "forced_stop_equivalent_operating_h",
    "Forced-stop wear",
    "equivalent hours",
  ],
  [
    "methane_endowment_value_eur_per_kg_ch4_equivalent",
    "Gas depletion value",
    "€/kg CH₄ equivalent",
  ],
  [
    "battery_endowment_value_eur_per_kwh",
    "Battery depletion value",
    "€/kWh stored",
  ],
];

export function Economics() {
  const w = useWorkspace();
  const research = w.recording?.schema === "plant-research-playback-v1" ? w.recording.provenance : null;
  const frame = w.frame!;
  const [period, setPeriod] = useState<EconomicPeriod>("run");
  const startIndex = periodStartIndex(w.index, period);
  const opening = useBoundary(startIndex);
  const runOpening = useBoundary(0);
  const previous = useBoundary(Math.max(0, w.index - 1));
  const summary = opening.frame ? economicSummary(frame, opening.frame) : null;
  const last = previous.frame ? economicSummary(frame, previous.frame) : null;
  const start = opening.frame?.clock.elapsed_hours ?? startIndex / 4;
  const end = frame.clock.elapsed_hours;
  const history = w.snapshot!.history.filter(
    (point) => point.elapsed_hours >= start && point.elapsed_hours <= end,
  );
  const settlements = frame.delivery.commitments.filter(
    (item) =>
      item.status !== "pending" &&
      item.due_hours > start &&
      item.due_hours <= end,
  );
  const changes =
    last?.entries
      .filter((entry) => Math.abs(entry.value) >= 0.005)
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 3) ?? [];
  const events = w
    .snapshot!.events.filter(
      (event) =>
        event.at_hours > start &&
        event.at_hours <= end &&
        event.decision_index <= w.index,
    )
    .slice(-4)
    .reverse();
  return (
    <div className="ec-page">
      <header className="ec-heading">
        <div>
          <h1>Economics</h1>
          <p>
            Delivery service, energy use and the resources behind the result.
          </p>
        </div>
        <fieldset className="ec-period">
          <legend>Reporting period</legend>
          {(
            [
              ["run", "Run so far"],
              ["day", "Last 24 hours"],
              ["interval", "Last interval"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              aria-pressed={period === value}
              onClick={() => setPeriod(value)}
            >
              {label}
            </button>
          ))}
        </fieldset>
      </header>
      {w.siteForecast && <p className="w-note">{w.siteForecast.site.name} · Weather drives the simulated plant. Electricity charges are model assumptions; no local market prices were downloaded.</p>}
      {research && <p className="w-note">Dedicated off-grid PV · €0/MWh marginal electricity charge. Fixed solar capital and operating costs are excluded from this operating contribution.</p>}
      <div className="ec-period-caption">
        <span>
          {w.time(start)} – {w.time(end)}{" "}
          <span className="ec-muted">({duration(end - start)})</span>
        </span>
        <span className="ec-muted">Illustrative simulation economics</span>
      </div>
      {!summary ? (
        <div className="ec-loading" role={opening.error ? "alert" : "status"}>
          <p>{opening.error ?? "Loading the period’s opening reading…"}</p>
          {opening.error && (
            <button className="w-button" onClick={opening.retry}>
              Retry reading
            </button>
          )}
        </div>
      ) : (
        <>
          <section
            className="ec-result"
            aria-label="Operating result for the selected period"
          >
            <div>
              <span>Modelled operating value</span>
              <strong>{signed(summary.netValue)}</strong>
              <p>Includes wear and inventory adjustments</p>
            </div>
            <div>
              <span>Delivery fulfilment</span>
              <strong
                className={
                  summary.fillRate === null ? "ec-result-empty" : undefined
                }
              >
                {summary.fillRate === null
                  ? "No deliveries due"
                  : `${n(summary.fillRate * 100)}%`}
              </strong>
              <p>
                {summary.settledCount
                  ? `${n(summary.deliveredKg)} / ${n(summary.requestedKg)} kg · ${summary.metCount} of ${summary.settledCount} commitments met`
                  : "Value is booked when a delivery settles"}
              </p>
            </div>
            <div>
              <span>Unfulfilled methane</span>
              <strong
                className={summary.unmetKg > 0 ? "ec-caution" : undefined}
              >
                {n(summary.unmetKg)} <small>kg</small>
              </strong>
              <p>From settled commitments in this period</p>
            </div>
          </section>
          <div className="ec-main-grid">
            <ValueBreakdown summary={summary} frame={frame} />
            <aside className="ec-context">
              <Inventory frame={frame} openingFrame={runOpening.frame} />
              <section
                className="ec-exposure"
                aria-labelledby="ec-exposure-title"
              >
                <h2 id="ec-exposure-title">Equipment behind the cost</h2>
                <p className="ec-muted">In the selected period</p>
                <dl>
                  <div>
                    <dt>Methane produced</dt>
                    <dd>{quantity(summary.producedKg, "kg")}</dd>
                  </div>
                  <div>
                    <dt>Electrolyser at low load</dt>
                    <dd>{quantity(summary.lowLoadHours, "h")}</dd>
                  </div>
                  <div>
                    <dt>Forced / controlled stops</dt>
                    <dd>
                      {summary.forcedStops === null ||
                      summary.controlledStops === null
                        ? "Not recorded"
                        : `${n(summary.forcedStops)} / ${n(summary.controlledStops)}`}
                    </dd>
                  </div>
                  <div>
                    <dt>Startup & standby energy</dt>
                    <dd>{quantity(summary.nonproductiveKwh, "kWh")}</dd>
                  </div>
                </dl>
                <p className="ec-muted">
                  Startup and standby energy help explain usage; they are not an
                  extra charge in the ledger.
                </p>
                <Link className="ec-link" to="/plant">
                  Inspect equipment
                </Link>
              </section>
            </aside>
          </div>
          <section className="ec-timeline" aria-labelledby="ec-timeline-title">
            <h2 id="ec-timeline-title">When the value changed</h2>
            <p className="ec-muted">
              Production incurs costs between deliveries. Markers show delivery
              settlements; select one to inspect that time.
            </p>
            <Plot
              title="Operating value within the period"
              unit="EUR"
              domain={[start, end]}
              cursor={end}
              onSelect={(hours) => void w.selectFrame(Math.round(hours * 4))}
              series={[
                {
                  label: "Modelled value",
                  color: "var(--stored)",
                  points: history.map((point) => ({
                    x: point.elapsed_hours,
                    y:
                      point.net_value_eur -
                      opening.frame!.economics.net_value_eur,
                  })),
                },
              ]}
              markers={settlements.map((item) => ({
                x: item.due_hours,
                label: `${w.time(item.due_hours)}: ${n(item.delivered_kg ?? 0)} kg delivered, ${n(item.unmet_kg ?? 0)} kg unmet`,
                color:
                  item.status === "shortfall" ? "var(--alarm)" : "var(--ch4)",
              }))}
            />
            <div className="ec-timeline-detail">
              <div>
                <h3>Latest interval at the cursor</h3>
                <p className="ec-muted">
                  {w.index
                    ? `${w.time(end - 0.25)} – ${w.time(end)}`
                    : "At run opening"}
                </p>
                {last ? (
                  <>
                    <div className="ec-change-total">
                      <span>Change in value</span>
                      <strong>{signed(last.netValue)}</strong>
                    </div>
                    {changes.length ? (
                      <ul className="ec-changes">
                        {changes.map((entry) => (
                          <li key={entry.id}>
                            <span>{LABELS[entry.id] ?? entry.label}</span>
                            <strong>{signed(entry.value)}</strong>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="ec-muted">No value changes recorded.</p>
                    )}
                    {changes.length > 0 && (
                      <p className="ec-muted">
                        Largest contributions to this interval’s change.
                      </p>
                    )}
                  </>
                ) : (
                  <p className="ec-muted">
                    {previous.error ?? "Loading the previous reading…"}
                    {previous.error && (
                      <button className="w-button" onClick={previous.retry}>
                        Retry reading
                      </button>
                    )}
                  </p>
                )}
                <Link className="ec-link" to="/planning">
                  Inspect the decision at this time
                </Link>
              </div>
              <div>
                <h3>Recent plant events</h3>
                <p className="ec-muted">
                  Events in the period provide operating context.
                </p>
                {events.length ? (
                  <ul className="ec-events">
                    {events.map((event) => (
                      <li key={event.id}>
                        <button
                          onClick={() =>
                            void w.selectFrame(event.decision_index)
                          }
                        >
                          <time>{w.time(event.at_hours)}</time>
                          <span>{event.title}</span>
                          <span aria-hidden="true">↗</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="ec-muted">No events recorded in this period.</p>
                )}
                <Link className="ec-link" to="/events">
                  Inspect all events
                </Link>
              </div>
            </div>
          </section>
        </>
      )}
      <details className="ec-assumptions">
        <summary>Cost assumptions & interpretation</summary>
        <p>
          {frame.economics.assumption_note} This is a modelled operating score,
          not commercial profit. {research
            ? "Dedicated off-grid PV supplies this mission. The marginal electricity charge is €0/MWh, with no export opportunity. Fixed solar capital and operating costs have not been estimated and are excluded."
            : "Electricity is valued at its forgone opportunity; the plant has no grid-trading settlement. Negative prices are floored at zero for this cost."} Wear is an estimated replacement reserve, not measured equipment life.
        </p>
        <p>
          Inventory charges track depletion below opening resources and reverse
          as stocks recover. Gas inventory includes methane plus the
          methane-equivalent value of the limiting H₂ / CO₂ feedstocks. These
          adjustments are already part of the net value.
        </p>
        <dl>
          {ASSUMPTIONS.map(([key, label, unit]) => (
            <div key={key}>
              <dt>{label}</dt>
              <dd>
                {assumption(frame, key) === null
                  ? "Not recorded"
                  : `${n(assumption(frame, key)!, 4)} ${unit}`}
              </dd>
            </div>
          ))}
        </dl>
      </details>
    </div>
  );
}
