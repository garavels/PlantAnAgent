import { useState } from "react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import type { UnitId } from "../components/plant-3d/selection";
import { useWorkspace } from "./context";
import { Scene, ProcessDiagram } from "./Scene";
import { Plot } from "./Plot";
import { duration, eur, n, time, words } from "./format";
import type { V2DemoFrame } from "../types/v2";

const UNITS: [UnitId, string][] = [
  ["solar", "Solar array"],
  ["battery", "Battery"],
  ["electrolyser", "Electrolyser"],
  ["dac", "Direct air capture"],
  ["h2Tank", "Hydrogen storage"],
  ["co2Tank", "CO₂ storage"],
  ["methanator", "Methanator"],
  ["ch4Tank", "Methane storage"],
];
const colors = ["var(--stored)", "var(--h2)", "var(--co2)", "var(--ch4)"];
const CONTROL_MODES = { scheduled: "Scheduled", recovery: "Recovery", fallback: "Fallback" };
const metric = (frame: V2DemoFrame, key: string) =>
  Number(frame.metrics[key] ?? 0);
function Heading({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="w-page-heading">
      <div>
        <h1>{title}</h1>
      </div>
      {children}
    </div>
  );
}
function Stat({
  label,
  value,
  note,
}: {
  label: string;
  value: ReactNode;
  note?: string;
}) {
  return (
    <div className="w-stat">
      <span>{label}</span>
      <strong>{value}</strong>
      {note && <small>{note}</small>}
    </div>
  );
}
function WeatherLabel({ frame }: { frame: V2DemoFrame }) {
  const weather = frame.weather;
  return (
    <>
      <strong>
        {weather.status === "night"
          ? "Night at the plant"
          : weather.attenuation === null
            ? "Solar reduction unattributed"
            : `${n(weather.attenuation * 100, 0)}% below clear-sky potential`}
      </strong>
      <span>
        {n(weather.available_kw, 0)} kW available
        {weather.status === "night"
          ? " · cloud conditions unobserved"
          : ` / ${n(weather.clear_sky_kw, 0)} kW clear-sky potential`}
      </span>
    </>
  );
}

export function Overview() {
  const w = useWorkspace(),
    f = w.frame!,
    s = f.state,
    d = f.delivery;
  const events = w
    .snapshot!.events.filter((e) => e.decision_index <= w.index)
    .slice(-3)
    .reverse();
  return (
    <div className="w-overview">
      <Heading title="Overview" />
      <div className="w-overview-hero">
        <div className="w-overview-mission">
          <span className="w-eyebrow">Next methane delivery</span>
          <h2>
            {d.next_due_hours === null ? (
              "Mission settled"
            ) : (
              <>
                {n(d.next_requested_kg, 0)} <small>kg CH₄</small>
              </>
            )}
          </h2>
          <p>
            {d.next_due_hours === null
              ? "Inspect the final delivery and resource settlements."
              : `${time(d.next_due_hours)} · in ${duration(d.next_due_hours - f.clock.elapsed_hours)}`}
          </p>
          <div
            className="w-progress"
            role="img"
            aria-label={`${n(s.ch4_level_kg)} kg currently stored for a ${d.next_requested_kg} kg next request`}
          >
            <span
              style={{
                width: `${Math.min(100, (s.ch4_level_kg / Math.max(1, d.next_requested_kg)) * 100)}%`,
              }}
            />
          </div>
          <span>{n(s.ch4_level_kg)} kg currently stored</span>

          <Link to="/deliveries" className="w-link">
            Delivery schedule
          </Link>
        </div>
        <div className="w-overview-scene">
          <Scene compact />
          <div className="w-mission-weather">
            <WeatherLabel frame={f} />
            <Link to="/weather" className="w-link">
              Weather details
            </Link>
          </div>
          <Link className="w-scene-open" to="/plant">
            Plant details ↗
          </Link>
          <span className="w-scene-caption">Synthetic weather overlay</span>
        </div>
      </div>
      <div className="w-overview-bottom">
        <div className="w-stats w-stats--three">
          <Stat
            label="Battery reserve"
            value={`${n(s.battery_soc * 100)}%`}
            note={`${n(s.battery_energy_kwh, 0)} kWh stored`}
          />
          <Stat
            label="Electrolyser"
            value={words(f.commitment?.mode ?? "Unavailable")}
            note={`${n(s.h2_production_kg_h)} kg H₂/h`}
          />
          <Stat
            label={f.operating_policy ? "Inventory recovery" : "Daily resource return"}
            value={f.operating_policy ? (f.decision?.operation ? CONTROL_MODES[f.decision.operation.mode] : "Monitoring") : `In ${duration(f.clock.hours_to_cycle_end)}`}
            note={`${f.resources.filter((r) => r.margin < -1e-6).length} resources below ${f.operating_policy ? "opening reference" : "return minimum"}`}
          />
        </div>
        <div className="w-overview-events">
          <div className="w-panel-heading">
            <h3>Recent events</h3>
            <Link to="/events">All events</Link>
          </div>
          {events.map((e) => (
            <button
              key={e.id}
              onClick={() => {
                void w.selectFrame(e.decision_index);
                if (e.unit) w.setSelectedUnit(e.unit as UnitId);
              }}
            >
              <time>{time(e.at_hours)}</time>
              <span>{e.title}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function EquipmentInspector() {
  const w = useWorkspace(),
    f = w.frame!,
    s = f.state,
    unit = w.selectedUnit ?? "electrolyser",
    c = f.commitment;
  const label = UNITS.find((u) => u[0] === unit)?.[1] ?? "Power bus";
  const rows: [string, string][] =
    unit === "electrolyser"
      ? [
          ["Mode", words(c?.mode ?? "Unavailable")],
          ["Actual power", `${n(s.electrolyser_power_kw)} kW`],
          ["Hydrogen output", `${n(s.h2_production_kg_h)} kg/h`],
          ["Startup remaining", duration((c?.startup_remaining_s ?? 0) / 3600)],
          [
            "Minimum up remaining",
            duration((c?.min_up_remaining_s ?? 0) / 3600),
          ],
          [
            "Minimum down remaining",
            duration((c?.min_down_remaining_s ?? 0) / 3600),
          ],
          [
            "Cooldown remaining",
            duration((c?.cooldown_remaining_s ?? 0) / 3600),
          ],
          ["Ramp up headroom", `${n(c?.ramp_up_headroom_kw ?? 0)} kW`],
          [
            "Cold / hot starts",
            `${c?.cold_starts ?? 0} / ${c?.hot_starts ?? 0}`,
          ],
          [
            "Forced / aborted stops",
            `${c?.forced_stops ?? 0} / ${c?.aborted_starts ?? 0}`,
          ],
          ["Startup energy", `${n(c?.startup_energy_kwh ?? 0)} kWh`],
        ]
      : unit === "solar"
        ? [
            ["Available power", `${n(s.solar_available_kw)} kW`],
            ["Clear-sky potential", `${n(s.solar_potential_kw)} kW`],
            ["Unused available power", `${n(s.solar_curtailed_kw)} kW`],
            ["Weather observation", f.weather.status],
          ]
        : unit === "battery"
          ? [
              ["State of charge", `${n(s.battery_soc * 100)}%`],
              ["Stored energy", `${n(s.battery_energy_kwh)} kWh`],
              ["Signed battery power", `${n(s.battery_power_kw)} kW`],
              ["Throughput", `${n(metric(f, "battery_throughput_kwh"))} kWh`],
            ]
          : unit === "dac"
            ? [
                ["Status", s.dac_online ? "Online" : "Offline"],
                ["Power", `${n(s.dac_power_kw)} kW`],
                ["CO₂ captured", `${n(s.co2_capture_kg_h)} kg/h`],
              ]
            : unit === "methanator"
              ? [
                  ["Status", s.methanator_online ? "Online" : "Offline"],
                  ["Power", `${n(s.methanator_power_kw)} kW`],
                  ["Methane output", `${n(s.ch4_production_kg_h)} kg/h`],
                ]
              : unit === "hub"
                ? [
                    ["Solar available", `${n(s.solar_available_kw)} kW`],
                    [
                      "Battery support",
                      `${n(s.battery_to_electrolyser_kw)} kW`,
                    ],
                    ["Curtailed", `${n(s.solar_curtailed_kw)} kW`],
                  ]
                : [
                    [
                      "Stored",
                      `${n(unit === "h2Tank" ? s.h2_level_kg : unit === "co2Tank" ? s.co2_level_kg : s.ch4_level_kg)} kg`,
                    ],
                    [
                      "Capacity",
                      `${n(unit === "h2Tank" ? s.h2_capacity_kg : unit === "co2Tank" ? s.co2_capacity_kg : s.ch4_capacity_kg)} kg`,
                    ],
                  ];
  return (
    <aside className="w-inspector">
      <div className="w-panel-heading">
        <h2>{label}</h2>
        <span>{time(f.clock.elapsed_hours)}</span>
      </div>
      <dl className="w-properties">
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>

      <Link to="/planning" className="w-link">
        Controller decision
      </Link>
    </aside>
  );
}

export function Plant() {
  const w = useWorkspace();
  const [diagram, setDiagram] = useState(false);
  return (
    <>
      <Heading title="Plant">
        <div className="w-segment">
          <button aria-pressed={!diagram} onClick={() => setDiagram(false)}>
            3D plant
          </button>
          <button aria-pressed={diagram} onClick={() => setDiagram(true)}>
            Process diagram
          </button>
        </div>
      </Heading>
      <div className="w-plant-layout">
        <section className="w-plant-main">
          {diagram ? (
            <ProcessDiagram frame={w.frame!} snapshot={w.snapshot!} />
          ) : (
            <Scene />
          )}
          <div className="w-unit-picker" aria-label="Select equipment">
            {UNITS.map(([id, label]) => (
              <button
                key={id}
                aria-pressed={w.selectedUnit === id}
                onClick={() => w.setSelectedUnit(id)}
              >
                {label}
              </button>
            ))}
          </div>
          <p className="w-note">
            {diagram ? "Process state" : "Synthetic weather overlay"}
          </p>
        </section>
        <EquipmentInspector />
      </div>
    </>
  );
}

export function Weather() {
  const w = useWorkspace(),
    f = w.frame!,
    [windowIndex, setWindowIndex] = useState(0),
    [compare, setCompare] = useState(false);
  const forecast = f.forecasts,
    chosen = forecast.solar[windowIndex],
    history = w.snapshot!.history.slice(0, w.index + 1);
  const issue = forecast.issued_at_hours;
  const previous = f.decision?.input_forecasts;
  const previousSolar =
    compare && previous
      ? [
          {
            label: `Previous issue (${time(previous.issued_at_hours)})`,
            color: "var(--co2)",
            dashed: true,
            step: true,
            points: previous.solar.flatMap((p) => [
              { x: previous.issued_at_hours + p.offset_hours, y: p.mean_kw },
              {
                x: previous.issued_at_hours + p.end_offset_hours,
                y: p.mean_kw,
              },
            ]),
          },
        ]
      : [];
  return (
    <>
      <Heading title="Weather" />
      <div className="w-weather-layout">
        <div className="w-weather-map">
          <Scene overhead />
          <div className="w-map-label">
            <span>Synthetic weather view</span>
            <strong>Plant and solar field</strong>
            <small>Illustrative cloud shapes · no satellite observations</small>
          </div>
          <div className="w-map-readout">
            <WeatherLabel frame={f} />
          </div>
        </div>
        <aside className="w-weather-forecast">
          <div className="w-panel-heading">
            <h2>Solar forecast</h2>
            <span>Issued {time(issue)}</span>
          </div>

          <div className="w-forecast-windows">
            {forecast.solar.map((p, i) => (
              <button
                key={i}
                aria-pressed={windowIndex === i}
                onClick={() => setWindowIndex(i)}
              >
                <span>
                  +{p.offset_hours}–{p.end_offset_hours} h
                </span>
                <strong>
                  {n(p.mean_kw, 0)} <small>kW</small>
                </strong>
                <small>p10 {n(p.p10_kw, 0)} kW</small>
                <span
                  className="w-availability"
                  style={{
                    width: `${Math.min(100, (p.mean_kw / w.snapshot!.config.solar.peak_kw) * 100)}%`,
                  }}
                />
              </button>
            ))}
          </div>
          <div className="w-forecast-selected">
            <span>
              {time(issue + chosen.offset_hours)} —{" "}
              {time(issue + chosen.end_offset_hours)}
            </span>
            <strong>{n(chosen.mean_kw, 0)} kW expected mean</strong>
            <p>p10 {n(chosen.p10_kw, 0)} kW · window mean</p>
          </div>
          <Link to="/planning" className="w-link">
            Production plan
          </Link>
        </aside>
      </div>
      <label className="w-check w-section">
        <input
          type="checkbox"
          checked={compare}
          disabled={!previous}
          onChange={(e) => setCompare(e.target.checked)}
        />
        Previous forecast issue
      </label>
      <div className="w-two-charts">
        <Plot
          title="Observed solar conditions"
          unit="kW"
          series={[
            {
              label: "Available",
              color: "var(--solar)",
              points: history.map((p) => ({
                x: p.elapsed_hours,
                y: p.solar_kw,
              })),
            },
            {
              label: "Clear-sky potential",
              color: "var(--text-dim)",
              dashed: true,
              points: history.map((p) => ({
                x: p.elapsed_hours,
                y: p.clear_sky_kw,
              })),
            },
          ]}
          cursor={f.clock.elapsed_hours}
          onSelect={(x) => void w.selectFrame(Math.round(x * 4))}
        />
        <Plot
          title="Issued solar forecast"
          unit="kW, window mean"
          domain={[issue, issue + 24]}
          series={[
            {
              label: "Mean",
              color: "var(--solar)",
              step: true,
              points: forecast.solar.flatMap((p) => [
                { x: issue + p.offset_hours, y: p.mean_kw },
                { x: issue + p.end_offset_hours, y: p.mean_kw },
              ]),
            },
            {
              label: "p10",
              color: "var(--text-dim)",
              dashed: true,
              step: true,
              points: forecast.solar.flatMap((p) => [
                { x: issue + p.offset_hours, y: p.p10_kw },
                { x: issue + p.end_offset_hours, y: p.p10_kw },
              ]),
            },
            ...previousSolar,
          ]}
        />
      </div>
      <p className="w-note">
        Window averages · site-wide solar availability · no spatial forecast
      </p>
    </>
  );
}

function OperatingPlan({ frame }: { frame: V2DemoFrame }) {
  const operation = frame.decision?.operation;
  if (!operation) return null;
  return (
    <section className="w-section" aria-label="Operating outlook">
      <div className="w-panel-heading">
        <h2>Operating outlook</h2>
        <span>Issued {time(operation.issued_at_hours)}</span>
      </div>
      <div className="w-stats w-stats--three">
        <Stat label="Control mode" value={CONTROL_MODES[operation.mode]}
          note={operation.mode === "fallback" ? `Planner: ${words(operation.solver_failure?.reason ?? "unavailable")}` : "Mean forecasts · operating targets"} />
        <Stat label="Predicted delivery shortfall"
          value={operation.projected_shortfall_kg === null ? "Unavailable" : `${n(operation.projected_shortfall_kg)} kg`}
          note={`Next ${n(operation.horizon_hours, 1)} hours`} />
        <Stat label="Inventory outlook" value={time(operation.issued_at_hours + operation.horizon_hours)}
          note="Fixed opening references" />
      </div>
      <div className="w-table">
        <table>
          <thead><tr><th>Resource</th><th>Opening reference</th><th>Projected inventory</th><th>Projected deficit</th></tr></thead>
          <tbody>{operation.resources.map((resource) => (
            <tr key={resource.id}>
              <th>{resource.label}</th>
              <td>{n(resource.reference)} {resource.unit}</td>
              <td>{resource.projected === null ? "—" : `${n(resource.projected)} ${resource.unit}`}</td>
              <td className={(resource.projected_deficit ?? 0) > 1e-6 ? "w-amber" : undefined}>
                {resource.projected_deficit === null ? "—" : `${n(resource.projected_deficit)} ${resource.unit}`}
              </td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </section>
  );
}

export function Planning() {
  const w = useWorkspace(),
    f = w.frame!,
    d = f.decision,
    plan = f.plan,
    forecast = d?.input_forecasts ?? f.forecasts;
  const [detail, setDetail] = useState(false),
    issue = forecast.issued_at_hours,
    domain: [number, number] = [issue, issue + 24];
  const predicted = plan?.points[0];
  const comparison = predicted
    ? ([
        ["Battery", "kWh", predicted.battery_kwh, f.state.battery_energy_kwh],
        ["Hydrogen", "kg", predicted.h2_kg, f.state.h2_level_kg],
        ["Carbon dioxide", "kg", predicted.co2_kg, f.state.co2_level_kg],
        ["Methane", "kg", predicted.ch4_kg, f.state.ch4_level_kg],
      ] as const)
    : [];
  return (
    <>
      <Heading title="Planning">
        <span className="w-tag">
          {d ? `Decision ${time(d.at_hours)} · ` : ""}
          {w.snapshot!.controller.id === "H"
            ? "Forecast-blind heuristic"
            : "Economic MPC · mean forecasts"}
        </span>
      </Heading>
      <OperatingPlan frame={f} />
      <div className="w-plan-top">
        <Plot
          title="Public solar at decision time"
          unit="kW"
          domain={domain}
          series={[
            {
              label: "Mean",
              color: "var(--solar)",
              step: true,
              points: forecast.solar.flatMap((p) => [
                { x: issue + p.offset_hours, y: p.mean_kw },
                { x: issue + p.end_offset_hours, y: p.mean_kw },
              ]),
            },
            {
              label: "p10",
              color: "var(--text-dim)",
              dashed: true,
              step: true,
              points: forecast.solar.flatMap((p) => [
                { x: issue + p.offset_hours, y: p.p10_kw },
                { x: issue + p.end_offset_hours, y: p.p10_kw },
              ]),
            },
          ]}
        />
        <Plot
          title="Public market at decision time"
          unit="EUR/MWh"
          domain={domain}
          series={[
            {
              label: "Mean price",
              color: "var(--co2)",
              step: true,
              points: forecast.prices.flatMap((p) => [
                { x: issue + p.offset_hours, y: p.mean },
                { x: issue + p.offset_hours + 1, y: p.mean },
              ]),
            },
          ]}
          band={forecast.prices.flatMap((p) => [
            { x: issue + p.offset_hours, low: p.p10, high: p.p90 },
            { x: issue + p.offset_hours + 1, low: p.p10, high: p.p90 },
          ])}
        />
      </div>
      {plan ? (
        <>
          <Plot
            title="Retained equipment schedule"
            unit="kW target"
            domain={domain}
            cursor={f.clock.elapsed_hours}
            series={(
              [
                "electrolyser_kw",
                "dac_kw",
                "methanator_kw",
                "battery_charge_kw",
              ] as const
            ).map((key, i) => ({
              label: [
                "Electrolyser",
                "DAC",
                "Methanator",
                "Battery charge ceiling",
              ][i],
              color: [
                "var(--feed)",
                "var(--co2)",
                "var(--ch4)",
                "var(--stored)",
              ][i],
              step: true,
              points: plan.points.map((p) => ({
                x: plan.issued_at_hours + p.offset_hours,
                y: p[key],
              })),
            }))}
          />
          <div className="w-two-charts">
            <Plot
              title="Planned gas inventories"
              unit="kg"
              domain={domain}
              series={(["h2_kg", "co2_kg", "ch4_kg"] as const).map(
                (key, i) => ({
                  label: ["H₂", "CO₂", "CH₄"][i],
                  color: colors[i + 1],
                  points: plan.points.map((p) => ({
                    x: plan.issued_at_hours + p.offset_hours + 0.25,
                    y: p[key],
                  })),
                }),
              )}
            />
            <Plot
              title="Planned battery inventory"
              unit="kWh"
              domain={domain}
              series={[
                {
                  label: "Battery",
                  color: "var(--stored)",
                  points: plan.points.map((p) => ({
                    x: plan.issued_at_hours + p.offset_hours + 0.25,
                    y: p.battery_kwh,
                  })),
                },
                {
                  label: f.operating_policy ? "Opening inventory reference" : "Daily return minimum",
                  color: "var(--text-dim)",
                  dashed: true,
                  points: [
                    { x: issue, y: f.resources[0].required },
                    { x: issue + 24, y: f.resources[0].required },
                  ],
                },
              ]}
            />
          </div>
        </>
      ) : (
        <div className="w-empty">
          <h2>
            {d?.operation?.mode === "fallback" ? "Heuristic fallback" : d ? "Immediate targets" : "No active plan"}
          </h2>
          <p>
            {d?.operation?.mode === "fallback" ? "Planner unavailable for this interval. Next planning attempt at the following decision." : d
              ? "It does not consume these forecasts or publish a forward schedule. Its requested and applied targets remain fully inspectable below."
              : "Run the economic MPC to inspect a real 24-hour schedule."}
          </p>
        </div>
      )}
      {predicted && (
        <section className="w-section">
          <h2>Predicted / actual inventory</h2>
          <p className="w-note">
            Interval ending {time(f.clock.elapsed_hours)} · forecast/model and
            execution differences combined
          </p>
          <div className="w-table">
            <table>
              <thead>
                <tr>
                  <th>Resource</th>
                  <th>Predicted</th>
                  <th>Actual</th>
                  <th>Actual − predicted</th>
                </tr>
              </thead>
              <tbody>
                {comparison.map(([label, unit, expected, actual]) => (
                  <tr key={label}>
                    <th>{label}</th>
                    <td>
                      {n(expected)} {unit}
                    </td>
                    <td>
                      {n(actual)} {unit}
                    </td>
                    <td>
                      {n(actual - expected)} {unit}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="w-plan-obligations">
            {f.delivery.commitments
              .filter((c) => c.due_hours > issue && c.due_hours <= issue + 24)
              .map((c) => (
                <span key={c.due_hours}>
                  {time(c.due_hours)} · {n(c.requested_kg, 0)} kg due
                </span>
              ))}
            <span>
              {f.operating_policy ? "Inventory review: " : "Resource return: "}
              {time(f.clock.elapsed_hours + f.clock.hours_to_cycle_end)}
            </span>
          </div>
        </section>
      )}
      {d && (
        <section className="w-decision">
          <div className="w-panel-heading">
            <h2>Command execution</h2>
            <span>
              {n(d.computation_ms, 0)} ms computation · {d.guard_adjustments}{" "}
              guard adjustments
            </span>
          </div>
          <div className="w-targets">
            {d.targets.map((t) => (
              <div key={t.id}>
                <span>{t.label}</span>
                <strong>
                  {n(t.requested_kw)} <small>requested kW</small>
                </strong>
                <span>{n(t.applied_mean_kw)} kW applied mean</span>
              </div>
            ))}
          </div>
          <p className="w-note">
            15-minute target · 1-minute dispatch and guard · battery ceiling
            excludes recovered spill
          </p>
          <details className="w-section">
            <summary>Controller notes</summary>
            <ul className="w-explanations">
              {d.notes.map((note, i) => (
                <li key={i}>{note}</li>
              ))}
            </ul>
          </details>
          <button
            className="w-button"
            aria-expanded={detail}
            onClick={() => setDetail(!detail)}
          >
            {detail ? "Hide minute execution" : "Minute execution"}
          </button>
          {detail && (
            <div className="w-table">
              <table>
                <thead>
                  <tr>
                    <th>Minute</th>
                    <th>Solar</th>
                    <th>Electrolyser</th>
                    <th>DAC</th>
                    <th>Methanator</th>
                    <th>Guard notes</th>
                  </tr>
                </thead>
                <tbody>
                  {d.minutes.map((m, i) => (
                    <tr key={i}>
                      <th>{i + 1}</th>
                      <td>{n(m.solar_kw)} kW</td>
                      <td>{n(m.electrolyser_kw)} kW</td>
                      <td>{n(m.dac_kw)} kW</td>
                      <td>{n(m.methanator_kw)} kW</td>
                      <td>{m.adjustments.join("; ") || "No adjustment"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}
    </>
  );
}

export function Deliveries() {
  const w = useWorkspace(),
    f = w.frame!,
    [day, setDay] = useState<number | null>(null),
    active = day ?? Math.min(7, Math.floor(f.clock.elapsed_hours / 24) + 1);
  const history = w.snapshot!.history.slice(0, w.index + 1);
  return (
    <>
      <Heading title="Deliveries & resources" />
      <div className="w-stats">
        <Stat label="Delivered" value={`${n(f.delivery.delivered_kg)} kg`} />
        <Stat
          label="Settled shortfall"
          value={`${n(f.delivery.unmet_kg)} kg`}
        />
        <Stat
          label="Produced in this run"
          value={`${n(metric(f, "ch4_produced_kg"))} kg`}
        />
        <Stat
          label="Currently stored"
          value={`${n(f.state.ch4_level_kg)} kg`}
        />
      </div>
      <section className="w-section">
        <div className="w-panel-heading">
          <h2>Delivery schedule</h2>
          <div className="w-days" aria-label="Select delivery day">
            {Array.from({ length: 7 }, (_, i) => (
              <button
                key={i}
                aria-pressed={active === i + 1}
                onClick={() => setDay(i + 1)}
              >
                Day {i + 1}
              </button>
            ))}
          </div>
        </div>
        <div className="w-delivery-cards">
          {f.delivery.commitments
            .filter(
              (d) =>
                d.due_hours > (active - 1) * 24 && d.due_hours <= active * 24,
            )
            .map((d) => (
              <article key={d.due_hours}>
                <time>{time(d.due_hours)}</time>
                <span className={`w-outcome w-outcome--${d.status}`}>
                  {words(d.status)}
                </span>
                <strong>
                  {n(d.delivered_kg ?? d.requested_kg)} <small>kg</small>
                </strong>
                <p>
                  {d.delivered_kg === null
                    ? "Requested · awaiting settlement"
                    : `${n(d.requested_kg)} kg requested · ${n(d.unmet_kg ?? 0)} kg shortfall`}
                </p>
                {d.status === "pending" && f.decision?.operation?.deliveries
                  .filter((prediction) => prediction.due_hours === d.due_hours)
                  .map((prediction) => <p key={prediction.due_hours} className={prediction.shortfall_kg > 1e-6 ? "w-amber" : undefined}>
                    Forecast {n(prediction.predicted_kg)} kg · {n(prediction.shortfall_kg)} kg shortfall
                  </p>)}
                {d.status !== "pending" && (
                  <button
                    className="w-button"
                    onClick={() => void w.selectFrame(d.due_hours * 4)}
                  >
                    Inspect settlement
                  </button>
                )}
              </article>
            ))}
        </div>
      </section>
      <section className="w-section">
        <div className="w-panel-heading">
          <h2>{f.operating_policy ? "Inventory recovery" : "Daily resource returns"}</h2>
          <span>{f.operating_policy ? "Review" : "Due"} in {duration(f.clock.hours_to_cycle_end)}</span>
        </div>
        <div className="w-resources">
          {f.resources.map((r, i) => (
            <div key={r.id}>
              <span>{r.label}</span>
              <strong>
                {n(r.current)} <small>{r.unit}</small>
              </strong>
              <div className="w-resource-track">
                <span
                  style={{
                    width: `${Math.min(100, (r.current / Math.max(1, r.required)) * 75)}%`,
                    background: colors[i],
                  }}
                />
                <i />
              </div>
              <small>
                {f.operating_policy ? "Opening reference" : "Return minimum"} {n(r.required)} {r.unit}
              </small>
              <p className={r.margin < -1e-6 ? "w-amber" : "w-green"}>
                {n(Math.abs(r.margin))} {r.unit}{" "}
                {r.margin < -1e-6 ? "to restore" : f.operating_policy ? "above reference" : "above minimum"}
              </p>
            </div>
          ))}
        </div>
        <p className="w-note">{f.operating_policy ? "Daily inventory review · fixed opening reference" : "Daily settlement · fixed opening reference"}</p>
        <div className="w-table">
          <table>
            <thead>
              <tr>
                <th>Cycle</th>
                {f.resources.map((r) => (
                  <th key={r.id}>{r.label}</th>
                ))}
                <th>Settlement</th>
              </tr>
            </thead>
            <tbody>
              {f.cycle_settlements.length ? (
                f.cycle_settlements.map((c) => (
                  <tr key={c.cycle_index}>
                    <th>
                      <button
                        className="w-text-button"
                        onClick={() => void w.selectFrame(c.cycle_index * 96)}
                      >
                        Day {c.cycle_index}
                      </button>
                    </th>
                    {f.resources.map((r) => (
                      <td key={r.id}>
                        {n(c.actual[r.id])} / {n(c.targets[r.id])} {r.unit}
                      </td>
                    ))}
                    <td>{f.operating_policy ? c.feasible ? "Restored" : "Deficit" : c.feasible ? "Met" : "Failed"}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={6}>{f.operating_policy ? "No daily inventory review yet." : "No daily return has settled at this time."}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
      <div className="w-two-charts">
        <Plot
          title="Gas inventory history"
          unit="kg"
          cursor={f.clock.elapsed_hours}
          onSelect={(x) => void w.selectFrame(Math.round(x * 4))}
          series={(["h2_kg", "co2_kg", "ch4_kg"] as const).map((key, i) => ({
            label: ["H₂", "CO₂", "CH₄"][i],
            color: colors[i + 1],
            points: history.map((p) => ({ x: p.elapsed_hours, y: p[key] })),
          }))}
        />
        <Plot
          title="Battery history"
          unit="% charge"
          series={[
            {
              label: "Battery",
              color: "var(--stored)",
              points: history.map((p) => ({
                x: p.elapsed_hours,
                y: p.battery_soc * 100,
              })),
            },
          ]}
        />
      </div>
    </>
  );
}

export function Economics() {
  const w = useWorkspace(),
    f = w.frame!,
    history = w.snapshot!.history.slice(0, w.index + 1),
    max = Math.max(1, ...f.economics.entries.map((e) => Math.abs(e.value_eur)));
  return (
    <>
      <Heading title="Economics" />
      <div className="w-economics-layout">
        <section className="w-value">
          <span>Net simulated value</span>
          <strong>{eur(f.economics.net_value_eur)}</strong>
          <p>Cumulative at {time(f.clock.elapsed_hours)}</p>
          <div className="w-properties">
            <Stat
              label="Delivered methane"
              value={`${n(f.delivery.delivered_kg)} kg`}
            />
            <Stat
              label="Battery throughput"
              value={`${n(metric(f, "battery_throughput_kwh"))} kWh`}
            />
            <Stat
              label="Startup & standby energy"
              value={`${n((f.commitment?.startup_energy_kwh ?? 0) + (f.commitment?.standby_energy_kwh ?? 0))} kWh`}
            />
          </div>
        </section>
        <section className="w-value-breakdown">
          <h2>Value contributions</h2>
          {f.economics.entries.map((e) => (
            <div key={e.id} className="w-value-row">
              <span>{e.label}</span>
              <strong className={e.value_eur > 0 ? "w-green" : ""}>
                {eur(e.value_eur)}
              </strong>
              <div>
                <i
                  style={{
                    width: `${(Math.abs(e.value_eur) / max) * 100}%`,
                    background:
                      e.value_eur > 0 ? "var(--stored)" : "var(--solar)",
                  }}
                />
              </div>
            </div>
          ))}
          <div className="w-value-total">
            <span>Total</span>
            <strong>{eur(f.economics.net_value_eur)}</strong>
          </div>
        </section>
      </div>
      <Plot
        title="Realized operating value"
        unit="EUR"
        cursor={f.clock.elapsed_hours}
        onSelect={(x) => void w.selectFrame(Math.round(x * 4))}
        series={[
          {
            label: "Net value",
            color: "var(--stored)",
            points: history.map((p) => ({
              x: p.elapsed_hours,
              y: p.net_value_eur,
            })),
          },
        ]}
      />
      <details className="w-section">
        <summary>Economic assumptions and interpretation</summary>
        <p>
          {f.economics.assumption_note} Renewable opportunity cost values
          accepted solar; it is not a grid trading settlement. Wear reserves are
          model assumptions, not measured equipment life.
        </p>
        <dl className="w-assumptions">
          {Object.entries(f.assumptions)
            .filter(([, v]) => typeof v === "number")
            .map(([k, v]) => (
              <div key={k}>
                <dt>{words(k)}</dt>
                <dd>{n(Number(v), 4)}</dd>
              </div>
            ))}
        </dl>
      </details>
    </>
  );
}

export function Events() {
  const w = useWorkspace(),
    [filter, setFilter] = useState("all"),
    [search, setSearch] = useState(""),
    [retrospective, setRetrospective] = useState(false);
  const items = w
    .snapshot!.events.filter(
      (e) =>
        (retrospective || e.decision_index <= w.index) &&
        (filter === "all" || e.kind === filter) &&
        `${e.title} ${e.detail}`.toLowerCase().includes(search.toLowerCase()),
    )
    .slice()
    .reverse();
  return (
    <>
      <Heading title="Events" />
      <div className="w-filters">
        <label>
          Event type
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            {[
              "all",
              "weather",
              "equipment",
              "warning",
              "recovery",
              "delivery",
              "cycle",
              "guard",
              "run",
            ].map((k) => (
              <option key={k}>{k}</option>
            ))}
          </select>
        </label>
        <label>
          Search events
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Equipment, delivery, guard…"
          />
        </label>
        <label className="w-check">
          <input
            type="checkbox"
            checked={retrospective}
            onChange={(e) => setRetrospective(e.target.checked)}
          />
          Entire retained run
        </label>
      </div>
      {retrospective && (
        <p className="w-note">
          Retrospective view: later events were not known to the controller at
          the selected time.
        </p>
      )}
      <div className="w-event-list">
        {items.length ? (
          items.map((e) => (
            <button
              key={e.id}
              onClick={() => {
                w.setPlaying(false);
                void w.selectFrame(e.decision_index);
                if (e.unit) w.setSelectedUnit(e.unit as UnitId);
              }}
            >
              <time>
                {time(e.at_hours)}
                <small>{duration(e.at_hours)} elapsed</small>
              </time>
              <span className={`w-event-kind w-event-kind--${e.kind}`}>
                {e.kind}
              </span>
              <div>
                <strong>{e.title}</strong>
                <p>{e.detail}</p>
              </div>
              <span aria-hidden>↗</span>
            </button>
          ))
        ) : (
          <div className="w-empty">
            <h2>No matching events</h2>
          </div>
        )}
      </div>
      <p className="w-note">
        Health and mode changes sampled every 15 minutes. Guard detail:
        Planning.
      </p>
    </>
  );
}

export function Evidence() {
  const w = useWorkspace(),
    f = w.frame!,
    snap = w.snapshot!;
  return (
    <>
      <Heading title="Evidence" />
      {snap.error && (
        <details className="w-section">
          <summary>
            Controller stop · {time(snap.frame.clock.elapsed_hours)}
          </summary>
          <p>{snap.error}</p>
        </details>
      )}
      <div className="w-evidence-status">
        <span className="w-tag">Disposable synthetic fixture</span>
        <h2>
          {w.recording
            ? "Public recording · integrity checked"
            : "Interactive simulation · recorded boundaries"}
        </h2>
        <p>
          {w.recording
            ? "The server checked the recording checksum and boundary sequence. This does not establish controller superiority or verify a research experiment."
            : "Commands cross the common dispatcher, independent guard and simulator. This run is for inspection and has not been admitted as research evidence."}
        </p>
      </div>
      <div className="w-table">
        <table>
          <thead>
            <tr>
              <th>Controller</th>
              <th>Information used</th>
              <th>Current availability</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>Heuristic · H</th>
              <td>Current physical observation</td>
              <td>Interactive</td>
            </tr>
            <tr>
              <th>Service MPC · M24</th>
              <td>Physical observation + public solar forecast</td>
              <td>Offline research implementation</td>
            </tr>
            <tr>
              <th>Economic MPC · MPC</th>
              <td>Public solar/market means · rolling inventory recovery</td>
              <td>Interactive · explicit heuristic fallback</td>
            </tr>
            <tr>
              <th>Research MPC · E-beta</th>
              <td>Public p10 solar · hard daily resource returns</td>
              <td>Offline research · historical recordings retained</td>
            </tr>
            <tr>
              <th>PPO · three families</th>
              <td>Full public beta observation</td>
              <td>Infrastructure ready; registered study pending</td>
            </tr>
          </tbody>
        </table>
      </div>
      <section className="w-section">
        <div className="w-panel-heading">
          <h2>Recorded outcomes</h2>
          <button className="w-button" onClick={() => void w.reloadCatalog()}>
            Refresh recordings
          </button>
        </div>
        <p className="w-note">
          Same fixed scenario; different controller information. Descriptive
          outcomes only. Failed cycles, shortfalls and stopped runs remain
          visible; no winner is selected.
        </p>
        <div className="w-table">
          <table>
            <thead>
              <tr>
                <th>Controller</th>
                <th>Run</th>
                <th>Retained hours</th>
                <th>Delivered / unmet kg</th>
                <th>Daily deficits</th>
                <th>Fallback decisions</th>
                <th>Net simulated value</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {w.catalog.map((c) => (
                <tr key={c.id}>
                  <th>
                    <button
                      className="w-text-button"
                      disabled={!c.available}
                      onClick={() => void w.loadRecording(c.id)}
                    >
                      {c.controller ?? "Unavailable"}
                    </button>
                  </th>
                  <td>{c.id.slice(0, 8)}</td>
                  <td>{c.hours ?? "—"}</td>
                  <td>
                    {c.delivered_kg === undefined
                      ? "—"
                      : `${n(c.delivered_kg)} / ${n(c.unmet_kg ?? 0)}`}
                  </td>
                  <td>{c.failed_cycles ?? "—"}</td>
                  <td>{c.fallback_decisions ?? "—"}</td>
                  <td>
                    {c.net_value_eur === undefined ? "—" : eur(c.net_value_eur)}
                  </td>
                  <td>{c.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="w-section">
        <h2>Run identity</h2>
        <dl className="w-evidence-grid">
          {Object.entries({
            "Run identity": snap.run_id,
            Controller: snap.controller.label,
            Task: snap.environment.id,
            "Task digest": snap.environment.task_digest,
            "Interface digest": snap.environment.interface_digest,
            "Public interface": `${snap.environment.observation_count} observations / ${snap.environment.action_count} actions`,
            "Selected decision": `${w.index} / ${snap.environment.horizon_hours * 4}`,
            "Command digest": f.decision?.command_digest ?? "No command yet",
            "Execution digest":
              f.decision?.execution_digest ?? "No execution yet",
            "Recording checksum":
              w.recording?.sha256 ??
              "Export to seal the current public history",
          }).map(([k, v]) => (
            <div key={k}>
              <dt>{k}</dt>
              <dd>{v}</dd>
            </div>
          ))}
        </dl>
        <button className="w-button" onClick={() => void w.exportRecording()}>
          Download this public recording
        </button>
      </section>
      <div className="w-capabilities">
        <section>
          <h3>Active capabilities</h3>
          <p>
            Plant physics, forecasts, recurring deliveries, daily inventory reviews,
            heuristic/economic MPC, execution guard, observed events and public
            replay.
          </p>
        </section>
        <section>
          <h3>Pending integration</h3>
          <p>
            Registered PPO training and selection, HEPA predictions,
            natural-language scenario authoring, supervisory takeover, site
            calibration and hardware-in-the-loop validation.
          </p>
          <p>
            These layers are not acting on this plant. Offline
            future-information planners and LP bounds retain their separate
            research role.
          </p>
        </section>
      </div>
    </>
  );
}
