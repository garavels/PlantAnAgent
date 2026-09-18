import { useState } from "react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import type { UnitId } from "../components/plant-3d/selection";
import { useWorkspace } from "./context";
import { Scene } from "./Scene";
import { EquipmentInspector } from "./EquipmentInspector";
import { EQUIPMENT } from "./equipmentCatalog";
import { Plot } from "./Plot";
import { duration, n, words } from "./format";
import type { V2DemoFrame } from "../types/v2";
import { ResearchMissionSummary } from "./ResearchPlayback";
import { featuredRecording } from "./recording";
import { SiteWeatherPanel } from "./SiteWeatherPanel";
import { AgentGuide, AgentViewToggle } from "./AgentGuide";
import { cloudReadout, missionDays, weatherReadout } from "./siteWeatherModel";
import { latestPlanDelivery, plannedInventoryAtFrame, retainedPlanTiming } from "./planningModel";
import "./planning.css";

const UNITS = EQUIPMENT;
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
  const readout = weatherReadout(frame.weather);
  return <><strong>{readout.heading}</strong><span>{readout.detail}</span></>;
}

export function Overview() {
  const w = useWorkspace(),
    f = w.frame!,
    s = f.state,
    d = f.delivery;
  const featured = featuredRecording(w.catalog);
  const events = w
    .snapshot!.events.filter((e) => e.decision_index <= w.index)
    .slice(-3)
    .reverse();
  return (
    <div className="w-overview">
      <Heading title="Overview" />
      {w.recording?.schema === "plant-research-playback-v1" ? (
        <ResearchMissionSummary provenance={w.recording.provenance} onInspectResult={() => {
          w.setPlaying(false);
          void w.selectFrame(w.latestIndex);
        }} />
      ) : !w.snapshot?.site_forecast && featured && (
        <div className="w-featured-recording">
          <div><strong>Successful trained PPO week</strong><p>25 kg CH₄/h · all deliveries and daily returns met · 14/16 untouched missions passed</p></div>
          <button className="w-button w-button--primary" disabled={w.reading} onClick={() => void w.loadRecording(featured.id)}>Open PPO demo</button>
        </div>
      )}
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
              : `${w.time(d.next_due_hours)} · in ${duration(d.next_due_hours - f.clock.elapsed_hours)}`}
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
          <span className="w-scene-caption">{w.snapshot?.site_forecast ? `${w.snapshot.site_forecast.site.name} · forecast-driven sunlight` : "Synthetic weather overlay"}</span>
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
              <time>{w.time(e.at_hours)}</time>
              <span>{e.title}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export function Plant() {
  const w = useWorkspace();
  return (
    <>
      <Heading title="Plant">
        <div className="w-plant-view-controls">
          <AgentViewToggle />
        </div>
      </Heading>
      <div className="w-plant-layout">
        <section className={`w-plant-main${w.agent.visible ? " w-plant-main--agent" : ""}`}>
          <Scene />
          <AgentGuide />
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
            {w.snapshot?.site_forecast ? `${w.snapshot.site_forecast.site.name} · Sun position follows the site and date; light follows simulated irradiance. ${cloudReadout(w.frame?.weather.site_sample).available ? cloudReadout(w.frame?.weather.site_sample).caption : "Cloud cover unavailable"}.` : "Synthetic weather overlay"}
          </p>
        </section>
        <EquipmentInspector key={w.selectedUnit ?? "electrolyser"} frame={w.frame!} config={w.snapshot!.config} unit={w.selectedUnit ?? "electrolyser"} />
      </div>
    </>
  );
}

export function Weather() {
  const w = useWorkspace();
  return w.snapshot?.site_forecast
    ? <><Heading title="Weather" /><SiteWeatherPanel forecast={w.snapshot.site_forecast} /></>
    : <SyntheticWeather />;
}

function SyntheticWeather() {
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
            label: `Previous issue (${w.time(previous.issued_at_hours)})`,
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
      <div className="sw-connection"><p><strong>No geographic site is active.</strong> This mission uses synthetic weather. Choose a location and select “Use in workspace” to drive the plant with that site's forecast.</p><Link className="w-button" to="/site">Choose plant location</Link></div>
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
            <span>{f.clock.elapsed_hours >= w.snapshot!.environment.horizon_hours ? "Last retained issue" : "Issued"} {w.time(issue)}</span>
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
              {w.time(issue + chosen.offset_hours)} —{" "}
              {w.time(issue + chosen.end_offset_hours)}
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

export function Planning() {
  const w = useWorkspace();
  const f = w.frame!;
  const snapshot = w.snapshot!;
  const plan = f.plan;
  const operation = f.decision?.operation;
  const timing = retainedPlanTiming(plan, snapshot.environment.decision_minutes);
  const projection = latestPlanDelivery(f);
  const predicted = plannedInventoryAtFrame(f, snapshot.environment.decision_minutes);
  const now = f.clock.elapsed_hours;
  const hasSchedule = plan && timing && timing.endHours > now && operation?.mode !== "fallback";
  const decidesOneInterval = snapshot.controller.id === "PPO" || snapshot.controller.id === "H";
  const nextDue = f.delivery.next_due_hours;
  const comparison = predicted ? ([
    ["Battery", "kWh", predicted.battery_kwh, f.state.battery_energy_kwh],
    ["Hydrogen", "kg", predicted.h2_kg, f.state.h2_level_kg],
    ["Carbon dioxide", "kg", predicted.co2_kg, f.state.co2_level_kg],
    ["Methane", "kg", predicted.ch4_kg, f.state.ch4_level_kg],
  ] as const) : [];

  return (
    <div className="w-planning">
      <Heading title="Planning">
        {plan && <span className="w-planning-stamp">Plan created {w.time(plan.issued_at_hours)}</span>}
      </Heading>

      <section className="w-planning-objective" aria-label="Next delivery and expected outcome">
        <div>
          <h2>Next delivery</h2>
          <p className="w-planning-delivery">{nextDue === null ? "All deliveries settled" : <>
            <strong>{n(f.delivery.next_requested_kg, 0)} kg CH₄</strong>
            <span>by {w.time(nextDue)}</span>
          </>}</p>
          <Link to="/deliveries" className="w-link">Delivery schedule</Link>
        </div>
        {hasSchedule && nextDue !== null && <div className="w-planning-outcome">
          <h2>Latest plan predicts</h2>
          {projection ? <>
            <p className="w-planning-delivery"><strong>{n(projection.delivery.predicted_kg)} kg delivered</strong></p>
            <p className={projection.delivery.shortfall_kg > 1e-6 ? "w-amber" : "w-planning-covered"}>
              {projection.delivery.shortfall_kg > 1e-6
                ? `${projection.delivery.shortfall_kg < 0.1 ? "<0.1" : n(projection.delivery.shortfall_kg)} kg shortfall for this delivery`
                : "Full delivery expected"}
            </p>
          </> : <p>No prediction recorded for this delivery.</p>}
        </div>}
      </section>

      {hasSchedule ? <>
        <section className="w-planning-schedule" aria-label="Planned operation">
          <p className="w-planning-range">{w.time(now)} to {w.time(timing.endHours)}</p>
          <Plot
            title="Equipment schedule"
            unit="kW target"
            compact
            markerLabel="Delivery deadline"
            domain={[0, timing.endHours - now]}
            cursor={0}
            xLabel="Hours from selected time"
            markers={[
              { x: 0, label: `Selected time: ${w.time(now)}`, color: "var(--text)" },
              ...f.delivery.commitments
                .filter((delivery) => delivery.due_hours > now && delivery.due_hours <= timing.endHours)
                .map((delivery) => ({
                  x: delivery.due_hours - now,
                  label: `${n(delivery.requested_kg, 0)} kg delivery at ${w.time(delivery.due_hours)}`,
                  color: "var(--ch4)",
                })),
            ]}
            series={([
              ["electrolyser_kw", "Electrolyser", "var(--feed)"],
              ["dac_kw", "Air capture", "var(--co2)"],
              ["methanator_kw", "Methanator", "var(--ch4)"],
              ["battery_charge_kw", "Battery charge ceiling", "var(--stored)"],
            ] as const).map(([key, label, color]) => ({
              label, color, step: true,
              points: plan.points
                .filter((point) => plan.issued_at_hours + point.offset_hours + timing.intervalHours > now)
                .flatMap((point) => [
                  { x: Math.max(0, plan.issued_at_hours + point.offset_hours - now), y: point[key] },
                  { x: plan.issued_at_hours + point.offset_hours + timing.intervalHours - now, y: point[key] },
                ]),
            }))}
          />
          <p className="w-planning-caption"><span className="w-planning-deadline-dot" />Delivery deadlines · Targets may change when the controller replans.</p>
        </section>

        <details className="w-planning-details">
          <summary>Plan details</summary>
          <p className="w-note">Saved at {w.time(plan.issued_at_hours)}. These projections belong to that plan.</p>
          <div className="w-two-charts">
            <Plot
              title="Predicted gas in storage"
              unit="kg"
              domain={timing.domain}
              cursor={now}
              series={(["h2_kg", "co2_kg", "ch4_kg"] as const).map((key, i) => ({
                label: ["H₂", "CO₂", "CH₄"][i], color: colors[i + 1],
                points: plan.points.map((point) => ({
                  x: plan.issued_at_hours + point.offset_hours + timing.intervalHours, y: point[key],
                })),
              }))}
            />
            <Plot
              title="Predicted battery energy"
              unit="kWh"
              domain={timing.domain}
              cursor={now}
              series={[{
                label: "Battery", color: "var(--stored)",
                points: plan.points.map((point) => ({
                  x: plan.issued_at_hours + point.offset_hours + timing.intervalHours, y: point.battery_kwh,
                })),
              }]}
            />
          </div>
          {operation && operation.resources.length > 0 && <section className="w-planning-detail-section">
            <h3>Inventory at the end of the plan</h3>
            <p className="w-note">Projected for {w.time(operation.issued_at_hours + operation.horizon_hours)}. Opening references are the inventory levels at the start of the mission.</p>
            <div className="w-table"><table>
              <thead><tr><th scope="col">Resource</th><th scope="col">Opening reference</th><th scope="col">Projected inventory</th><th scope="col">Projected deficit</th></tr></thead>
              <tbody>{operation.resources.map((resource) => <tr key={resource.id}>
                <th scope="row">{resource.label}</th>
                <td>{n(resource.reference)} {resource.unit}</td>
                <td>{resource.projected === null ? "—" : `${n(resource.projected)} ${resource.unit}`}</td>
                <td className={(resource.projected_deficit ?? 0) > 1e-6 ? "w-amber" : undefined}>
                  {resource.projected_deficit === null ? "—" : `${n(resource.projected_deficit)} ${resource.unit}`}
                </td>
              </tr>)}</tbody>
            </table></div>
          </section>}
          {predicted && <section className="w-planning-detail-section">
            <h3>Plan compared with the simulation</h3>
            <p className="w-note">Storage at {w.time(now)}. Both columns refer to the same time.</p>
            <div className="w-table"><table>
              <thead><tr><th scope="col">Resource</th><th scope="col">Plan predicted</th><th scope="col">Simulation result</th><th scope="col">Difference</th></tr></thead>
              <tbody>{comparison.map(([label, unit, expected, actual]) => <tr key={label}>
                <th scope="row">{label}</th><td>{n(expected)} {unit}</td><td>{n(actual)} {unit}</td><td>{n(actual - expected)} {unit}</td>
              </tr>)}</tbody>
            </table></div>
          </section>}
          <div className="w-planning-links">
            <Link to="/weather" className="w-link">Weather forecasts</Link>
            <Link to="/plant" className="w-link" onClick={() => w.setSelectedUnit("hub")}>Equipment commands</Link>
          </div>
        </details>
      </> : <section className="w-planning-empty" aria-label="Plan availability">
        <p>{decidesOneInterval
          ? `This controller decides every ${snapshot.environment.decision_minutes} minutes and publishes no forward schedule.`
          : operation?.mode === "fallback"
            ? "The planner is unavailable. The plant is using fallback commands."
            : !f.decision
              ? w.recording && w.latestIndex === 0
                ? "No plan was recorded. This recording contains only the initial state."
                : w.recording || w.latestIndex > 0
                  ? "No plan at the starting state. Move forward in the timeline to inspect a recorded plan."
                  : "No plan yet. Advance the simulation."
              : nextDue === null
                ? "The mission has ended. Use the timeline to inspect earlier plans."
                : "No forward schedule was recorded for this interval."}</p>
        {(decidesOneInterval || f.decision) && <Link to="/plant" className="w-link" onClick={() => w.setSelectedUnit("hub")}>View equipment commands</Link>}
      </section>}
    </div>
  );
}

export function Deliveries() {
  const w = useWorkspace(),
    f = w.frame!,
    [day, setDay] = useState<number | null>(null),
    days = missionDays(w.snapshot!.environment.horizon_hours),
    active = Math.min(days, day ?? Math.floor(f.clock.elapsed_hours / 24) + 1);
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
            {Array.from({ length: days }, (_, i) => (
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
                <time>{w.time(d.due_hours)}</time>
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

export { Economics } from "./Economics";

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
                {w.time(e.at_hours)}
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
