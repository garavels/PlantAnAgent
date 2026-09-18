import type { ReactNode } from "react";
import type { UnitId } from "../components/plant-3d/selection";
import type { PlantConfig } from "../types/plant";
import type { V2DemoFrame } from "../types/v2";
import { EquipmentDecision } from "./EquipmentDecision";
import { EQUIPMENT } from "./equipmentCatalog";
import { duration, n, words } from "./format";
import { equipmentTime } from "./equipmentTime";
import "./equipment.css";


type Row = [string, string];
const power = (value: number) => `${n(value)} kW`;
const mass = (value: number) => `${n(value)} kg`;
const rate = (value: number) => `${n(value)} kg/h`;
const direction = (value: number) => value > 0.05 ? "Filling" : value < -0.05 ? "Draining" : "Balanced";

function Readings({ rows }: { rows: Row[] }) {
  return <dl className="w-equipment-readings">{rows.map(([label, value]) => (
    <div key={label}><dt>{label}</dt><dd>{value}</dd></div>
  ))}</dl>;
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return <section className="w-equipment-section"><h3>{title}</h3>{children}</section>;
}

function Meter({ value, label }: { value: number; label: string }) {
  return <div className="w-equipment-meter" role="img" aria-label={label}>
    <span style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
  </div>;
}

function InventoryReference({ frame, id }: { frame: V2DemoFrame; id: string }) {
  const resource = frame.resources.find((item) => item.id === id);
  if (!resource) return null;
  const referenceName = frame.operating_policy ? "Opening reference" : "Daily return minimum";
  const position = Math.abs(resource.margin) < 0.05 ? "At reference"
    : `${n(Math.abs(resource.margin))} ${resource.unit} ${resource.margin > 0 ? "above" : "below"}`;
  return <div className="w-equipment-reference">
    <Readings rows={[[referenceName, `${n(resource.required)} ${resource.unit}`], ["Inventory position", position]]} />
    {!frame.operating_policy && <p className="w-equipment-note">Return due in {duration(frame.clock.hours_to_cycle_end)}.</p>}
  </div>;
}

function StorageBalance({ frame, unit, inflow, outflow }: {
  frame: V2DemoFrame; unit: "h2Tank" | "co2Tank"; inflow: number; outflow: number;
}) {
  const net = inflow - outflow;
  return <Section title="Storage balance">
    <Readings rows={[
      [unit === "h2Tank" ? "From electrolyser" : "From air capture", rate(inflow)],
      ["To methanator", rate(outflow)],
      [net < -0.05 ? "Net outflow" : "Net inflow", rate(Math.abs(net))],
    ]} />
    <InventoryReference frame={frame} id={unit === "h2Tank" ? "h2" : "co2"} />
  </Section>;
}

function NextDelivery({ frame }: { frame: V2DemoFrame }) {
  const delivery = frame.delivery;
  if (delivery.next_due_hours === null) return <Section title="Next delivery">
    <p className="w-equipment-note">No deliveries remaining in this mission.</p>
  </Section>;
  const covered = Math.min(delivery.next_requested_kg, Math.max(0, frame.state.ch4_level_kg));
  const remaining = Math.max(0, delivery.next_requested_kg - covered);
  return <Section title="Next delivery">
    <p className="w-equipment-delivery-time">{equipmentTime(frame, delivery.next_due_hours)}
      <span>In {duration(delivery.next_due_hours - frame.clock.elapsed_hours)}</span>
    </p>
    <Readings rows={[
      ["Delivery quantity", mass(delivery.next_requested_kg)],
      ["Covered by current stock", mass(covered)],
      ["Still to produce", mass(remaining)],
    ]} />
    <Meter value={delivery.next_requested_kg > 0 ? covered / delivery.next_requested_kg : 1}
      label={`${mass(covered)} of ${mass(delivery.next_requested_kg)} covered by current stock`} />
    <p className="w-equipment-note">{remaining < 0.05 ? "Current stock covers this delivery." : "Coverage uses current stock; future production is not included."}</p>
  </Section>;
}

function PowerAllocation({ frame, hub }: { frame: V2DemoFrame; hub: boolean }) {
  const s = frame.state;
  return <Section title="Power allocation">
    <Readings rows={[
      ["Electrolyser", power(hub ? s.electrolyser_power_kw : s.solar_to_electrolyser_kw)],
      ["Direct air capture", power(s.solar_to_dac_kw)],
      ["Methanator", power(s.solar_to_methanator_kw)],
      ["Battery charging", power(s.solar_to_battery_kw)],
      ...(hub ? [["Battery support to electrolyser", power(s.battery_to_electrolyser_kw)] as Row] : []),
    ]} />
    <p className="w-equipment-note">{hub
      ? "Electrolyser consumption includes battery support."
      : "Solar destinations in the current sample. Battery support is separate."}</p>
  </Section>;
}

export function EquipmentInspector({ frame, config, unit }: {
  frame: V2DemoFrame; config: PlantConfig; unit: UnitId;
}) {
  const s = frame.state, c = frame.commitment;
  const siteWeather = frame.weather.reference_kind === "forecast_pv";
  const name = EQUIPMENT.find(([id]) => id === unit)![1];
  const h2Use = s.ch4_production_kg_h * config.methanator.h2_kg_per_kg_ch4;
  const co2Use = s.ch4_production_kg_h * config.methanator.co2_kg_per_kg_ch4;
  let status = "", headline = "", value = "", valueUnit = "", capacity: number | undefined;
  let fill: number | undefined;
  let rows: Row[] = [], details: Row[] = [], limits: Row[] = [];

  switch (unit) {
    case "solar":
      status = frame.weather.status === "night" ? siteWeather ? "No forecast PV" : "Night" : s.solar_available_kw > 0.05 ? "Generating" : "Idle";
      headline = "Available solar"; value = n(s.solar_available_kw); valueUnit = "kW";
      rows = [["Being used", power(Math.max(0, s.solar_available_kw - s.solar_curtailed_kw))], ["Curtailed", power(s.solar_curtailed_kw)]];
      if (!siteWeather && frame.weather.attenuation !== null && frame.weather.attenuation > 0.001 && frame.weather.status !== "night") {
        rows.push(["Below clear-sky potential", `${n(frame.weather.attenuation * 100)}%`]);
      }
      if (siteWeather && frame.weather.clear_sky_kw > 0.05 && frame.weather.available_kw < frame.weather.clear_sky_kw - 0.05) {
        rows.push(["Below forecast PV", `${n((1 - frame.weather.available_kw / frame.weather.clear_sky_kw) * 100)}%`]);
      }
      details = [[siteWeather ? "Forecast PV reference" : "Clear-sky potential", power(siteWeather ? frame.weather.clear_sky_kw : s.solar_potential_kw)], ["Array rating", power(config.solar.peak_kw)]];
      break;
    case "hub": {
      const consumption = s.electrolyser_power_kw + s.dac_power_kw + s.methanator_power_kw;
      status = consumption + s.solar_to_battery_kw > 0.05 ? "Distributing power" : "Idle";
      headline = "Available supply"; value = n(s.solar_available_kw + s.battery_to_electrolyser_kw); valueUnit = "kW";
      rows = [["Process consumption", power(consumption)], ["Battery charging", power(s.solar_to_battery_kw)], ["Unused solar", power(s.solar_curtailed_kw)]];
      break;
    }
    case "battery":
      status = s.battery_power_kw > 0.05 ? "Charging" : s.battery_power_kw < -0.05 ? "Discharging" : "Idle";
      if (s.solar_to_battery_kw > 0.05 && s.battery_to_electrolyser_kw > 0.05) status = "Charging and discharging";
      headline = "State of charge"; value = n(s.battery_soc * 100); valueUnit = "%"; fill = s.battery_soc;
      rows = [
        ["Stored energy", `${n(s.battery_energy_kwh)} / ${n(s.battery_capacity_kwh)} kWh`],
        [s.battery_power_kw < -0.05 ? "Net discharge" : "Net charging", power(Math.abs(s.battery_power_kw))],
        ["Reserve above physical minimum", `${n(Math.max(0, s.battery_energy_kwh - s.battery_capacity_kwh * config.battery.min_soc))} kWh`],
      ];
      details = [
        ["Solar charging now", power(s.solar_to_battery_kw)],
        ["Support to electrolyser now", power(s.battery_to_electrolyser_kw)],
        ["Physical minimum charge", `${n(config.battery.min_soc * 100)}%`],
        ["Maximum charge / discharge", `${n(config.battery.max_charge_kw)} / ${n(config.battery.max_discharge_kw)} kW`],
      ];
      if (typeof frame.metrics.battery_throughput_kwh === "number") details.push(["Cumulative throughput", `${n(frame.metrics.battery_throughput_kwh)} kWh`]);
      if (s.battery_soc <= config.battery.min_soc + 1e-6) limits.push(["Reserve", "At physical minimum"]);
      break;
    case "electrolyser":
      status = c ? words(c.mode) : s.electrolyser_online ? "Running" : "Idle";
      headline = "Hydrogen production"; value = n(s.h2_production_kg_h); valueUnit = "kg/h";
      rows = [["Actual power", power(s.electrolyser_power_kw)], ["Load", `${n(s.electrolyser_load_frac * 100)}%`]];
      details = [
        ["Rated power", power(config.electrolyser.rated_kw)],
        ["Minimum stable load", `${n(Number(frame.commitment_profile?.min_stable_load_frac ?? config.electrolyser.min_load_frac) * 100)}%`],
        ["Stack temperature", `${n(s.electrolyser_temperature_c)} °C`],
        ["Relative yield", `${n(s.electrolyser_efficiency * 100)}% of nameplate`],
        ["Solar supply", power(s.solar_to_electrolyser_kw)], ["Battery supply", power(s.battery_to_electrolyser_kw)],
      ];
      if (c) {
        const timers: [string, number][] = [["Startup remaining", c.startup_remaining_s], ["Must stay on for", c.min_up_remaining_s], ["Restart allowed in", c.min_down_remaining_s], ["Cooldown remaining", c.cooldown_remaining_s]];
        limits = timers.filter(([, seconds]) => seconds > 0).map(([label, seconds]) => [label, duration(seconds / 3600)]);
        if (c.mode === "running") {
          const minPower = Number(frame.commitment_profile?.min_stable_load_frac ?? config.electrolyser.min_load_frac) * config.electrolyser.rated_kw;
          if (c.ramp_up_headroom_kw + 0.05 < config.electrolyser.rated_kw - s.electrolyser_power_kw) {
            limits.push(["Maximum rise next minute", power(c.ramp_up_headroom_kw)]);
          }
          if (c.ramp_down_headroom_kw + 0.05 < s.electrolyser_power_kw - minPower) {
            limits.push(["Maximum fall next minute", power(c.ramp_down_headroom_kw)]);
          }
        }
        details.push(["Next-minute ramp up / down", `${n(c.ramp_up_headroom_kw)} / ${n(c.ramp_down_headroom_kw)} kW`],
          ["Cold / hot starts", `${c.cold_starts} / ${c.hot_starts}`], ["Forced / aborted stops", `${c.forced_stops} / ${c.aborted_starts}`],
          ["Cumulative startup energy", `${n(c.startup_energy_kwh)} kWh`]);
      }
      break;
    case "dac":
      status = s.dac_online ? "Capturing" : "Idle";
      headline = "CO₂ capture"; value = n(s.co2_capture_kg_h); valueUnit = "kg/h";
      rows = [["Actual power", power(s.dac_power_kw)], ["CO₂ storage space", mass(Math.max(0, s.co2_capacity_kg - s.co2_level_kg))]];
      details = [["Rated power", power(config.dac.rated_kw)], ["Load", `${n(s.dac_load_frac * 100)}%`]];
      break;
    case "methanator":
      status = s.methanator_online ? "Producing" : "Idle";
      headline = "Methane production"; value = n(s.ch4_production_kg_h); valueUnit = "kg/h";
      rows = [["Actual power", power(s.methanator_power_kw)], ["Hydrogen consumption", rate(h2Use)], ["CO₂ consumption", rate(co2Use)]];
      details = [["Rated power", power(config.methanator.rated_kw)], ["Load", `${n(s.methanator_load_frac * 100)}%`],
        ["Hydrogen available", mass(s.h2_level_kg)], ["CO₂ available", mass(s.co2_level_kg)],
        ["Methane storage space", mass(Math.max(0, s.ch4_capacity_kg - s.ch4_level_kg))]];
      break;
    case "h2Tank":
    case "co2Tank":
    case "ch4Tank": {
      const stored = unit === "h2Tank" ? s.h2_level_kg : unit === "co2Tank" ? s.co2_level_kg : s.ch4_level_kg;
      capacity = unit === "h2Tank" ? s.h2_capacity_kg : unit === "co2Tank" ? s.co2_capacity_kg : s.ch4_capacity_kg;
      const net = unit === "h2Tank" ? s.h2_production_kg_h - h2Use : s.co2_capture_kg_h - co2Use;
      status = unit === "ch4Tank" ? (s.ch4_production_kg_h > 0.05 ? "Receiving methane" : "No production inflow") : direction(net);
      headline = "Stored inventory"; value = n(stored); valueUnit = "kg"; fill = capacity > 0 ? stored / capacity : 0;
      if (unit === "ch4Tank") rows = [["From methanator", rate(s.ch4_production_kg_h)]];
      details = [["Free capacity", mass(Math.max(0, capacity - stored))]];
      if (frame.decision) {
        const start = frame.decision.input_state;
        const previous = unit === "h2Tank" ? start.h2_level_kg : unit === "co2Tank" ? start.co2_level_kg : start.ch4_level_kg;
        const change = stored - previous;
        details.push(["Last interval", `${equipmentTime(frame, frame.decision.at_hours)}–${equipmentTime(frame, frame.clock.elapsed_hours)}`],
          ["Inventory change", `${change >= 0.05 ? "+" : ""}${mass(change)}`]);
      }
      break;
    }
  }

  // Report observed inventory conditions without attributing a controller motive.
  if ((unit === "electrolyser" || unit === "h2Tank") && s.h2_capacity_kg - s.h2_level_kg <= 1e-6) limits.push(["Hydrogen storage", "Full"]);
  if ((unit === "dac" || unit === "co2Tank") && s.co2_capacity_kg - s.co2_level_kg <= 1e-6) limits.push(["CO₂ storage", "Full"]);
  if ((unit === "methanator" || unit === "ch4Tank") && s.ch4_capacity_kg - s.ch4_level_kg <= 1e-6) limits.push(["Methane storage", "Full"]);
  if (unit === "methanator" && s.h2_level_kg <= 1e-6) limits.push(["Hydrogen storage", "Empty"]);
  if (unit === "methanator" && s.co2_level_kg <= 1e-6) limits.push(["CO₂ storage", "Empty"]);

  // Health reports use the simulator's snake_case unit identifiers.
  const healthKey = unit === "h2Tank" ? "h2_tank" : unit === "co2Tank" ? "co2_tank" : unit === "ch4Tank" ? "ch4_tank" : unit;
  const health = s.health?.[healthKey];
  const impaired = health && (health.offline || health.status !== "nominal");
  if (health?.offline || health?.status === "offline") status = "Offline · fault";
  else if (health?.status === "degraded") status += " · degraded";

  return <aside className="w-inspector" data-unit={unit} aria-label={`${name} inspector`}>
    <div className="w-panel-heading"><h2>{name}</h2></div>
    <p className={`w-equipment-status${impaired ? " w-equipment-status--warning" : ""}`}>{status.charAt(0).toUpperCase() + status.slice(1)}</p>
    <p className="w-inspector-caption">{equipmentTime(frame, frame.clock.elapsed_hours)} · {frame.decision ? `${n(config.dt_s / 60)}-minute sample` : "Initial state"}</p>
    <div className="w-equipment-primary">
      <span>{headline}</span><strong>{value} <small>{valueUnit}</small></strong>
      {capacity !== undefined && <span>of {mass(capacity)} · {n((fill ?? 0) * 100)}% full</span>}
      {fill !== undefined && <Meter value={fill} label={`${n(fill * 100)}% full`} />}
    </div>
    {rows.length > 0 && <Readings rows={rows} />}
    {limits.length > 0 && <Section title="Active limits"><Readings rows={limits} /></Section>}
    {(unit === "solar" || unit === "hub") && <PowerAllocation frame={frame} hub={unit === "hub"} />}
    {unit === "h2Tank" && <StorageBalance frame={frame} unit={unit} inflow={s.h2_production_kg_h} outflow={h2Use} />}
    {unit === "co2Tank" && <StorageBalance frame={frame} unit={unit} inflow={s.co2_capture_kg_h} outflow={co2Use} />}
    {unit === "ch4Tank" && <NextDelivery frame={frame} />}
    <EquipmentDecision frame={frame} unit={unit} />
    {details.length > 0 && <details className="w-equipment-details">
      <summary>Details</summary>
      <Readings rows={details} />
      {unit === "battery" && <InventoryReference frame={frame} id="battery" />}
      {unit === "ch4Tank" && <>
        <InventoryReference frame={frame} id="ch4" />
        <p className="w-equipment-note">Inventory change includes production and scheduled deliveries.</p>
      </>}
    </details>}
  </aside>;
}
