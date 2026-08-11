/** Single-line diagram of the plant, driven entirely by simulator state.
 *
 *      Solar ──┬──> Battery ──┐ (support)
 *              │              v
 *              └──────> Electrolyser ──> H2 tank
 *
 * Conduit brightness and dot speed track power; every number on screen is a
 * field of `PlantState`. No layout knowledge leaks back into the simulator,
 * which is what will let a 3D view render the same state differently.
 */

import type { ReactNode } from 'react'

import { intensity, kg, kw, kwh, pct } from '../../lib/format'
import type { PlantConfig, PlantState } from '../../types/plant'
import './plant-flow.css'

const VIEW = { w: 1180, h: 470 }

const LEGEND = [
  ['var(--solar)', 'Solar generation'],
  ['var(--stored)', 'Stored energy'],
  ['var(--feed)', 'Stack feed'],
  ['var(--h2)', 'Hydrogen'],
  ['var(--waste)', 'Curtailed'],
] as const

interface ConduitProps {
  d: string
  power: number
  reference: number
  accent: string
  label: string
  at: [number, number]
  anchor?: 'start' | 'middle' | 'end'
  dashed?: boolean
}

/** One line on the diagram: static rail + travelling energy dots. */
function Conduit({ d, power, reference, accent, label, at, anchor = 'middle', dashed }: ConduitProps) {
  const live = power > 0.5
  const load = intensity(power, reference)
  return (
    <g className="conduit" style={{ color: accent }}>
      <path className={`conduit__rail${dashed ? ' conduit__rail--dashed' : ''}`} d={d} />
      <path
        className="conduit__flow"
        d={d}
        style={{
          opacity: live ? 0.4 + 0.6 * load : 0,
          strokeWidth: live ? 1.6 + 1.8 * load : 1.6,
          animationDuration: `${(1.6 - 1.1 * load).toFixed(2)}s`,
        }}
      />
      <text className={`conduit__label mono${live ? ' conduit__label--live' : ''}`} x={at[0]} y={at[1]} textAnchor={anchor}>
        {label}
      </text>
    </g>
  )
}

interface NodeProps {
  x: number
  y: number
  w: number
  h: number
  title: string
  accent: string
  status: { text: string; live: boolean }
  value: string
  unit: string
  fill: number
  footer: string
  marker?: { at: number; label: string }
  children?: ReactNode
}

/** A unit of equipment: header, status, primary readout, level bar. */
function EquipmentNode({
  x,
  y,
  w,
  h,
  title,
  accent,
  status,
  value,
  unit,
  fill,
  footer,
  marker,
  children,
}: NodeProps) {
  const pad = 18
  const barY = y + h - 40
  const barW = w - pad * 2
  return (
    <g className="node" style={{ color: accent }}>
      <rect className="node__body" x={x} y={y} width={w} height={h} rx={12} />

      <circle className="node__dot" cx={x + pad + 3} cy={y + 24} r={3.5} />
      <text className="node__title" x={x + pad + 15} y={y + 28}>
        {title}
      </text>
      <text className={`node__status${status.live ? ' node__status--live' : ''}`} x={x + w - pad} y={y + 28} textAnchor="end">
        {status.text}
      </text>

      <text className="node__value mono" x={x + pad} y={y + 80}>
        {value}
        <tspan className="node__unit" dx="10">
          {unit}
        </tspan>
      </text>

      <rect className="node__bar-bg" x={x + pad} y={barY} width={barW} height={3} rx={1.5} />
      <rect
        className="node__bar-fill"
        x={x + pad}
        y={barY}
        width={Math.max(0, Math.min(1, fill)) * barW}
        height={3}
        rx={1.5}
      />
      {marker && (
        <g className="node__marker">
          <path d={`M${x + pad + marker.at * barW} ${barY - 4} v 11`} />
          <text x={x + pad + marker.at * barW} y={barY - 9} textAnchor="middle">
            {marker.label}
          </text>
        </g>
      )}
      <text className="node__foot mono" x={x + pad} y={y + h - 16}>
        {footer}
      </text>
      {children}
    </g>
  )
}

export function PlantFlow({ state, config }: { state: PlantState; config: PlantConfig | null }) {
  const peakKw = config?.solar.peak_kw ?? 5000
  const ratedKw = config?.electrolyser.rated_kw ?? 3000
  const minLoadKw = config?.electrolyser_min_load_kw ?? 450
  const chargeKw = config?.battery.max_charge_kw ?? 2500
  const ratedKgH = ratedKw / (config?.electrolyser.specific_energy_kwh_per_kg ?? 52)

  const charging = state.battery_power_kw > 0.5
  const discharging = state.battery_power_kw < -0.5
  const daylight = state.solar_available_kw > 0.5

  const stackStatus = state.electrolyser_online
    ? 'Online'
    : daylight || discharging
      ? 'Below min load'
      : 'Standby'

  return (
    <svg className="flow" viewBox={`0 0 ${VIEW.w} ${VIEW.h}`} preserveAspectRatio="xMidYMid meet">
      <g className="legend">
        {LEGEND.map(([accent, name], index) => (
          <g key={name} style={{ color: accent }} transform={`translate(0 ${index * 21})`}>
            <circle className="legend__dot" cx={26} cy={30} r={3} />
            <text className="legend__name" x={40} y={34}>
              {name}
            </text>
          </g>
        ))}
      </g>

      {/* --- conduits (behind the equipment) --- */}
      <Conduit
        d="M244 235 H 330"
        power={state.solar_available_kw}
        reference={peakKw}
        accent="var(--solar)"
        label={`${kw(state.solar_available_kw)} kW`}
        at={[287, 222]}
      />
      <Conduit
        d="M330 235 V 118 H 430"
        power={state.solar_to_battery_kw}
        reference={chargeKw}
        accent="var(--stored)"
        label={`${kw(state.solar_to_battery_kw)} kW`}
        at={[344, 108]}
        anchor="start"
      />
      <Conduit
        d="M330 235 V 352 H 430"
        power={state.solar_to_electrolyser_kw}
        reference={ratedKw}
        accent="var(--feed)"
        label={`${kw(state.solar_to_electrolyser_kw)} kW`}
        at={[344, 373]}
        anchor="start"
      />
      <Conduit
        d="M555 194 V 276"
        power={state.battery_to_electrolyser_kw}
        reference={chargeKw}
        accent="var(--stored)"
        label={`${kw(state.battery_to_electrolyser_kw)} kW`}
        at={[569, 245]}
        anchor="start"
      />
      <Conduit
        d="M680 352 H 762 V 270 H 830"
        power={state.h2_production_kg_h}
        reference={ratedKgH}
        accent="var(--h2)"
        label={`${state.h2_production_kg_h.toFixed(1)} kg/h`}
        at={[721, 341]}
      />
      <Conduit
        d="M287 235 V 424"
        power={state.solar_curtailed_kw}
        reference={peakKw}
        accent="var(--waste)"
        label={`${kw(state.solar_curtailed_kw)} kW`}
        at={[275, 400]}
        anchor="end"
        dashed
      />

      {/* bus junction + annotations */}
      <circle className="flow__junction" cx={330} cy={235} r={3.5} />
      <text className="flow__axis" x={318} y={186} transform="rotate(-90 318 186)">
        DC bus
      </text>
      <g className="flow__vent">
        <path d="M273 424 H 301 M279 431 l3 7 M288 431 l3 7 M297 431 l3 7" />
        <text x={287} y={455} textAnchor="middle">
          Curtailed
        </text>
      </g>
      <text className="flow__axis" x={569} y={218} textAnchor="start">
        Support
      </text>

      {/* --- equipment --- */}
      <EquipmentNode
        x={24}
        y={160}
        w={220}
        h={150}
        title="Solar array"
        accent="var(--solar)"
        status={{ text: daylight ? 'Generating' : 'Night', live: daylight }}
        value={kw(state.solar_available_kw)}
        unit="kW"
        fill={state.solar_available_kw / peakKw}
        footer={`Peak ${kw(peakKw)} kW · ${state.clock}`}
      />

      <EquipmentNode
        x={430}
        y={44}
        w={250}
        h={150}
        title="Battery"
        accent="var(--stored)"
        status={{
          text: charging ? 'Charging' : discharging ? 'Discharging' : 'Idle',
          live: charging || discharging,
        }}
        value={pct(state.battery_soc)}
        unit="%"
        fill={state.battery_soc}
        footer={`${kwh(state.battery_energy_kwh)} / ${kwh(config?.battery.capacity_kwh ?? 0)} kWh · ${
          state.battery_power_kw >= 0 ? '+' : ''
        }${kw(state.battery_power_kw)} kW`}
      />

      <EquipmentNode
        x={430}
        y={276}
        w={250}
        h={150}
        title="Electrolyser"
        accent="var(--feed)"
        status={{ text: stackStatus, live: state.electrolyser_online }}
        value={kw(state.electrolyser_power_kw)}
        unit="kW"
        fill={state.electrolyser_power_kw / ratedKw}
        marker={{ at: minLoadKw / ratedKw, label: 'min' }}
        footer={`Load ${pct(state.electrolyser_load_frac, 0)}% · rated ${kw(ratedKw)} kW`}
      />

      {/* H2 tank: vertical vessel, level rendered as a real level */}
      <g className="node node--tank" style={{ color: 'var(--h2)' }}>
        <rect className="node__body" x={830} y={140} width={210} height={260} rx={12} />
        <circle className="node__dot" cx={851} cy={168} r={3.5} />
        <text className="node__title" x={863} y={172}>
          H₂ storage
        </text>
        <text
          className={`node__status${state.h2_fill > 0.95 ? '' : ' node__status--live'}`}
          x={1022}
          y={172}
          textAnchor="end"
        >
          {state.h2_fill > 0.95 ? 'Full' : 'Filling'}
        </text>

        <g className="tank">
          <rect className="tank__shell" x={851} y={196} width={56} height={182} rx={10} />
          <rect
            className="tank__level"
            x={851}
            y={196 + (1 - Math.min(1, state.h2_fill)) * 182}
            width={56}
            height={Math.min(1, state.h2_fill) * 182}
            rx={10}
          />
          {[0.25, 0.5, 0.75].map((tick) => (
            <path key={tick} className="tank__tick" d={`M911 ${378 - tick * 182} h 7`} />
          ))}
        </g>

        <text className="node__value node__value--tank mono" x={932} y={252}>
          {kg(state.h2_level_kg)}
        </text>
        <text className="node__unit" x={934} y={274}>
          kg stored
        </text>
        <text className="node__rate mono" x={932} y={314}>
          +{state.h2_production_kg_h.toFixed(1)}
        </text>
        <text className="node__unit" x={934} y={334}>
          kg/h
        </text>
        <text className="node__foot mono" x={932} y={374}>
          {pct(state.h2_fill, 0)}% of {kw(state.h2_capacity_kg)} kg
        </text>
      </g>
    </svg>
  )
}
