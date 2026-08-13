/** Single-line diagram of the plant, driven entirely by simulator state.
 *
 *      Solar ──┬──> Battery ──┐ (support)
 *              │              v
 *              ├──> Electrolyser ──> H2 tank ──┐
 *              │                               ├──> Methanator ──> CH4 tank
 *              ├──> DAC ──────────> CO2 tank ──┘
 *              │
 *              └──────────(power)──────────────┘
 *
 * Conduit brightness and dot speed track throughput; every number on screen is
 * a field of `PlantState` (rates derived from it via `PlantConfig` recipes).
 * No layout knowledge leaks back into the simulator, which is what lets the 3D
 * view render the same state differently.
 */

import type { ReactNode } from 'react'

import { intensity, kg, kw, kwh, pct } from '../../lib/format'
import type { PlantConfig, PlantState } from '../../types/plant'
import './plant-flow.css'

const VIEW = { w: 1560, h: 648 }

const LEGEND = [
  ['var(--solar)', 'Solar generation'],
  ['var(--stored)', 'Stored energy'],
  ['var(--feed)', 'Stack feed'],
  ['var(--h2)', 'Hydrogen'],
  ['var(--co2)', 'Carbon dioxide'],
  ['var(--ch4)', 'Methane'],
  ['var(--waste)', 'Curtailed'],
] as const

interface ConduitSpec {
  id: string
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
function ConduitRail({ d, power, reference, accent, dashed }: ConduitSpec) {
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
    </g>
  )
}

/** A conduit's reading, drawn in a layer above the equipment.
 *
 * The nodes are opaque, and a three-digit rate is wider than the gaps between
 * them, so a label painted underneath one silently loses its leading digits --
 * `412.1 kg/h` reading as `12.1 kg/h`. Labels therefore sit on top, with a halo
 * in the card colour so they stay legible wherever they land.
 */
function ConduitLabel({ power, accent, label, at, anchor = 'middle' }: ConduitSpec) {
  const live = power > 0.5
  return (
    <text
      className={`conduit__label mono${live ? ' conduit__label--live' : ''}`}
      style={{ color: accent }}
      x={at[0]}
      y={at[1]}
      textAnchor={anchor}
    >
      {label}
    </text>
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

interface TankNodeProps {
  x: number
  y: number
  w: number
  h: number
  title: string
  accent: string
  status: { text: string; live: boolean }
  levelKg: number
  capacityKg: number
  fill: number
  /** Signed net flow, kg/h: positive fills, negative drains. */
  netKgH: number
}

/** A storage vessel: the level is rendered as a real level, like the 3D twin. */
function TankNode({ x, y, w, h, title, accent, status, levelKg, capacityKg, fill, netKgH }: TankNodeProps) {
  const vesselH = h - 78
  const clamped = Math.max(0, Math.min(1, fill))
  return (
    <g className="node node--tank" style={{ color: accent }}>
      <rect className="node__body" x={x} y={y} width={w} height={h} rx={12} />
      <circle className="node__dot" cx={x + 21} cy={y + 28} r={3.5} />
      <text className="node__title" x={x + 33} y={y + 32}>
        {title}
      </text>
      <text
        className={`node__status${status.live ? ' node__status--live' : ''}`}
        x={x + w - 18}
        y={y + 32}
        textAnchor="end"
      >
        {status.text}
      </text>

      <g className="tank">
        <rect className="tank__shell" x={x + 21} y={y + 56} width={56} height={vesselH} rx={10} />
        <rect
          className="tank__level"
          x={x + 21}
          y={y + 56 + (1 - clamped) * vesselH}
          width={56}
          height={clamped * vesselH}
          rx={10}
        />
        {[0.25, 0.5, 0.75].map((tick) => (
          <path key={tick} className="tank__tick" d={`M${x + 81} ${y + 56 + vesselH - tick * vesselH} h 7`} />
        ))}
      </g>

      <text className="node__value node__value--tank mono" x={x + 102} y={y + 112}>
        {kg(levelKg)}
      </text>
      <text className="node__unit" x={x + 104} y={y + 134}>
        kg stored
      </text>
      <text className="node__rate mono" x={x + 102} y={y + 174}>
        {netKgH >= 0 ? '+' : '−'}
        {Math.abs(netKgH).toFixed(1)}
      </text>
      <text className="node__unit" x={x + 104} y={y + 194}>
        kg/h net
      </text>
      <text className="node__foot mono" x={x + 102} y={y + h - 26}>
        {pct(clamped, 0)}% of {kw(capacityKg)} kg
      </text>
    </g>
  )
}

export function PlantFlow({ state, config }: { state: PlantState; config: PlantConfig | null }) {
  const peakKw = config?.solar.peak_kw ?? 5000
  const ratedKw = config?.electrolyser.rated_kw ?? 3000
  const minLoadKw = config?.electrolyser_min_load_kw ?? 450
  const chargeKw = config?.battery.max_charge_kw ?? 2500
  const ratedKgH = ratedKw / (config?.electrolyser.specific_energy_kwh_per_kg ?? 52)
  const dacRatedKw = config?.dac.rated_kw ?? 1200
  const dacRatedKgH = dacRatedKw / (config?.dac.specific_energy_kwh_per_kg ?? 2)
  const metRatedKw = config?.methanator.rated_kw ?? 180
  const metRatedKgH = metRatedKw / (config?.methanator.specific_energy_kwh_per_kg ?? 1.2)
  const h2PerCh4 = config?.methanator.h2_kg_per_kg_ch4 ?? 0.5
  const co2PerCh4 = config?.methanator.co2_kg_per_kg_ch4 ?? 2.75

  // gas flows into the reactor follow the recipe, in the reactor's own output
  const h2ToMetKgH = state.ch4_production_kg_h * h2PerCh4
  const co2ToMetKgH = state.ch4_production_kg_h * co2PerCh4

  const charging = state.battery_power_kw > 0.5
  const discharging = state.battery_power_kw < -0.5
  const daylight = state.solar_available_kw > 0.5

  const stackStatus = state.electrolyser_online
    ? 'Online'
    : daylight || discharging
      ? 'Below min load'
      : 'Standby'

  const dacStatus = state.dac_online ? 'Capturing' : state.co2_fill > 0.995 ? 'Tank full' : daylight ? 'Standby' : 'Night'

  const metStatus = state.methanator_online
    ? 'Running'
    : state.h2_level_kg < 0.05 || state.co2_level_kg < 0.05
      ? 'No feedstock'
      : 'Standby'

  const tankStatus = (fill: number, netKgH: number) =>
    fill > 0.95 && netKgH >= 0
      ? { text: 'Full', live: false }
      : netKgH > 0.05
        ? { text: 'Filling', live: true }
        : netKgH < -0.05
          ? { text: 'Draining', live: true }
          : { text: 'Holding', live: false }

  const conduits: ConduitSpec[] = [
    {
      id: 'solar',
      d: 'M244 265 H 330',
      power: state.solar_available_kw,
      reference: peakKw,
      accent: 'var(--solar)',
      label: `${kw(state.solar_available_kw)} kW`,
      at: [287, 252],
    },
    {
      id: 'solar-battery',
      d: 'M330 265 V 94 H 420',
      power: state.solar_to_battery_kw,
      reference: chargeKw,
      accent: 'var(--stored)',
      label: `${kw(state.solar_to_battery_kw)} kW`,
      at: [344, 84],
      anchor: 'start',
    },
    {
      id: 'solar-stack',
      d: 'M330 265 V 280 H 420',
      power: state.solar_to_electrolyser_kw,
      reference: ratedKw,
      accent: 'var(--feed)',
      label: `${kw(state.solar_to_electrolyser_kw)} kW`,
      at: [344, 301],
      anchor: 'start',
    },
    {
      id: 'solar-dac',
      d: 'M330 265 V 460 H 420',
      power: state.solar_to_dac_kw,
      reference: dacRatedKw,
      accent: 'var(--co2)',
      label: `${kw(state.solar_to_dac_kw)} kW`,
      at: [344, 481],
      anchor: 'start',
    },
    {
      id: 'battery-stack',
      d: 'M545 164 V 210',
      power: state.battery_to_electrolyser_kw,
      reference: chargeKw,
      accent: 'var(--stored)',
      label: `${kw(state.battery_to_electrolyser_kw)} kW`,
      at: [559, 196],
      anchor: 'start',
    },
    {
      id: 'stack-h2',
      d: 'M670 280 H 700 V 190 H 742',
      power: state.h2_production_kg_h,
      reference: ratedKgH,
      accent: 'var(--h2)',
      label: `${state.h2_production_kg_h.toFixed(1)} kg/h`,
      at: [736, 178],
      anchor: 'end',
    },
    {
      id: 'dac-co2',
      d: 'M670 460 H 700 V 480 H 742',
      power: state.co2_capture_kg_h,
      reference: dacRatedKgH,
      accent: 'var(--co2)',
      label: `${state.co2_capture_kg_h.toFixed(1)} kg/h`,
      at: [736, 552],
      anchor: 'end',
    },
    {
      id: 'h2-methanator',
      d: 'M950 190 H 985 V 280 H 1022',
      power: h2ToMetKgH,
      reference: ratedKgH,
      accent: 'var(--h2)',
      label: `${h2ToMetKgH.toFixed(1)} kg/h`,
      at: [1018, 262],
      anchor: 'end',
    },
    {
      id: 'co2-methanator',
      d: 'M950 480 H 990 V 345 H 1022',
      power: co2ToMetKgH,
      reference: dacRatedKgH,
      accent: 'var(--co2)',
      label: `${co2ToMetKgH.toFixed(1)} kg/h`,
      at: [1018, 424],
      anchor: 'end',
    },
    {
      id: 'methanator-ch4',
      d: 'M1260 280 H 1332',
      power: state.ch4_production_kg_h,
      reference: metRatedKgH,
      accent: 'var(--ch4)',
      label: `${state.ch4_production_kg_h.toFixed(1)} kg/h`,
      at: [1296, 268],
    },
    {
      id: 'solar-methanator',
      d: 'M330 460 V 612 H 1140 V 350',
      power: state.solar_to_methanator_kw,
      reference: metRatedKw,
      accent: 'var(--feed)',
      label: `${kw(state.solar_to_methanator_kw)} kW`,
      at: [660, 600],
      anchor: 'end',
    },
    {
      id: 'curtailed',
      d: 'M287 265 V 556',
      power: state.solar_curtailed_kw,
      reference: peakKw,
      accent: 'var(--waste)',
      label: `${kw(state.solar_curtailed_kw)} kW`,
      at: [275, 528],
      anchor: 'end',
      dashed: true,
    },
  ]

  return (
    <div className="flow-frame">
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

      {/* --- conduits (behind the equipment; their labels go on top) --- */}
      {conduits.map((conduit) => (
        <ConduitRail key={conduit.id} {...conduit} />
      ))}

      {/* bus junction + annotations */}
      <circle className="flow__junction" cx={330} cy={265} r={3.5} />
      <text className="flow__axis" x={318} y={352} transform="rotate(-90 318 352)">
        DC bus
      </text>
      <g className="flow__vent">
        <path d="M273 556 H 301 M279 563 l3 7 M288 563 l3 7 M297 563 l3 7" />
        <text x={287} y={591} textAnchor="middle">
          Curtailed
        </text>
      </g>
      <text className="flow__axis" x={531} y={196} textAnchor="end">
        Support
      </text>
      <text className="flow__axis" x={735} y={630} textAnchor="middle">
        Methanator power
      </text>

      {/* --- equipment --- */}
      <EquipmentNode
        x={24}
        y={190}
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
        x={420}
        y={24}
        w={250}
        h={140}
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
        x={420}
        y={210}
        w={250}
        h={140}
        title="Electrolyser"
        accent="var(--feed)"
        status={{ text: stackStatus, live: state.electrolyser_online }}
        value={kw(state.electrolyser_power_kw)}
        unit="kW"
        fill={state.electrolyser_power_kw / ratedKw}
        marker={{ at: minLoadKw / ratedKw, label: 'min' }}
        footer={`Load ${pct(state.electrolyser_load_frac, 0)}% · rated ${kw(ratedKw)} kW`}
      />

      <EquipmentNode
        x={420}
        y={390}
        w={250}
        h={140}
        title="DAC"
        accent="var(--co2)"
        status={{ text: dacStatus, live: state.dac_online }}
        value={kw(state.dac_power_kw)}
        unit="kW"
        fill={state.dac_power_kw / dacRatedKw}
        footer={`Capture ${state.co2_capture_kg_h.toFixed(1)} kg/h · rated ${kw(dacRatedKw)} kW`}
      />

      <TankNode
        x={740}
        y={70}
        w={210}
        h={240}
        title="H₂ storage"
        accent="var(--h2)"
        status={tankStatus(state.h2_fill, state.h2_production_kg_h - h2ToMetKgH)}
        levelKg={state.h2_level_kg}
        capacityKg={state.h2_capacity_kg}
        fill={state.h2_fill}
        netKgH={state.h2_production_kg_h - h2ToMetKgH}
      />

      <TankNode
        x={740}
        y={360}
        w={210}
        h={240}
        title="CO₂ storage"
        accent="var(--co2)"
        status={tankStatus(state.co2_fill, state.co2_capture_kg_h - co2ToMetKgH)}
        levelKg={state.co2_level_kg}
        capacityKg={state.co2_capacity_kg}
        fill={state.co2_fill}
        netKgH={state.co2_capture_kg_h - co2ToMetKgH}
      />

      <EquipmentNode
        x={1022}
        y={210}
        w={238}
        h={140}
        title="Methanator"
        accent="var(--ch4)"
        status={{ text: metStatus, live: state.methanator_online }}
        value={kw(state.methanator_power_kw)}
        unit="kW"
        fill={state.methanator_power_kw / metRatedKw}
        footer={`${state.ch4_production_kg_h.toFixed(1)} kg CH₄/h · rated ${kw(metRatedKw)} kW`}
      />

      <TankNode
        x={1332}
        y={160}
        w={200}
        h={240}
        title="CH₄ storage"
        accent="var(--ch4)"
        status={tankStatus(state.ch4_fill, state.ch4_production_kg_h)}
        levelKg={state.ch4_level_kg}
        capacityKg={state.ch4_capacity_kg}
        fill={state.ch4_fill}
        netKgH={state.ch4_production_kg_h}
      />

      {/* --- readings, above the equipment so no digit is ever painted over --- */}
      {conduits.map((conduit) => (
        <ConduitLabel key={conduit.id} {...conduit} />
      ))}
    </svg>
    </div>
  )
}
