/** Where the power is going, and how much of the sun the plant kept.
 *
 * The capture percentage is the number an agent will eventually be scored on.
 * Everything here is derived from the current `PlantState` -- no history needed.
 */

import { kw, kwh, pct } from '../../lib/format'
import type { PlantState } from '../../types/plant'
import './balance-card.css'

export function BalanceCard({ state }: { state: PlantState }) {
  const produced = Math.max(1e-9, state.energy_produced_kwh)
  const capturedFrac = Math.max(0, produced - state.energy_curtailed_kwh) / produced

  // live destination mix: everything leaving the bus this timestep
  const routes = [
    { name: 'To battery', power: state.solar_to_battery_kw, tint: 'var(--stored-ink)' },
    { name: 'To stack', power: state.electrolyser_power_kw, tint: 'var(--feed-ink)' },
    { name: 'To DAC', power: state.dac_power_kw, tint: 'var(--co2-ink)' },
    { name: 'To methanator', power: state.methanator_power_kw, tint: 'var(--ch4-ink)' },
    { name: 'Curtailed', power: state.solar_curtailed_kw, tint: 'var(--waste-ink)' },
  ]
  const routed = Math.max(1e-9, routes.reduce((total, route) => total + route.power, 0))

  return (
    <section className="card card--bone balance">
      <header className="balance__head">
        <span className="label">Energy balance</span>
        <span className="balance__meta">Since reset</span>
      </header>

      <div className="balance__capture">
        <div className="balance__readout">
          <span className="balance__value mono">{pct(capturedFrac, 1)}</span>
          <span className="balance__unit">% of solar captured</span>
        </div>
        <div className="balance__split" role="img" aria-label={`${pct(capturedFrac, 0)} percent captured`}>
          <i className="balance__split-used" style={{ width: `${capturedFrac * 100}%` }} />
        </div>
      </div>

      <div className="balance__now">
        <div className="balance__now-head">
          <span className="label">Right now</span>
          <span className="balance__meta">kW</span>
        </div>
        {routes.map((route) => (
          <div className="balance__route" key={route.name}>
            <span className="balance__route-name">{route.name}</span>
            <span className="balance__route-bar">
              <i style={{ width: `${(route.power / routed) * 100}%`, background: route.tint }} />
            </span>
            <span className="balance__route-value mono">{kw(route.power)}</span>
          </div>
        ))}
      </div>

      <dl className="balance__rows">
        <div>
          <dt>Solar available</dt>
          <dd className="mono">{kwh(state.energy_produced_kwh)} kWh</dd>
        </div>
        <div>
          <dt>Curtailed</dt>
          <dd className="mono">{kwh(state.energy_curtailed_kwh)} kWh</dd>
        </div>
        <div>
          <dt>Hydrogen made</dt>
          <dd className="mono">{state.h2_produced_total_kg.toFixed(1)} kg</dd>
        </div>
        <div>
          <dt>CO₂ captured</dt>
          <dd className="mono">{state.co2_captured_total_kg.toFixed(1)} kg</dd>
        </div>
        <div>
          <dt>Methane made</dt>
          <dd className="mono">{state.ch4_produced_total_kg.toFixed(1)} kg</dd>
        </div>
      </dl>
    </section>
  )
}
