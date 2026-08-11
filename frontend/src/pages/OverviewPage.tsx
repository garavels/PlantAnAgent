/** Overview: what the plant is doing right now.
 *
 * Reads simulator state only -- four readouts, the single-line diagram, the
 * energy balance, and the operator deck. Future views (3D plant, metrics, agent)
 * sit alongside it and read the same `usePlant()` session.
 */

import { BalanceCard } from '../components/overview/BalanceCard'
import { ControlDeck } from '../components/overview/ControlDeck'
import { MetricCard } from '../components/overview/MetricCard'
import { PlantFlow } from '../components/overview/PlantFlow'
import { PageHeader } from '../components/shell/PageHeader'
import { kg, kw, pct } from '../lib/format'
import { usePlant } from '../state/plantContext'
import './overview.css'

export function OverviewPage() {
  const { state, frames, config, link, error, reconnect } = usePlant()

  if (!state) {
    return (
      <div className="card overview__blank">
        <span className="label">{link === 'down' ? 'No link to the simulator' : 'Linking to the simulator'}</span>
        {link === 'down' && (
          <>
            <p className="overview__hint">
              Start the dev API, then reconnect:
              <code>uv run uvicorn plant_backend.app:app --reload --port 8000</code>
            </p>
            <p className="overview__error mono">{error}</p>
            <button className="pill" onClick={reconnect}>
              Retry link
            </button>
          </>
        )}
      </div>
    )
  }

  const peakKw = config?.solar.peak_kw ?? 5000
  const ratedKw = config?.electrolyser.rated_kw ?? 3000
  const ratedKgH = ratedKw / (config?.electrolyser.specific_energy_kwh_per_kg ?? 52)

  return (
    <div className="overview">
      <PageHeader title="Overview">
        <span className="page-head__stat">
          <span className="page-head__value">{state.clock}</span>
          <span className="page-head__unit">Sim clock</span>
        </span>
        <span className="page-head__stat">
          <span className="page-head__value">{kg(state.h2_produced_total_kg)}</span>
          <span className="page-head__unit">kg H₂ total</span>
        </span>
      </PageHeader>

      {link === 'down' && (
        <div className="overview__banner">
          <span className="chip chip--alarm">
            <i className="chip__led" />
            Link down
          </span>
          <span className="mono">{error}</span>
          <button className="pill" onClick={reconnect}>
            Retry
          </button>
        </div>
      )}

      <section className="overview__metrics">
        <MetricCard
          index={0}
          label="Solar power"
          unit="kW available"
          value={kw(state.solar_available_kw)}
          fill={state.solar_available_kw / peakKw}
          footer={`Peak ${kw(peakKw)} kW · ${state.solar_available_kw > 0.5 ? 'daylight' : 'night'}`}
          trend={frames.map((frame) => frame.solar_available_kw)}
          trendMax={peakKw}
        />
        <MetricCard
          index={1}
          label="Battery"
          unit="% state of charge"
          value={pct(state.battery_soc)}
          fill={state.battery_soc}
          footer={`${kw(state.battery_energy_kwh)} kWh stored · ${
            state.battery_power_kw >= 0 ? '+' : ''
          }${kw(state.battery_power_kw)} kW`}
          trend={frames.map((frame) => frame.battery_soc)}
          trendMax={1}
        />
        <MetricCard
          index={2}
          label="H₂ production"
          unit="kg per hour"
          value={state.h2_production_kg_h.toFixed(1)}
          fill={state.electrolyser_load_frac}
          footer={`Stack ${state.electrolyser_online ? 'online' : 'offline'} · ${kw(
            state.electrolyser_power_kw,
          )} kW`}
          trend={frames.map((frame) => frame.h2_production_kg_h)}
          trendMax={ratedKgH}
        />
        <MetricCard
          index={3}
          label="H₂ storage"
          unit="kg in tank"
          value={kg(state.h2_level_kg)}
          fill={state.h2_fill}
          footer={`${pct(state.h2_fill, 0)}% of ${kw(state.h2_capacity_kg)} kg capacity`}
          trend={frames.map((frame) => frame.h2_level_kg)}
          trendMax={state.h2_capacity_kg}
        />
      </section>

      <section className="overview__middle">
        <div className="card overview__diagram">
          <header className="overview__diagram-head">
            <h3 className="overview__diagram-title">Plant flow</h3>
            <span className="label">Live power routing · {config?.dt_s ?? 60}s timestep</span>
          </header>
          <PlantFlow state={state} config={config} />
        </div>
        <BalanceCard state={state} />
      </section>

      <ControlDeck />
    </div>
  )
}
