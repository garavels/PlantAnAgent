/** 3D Plant: the digital twin as the primary surface.
 *
 * The scene fills the view; a quiet readout column rides its right edge and a
 * bone transport bar sits underneath. Every number comes from the shared
 * `usePlant()` session -- this page adds no state of its own beyond which
 * unit is pinned (owned by the scene).
 */

import { useEffect } from 'react'
import type { CSSProperties } from 'react'

import { PlantScene } from '../components/plant-3d/PlantScene'
import { PageHeader } from '../components/shell/PageHeader'
import { SPEEDS, TICK_MS } from '../hooks/usePlantSession'
import { kg, kw, pct } from '../lib/format'
import { usePlant } from '../state/plantContext'
import './plant-3d.css'

export function Plant3DPage() {
  const { state, config, link, error, playing, speed, togglePlay, stepOnce, reset, reconnect, setSpeed } =
    usePlant()

  // control-room reflexes: space runs, R resets
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target && ['INPUT', 'BUTTON', 'TEXTAREA'].includes(target.tagName)) return
      if (event.code === 'Space') {
        event.preventDefault()
        togglePlay()
      }
      if (event.key.toLowerCase() === 'r') reset()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [togglePlay, reset])

  if (!state) {
    return (
      <div className="card plant3d__blank">
        <span className="label">{link === 'down' ? 'No link to the simulator' : 'Linking to the simulator'}</span>
        {link === 'down' && (
          <>
            <p className="plant3d__error mono">{error}</p>
            <button className="pill" onClick={reconnect}>
              Retry link
            </button>
          </>
        )}
      </div>
    )
  }

  const peakKw = config?.solar.peak_kw ?? 5000
  const metRatedKgH = (config?.methanator.rated_kw ?? 180) / (config?.methanator.specific_energy_kwh_per_kg ?? 1.2)
  const offline = link !== 'online'

  const readouts = [
    {
      id: 'solar',
      label: 'Solar',
      accent: 'var(--solar)',
      value: kw(state.solar_available_kw),
      unit: 'kW',
      fill: state.solar_available_kw / peakKw,
      note: state.solar_available_kw > 0.5 ? 'generating' : 'night',
    },
    {
      id: 'battery',
      label: 'Battery',
      accent: 'var(--stored)',
      value: pct(state.battery_soc, 0),
      unit: '%',
      fill: state.battery_soc,
      note: `${state.battery_power_kw >= 0 ? '+' : ''}${kw(state.battery_power_kw)} kW`,
    },
    {
      id: 'stack',
      label: 'Electrolyser',
      accent: 'var(--feed)',
      value: kw(state.electrolyser_power_kw),
      unit: 'kW',
      fill: state.electrolyser_load_frac,
      note: state.electrolyser_online ? 'running' : 'standby',
    },
    {
      id: 'h2',
      label: 'H₂ storage',
      accent: 'var(--h2)',
      value: kg(state.h2_level_kg),
      unit: 'kg',
      fill: state.h2_fill,
      note: `${pct(state.h2_fill, 0)}% full`,
    },
    {
      id: 'co2',
      label: 'CO₂ storage',
      accent: 'var(--co2)',
      value: kg(state.co2_level_kg),
      unit: 'kg',
      fill: state.co2_fill,
      note: state.dac_online ? `capturing ${state.co2_capture_kg_h.toFixed(1)} kg/h` : 'DAC idle',
    },
    {
      id: 'ch4',
      label: 'CH₄ output',
      accent: 'var(--ch4)',
      value: state.ch4_production_kg_h.toFixed(1),
      unit: 'kg/h',
      fill: metRatedKgH > 0 ? state.ch4_production_kg_h / metRatedKgH : 0,
      note: state.methanator_online ? `${kg(state.ch4_level_kg)} kg stored` : 'reactor standby',
    },
  ]

  return (
    <div className="plant3d">
      <PageHeader title="3D Plant">
        <span className="page-head__stat">
          <span className="page-head__value">{state.clock}</span>
          <span className="page-head__unit">Sim clock</span>
        </span>
        <span className="page-head__stat">
          <span className={`chip ${playing ? 'chip--live' : 'chip--idle'}`}>
            <i className="chip__led" style={playing ? { animation: 'breathe 1.8s infinite' } : undefined} />
            {playing ? `Running ×${speed}` : 'Paused'}
          </span>
        </span>
      </PageHeader>

      <section className="plant3d__stage">
        <PlantScene state={state} config={config} playing={playing} />

        <aside className="plant3d__hud" aria-label="Live plant readouts">
          {readouts.map((r) => (
            <div key={r.id} className="plant3d__readout" style={{ '--carrier': r.accent } as CSSProperties}>
              <span className="plant3d__readout-label">
                <i className="plant3d__readout-dot" />
                {r.label}
              </span>
              <span className="plant3d__readout-value mono">
                {r.value}
                <em>{r.unit}</em>
              </span>
              <span className="plant3d__readout-bar">
                <span style={{ width: `${Math.min(100, Math.max(0, r.fill * 100))}%` }} />
              </span>
              <span className="plant3d__readout-note">{r.note}</span>
            </div>
          ))}
          {state.solar_curtailed_kw > 0.5 && (
            <div className="plant3d__curtail mono">{kw(state.solar_curtailed_kw)} kW curtailed</div>
          )}
        </aside>

        <span className="plant3d__hint">Drag to orbit · scroll to zoom · click a unit to inspect</span>
      </section>

      <section className="card card--bone plant3d__bar">
        <div className="plant3d__transport">
          <button className={`pill${playing ? ' pill--on' : ''}`} onClick={togglePlay} disabled={offline}>
            {playing ? (
              <svg viewBox="0 0 12 12" aria-hidden>
                <rect x="2.5" y="1.5" width="2.6" height="9" rx="0.8" />
                <rect x="7" y="1.5" width="2.6" height="9" rx="0.8" />
              </svg>
            ) : (
              <svg viewBox="0 0 12 12" aria-hidden>
                <path d="M3 1.7 10.2 6 3 10.3Z" />
              </svg>
            )}
            {playing ? 'Pause' : 'Play'}
          </button>
          <button className="pill" onClick={stepOnce} disabled={offline || playing}>
            Step
          </button>
          <button className="pill" onClick={reset} disabled={offline}>
            Reset
          </button>
        </div>
        <div className="plant3d__speeds">
          {SPEEDS.map((option) => (
            <button
              key={option}
              className={`plant3d__tick${speed === option ? ' plant3d__tick--on' : ''}`}
              onClick={() => setSpeed(option)}
            >
              ×{option}
            </button>
          ))}
          <span className="plant3d__speedhint">
            {((speed * (1000 / TICK_MS) * (config?.dt_s ?? 60)) / 60).toFixed(0)} sim-min per second
          </span>
        </div>
      </section>
    </div>
  )
}
