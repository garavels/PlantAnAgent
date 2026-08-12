/** Operator controls: run the clock, and split the available power.
 *
 * Each share slider *is* the action: the fraction of available solar offered to
 * that consumer. Shares are independent; if they sum past 100% the simulator
 * scales them down proportionally (shown dimmed). What is left unallocated is
 * curtailed. When an agent takes over, this panel becomes its supervisor rather
 * than its driver.
 */

import { useEffect } from 'react'

import { SPEEDS, TICK_MS } from '../../hooks/usePlantSession'
import { kw, pct } from '../../lib/format'
import { usePlant } from '../../state/plantContext'
import type { PlantAction } from '../../types/plant'
import './control-deck.css'

interface ShareRow {
  key: 'battery_share' | 'electrolyser_share' | 'dac_share' | 'methanator_share'
  label: string
  swatch: string
}

const SHARES: ShareRow[] = [
  { key: 'battery_share', label: 'Battery', swatch: 'deck__swatch--battery' },
  { key: 'electrolyser_share', label: 'Stack', swatch: 'deck__swatch--stack' },
  { key: 'dac_share', label: 'DAC', swatch: 'deck__swatch--dac' },
  { key: 'methanator_share', label: 'Methanator', swatch: 'deck__swatch--met' },
]

/** Mirror of `Action.sanitised`: what the simulator will actually apply. */
function normalised(action: PlantAction): PlantAction {
  const total = action.battery_share + action.electrolyser_share + action.dac_share + action.methanator_share
  if (total <= 1) return action
  const scale = (share: number) => share / total
  return {
    ...action,
    battery_share: scale(action.battery_share),
    electrolyser_share: scale(action.electrolyser_share),
    dac_share: scale(action.dac_share),
    methanator_share: scale(action.methanator_share),
  }
}

export function ControlDeck() {
  const { config, action, playing, speed, link, setAction, setSpeed, togglePlay, stepOnce, reset } = usePlant()

  const maxSupportKw = config?.battery.max_discharge_kw ?? 2500
  const simMinutesPerSecond = ((speed * (1000 / TICK_MS) * (config?.dt_s ?? 60)) / 60).toFixed(0)
  const offline = link !== 'online'
  const applied = normalised(action)
  const oversubscribed = applied !== action

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

  return (
    <section className="card card--bone deck">
      <div className="deck__group">
        <span className="label">Clock</span>
        <div className="deck__buttons">
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
        <div className="deck__speeds">
          {SPEEDS.map((option) => (
            <button
              key={option}
              className={`tick${speed === option ? ' tick--on' : ''}`}
              onClick={() => setSpeed(option)}
            >
              ×{option}
            </button>
          ))}
          <span className="deck__hint">{simMinutesPerSecond} sim-min per second</span>
        </div>
      </div>

      <div className="deck__group deck__group--wide">
        <div className="deck__row">
          <span className="label">Power allocation</span>
          <span className="deck__hint">{oversubscribed ? 'Over 100% — shares scale down to fit' : 'Share of available solar'}</span>
        </div>
        <div className="deck__shares">
          {SHARES.map(({ key, label, swatch }) => (
            <div className="deck__share" key={key}>
              <span className="deck__share-head">
                <em>
                  <i className={`deck__swatch ${swatch}`} />
                  {label}
                </em>
                <b className="mono">
                  {pct(action[key], 0)}%{oversubscribed && <i className="deck__applied mono"> → {pct(applied[key], 0)}%</i>}
                </b>
              </span>
              <input
                className={`slider slider--${key}`}
                type="range"
                min={0}
                max={100}
                step={1}
                value={Math.round(action[key] * 100)}
                onChange={(event) => setAction({ ...action, [key]: Number(event.target.value) / 100 })}
                aria-label={`Share of available solar power sent to the ${label.toLowerCase()}`}
              />
            </div>
          ))}
        </div>
      </div>

      <div className="deck__group">
        <div className="deck__row">
          <span className="label">Battery support</span>
          <span className="deck__values">
            <em>
              <b className="mono">{kw(action.battery_support_kw)}</b> kW
            </em>
          </span>
        </div>
        <input
          className="slider slider--support"
          type="range"
          min={0}
          max={maxSupportKw}
          step={50}
          value={action.battery_support_kw}
          onChange={(event) => setAction({ ...action, battery_support_kw: Number(event.target.value) })}
          aria-label="Battery discharge offered to the electrolyser"
        />
        <div className="deck__scale">
          <span>Discharge to keep the stack alive</span>
          <span className="mono">{kw(maxSupportKw)} kW</span>
        </div>
      </div>
    </section>
  )
}
