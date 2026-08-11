/** Operator controls: run the clock, and split the available power.
 *
 * The split slider *is* the action: everything left of centre buffers energy in
 * the battery, everything right of it goes to the stack. When an agent takes
 * over, this panel becomes its supervisor rather than its driver.
 */

import { useEffect } from 'react'

import { SPEEDS, TICK_MS } from '../../hooks/usePlantSession'
import { kw, pct } from '../../lib/format'
import { usePlant } from '../../state/plantContext'
import './control-deck.css'

export function ControlDeck() {
  const { config, action, playing, speed, link, setAction, setSpeed, togglePlay, stepOnce, reset } = usePlant()

  const maxSupportKw = config?.battery.max_discharge_kw ?? 2500
  const simMinutesPerSecond = ((speed * (1000 / TICK_MS) * (config?.dt_s ?? 60)) / 60).toFixed(0)
  const offline = link !== 'online'

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

  const setSplit = (electrolyserPercent: number) => {
    const share = electrolyserPercent / 100
    setAction({ ...action, electrolyser_share: share, battery_share: 1 - share })
  }

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
          <span className="deck__values">
            <em>
              <i className="deck__swatch deck__swatch--battery" />
              <b className="mono">{pct(action.battery_share, 0)}%</b> battery
            </em>
            <em>
              <i className="deck__swatch deck__swatch--stack" />
              <b className="mono">{pct(action.electrolyser_share, 0)}%</b> stack
            </em>
          </span>
        </div>
        <input
          className="slider slider--split"
          type="range"
          min={0}
          max={100}
          step={1}
          value={Math.round(action.electrolyser_share * 100)}
          onChange={(event) => setSplit(Number(event.target.value))}
          aria-label="Share of available solar power sent to the electrolyser"
        />
        <div className="deck__scale">
          <span>Charge the battery</span>
          <span>Make hydrogen</span>
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
