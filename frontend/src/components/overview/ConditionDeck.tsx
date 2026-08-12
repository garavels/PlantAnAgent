/** Abnormal operation: load a scenario, inject a fault, clear what is running.
 *
 * A demo/developer panel, not a scenario editor. Both catalogues come from the
 * simulator, so adding a fault type or a scenario there is enough to make it
 * appear here.
 */

import { useState } from 'react'

import { usePlant } from '../../state/plantContext'
import './condition-deck.css'

/** `2 h 30 m` for a duration in simulated seconds. */
function span(seconds: number | null): string {
  if (seconds === null) return 'until cleared'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.round((seconds % 3600) / 60)
  return hours > 0 ? `${hours} h${minutes ? ` ${minutes} m` : ''}` : `${minutes} m`
}

export function ConditionDeck() {
  const { link, scenario, scenarios, faults, faultKinds, injectFault, clearFault, clearFaults, loadScenario } =
    usePlant()
  const [choice, setChoice] = useState('')

  const offline = link !== 'online'
  const selected = choice || scenario || scenarios[0]?.id || ''
  const chosen = scenarios.find((entry) => entry.id === selected)
  const live = faults.filter((fault) => !fault.cleared)

  return (
    <section className="card card--bone conditions">
      <div className="conditions__group">
        <div className="conditions__row">
          <span className="label">Scenario</span>
          {scenario && <span className="conditions__hint">Running · {scenario}</span>}
        </div>
        <div className="conditions__pick">
          <select
            className="conditions__select"
            value={selected}
            onChange={(event) => setChoice(event.target.value)}
            disabled={offline || scenarios.length === 0}
            aria-label="Predefined scenario"
          >
            {scenarios.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.name}
              </option>
            ))}
          </select>
          <button className="pill" onClick={() => loadScenario(selected)} disabled={offline || !selected}>
            Run scenario
          </button>
        </div>
        {chosen && <p className="conditions__desc">{chosen.description}</p>}
      </div>

      <div className="conditions__group conditions__group--wide">
        <div className="conditions__row">
          <span className="label">Inject fault</span>
          <span className="conditions__hint">Applies to the running plant, at the current sim time</span>
        </div>
        <div className="conditions__faults">
          {faultKinds.map((kind) => (
            <button
              key={kind.type}
              className="conditions__inject"
              onClick={() => injectFault(kind.type)}
              disabled={offline}
              title={kind.description}
            >
              {kind.label}
            </button>
          ))}
        </div>
      </div>

      <div className="conditions__group">
        <div className="conditions__row">
          <span className="label">Active</span>
          <button className="pill" onClick={clearFaults} disabled={offline || live.length === 0}>
            Clear all
          </button>
        </div>
        <ul className="conditions__active">
          {live.length === 0 && <li className="conditions__none">Plant is running clean</li>}
          {live.map((fault) => (
            <li key={fault.id} className={`conditions__fault${fault.active ? ' conditions__fault--on' : ''}`}>
              <span className="conditions__fault-name">
                <i className={`conditions__led conditions__led--${fault.mode}`} />
                {fault.label}
              </span>
              <span className="conditions__fault-meta mono">
                {Math.round(fault.severity * 100)}% · {fault.active ? span(fault.duration_s) : 'scheduled'}
              </span>
              <button
                className="conditions__clear"
                onClick={() => clearFault(fault.id)}
                aria-label={`Clear ${fault.label}`}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}
