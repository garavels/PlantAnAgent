/** Abnormal operation: inject a fault, see what is running, clear it.
 *
 * A demo/developer panel, not a fault editor. The catalogue comes from the
 * simulator, so adding a fault type there is enough to make it appear here.
 */

import { usePlant } from '../../state/plantContext'
import './fault-deck.css'

/** `2 h 30 m` for a duration in simulated seconds. */
function span(seconds: number | null): string {
  if (seconds === null) return 'until cleared'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.round((seconds % 3600) / 60)
  return hours > 0 ? `${hours} h${minutes ? ` ${minutes} m` : ''}` : `${minutes} m`
}

export function FaultDeck() {
  const { link, faults, faultKinds, injectFault, clearFault, clearFaults } = usePlant()

  const offline = link !== 'online'
  const live = faults.filter((fault) => !fault.cleared)

  return (
    <section className="card card--bone faults">
      <div className="faults__group">
        <div className="faults__row">
          <span className="label">Inject fault</span>
          <span className="faults__hint">Applies to the running plant, at the current sim time</span>
        </div>
        <div className="faults__kinds">
          {faultKinds.map((kind) => (
            <button
              key={kind.type}
              className="faults__inject"
              onClick={() => injectFault(kind.type)}
              disabled={offline}
              title={kind.description}
            >
              {kind.label}
            </button>
          ))}
        </div>
      </div>

      <div className="faults__group">
        <div className="faults__row">
          <span className="label">Active</span>
          <button className="pill" onClick={clearFaults} disabled={offline || live.length === 0}>
            Clear all
          </button>
        </div>
        <ul className="faults__active">
          {live.length === 0 && <li className="faults__none">Plant is running clean</li>}
          {live.map((fault) => (
            <li key={fault.id} className={`faults__fault${fault.active ? ' faults__fault--on' : ''}`}>
              <span className="faults__fault-name">
                <i className={`faults__led faults__led--${fault.mode}`} />
                {fault.label}
              </span>
              <span className="faults__fault-meta mono">
                {Math.round(fault.severity * 100)}% · {fault.active ? span(fault.duration_s) : 'scheduled'}
              </span>
              <button
                className="faults__clear"
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
