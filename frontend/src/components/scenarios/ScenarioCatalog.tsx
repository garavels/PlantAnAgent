/** The simulator's predefined scenarios, one card each.
 *
 * A demo/developer catalogue, not a scenario editor: adding a scenario to the
 * simulator is enough to make it appear here.
 */

import { usePlant } from '../../state/plantContext'
import type { ScenarioDef } from '../../types/plant'
import './scenario-catalog.css'

/** `06:00` for a scenario's starting hour of the simulated day. */
function hour(value: number): string {
  const hours = Math.floor(value)
  const minutes = Math.round((value - hours) * 60)
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`
}

function initials(scenario: ScenarioDef): string[] {
  const { initial } = scenario
  const rows: string[] = []
  if (initial.battery_soc !== null) rows.push(`battery ${Math.round(initial.battery_soc * 100)}%`)
  if (initial.h2_kg !== null) rows.push(`H₂ ${initial.h2_kg} kg`)
  if (initial.co2_kg !== null) rows.push(`CO₂ ${initial.co2_kg} kg`)
  if (initial.ch4_kg !== null) rows.push(`CH₄ ${initial.ch4_kg} kg`)
  return rows
}

export function ScenarioCatalog() {
  const { link, scenario, scenarios, loadScenario } = usePlant()
  const offline = link !== 'online'

  return (
    <section className="catalog">
      {scenarios.length === 0 && <p className="catalog__empty label">The simulator advertises no scenarios</p>}
      {scenarios.map((entry, index) => {
        const running = entry.id === scenario
        return (
          <article
            key={entry.id}
            className={`card catalog__card${running ? ' catalog__card--on' : ''}`}
            style={{ animationDelay: `${index * 50}ms` }}
          >
            <header className="catalog__head">
              <h3 className="catalog__name">{entry.name}</h3>
              {running && (
                <span className="chip chip--live">
                  <i className="chip__led" />
                  Running
                </span>
              )}
            </header>

            <p className="catalog__desc">{entry.description}</p>

            <dl className="catalog__facts">
              <div>
                <dt className="label">Starts</dt>
                <dd className="mono">{hour(entry.start_hour)}</dd>
              </div>
              <div>
                <dt className="label">Faults</dt>
                <dd className="mono">{entry.faults.length}</dd>
              </div>
              <div>
                <dt className="label">Initial</dt>
                <dd className="catalog__initial">{initials(entry).join(' · ') || 'plant defaults'}</dd>
              </div>
            </dl>

            <button
              className={`pill${running ? '' : ' pill--on'}`}
              onClick={() => loadScenario(entry.id)}
              disabled={offline}
            >
              {running ? 'Restart scenario' : 'Run scenario'}
            </button>
          </article>
        )
      })}
    </section>
  )
}
