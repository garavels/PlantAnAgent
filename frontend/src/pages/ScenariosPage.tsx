/** Scenarios: pick the day the plant has to survive, then break it on purpose.
 *
 * Both catalogues come from the simulator; this page only offers them and shows
 * what is currently running.
 */

import { FaultDeck } from '../components/scenarios/FaultDeck'
import { ScenarioCatalog } from '../components/scenarios/ScenarioCatalog'
import { PageHeader } from '../components/shell/PageHeader'
import { usePlant } from '../state/plantContext'
import './scenarios.css'

export function ScenariosPage() {
  const { scenario, scenarios, faults, state } = usePlant()

  const running = scenarios.find((entry) => entry.id === scenario)
  const live = faults.filter((fault) => !fault.cleared)

  return (
    <div className="scenarios">
      <PageHeader title="Scenarios">
        <span className="page-head__stat">
          <span className={`chip ${running ? 'chip--live' : 'chip--idle'}`}>
            <i className="chip__led" />
            {running ? running.name : 'No scenario loaded'}
          </span>
        </span>
        <span className="page-head__stat">
          <span className="page-head__value">{live.length}</span>
          <span className="page-head__unit">Active faults</span>
        </span>
        <span className="page-head__stat">
          <span className="page-head__value">{state?.clock ?? '--'}</span>
          <span className="page-head__unit">Sim clock</span>
        </span>
      </PageHeader>

      <p className="scenarios__note">
        Running a scenario resets the plant to the scenario's starting hour and stores,
        then schedules its faults. Injected faults apply to the plant as it is now.
      </p>

      <ScenarioCatalog />

      <FaultDeck />
    </div>
  )
}
