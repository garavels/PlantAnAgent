/** Agent: what the supervisor makes of the plant.
 *
 * The supervisor runs in the backend, on situations rather than on timesteps, so
 * this page is a window and not a controller: it reads the snapshot that arrives
 * with every payload (status, latest incident, trace) and offers exactly two
 * gestures -- investigate now, and switch the supervisor off.
 *
 * Everything shown here is an *operational* record. The trace lines are the ones
 * the supervisor chose to log; its reasoning is not exposed, by design.
 */

import { PageHeader } from '../components/shell/PageHeader'
import { usePlant } from '../state/plantContext'
import type {
  AgentIncident,
  AgentStatus,
  AgentTraceEntry,
  ControlStatus,
  EventRisk,
  RiskLevel,
  RiskSnapshot,
  RiskTrend,
} from '../types/plant'
import './agent.css'

const STATUS_WORD: Record<AgentStatus, string> = {
  monitoring: 'Monitoring',
  investigating: 'Investigating',
  attention: 'Human attention required',
  disabled: 'Disabled',
}

const STATUS_CHIP: Record<AgentStatus, string> = {
  monitoring: 'chip--live',
  investigating: 'chip--warn',
  attention: 'chip--alarm',
  disabled: 'chip--idle',
}

/** V8: how a controller's health grade reads as a chip. */
const GRADE_CHIP: Record<string, string> = {
  good: 'chip--live',
  suspect: 'chip--warn',
  poor: 'chip--alarm',
  unrated: 'chip--idle',
}

/** Trace kinds that deserve emphasis over the run-of-the-mill tool calls. */
const LOUD: AgentTraceEntry['kind'][] = ['prediction', 'diagnosis', 'action', 'error']

/** V10: how a risk level reads. LOW is deliberately quiet. */
const LEVEL_CLASS: Record<RiskLevel, string> = {
  low: '',
  elevated: ' agent__risk-cell--elevated',
  high: ' agent__risk-cell--high',
  critical: ' agent__risk-cell--critical',
}

const TREND_MARK: Record<RiskTrend, string> = {
  stable: '',
  rising: '\u2191',
  falling: '\u2193',
}

function words(value: string): string {
  return value.replace(/_/g, ' ')
}

export function AgentPage() {
  const { agent, control, investigate, setAgentEnabled, state } = usePlant()

  if (!agent) {
    return (
      <div className="card agent__blank">
        <span className="label">Waiting for the supervisor</span>
      </div>
    )
  }

  const incident = agent.incident
  const trace = [...agent.trace].reverse()
  const history = [...agent.incidents].reverse()

  return (
    <div className="agent">
      <PageHeader title="Agent">
        <span className="page-head__stat">
          <span className={`chip ${STATUS_CHIP[agent.status]}`}>
            <i className="chip__led" />
            {STATUS_WORD[agent.status]}
          </span>
        </span>
        <span className="page-head__stat">
          <span className="page-head__value">{agent.runs}</span>
          <span className="page-head__unit">Runs</span>
        </span>
        <span className="page-head__stat">
          <span className="page-head__value">{agent.last_run_clock ?? '--'}</span>
          <span className="page-head__unit">Last run</span>
        </span>
        <span className="page-head__stat">
          <span className="page-head__value">{words(agent.engine)}</span>
          <span className="page-head__unit">Supervisor</span>
        </span>
      </PageHeader>

      {agent.human_intervention_required && (
        <div className="agent__banner">
          <span className="chip chip--alarm">
            <i className="chip__led" />
            Human intervention required
          </span>
          <span>{incident?.summary}</span>
        </div>
      )}

      <div className="agent__top">
        <IncidentCard incident={incident} />
        <HierarchyCard control={control} />
        <section className="card agent__deck">
          <header className="agent__head">
            <h3 className="agent__title">Supervision</h3>
            <span className="label">{state?.clock ?? ''}</span>
          </header>
          <p className="agent__note">
            The controller runs the plant every timestep. The supervisor is woken by
            situations -- a warning, a critical event, a degraded plant, a collapse in
            output -- at most once every {Math.round(agent.min_gap_s / 60)} simulated
            minutes, and never twice for the same trigger within{' '}
            {Math.round(agent.cooldown_s / 60)} simulated minutes.
          </p>
          <div className="agent__buttons">
            <button className="pill pill--on" onClick={investigate} disabled={agent.busy}>
              {agent.busy ? 'Investigating…' : 'Investigate current plant'}
            </button>
            <button className="pill" onClick={() => setAgentEnabled(!agent.enabled)}>
              {agent.enabled ? 'Switch supervisor off' : 'Switch supervisor on'}
            </button>
          </div>
          {agent.last_error && <p className="agent__error mono">{agent.last_error}</p>}
        </section>
      </div>

      <PredictiveRiskCard risk={agent.risk} />

      <div className="agent__bottom">
        <section className="card agent__trace">
          <header className="agent__head">
            <h3 className="agent__title">Trace</h3>
            <span className="label">Newest first</span>
          </header>
          <ul className="agent__lines">
            {trace.length === 0 && <li className="agent__empty">The supervisor has not run yet</li>}
            {trace.map((entry) => (
              <li
                key={entry.seq}
                className={`agent__line${LOUD.includes(entry.kind) ? ' agent__line--loud' : ''}`}
              >
                <span className="agent__clock mono">{entry.clock}</span>
                <span className="agent__kind">{entry.kind}</span>
                <span className="agent__message">{entry.message}</span>
              </li>
            ))}
          </ul>
        </section>

        <section className="card agent__history">
          <header className="agent__head">
            <h3 className="agent__title">Incidents</h3>
            <span className="label">{history.length}</span>
          </header>
          <ul className="agent__incidents">
            {history.length === 0 && <li className="agent__empty">None so far</li>}
            {history.map((entry) => (
              <li key={entry.id} className="agent__incident">
                <span className="agent__clock mono">{entry.opened_clock}</span>
                <span className="agent__category">{words(entry.category)}</span>
                <span className={`agent__state${entry.resolved ? '' : ' agent__state--open'}`}>
                  {entry.resolved ? 'closed' : 'open'}
                </span>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  )
}

/** V10: what HEPA says may happen, per event and horizon.
 *
 * A forecast is only shown when there is one: with no trained checkpoint the card
 * says so and stops there, because a plausible-looking table of numbers no model
 * produced is worse than no table. Levels colour the cells and an arrow marks a
 * risk that moved since the previous forecast -- the movement is usually the
 * information. Nothing here is a control: predictions reach the plant only through
 * the supervisor.
 */
function PredictiveRiskCard({ risk }: { risk: RiskSnapshot }) {
  const horizons = risk.horizons
  const byEvent = new Map<string, Map<string, EventRisk>>()
  for (const entry of risk.risks) {
    const row = byEvent.get(entry.event) ?? new Map<string, EventRisk>()
    row.set(entry.horizon, entry)
    byEvent.set(entry.event, row)
  }
  const fixture = risk.source.startsWith('fixture')

  return (
    <section className="card agent__risk">
      <header className="agent__head">
        <h3 className="agent__title">Predictive risk</h3>
        <span className="label">
          {risk.available ? `HEPA · ${risk.updated_clock ?? '--'}` : 'HEPA'}
        </span>
      </header>

      <div className="agent__risk-status">
        <span className={`chip ${risk.available ? 'chip--live' : 'chip--idle'}`}>
          <i className="chip__led" />
          {risk.available ? 'Active' : 'No trained model loaded'}
        </span>
        {fixture && <span className="label">{words(risk.source)} — test fixture</span>}
      </div>

      {!risk.available ? (
        <p className="agent__note">
          {risk.reason ??
            'No event-risk model is loaded, so no probabilities are shown. The supervisor is ' +
              'working from telemetry, events and diagnostics alone.'}
        </p>
      ) : (
        <>
          <table className="agent__risk-table">
            <thead>
              <tr>
                <th className="label">Event</th>
                {horizons.map((horizon) => (
                  <th key={horizon} className="label">
                    {horizon}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...byEvent.entries()].map(([event, row]) => (
                <tr key={event}>
                  <td className="agent__risk-event">{words(event)}</td>
                  {horizons.map((horizon) => {
                    const cell = row.get(horizon)
                    return (
                      <td
                        key={horizon}
                        className={`agent__risk-cell mono${
                          cell ? LEVEL_CLASS[cell.level] : ''
                        }`}
                      >
                        {cell ? `${Math.round(cell.probability * 100)}%` : '--'}
                        {cell && <span className="agent__risk-trend">{TREND_MARK[cell.trend]}</span>}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="agent__note">
            Probabilities, not certainties, refreshed every{' '}
            {Math.round(risk.refresh_s / 60)} simulated minutes. A level is when the
            supervisor looks; what it does is decided from telemetry and diagnostics.
          </p>
        </>
      )}
    </section>
  )
}

/** V8: the hierarchy in four lines -- who drives, how, how well, and last handover.
 *
 * The supervisor sits above a controller that runs every timestep, so "who is
 * driving" is the one fact this page was missing: an intervention that keeps RL
 * on the plant reads very differently from one that takes it off.
 */
function HierarchyCard({ control }: { control: ControlStatus | null }) {
  const hierarchy = control?.hierarchy
  const health = hierarchy?.health
  const grade = health?.grade ?? 'unrated'
  const handover = hierarchy?.last_switch ?? null
  return (
    <section className="card agent__hierarchy">
      <header className="agent__head">
        <h3 className="agent__title">Control</h3>
        <span className="label">{hierarchy ? words(hierarchy.operating_state) : '--'}</span>
      </header>
      <dl className="agent__facts">
        <div>
          <dt className="label">Active controller</dt>
          <dd className="agent__value">{control?.controller.label ?? '--'}</dd>
        </div>
        <div>
          <dt className="label">Health</dt>
          <dd className="agent__value">
            <span className={`chip ${GRADE_CHIP[grade] ?? 'chip--idle'}`}>
              <i className="chip__led" />
              {grade}
            </span>
          </dd>
        </div>
        <div className="agent__facts-wide">
          <dt className="label">Behaviour</dt>
          <dd className="agent__summary">
            {health?.summary ?? 'No controller history yet'}
          </dd>
        </div>
        <div className="agent__facts-wide">
          <dt className="label">Latest switch</dt>
          <dd className="agent__summary">
            {handover ? (
              <>
                <span className="agent__action mono">{handover.summary}</span>
                <span className="agent__switch-reason">
                  {handover.clock} &middot; {handover.by} &middot; {handover.reason}
                </span>
              </>
            ) : (
              'No handover this run'
            )}
          </dd>
        </div>
      </dl>
    </section>
  )
}

/** The latest situation: what it is, how sure the supervisor is, what it did. */
function IncidentCard({ incident }: { incident: AgentIncident | null }) {
  return (
    <section className="card agent__incident-card">
      <header className="agent__head">
        <h3 className="agent__title">Latest incident</h3>
        <span className="label">{incident ? incident.id : 'None'}</span>
      </header>

      {!incident ? (
        <p className="agent__note">
          Nothing has needed supervision. The plant is being run by the controller inside
          its envelope.
        </p>
      ) : (
        <dl className="agent__facts">
          <div>
            <dt className="label">Diagnosis</dt>
            <dd className="agent__value">{words(incident.category)}</dd>
          </div>
          <div>
            <dt className="label">Component</dt>
            <dd className="agent__value">{incident.component ? words(incident.component) : '--'}</dd>
          </div>
          <div>
            <dt className="label">Confidence</dt>
            <dd className="agent__value">{incident.confidence}</dd>
          </div>
          <div>
            <dt className="label">Triggered by</dt>
            <dd className="agent__value">{words(incident.trigger)}</dd>
          </div>
          <div className="agent__facts-wide">
            <dt className="label">Summary</dt>
            <dd className="agent__summary">{incident.summary || 'Investigating…'}</dd>
          </div>
          <div className="agent__facts-wide">
            <dt className="label">Action</dt>
            <dd className="agent__actions">
              {incident.actions.length === 0
                ? 'No change to the envelope'
                : incident.actions.map((action) => (
                    <span key={action} className="agent__action mono">
                      {action}
                    </span>
                  ))}
            </dd>
          </div>
        </dl>
      )}
    </section>
  )
}
