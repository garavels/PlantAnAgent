/** Describe a situation in words; get a scenario the simulator can already run.
 *
 * The model is only ever reached through the backend, which constrains it to the
 * simulator's own schema and validates what comes back. So this panel is a
 * review step, not an editor: the draft below is a real `Scenario`, and saving it
 * hands it to the same catalogue the predefined ones live in.
 */

import { useState } from 'react'

import { generateScenario } from '../../lib/api'
import { usePlant } from '../../state/plantContext'
import type { ScenarioDef, ScenarioFaultDef } from '../../types/plant'
import './scenario-composer.css'

const EXAMPLE =
  'Solar capacity suddenly drops by 50%, two sensors drift, and the DAC becomes degraded after 3 hours.'

/** `06:00` for an hour of the simulated day. */
function hour(value: number): string {
  const hours = Math.floor(value)
  const minutes = Math.round((value - hours) * 60)
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`
}

/** `+3 h 30 m` for an offset in simulated seconds. */
function offset(seconds: number | null): string {
  const total = Math.round((seconds ?? 0) / 60)
  const hours = Math.floor(total / 60)
  const minutes = total % 60
  if (hours === 0 && minutes === 0) return 'at the start'
  return `+${hours > 0 ? `${hours} h${minutes ? ` ${minutes} m` : ''}` : `${minutes} m`}`
}

function line(fault: ScenarioFaultDef): string {
  const duration = fault.duration_s === null ? 'until cleared' : `for ${offset(fault.duration_s).slice(1)}`
  const channel = fault.channel ? ` on ${fault.channel}` : ''
  return `${fault.type}${channel} · ${Math.round(fault.severity * 100)}% · ${offset(fault.start_s)} · ${duration}`
}

function initials(scenario: ScenarioDef): string {
  const { initial } = scenario
  const rows: string[] = []
  if (initial.battery_soc !== null) rows.push(`battery ${Math.round(initial.battery_soc * 100)}%`)
  if (initial.h2_kg !== null) rows.push(`H₂ ${initial.h2_kg} kg`)
  if (initial.co2_kg !== null) rows.push(`CO₂ ${initial.co2_kg} kg`)
  if (initial.ch4_kg !== null) rows.push(`CH₄ ${initial.ch4_kg} kg`)
  return rows.join(' · ') || 'plant defaults'
}

export function ScenarioComposer() {
  const { link, loadScenario, saveScenario } = usePlant()
  const [prompt, setPrompt] = useState('')
  const [draft, setDraft] = useState<ScenarioDef | null>(null)
  const [warnings, setWarnings] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const offline = link !== 'online'

  async function generate() {
    setBusy(true)
    setError(null)
    try {
      const result = await generateScenario(prompt)
      setDraft(result.scenario)
      setWarnings(result.warnings)
    } catch (reason) {
      setDraft(null)
      setWarnings([])
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setBusy(false)
    }
  }

  async function keep(run: boolean) {
    if (!draft) return
    setBusy(true)
    setError(null)
    try {
      const id = await saveScenario(draft)
      setDraft(null)
      setWarnings([])
      setPrompt('')
      if (run) loadScenario(id)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="card card--bone composer">
      <div className="composer__row">
        <span className="label">Generate scenario</span>
        <span className="composer__hint">
          Described in words, turned into the simulator's own faults and disturbances
        </span>
      </div>

      <textarea
        className="composer__prompt"
        rows={3}
        value={prompt}
        placeholder={EXAMPLE}
        onChange={(event) => setPrompt(event.target.value)}
        disabled={busy}
      />

      <div className="composer__actions">
        <button
          className="pill pill--on"
          onClick={() => void generate()}
          disabled={offline || busy || prompt.trim().length === 0}
        >
          {busy && !draft ? 'Generating…' : 'Generate scenario'}
        </button>
        <button className="pill" onClick={() => setPrompt(EXAMPLE)} disabled={busy}>
          Use the example
        </button>
      </div>

      {error && <p className="composer__error">{error}</p>}

      {draft && (
        <div className="composer__draft">
          <div className="composer__row">
            <h3 className="composer__name">{draft.name}</h3>
            <span className="chip chip--idle">
              <i className="chip__led" />
              For review
            </span>
          </div>
          <p className="composer__desc">{draft.description}</p>

          <dl className="composer__facts">
            <div>
              <dt className="label">Starts</dt>
              <dd className="mono">{hour(draft.start_hour)}</dd>
            </div>
            <div>
              <dt className="label">Initial</dt>
              <dd className="composer__initial">{initials(draft)}</dd>
            </div>
          </dl>

          <span className="label">Scheduled faults</span>
          <ul className="composer__faults">
            {draft.faults.length === 0 && <li className="composer__none">None — a nominal day</li>}
            {draft.faults.map((fault, index) => (
              <li key={`${fault.type}-${index}`} className="mono">
                {line(fault)}
              </li>
            ))}
          </ul>

          {warnings.length > 0 && (
            <ul className="composer__warnings">
              {warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          )}

          <div className="composer__actions">
            <button className="pill pill--on" onClick={() => void keep(true)} disabled={offline || busy}>
              Save and run
            </button>
            <button className="pill" onClick={() => void keep(false)} disabled={offline || busy}>
              Save only
            </button>
            <button
              className="pill"
              onClick={() => {
                setDraft(null)
                setWarnings([])
              }}
              disabled={busy}
            >
              Discard
            </button>
          </div>
        </div>
      )}
    </section>
  )
}
