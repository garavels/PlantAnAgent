/** Events: the simulator's own log, on a page of its own.
 *
 * The overview used to carry a cramped twelve-line feed. Here the whole ring
 * buffer is readable, filterable by level, newest first.
 */

import { useState } from 'react'

import { EventLog } from '../components/events/EventLog'
import { PageHeader } from '../components/shell/PageHeader'
import { usePlant } from '../state/plantContext'
import type { EventLevel } from '../types/plant'
import './events.css'

type Filter = 'all' | EventLevel

const FILTERS: { key: Filter; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'info', label: 'Info' },
  { key: 'warning', label: 'Warnings' },
  { key: 'critical', label: 'Critical' },
]

export function EventsPage() {
  const { events, state } = usePlant()
  const [filter, setFilter] = useState<Filter>('all')

  const count = (level: EventLevel) => events.filter((event) => event.level === level).length
  const shown = [...(filter === 'all' ? events : events.filter((event) => event.level === filter))].reverse()

  return (
    <div className="events-page">
      <PageHeader title="Events">
        <span className="page-head__stat">
          <span className="page-head__value">{events.length}</span>
          <span className="page-head__unit">Logged</span>
        </span>
        <span className="page-head__stat">
          <span className="page-head__value">{count('warning')}</span>
          <span className="page-head__unit">Warnings</span>
        </span>
        <span className="page-head__stat">
          <span className="page-head__value">{count('critical')}</span>
          <span className="page-head__unit">Critical</span>
        </span>
        <span className="page-head__stat">
          <span className="page-head__value">{state?.clock ?? '--'}</span>
          <span className="page-head__unit">Sim clock</span>
        </span>
      </PageHeader>

      <section className="card events-page__log">
        <header className="events-page__head">
          <h3 className="events-page__title">Simulator log</h3>
          <div className="events-page__filters">
            {FILTERS.map(({ key, label }) => (
              <button
                key={key}
                className={`pill${filter === key ? ' pill--on' : ''}`}
                onClick={() => setFilter(key)}
                aria-pressed={filter === key}
              >
                {label}
              </button>
            ))}
          </div>
          <span className="label">
            {shown.length} shown · newest first
          </span>
        </header>
        <EventLog events={shown} codes />
      </section>
    </div>
  )
}
