/** The simulator's log, newest first.
 *
 * Every line is emitted by the simulator itself -- the console never invents
 * an event from the numbers it happens to be drawing.
 */

import type { PlantEvent } from '../../types/plant'
import './event-log.css'

interface EventLogProps {
  events: PlantEvent[]
  /** Show the simulator's event code alongside the message. */
  codes?: boolean
}

export function EventLog({ events, codes = false }: EventLogProps) {
  return (
    <ul className={`log__list${codes ? ' log__list--coded' : ''}`}>
      {events.length === 0 && <li className="log__empty">Nothing reported yet</li>}
      {events.map((event) => (
        <li key={event.seq} className={`log__row log__row--${event.level}`}>
          <span className="log__clock mono">{event.clock}</span>
          <span className="log__level">{event.level}</span>
          {codes && <span className="log__code mono">{event.code}</span>}
          <span className="log__message">{event.message}</span>
        </li>
      ))}
    </ul>
  )
}
