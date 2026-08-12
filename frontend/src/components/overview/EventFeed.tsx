/** Recent simulator events, newest first.
 *
 * Every line is emitted by the simulator itself -- the console never invents
 * an event from the numbers it happens to be drawing.
 */

import { usePlant } from '../../state/plantContext'
import './event-feed.css'

interface EventFeedProps {
  /** How many of the most recent events to show. */
  limit?: number
}

export function EventFeed({ limit = 12 }: EventFeedProps) {
  const { events } = usePlant()
  const recent = events.slice(-limit).reverse()

  return (
    <section className="card events">
      <header className="events__head">
        <h3 className="events__title">Events</h3>
        <span className="label">From the simulator</span>
      </header>
      <ul className="events__list">
        {recent.length === 0 && <li className="events__empty">Nothing reported yet</li>}
        {recent.map((event) => (
          <li key={event.seq} className={`events__row events__row--${event.level}`}>
            <span className="events__clock mono">{event.clock}</span>
            <span className="events__level">{event.level}</span>
            <span className="events__message">{event.message}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}
