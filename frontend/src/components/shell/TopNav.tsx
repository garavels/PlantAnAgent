import { NavLink } from 'react-router-dom'

import { usePlant } from '../../state/plantContext'
import { VIEWS } from '../../views'
import './top-nav.css'

const LINK_CHIP = {
  connecting: { className: 'chip chip--warn', text: 'Linking' },
  online: { className: 'chip chip--live', text: 'Link ok' },
  down: { className: 'chip chip--alarm', text: 'Link down' },
} as const

/** Logo, views, and the two lamps that say whether anything is alive. */
export function TopNav() {
  const { playing, link } = usePlant()
  const linkChip = LINK_CHIP[link]

  return (
    <nav className="nav">
      <span className="nav__mark" aria-label="Plant An Agent">
        <svg viewBox="0 0 28 28" aria-hidden>
          <rect x="0.75" y="0.75" width="26.5" height="26.5" rx="9" fill="none" stroke="var(--accent)" />
          <path d="M9.5 5.5h4.5a8.5 8.5 0 0 1 0 17H9.5Z" fill="var(--accent)" />
        </svg>
      </span>

      <ul className="nav__list">
        {VIEWS.map((view) => (
          <li key={view.path}>
            {view.ready ? (
              <NavLink to={view.path} className="nav__link">
                {view.label}
              </NavLink>
            ) : (
              <span className="nav__link nav__link--locked" title={`${view.label} -- not built yet`}>
                {view.label}
                <i className="nav__soon" />
              </span>
            )}
          </li>
        ))}
      </ul>

      <div className="nav__status">
        <span className={`chip ${playing ? 'chip--live' : 'chip--idle'}`}>
          <i className="chip__led" style={playing ? { animation: 'breathe 1.8s infinite' } : undefined} />
          {playing ? 'Running' : 'Paused'}
        </span>
        <span className={linkChip.className}>
          <i className="chip__led" />
          {linkChip.text}
        </span>
      </div>
    </nav>
  )
}
