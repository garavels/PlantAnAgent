/** The console's view manifest.
 *
 * This single list drives both the router and the nav rail. Adding the 3D plant,
 * an agent console or a metrics view means writing the page, flipping `ready` to
 * true and pointing `element` at it -- nothing else moves.
 */

import type { ReactNode } from 'react'

import { OverviewPage } from './pages/OverviewPage'
import { Plant3DView } from './pages/Plant3DView'

export interface ViewDef {
  path: string
  label: string
  glyph: ReactNode
  ready: boolean
  element?: ReactNode
}

const stroke = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.6,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
}

export const VIEWS: ViewDef[] = [
  {
    path: '/',
    label: 'Overview',
    ready: true,
    element: <OverviewPage />,
    glyph: (
      <svg viewBox="0 0 20 20" {...stroke}>
        <rect x="2.5" y="2.5" width="6" height="6" />
        <rect x="11.5" y="2.5" width="6" height="6" />
        <rect x="2.5" y="11.5" width="6" height="6" />
        <rect x="11.5" y="11.5" width="6" height="6" />
      </svg>
    ),
  },
  {
    path: '/plant-3d',
    label: '3D Plant',
    ready: true,
    element: <Plant3DView />,
    glyph: (
      <svg viewBox="0 0 20 20" {...stroke}>
        <path d="M10 2.5 17 6.3v7.4L10 17.5 3 13.7V6.3Z" />
        <path d="M3 6.3 10 10l7-3.7M10 10v7.5" />
      </svg>
    ),
  },
  {
    path: '/agent',
    label: 'Agent',
    ready: false,
    glyph: (
      <svg viewBox="0 0 20 20" {...stroke}>
        <rect x="4" y="6.5" width="12" height="9" rx="2" />
        <path d="M10 3v3.5M7.5 10.5h.01M12.5 10.5h.01M8 13h4" />
      </svg>
    ),
  },
  {
    path: '/metrics',
    label: 'Metrics',
    ready: false,
    glyph: (
      <svg viewBox="0 0 20 20" {...stroke}>
        <path d="M2.5 16.5h15M5 13.5v-4M9 13.5V5M13 13.5v-6M17 13.5v-9" />
      </svg>
    ),
  },
  {
    path: '/events',
    label: 'Events',
    ready: false,
    glyph: (
      <svg viewBox="0 0 20 20" {...stroke}>
        <path d="M2.5 10h3.2l2-4.5 2.6 9L12.8 10h4.7" />
      </svg>
    ),
  },
  {
    path: '/scenarios',
    label: 'Scenarios',
    ready: false,
    glyph: (
      <svg viewBox="0 0 20 20" {...stroke}>
        <path d="M4 3.5v13M4 5h9l-1.6 2.5L13 10H4" />
      </svg>
    ),
  },
]

export const READY_VIEWS = VIEWS.filter((view) => view.ready)
