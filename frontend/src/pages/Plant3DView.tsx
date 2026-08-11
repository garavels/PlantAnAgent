/** Lazy wrapper for the twin: three.js stays out of the Overview's bundle,
 * and the view manifest keeps importing plain components. */

import { lazy, Suspense } from 'react'

// the twin's css is tiny; importing it here keeps the loading state styled
import './plant-3d.css'

const Plant3DPage = lazy(() =>
  import('./Plant3DPage').then((module) => ({ default: module.Plant3DPage })),
)

export function Plant3DView() {
  return (
    <Suspense
      fallback={
        <div className="card plant3d__blank">
          <span className="label">Loading the twin</span>
        </div>
      }
    >
      <Plant3DPage />
    </Suspense>
  )
}
