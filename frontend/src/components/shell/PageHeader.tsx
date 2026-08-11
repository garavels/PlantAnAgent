import type { ReactNode } from 'react'

import './page-header.css'

/** View title on the left, a live readout on the right. Reused by future pages. */
export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="page-head">
      <h2 className="page-head__title">{title}</h2>
      <div className="page-head__aside">{children}</div>
    </header>
  )
}
