import type { ReactNode } from 'react'

import { TopNav } from './TopNav'
import './app-shell.css'

/** The console: a dark panel inset on the sage room, with a view slot. */
export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="shell">
      <TopNav />
      <main className="shell__view">{children}</main>
    </div>
  )
}
