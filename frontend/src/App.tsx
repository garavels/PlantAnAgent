import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'

import { AppShell } from './components/shell/AppShell'
import { PlantProvider } from './state/PlantProvider'
import { READY_VIEWS } from './views'

export default function App() {
  return (
    <PlantProvider>
      {/* the router's default transition-priority navigation never commits while
          the stream re-renders the twin, so navigation runs at normal priority */}
      <BrowserRouter useTransitions={false}>
        <AppShell>
          <Routes>
            {READY_VIEWS.map((view) => (
              <Route key={view.path} path={view.path} element={view.element} />
            ))}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AppShell>
      </BrowserRouter>
    </PlantProvider>
  )
}
