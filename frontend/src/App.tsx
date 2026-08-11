import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'

import { AppShell } from './components/shell/AppShell'
import { PlantProvider } from './state/PlantProvider'
import { READY_VIEWS } from './views'

export default function App() {
  return (
    <PlantProvider>
      <BrowserRouter>
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
