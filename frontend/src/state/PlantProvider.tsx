import type { ReactNode } from 'react'

import { usePlantSession } from '../hooks/usePlantSession'
import { PlantContext } from './plantContext'

/** One plant session, shared by every view. */
export function PlantProvider({ children }: { children: ReactNode }) {
  return <PlantContext.Provider value={usePlantSession()}>{children}</PlantContext.Provider>
}
