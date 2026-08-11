/** Access to the shared plant session.
 *
 * Extension point: the 3D plant, metrics and agent views all read from here, so
 * they stay pure functions of simulator state and never step the plant themselves.
 */

import { createContext, useContext } from 'react'

import type { PlantSession } from '../hooks/usePlantSession'

export const PlantContext = createContext<PlantSession | null>(null)

export function usePlant(): PlantSession {
  const session = useContext(PlantContext)
  if (!session) throw new Error('usePlant must be used inside <PlantProvider>')
  return session
}
