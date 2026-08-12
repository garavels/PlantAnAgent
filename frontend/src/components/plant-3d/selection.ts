/** Selection + hover plumbing shared by every 3D unit.
 *
 * `PlantScene` owns the selected unit id; units register their inspect
 * content (name + live value rows) through `UnitChrome`, which renders the
 * floating name plate and handles pointer affordances.
 */

import { createContext, useContext } from 'react'

export type UnitId =
  | 'solar'
  | 'hub'
  | 'battery'
  | 'electrolyser'
  | 'h2Tank'
  | 'dac'
  | 'co2Tank'
  | 'methanator'
  | 'ch4Tank'

export interface UnitSelection {
  selected: UnitId | null
  select: (id: UnitId | null) => void
  /** Freezes flow packet motion (sim paused or reduced motion). */
  flowsFrozen: boolean
}

export const UnitSelectionContext = createContext<UnitSelection>({
  selected: null,
  select: () => undefined,
  flowsFrozen: false,
})

export const useUnitSelection = () => useContext(UnitSelectionContext)
