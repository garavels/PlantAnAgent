/** Unit chrome: hover/select behaviour and the floating name plate.
 *
 * Wraps a unit's meshes. Hover raises a subtle emissive sheen on the unit and
 * wakes its name plate; click pins the plate open with live values; clicking
 * empty ground (or pressing Escape) clears the pin. Plates are DOM overlays
 * with pointer events disabled -- orbiting never fights the labels.
 */

import { Html, useCursor } from '@react-three/drei'
import { useState } from 'react'
import type { CSSProperties, ReactNode } from 'react'
import type { ThreeEvent } from '@react-three/fiber'

import { useUnitSelection } from './selection'
import type { UnitId } from './selection'

export interface InspectRow {
  label: string
  value: string
}

interface UnitChromeProps {
  id: UnitId
  /** Display name on the plate, e.g. "Electrolyser". */
  name: string
  /** Carrier tint for the plate's status dot. */
  accent: string
  /** Height above the unit origin where the plate floats. */
  plateHeight: number
  /** Sideways shift, for plates that would collide with the HUD. */
  plateOffset?: [number, number]
  /** Live rows revealed while hovered or pinned. */
  rows: InspectRow[]
  children: (active: boolean) => ReactNode
}

export function UnitChrome({ id, name, accent, plateHeight, plateOffset, rows, children }: UnitChromeProps) {
  const { selected, select } = useUnitSelection()
  const [hovered, setHovered] = useState(false)
  useCursor(hovered)

  const pinned = selected === id
  const active = hovered || pinned

  const onClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    select(pinned ? null : id)
  }

  return (
    <group
      onClick={onClick}
      onPointerOver={(event) => {
        event.stopPropagation()
        setHovered(true)
      }}
      onPointerOut={() => setHovered(false)}
    >
      {children(active)}
      <Html
        position={[plateOffset?.[0] ?? 0, plateHeight, plateOffset?.[1] ?? 0]}
        center
        distanceFactor={13}
        zIndexRange={[20, 0]}
        className="unit-plate-anchor"
      >
        <div
          className={`unit-plate${active ? ' unit-plate--on' : ''}`}
          style={{ '--unit-accent': accent } as CSSProperties}
        >
          <span className="unit-plate__head">
            <i className="unit-plate__dot" />
            {name}
          </span>
          <dl className={`unit-plate__rows${active ? ' unit-plate__rows--on' : ''}`}>
            {rows.map((row) => (
              <div key={row.label} className="unit-plate__row">
                <dt>{row.label}</dt>
                <dd className="mono">{row.value}</dd>
              </div>
            ))}
          </dl>
        </div>
      </Html>
    </group>
  )
}
