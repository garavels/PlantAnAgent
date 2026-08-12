/** Unit chrome: hover/select behaviour and the floating name plate.
 *
 * Wraps a unit's meshes. Hover raises a subtle emissive sheen on the unit and
 * wakes its name plate; click pins the plate open with live values; clicking
 * empty ground (or pressing Escape) clears the pin. Plates are DOM overlays
 * with pointer events disabled -- orbiting never fights the labels.
 *
 * A unit that is degraded or offline also gets a warning beacon floating over
 * it and a tinted plate, so an abnormal plant reads at a glance from any
 * camera angle.
 */

import { Html, useCursor } from '@react-three/drei'
import { useRef, useState } from 'react'
import type { CSSProperties, ReactNode } from 'react'
import { useFrame, type ThreeEvent } from '@react-three/fiber'
import * as THREE from 'three'

import type { UnitStatus } from '../../types/plant'
import { statusTint } from './palette'
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
  /** Condition of the unit; anything but `nominal` raises a beacon. */
  status?: UnitStatus
  children: (active: boolean) => ReactNode
}

/** A slowly pulsing warning marker: amber when degraded, red when down. */
function Beacon({ tint, height }: { tint: string; height: number }) {
  const mat = useRef<THREE.MeshStandardMaterial>(null)
  const mesh = useRef<THREE.Mesh>(null)

  useFrame((clock) => {
    const pulse = 0.55 + 0.45 * Math.sin(clock.clock.elapsedTime * 2.6)
    if (mat.current) mat.current.emissiveIntensity = 1.4 + pulse * 2.6
    if (mesh.current) mesh.current.rotation.y += 0.01
  })

  return (
    <mesh ref={mesh} position={[0, height, 0]} raycast={() => null}>
      <octahedronGeometry args={[0.42, 0]} />
      <meshStandardMaterial ref={mat} color={tint} emissive={tint} emissiveIntensity={2} toneMapped={false} />
    </mesh>
  )
}

export function UnitChrome({ id, name, accent, plateHeight, plateOffset, rows, status = 'nominal', children }: UnitChromeProps) {
  const { selected, select } = useUnitSelection()
  const [hovered, setHovered] = useState(false)
  useCursor(hovered)

  const pinned = selected === id
  const active = hovered || pinned
  const tint = statusTint(status)

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
      {tint && <Beacon tint={tint} height={plateHeight - 0.7} />}
      <Html
        position={[plateOffset?.[0] ?? 0, plateHeight, plateOffset?.[1] ?? 0]}
        center
        distanceFactor={13}
        zIndexRange={[20, 0]}
        className="unit-plate-anchor"
      >
        <div
          className={`unit-plate${active ? ' unit-plate--on' : ''}${tint ? ` unit-plate--${status}` : ''}`}
          style={{ '--unit-accent': tint ?? accent } as CSSProperties}
        >
          <span className="unit-plate__head">
            <i className="unit-plate__dot" />
            {name}
            {tint && <em className="unit-plate__flag">{status}</em>}
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
