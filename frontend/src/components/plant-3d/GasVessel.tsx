/** A gas storage vessel: a vertical pressure tank with a true rendered level.
 *
 * The shell is a faint glass so the stored gas reads as a physical column of
 * light inside it -- the fill height IS the tank's fill fraction, eased. A
 * bright meniscus ring marks the surface, and quarter-height gauge rings give
 * it scale. One component serves H2, CO2 and CH4; only the accent, size and
 * site position differ.
 */

import { useFrame } from '@react-three/fiber'
import { useRef } from 'react'
import * as THREE from 'three'

import { SURFACE } from './palette'
import type { UnitId } from './selection'
import { UnitChrome, type InspectRow } from './UnitChrome'
import { damp, type TankVisual } from './visuals'

const LEG_H = 0.7

interface GasVesselProps {
  id: UnitId
  name: string
  accent: string
  /** Site position, from layout.UNIT_POSITIONS. */
  at: { x: number; z: number }
  /** Shell radius / straight shell height, in meters. */
  radius: number
  height: number
  tank: TankVisual
  plateOffset?: [number, number]
}

export function GasVessel({ id, name, accent, at, radius, height, tank, plateOffset }: GasVesselProps) {
  const gasRef = useRef<THREE.Mesh>(null)
  const gasMatRef = useRef<THREE.MeshStandardMaterial>(null)
  const ringRef = useRef<THREE.Mesh>(null)
  const level = useRef(tank.fill)
  const innerR = radius - 0.16

  useFrame((_, delta) => {
    level.current = damp(level.current, tank.fill, delta, 3)
    const l = level.current
    const gas = gasRef.current
    if (gas) {
      const h = Math.max(0.02, l * height)
      gas.scale.y = h
      gas.position.y = LEG_H + h / 2
    }
    if (ringRef.current) ringRef.current.position.y = LEG_H + Math.max(0.02, l * height)
    if (gasMatRef.current) {
      // fuller tank reads brighter; a live inflow adds a soft shimmer of brightness
      gasMatRef.current.emissiveIntensity = 0.35 + l * 0.5 + (Math.abs(tank.netKgH) > 0.05 ? 0.25 : 0)
      gasMatRef.current.opacity = 0.5 + l * 0.3
    }
  })

  const rows: InspectRow[] = [
    { label: 'Stored', value: `${tank.kg.toFixed(1)} kg` },
    { label: 'Fill', value: `${Math.round(tank.fill * 100)} % of ${Math.round(tank.capacityKg).toLocaleString('en-US')} kg` },
    {
      label: 'Net flow',
      value: `${tank.netKgH >= 0 ? '+' : '−'}${Math.abs(tank.netKgH).toFixed(1)} kg/h`,
    },
  ]

  return (
    <group position={[at.x, 0, at.z]}>
      <UnitChrome
        id={id}
        name={name}
        accent={accent}
        plateHeight={height + LEG_H + 1.6}
        plateOffset={plateOffset}
        rows={rows}
      >
        {(active) => (
          <group>
            {/* legs */}
            {[0, 1, 2, 3].map((i) => {
              const a = (i / 4) * Math.PI * 2 + Math.PI / 4
              return (
                <mesh key={i} position={[Math.cos(a) * (radius - 0.25), LEG_H / 2, Math.sin(a) * (radius - 0.25)]} castShadow>
                  <cylinderGeometry args={[0.09, 0.11, LEG_H, 8]} />
                  <meshStandardMaterial color={SURFACE.steelDark} roughness={0.6} metalness={0.45} />
                </mesh>
              )
            })}

            {/* stored gas: the level is the state */}
            <mesh ref={gasRef} position={[0, LEG_H + height / 2, 0]}>
              <cylinderGeometry args={[innerR, innerR, 1, 28]} />
              <meshStandardMaterial
                ref={gasMatRef}
                color={accent}
                emissive={accent}
                emissiveIntensity={0.4}
                transparent
                opacity={0.55}
                toneMapped={false}
                depthWrite={false}
              />
            </mesh>
            {/* meniscus ring at the gas surface */}
            <mesh ref={ringRef} position={[0, LEG_H, 0]} rotation={[Math.PI / 2, 0, 0]}>
              <torusGeometry args={[innerR - 0.02, 0.035, 8, 40]} />
              <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={1.8} toneMapped={false} />
            </mesh>

            {/* translucent shell */}
            <mesh position={[0, LEG_H + height / 2, 0]} castShadow>
              <cylinderGeometry args={[radius, radius, height, 28, 1, true]} />
              <meshPhysicalMaterial
                color={active ? '#cfd8c6' : '#aab3a2'}
                roughness={0.18}
                metalness={0.1}
                transparent
                opacity={active ? 0.3 : 0.2}
                side={THREE.DoubleSide}
                depthWrite={false}
              />
            </mesh>
            {/* top and bottom caps (solid) */}
            <mesh position={[0, LEG_H + height, 0]} castShadow>
              <sphereGeometry args={[radius, 28, 14, 0, Math.PI * 2, 0, Math.PI / 2]} />
              <meshStandardMaterial color={SURFACE.steelLight} roughness={0.35} metalness={0.55} />
            </mesh>
            <mesh position={[0, LEG_H + 0.05, 0]}>
              <cylinderGeometry args={[radius, radius * 0.98, 0.1, 28]} />
              <meshStandardMaterial color={SURFACE.steel} roughness={0.45} metalness={0.5} />
            </mesh>
            {/* shell hoops give the vessel scale */}
            {[0.25, 0.5, 0.75].map((f) => (
              <mesh key={f} position={[0, LEG_H + f * height, 0]} rotation={[Math.PI / 2, 0, 0]}>
                <torusGeometry args={[radius + 0.02, 0.03, 8, 40]} />
                <meshStandardMaterial color={SURFACE.steel} roughness={0.5} metalness={0.5} />
              </mesh>
            ))}
            {/* top valve + feed stub */}
            <mesh position={[0, LEG_H + height + radius * 0.62, 0]}>
              <cylinderGeometry args={[0.12, 0.16, 0.35, 10]} />
              <meshStandardMaterial color={SURFACE.steelDark} roughness={0.5} metalness={0.5} />
            </mesh>
            <mesh position={[-radius - 0.2, 0.75, 0]} rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.09, 0.09, 0.7, 10]} />
              <meshStandardMaterial color={SURFACE.pipe} roughness={0.5} metalness={0.5} />
            </mesh>
          </group>
        )}
      </UnitChrome>
    </group>
  )
}
