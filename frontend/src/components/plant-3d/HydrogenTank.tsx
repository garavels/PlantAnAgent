/** The H2 tank: a vertical pressure vessel with a true rendered level.
 *
 * The shell is a faint glass so the stored gas reads as a physical column of
 * light inside it -- the fill height IS `h2_fill`, eased. A bright meniscus
 * ring marks the surface, and quarter-height gauge rings give it scale.
 */

import { useFrame } from '@react-three/fiber'
import { useRef } from 'react'
import * as THREE from 'three'

import { UNIT_POSITIONS } from './layout'
import { CARRIER, SURFACE } from './palette'
import { UnitChrome, type InspectRow } from './UnitChrome'
import { damp, type PlantVisuals } from './visuals'

const TANK_R = 1.5
const TANK_H = 5.2 // straight shell height; gas column lives inside
const LEG_H = 0.7
const INNER_R = TANK_R - 0.16

interface HydrogenTankProps {
  tank: PlantVisuals['tank']
}

export function HydrogenTank({ tank }: HydrogenTankProps) {
  const gasRef = useRef<THREE.Mesh>(null)
  const gasMatRef = useRef<THREE.MeshStandardMaterial>(null)
  const ringRef = useRef<THREE.Mesh>(null)
  const level = useRef(tank.fill)

  useFrame((_, delta) => {
    level.current = damp(level.current, tank.fill, delta, 3)
    const l = level.current
    const gas = gasRef.current
    if (gas) {
      const h = Math.max(0.02, l * TANK_H)
      gas.scale.y = h
      gas.position.y = LEG_H + h / 2
    }
    if (ringRef.current) ringRef.current.position.y = LEG_H + Math.max(0.02, l * TANK_H)
    if (gasMatRef.current) {
      // fuller tank reads brighter; a live inflow adds a soft shimmer of brightness
      gasMatRef.current.emissiveIntensity = 0.35 + l * 0.5 + (tank.kgH > 0.05 ? 0.25 : 0)
      gasMatRef.current.opacity = 0.5 + l * 0.3
    }
  })

  const rows: InspectRow[] = [
    { label: 'Stored', value: `${tank.kg.toFixed(1)} kg` },
    { label: 'Fill', value: `${Math.round(tank.fill * 100)} % of ${Math.round(tank.capacityKg).toLocaleString('en-US')} kg` },
    { label: 'Inflow', value: `+${tank.kgH.toFixed(1)} kg/h` },
  ]

  return (
    <group position={[UNIT_POSITIONS.h2Tank.x, 0, UNIT_POSITIONS.h2Tank.z]}>
      <UnitChrome
        id="tank"
        name="H₂ storage"
        accent={CARRIER.h2}
        plateHeight={TANK_H + LEG_H + 1.6}
        plateOffset={[-4.4, 0]}
        rows={rows}
      >
        {(active) => (
          <group>
            {/* legs */}
            {[0, 1, 2, 3].map((i) => {
              const a = (i / 4) * Math.PI * 2 + Math.PI / 4
              return (
                <mesh key={i} position={[Math.cos(a) * (TANK_R - 0.25), LEG_H / 2, Math.sin(a) * (TANK_R - 0.25)]} castShadow>
                  <cylinderGeometry args={[0.09, 0.11, LEG_H, 8]} />
                  <meshStandardMaterial color={SURFACE.steelDark} roughness={0.6} metalness={0.45} />
                </mesh>
              )
            })}

            {/* stored gas: the level is the state */}
            <mesh ref={gasRef} position={[0, LEG_H + TANK_H / 2, 0]}>
              <cylinderGeometry args={[INNER_R, INNER_R, 1, 28]} />
              <meshStandardMaterial
                ref={gasMatRef}
                color={CARRIER.h2}
                emissive={CARRIER.h2}
                emissiveIntensity={0.4}
                transparent
                opacity={0.55}
                toneMapped={false}
                depthWrite={false}
              />
            </mesh>
            {/* meniscus ring at the gas surface */}
            <mesh ref={ringRef} position={[0, LEG_H, 0]} rotation={[Math.PI / 2, 0, 0]}>
              <torusGeometry args={[INNER_R - 0.02, 0.035, 8, 40]} />
              <meshStandardMaterial color={CARRIER.h2} emissive={CARRIER.h2} emissiveIntensity={1.8} toneMapped={false} />
            </mesh>

            {/* translucent shell */}
            <mesh position={[0, LEG_H + TANK_H / 2, 0]} castShadow>
              <cylinderGeometry args={[TANK_R, TANK_R, TANK_H, 28, 1, true]} />
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
            <mesh position={[0, LEG_H + TANK_H, 0]} castShadow>
              <sphereGeometry args={[TANK_R, 28, 14, 0, Math.PI * 2, 0, Math.PI / 2]} />
              <meshStandardMaterial color={SURFACE.steelLight} roughness={0.35} metalness={0.55} />
            </mesh>
            <mesh position={[0, LEG_H + 0.05, 0]}>
              <cylinderGeometry args={[TANK_R, TANK_R * 0.98, 0.1, 28]} />
              <meshStandardMaterial color={SURFACE.steel} roughness={0.45} metalness={0.5} />
            </mesh>
            {/* shell hoops give the vessel scale */}
            {[0.25, 0.5, 0.75].map((f) => (
              <mesh key={f} position={[0, LEG_H + f * TANK_H, 0]} rotation={[Math.PI / 2, 0, 0]}>
                <torusGeometry args={[TANK_R + 0.02, 0.03, 8, 40]} />
                <meshStandardMaterial color={SURFACE.steel} roughness={0.5} metalness={0.5} />
              </mesh>
            ))}
            {/* top valve + feed stub */}
            <mesh position={[0, LEG_H + TANK_H + TANK_R * 0.62, 0]}>
              <cylinderGeometry args={[0.12, 0.16, 0.35, 10]} />
              <meshStandardMaterial color={SURFACE.steelDark} roughness={0.5} metalness={0.5} />
            </mesh>
            <mesh position={[-TANK_R - 0.2, 0.75, 0]} rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.09, 0.09, 0.7, 10]} />
              <meshStandardMaterial color={SURFACE.pipe} roughness={0.5} metalness={0.5} />
            </mesh>
          </group>
        )}
      </UnitChrome>
    </group>
  )
}
