/** The DAC: an air-contactor row with two big intake fans on the roof.
 *
 * While the unit captures, the fan rotors turn (speed tracks load) and the
 * intake grilles glow faintly with the CO2 carrier tint; off, the fans rest
 * and the box goes quiet. Deliberately boxy -- it reads as HVAC hardware, not
 * as process plant, which sets it apart from the reactor and the stacks.
 */

import { useFrame } from '@react-three/fiber'
import { useRef } from 'react'
import * as THREE from 'three'

import { UNIT_POSITIONS } from './layout'
import { CARRIER, SURFACE, statusTint } from './palette'
import { useUnitSelection } from './selection'
import { UnitChrome, type InspectRow } from './UnitChrome'
import { damp, type PlantVisuals } from './visuals'

const BODY = { w: 4.4, h: 2.2, d: 2.6 }
const FAN_R = 0.78
const FANS = [-1.1, 1.1]

interface DacUnitProps {
  dac: PlantVisuals['dac']
}

export function DacUnit({ dac }: DacUnitProps) {
  const { flowsFrozen } = useUnitSelection()
  const rotorsRef = useRef<(THREE.Group | null)[]>([])
  const grilleMatRef = useRef<THREE.MeshStandardMaterial>(null)
  const lampMatRef = useRef<THREE.MeshStandardMaterial>(null)
  const spin = useRef(0)
  const tint = statusTint(dac.status)

  useFrame((_, delta) => {
    spin.current = damp(spin.current, dac.online ? 0.35 + dac.loadFrac * 0.65 : 0, delta, 3)
    const s = spin.current
    if (!flowsFrozen && s > 0.001) {
      for (const rotor of rotorsRef.current) {
        if (rotor) rotor.rotation.y -= s * delta * 9
      }
    }
    if (grilleMatRef.current) grilleMatRef.current.emissiveIntensity = 0.04 + s * 0.9
    if (lampMatRef.current) {
      lampMatRef.current.emissive.set(tint ?? (dac.online ? CARRIER.co2 : '#5a6156'))
      lampMatRef.current.emissiveIntensity = tint ? 1.4 : dac.online ? 0.5 + s * 1.6 : 0.25
    }
  })

  const rows: InspectRow[] = [
    { label: 'Power', value: `${Math.round(dac.powerKw).toLocaleString('en-US')} kW` },
    { label: 'Status', value: dac.status === 'offline' ? 'Offline' : dac.online ? 'Capturing' : 'Standby' },
    { label: 'Load', value: `${Math.round(dac.loadFrac * 100)} % of ${Math.round(dac.ratedKw).toLocaleString('en-US')} kW` },
    { label: 'CO₂ captured', value: `${dac.co2kgH.toFixed(1)} kg/h` },
  ]

  return (
    <group position={[UNIT_POSITIONS.dac.x, 0, UNIT_POSITIONS.dac.z]}>
      <UnitChrome id="dac" name="DAC" accent={CARRIER.co2} status={dac.status} plateHeight={4.4} rows={rows}>
        {(active) => (
          <group>
            {/* plinth */}
            <mesh position={[0, 0.07, 0]} receiveShadow>
              <boxGeometry args={[BODY.w + 0.5, 0.14, BODY.d + 0.5]} />
              <meshStandardMaterial color={SURFACE.padLine} roughness={0.95} />
            </mesh>
            {/* contactor body */}
            <mesh position={[0, BODY.h / 2 + 0.14, 0]} castShadow>
              <boxGeometry args={[BODY.w, BODY.h, BODY.d]} />
              <meshStandardMaterial
                color={SURFACE.body}
                roughness={0.62}
                metalness={0.3}
                emissive={CARRIER.co2}
                emissiveIntensity={active ? 0.05 : 0}
              />
            </mesh>
            {/* intake grilles on the long faces: slats with a carrier glow */}
            {[-1, 1].map((side) => (
              <group key={side} position={[0, BODY.h / 2 + 0.14, (side * (BODY.d + 0.05)) / 2]}>
                <mesh>
                  <boxGeometry args={[BODY.w - 0.7, BODY.h - 0.7, 0.05]} />
                  <meshStandardMaterial
                    ref={side === 1 ? grilleMatRef : undefined}
                    color="#151a15"
                    emissive={CARRIER.co2}
                    emissiveIntensity={0.04}
                    toneMapped={false}
                  />
                </mesh>
                {Array.from({ length: 6 }, (_, i) => (
                  <mesh key={i} position={[0, -BODY.h / 2 + 0.5 + i * 0.26, side * 0.035]}>
                    <boxGeometry args={[BODY.w - 0.6, 0.05, 0.04]} />
                    <meshStandardMaterial color={SURFACE.steelDark} roughness={0.7} metalness={0.4} />
                  </mesh>
                ))}
              </group>
            ))}

            {/* roof fans: ring housing + spinning rotor */}
            {FANS.map((x, i) => (
              <group key={x} position={[x, BODY.h + 0.14, 0]}>
                <mesh position={[0, 0.1, 0]} castShadow>
                  <cylinderGeometry args={[FAN_R + 0.1, FAN_R + 0.12, 0.34, 24]} />
                  <meshStandardMaterial color={SURFACE.steel} roughness={0.5} metalness={0.5} />
                </mesh>
                <mesh position={[0, 0.28, 0]} rotation={[Math.PI / 2, 0, 0]}>
                  <torusGeometry args={[FAN_R, 0.045, 8, 32]} />
                  <meshStandardMaterial color={SURFACE.steelLight} roughness={0.45} metalness={0.55} />
                </mesh>
                <group ref={(el) => (rotorsRef.current[i] = el)} position={[0, 0.24, 0]}>
                  {[0, 1, 2].map((blade) => (
                    <mesh key={blade} rotation={[0, (blade / 3) * Math.PI * 2, 0]}>
                      <boxGeometry args={[FAN_R * 1.7, 0.03, 0.16]} />
                      <meshStandardMaterial color={SURFACE.steelDark} roughness={0.55} metalness={0.5} />
                    </mesh>
                  ))}
                  <mesh>
                    <cylinderGeometry args={[0.14, 0.14, 0.12, 12]} />
                    <meshStandardMaterial color={SURFACE.steelLight} roughness={0.4} metalness={0.6} />
                  </mesh>
                </group>
              </group>
            ))}

            {/* status lamp */}
            <mesh position={[BODY.w / 2 - 0.4, BODY.h + 0.42, BODY.d / 2 - 0.4]}>
              <sphereGeometry args={[0.11, 12, 12]} />
              <meshStandardMaterial
                ref={lampMatRef}
                color={CARRIER.co2}
                emissive={CARRIER.co2}
                emissiveIntensity={0.25}
                toneMapped={false}
              />
            </mesh>

            {/* CO2 outlet stub toward the tank */}
            <mesh position={[BODY.w / 2 + 0.25, 0.75, 0]} rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.09, 0.09, 0.8, 10]} />
              <meshStandardMaterial color={SURFACE.pipe} roughness={0.5} metalness={0.5} />
            </mesh>
          </group>
        )}
      </UnitChrome>
    </group>
  )
}
