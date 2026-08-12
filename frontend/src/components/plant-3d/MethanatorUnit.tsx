/** The methanator: a reactor tower on a small skid, fed by H2 and CO2.
 *
 * Where the electrolyser is a row of horizontal cell blocks, the reactor is a
 * single vertical column with a glowing sight-glass strip down its side: the
 * strip's brightness tracks production. When the reactor is starved (a
 * feedstock tank is empty) the status lamp takes the alarm tint -- the one
 * place the scene borrows from the console's alarm vocabulary.
 */

import { useFrame } from '@react-three/fiber'
import { useRef } from 'react'
import * as THREE from 'three'

import { UNIT_POSITIONS } from './layout'
import { CARRIER, SURFACE } from './palette'
import { UnitChrome, type InspectRow } from './UnitChrome'
import { damp, type PlantVisuals } from './visuals'

const SKID = { w: 5.2, d: 2.8 }
const TOWER_R = 0.85
const TOWER_H = 3.4

interface MethanatorUnitProps {
  methanator: PlantVisuals['methanator']
}

export function MethanatorUnit({ methanator }: MethanatorUnitProps) {
  const stripMatRef = useRef<THREE.MeshStandardMaterial>(null)
  const windowMatRef = useRef<THREE.MeshStandardMaterial>(null)
  const lampMatRef = useRef<THREE.MeshStandardMaterial>(null)
  const load = useRef(0)

  useFrame((_, delta) => {
    load.current = damp(load.current, methanator.online ? methanator.loadFrac : 0, delta, 4)
    const l = load.current
    if (stripMatRef.current) stripMatRef.current.emissiveIntensity = 0.06 + l * 2.1
    if (windowMatRef.current) windowMatRef.current.emissiveIntensity = 0.05 + l * 1.1
    if (lampMatRef.current) {
      lampMatRef.current.emissive.set(methanator.starved ? '#d98c7a' : methanator.online ? CARRIER.ch4 : '#5a6156')
      lampMatRef.current.emissiveIntensity = methanator.online ? 0.6 + l * 1.8 : methanator.starved ? 0.9 : 0.25
    }
  })

  const status = methanator.online ? 'Running' : methanator.starved ? 'No feedstock' : 'Standby'
  const rows: InspectRow[] = [
    { label: 'Power', value: `${Math.round(methanator.powerKw).toLocaleString('en-US')} kW` },
    { label: 'Status', value: status },
    { label: 'Load', value: `${Math.round(methanator.loadFrac * 100)} % of ${Math.round(methanator.ratedKw).toLocaleString('en-US')} kW` },
    { label: 'CH₄ output', value: `${methanator.ch4kgH.toFixed(1)} kg/h` },
  ]

  return (
    <group position={[UNIT_POSITIONS.methanator.x, 0, UNIT_POSITIONS.methanator.z]}>
      <UnitChrome id="methanator" name="Methanator" accent={CARRIER.ch4} plateHeight={5.4} rows={rows}>
        {(active) => (
          <group>
            {/* skid base */}
            <mesh position={[0, 0.18, 0]} receiveShadow castShadow>
              <boxGeometry args={[SKID.w, 0.36, SKID.d]} />
              <meshStandardMaterial color={SURFACE.steelDark} roughness={0.7} metalness={0.4} />
            </mesh>

            {/* reactor tower */}
            <group position={[-0.9, 0, -0.2]}>
              <mesh position={[0, 0.36 + TOWER_H / 2, 0]} castShadow>
                <cylinderGeometry args={[TOWER_R, TOWER_R, TOWER_H, 20]} />
                <meshStandardMaterial
                  color={SURFACE.body}
                  roughness={0.55}
                  metalness={0.45}
                  emissive={CARRIER.ch4}
                  emissiveIntensity={active ? 0.05 : 0}
                />
              </mesh>
              <mesh position={[0, 0.36 + TOWER_H, 0]} castShadow>
                <sphereGeometry args={[TOWER_R, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2]} />
                <meshStandardMaterial color={SURFACE.steelLight} roughness={0.4} metalness={0.55} />
              </mesh>
              {/* sight-glass strip: the reactor's glow */}
              <mesh position={[0, 0.36 + TOWER_H / 2, TOWER_R + 0.015]}>
                <boxGeometry args={[0.22, TOWER_H - 0.7, 0.03]} />
                <meshStandardMaterial
                  ref={stripMatRef}
                  color="#1a1a16"
                  emissive={CARRIER.ch4}
                  emissiveIntensity={0.06}
                  toneMapped={false}
                />
              </mesh>
              {/* flange rings */}
              {[0.28, 0.72].map((f) => (
                <mesh key={f} position={[0, 0.36 + f * TOWER_H, 0]} rotation={[Math.PI / 2, 0, 0]}>
                  <torusGeometry args={[TOWER_R + 0.03, 0.045, 8, 28]} />
                  <meshStandardMaterial color={SURFACE.steel} roughness={0.5} metalness={0.5} />
                </mesh>
              ))}
            </group>

            {/* feed manifold: two inlet stubs (H2 east, CO2 north), one outlet */}
            <mesh position={[1.6, 0.75, 0.3]} rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.09, 0.09, 1.1, 10]} />
              <meshStandardMaterial color={SURFACE.pipe} roughness={0.5} metalness={0.5} />
            </mesh>
            <mesh position={[-0.2, 0.75, -1.2]} rotation={[Math.PI / 2, 0, 0]}>
              <cylinderGeometry args={[0.09, 0.09, 1.0, 10]} />
              <meshStandardMaterial color={SURFACE.pipe} roughness={0.5} metalness={0.5} />
            </mesh>
            <mesh position={[1.9, 0.75, -0.6]} rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.11, 0.11, 1.2, 10]} />
              <meshStandardMaterial color={SURFACE.pipe} roughness={0.5} metalness={0.5} />
            </mesh>

            {/* control cabinet with a lit window */}
            <group position={[SKID.w / 2 - 0.7, 0, 0.75]}>
              <mesh position={[0, 1.0, 0]} castShadow>
                <boxGeometry args={[1.0, 1.7, 0.95]} />
                <meshStandardMaterial color={SURFACE.bodyDark} roughness={0.6} metalness={0.35} />
              </mesh>
              <mesh position={[0, 1.24, 0.49]}>
                <boxGeometry args={[0.62, 0.44, 0.02]} />
                <meshStandardMaterial
                  ref={windowMatRef}
                  color="#14110c"
                  emissive={CARRIER.ch4}
                  emissiveIntensity={0.05}
                  toneMapped={false}
                />
              </mesh>
            </group>

            {/* status lamp on a small mast */}
            <mesh position={[-SKID.w / 2 + 0.4, 1.9, 0.9]}>
              <cylinderGeometry args={[0.035, 0.045, 3.1, 8]} />
              <meshStandardMaterial color={SURFACE.steel} roughness={0.6} metalness={0.5} />
            </mesh>
            <mesh position={[-SKID.w / 2 + 0.4, 3.5, 0.9]}>
              <sphereGeometry args={[0.11, 12, 12]} />
              <meshStandardMaterial
                ref={lampMatRef}
                color={CARRIER.ch4}
                emissive={CARRIER.ch4}
                emissiveIntensity={0.25}
                toneMapped={false}
              />
            </mesh>
          </group>
        )}
      </UnitChrome>
    </group>
  )
}
