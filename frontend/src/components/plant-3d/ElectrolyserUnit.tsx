/** The electrolyser: a process skid with cell stacks and separator vessels.
 *
 * When the stack runs, the seams between the cell blocks glow with the stack
 * feed tint, proportional to load; below minimum load the skid goes dark and
 * cold. This is the plant's heartbeat, so it carries the strongest glow.
 */

import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

import { UNIT_POSITIONS } from './layout'
import { CARRIER, SURFACE, statusTint } from './palette'
import { UnitChrome, type InspectRow } from './UnitChrome'
import { damp, type PlantVisuals } from './visuals'

const SKID = { w: 7.2, d: 3.0 }
const STACKS = 3

interface ElectrolyserUnitProps {
  electrolyser: PlantVisuals['electrolyser']
  minLoadFrac: number
}

export function ElectrolyserUnit({ electrolyser, minLoadFrac }: ElectrolyserUnitProps) {
  // One shared seam material: every stack face brightens together.
  const seamMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#202724',
        emissive: CARRIER.feed,
        emissiveIntensity: 0.06,
        toneMapped: false,
      }),
    [],
  )
  const windowMatRef = useRef<THREE.MeshStandardMaterial>(null)
  const lampMatRef = useRef<THREE.MeshStandardMaterial>(null)
  const load = useRef(0)
  const tint = statusTint(electrolyser.status)

  useFrame((_, delta) => {
    load.current = damp(load.current, electrolyser.online ? electrolyser.loadFrac : 0, delta, 4)
    const l = load.current
    seamMat.emissiveIntensity = 0.06 + l * 2.3
    if (windowMatRef.current) windowMatRef.current.emissiveIntensity = 0.05 + l * 1.1
    if (lampMatRef.current) {
      lampMatRef.current.emissive.set(tint ?? (electrolyser.online ? CARRIER.feed : '#5a6156'))
      lampMatRef.current.emissiveIntensity = tint ? 1.4 : electrolyser.online ? 0.6 + l * 1.8 : 0.25
    }
  })

  const status =
    electrolyser.status === 'offline' ? 'Offline' : electrolyser.online ? 'Running' : 'Standby'
  const rows: InspectRow[] = [
    { label: 'Power', value: `${Math.round(electrolyser.powerKw).toLocaleString('en-US')} kW` },
    { label: 'Status', value: status },
    { label: 'Load', value: `${Math.round(electrolyser.loadFrac * 100)} % of ${Math.round(electrolyser.ratedKw).toLocaleString('en-US')} kW` },
    { label: 'H₂ output', value: `${electrolyser.h2kgH.toFixed(1)} kg/h` },
    { label: 'Efficiency', value: `${Math.round(electrolyser.efficiency * 100)} % of nameplate` },
    { label: 'Stack temp', value: `${electrolyser.temperatureC.toFixed(0)} °C` },
    { label: 'Min load', value: `${Math.round(minLoadFrac * 100)} %` },
  ]

  const stackW = 1.5
  const gap = 0.32
  const block0 = -((STACKS - 1) * (stackW + gap)) / 2

  return (
    <group position={[UNIT_POSITIONS.electrolyser.x, 0, UNIT_POSITIONS.electrolyser.z]}>
      <UnitChrome id="electrolyser" name="Electrolyser" accent={CARRIER.feed} status={electrolyser.status} plateHeight={4.6} rows={rows}>
        {(active) => (
          <group>
            {/* skid base */}
            <mesh position={[0, 0.18, 0]} receiveShadow castShadow>
              <boxGeometry args={[SKID.w, 0.36, SKID.d]} />
              <meshStandardMaterial color={SURFACE.steelDark} roughness={0.7} metalness={0.4} />
            </mesh>

            {/* cell stack blocks with glowing seams */}
            {Array.from({ length: STACKS }, (_, i) => (
              <group key={i} position={[block0 + i * (stackW + gap) - 0.4, 0, -0.35]}>
                <mesh position={[0, 1.35, 0]} castShadow>
                  <boxGeometry args={[stackW, 2.0, 1.7]} />
                  <meshStandardMaterial
                    color={SURFACE.body}
                    roughness={0.55}
                    metalness={0.4}
                    emissive={CARRIER.feed}
                    emissiveIntensity={active ? 0.05 : 0}
                  />
                </mesh>
                <mesh position={[0, 1.35, 0.87]} material={seamMat}>
                  <boxGeometry args={[stackW - 0.3, 1.5, 0.03]} />
                </mesh>
                {/* top manifold */}
                <mesh position={[0, 2.5, 0]}>
                  <cylinderGeometry args={[0.16, 0.16, stackW - 0.2, 10]} />
                  <meshStandardMaterial color={SURFACE.steel} roughness={0.45} metalness={0.6} />
                </mesh>
              </group>
            ))}

            {/* separator vessels behind the stacks */}
            {[1.9, 2.9].map((x, i) => {
              const vesselH = 2.3 - i * 0.4
              return (
                <group key={x} position={[x, 0, -0.9]}>
                  <mesh position={[0, 0.36 + vesselH / 2, 0]} castShadow>
                    <cylinderGeometry args={[0.42 - i * 0.08, 0.42 - i * 0.08, vesselH, 16]} />
                    <meshStandardMaterial color={SURFACE.steelLight} roughness={0.4} metalness={0.55} />
                  </mesh>
                  <mesh position={[0, 0.36 + vesselH, 0]}>
                    <sphereGeometry args={[0.42 - i * 0.08, 16, 10, 0, Math.PI * 2, 0, Math.PI / 2]} />
                    <meshStandardMaterial color={SURFACE.steelLight} roughness={0.4} metalness={0.55} />
                  </mesh>
                </group>
              )
            })}

            {/* control cabinet with a lit window */}
            <group position={[SKID.w / 2 - 0.75, 0, 0.75]}>
              <mesh position={[0, 1.05, 0]} castShadow>
                <boxGeometry args={[1.1, 1.8, 1.0]} />
                <meshStandardMaterial color={SURFACE.bodyDark} roughness={0.6} metalness={0.35} />
              </mesh>
              <mesh position={[0, 1.3, 0.51]}>
                <boxGeometry args={[0.7, 0.5, 0.02]} />
                <meshStandardMaterial
                  ref={windowMatRef}
                  color="#101511"
                  emissive={CARRIER.feed}
                  emissiveIntensity={0.05}
                  toneMapped={false}
                />
              </mesh>
            </group>

            {/* status lamp on a small mast rising from the skid base */}
            <mesh position={[-SKID.w / 2 + 0.5, 2.16, 0.9]}>
              <cylinderGeometry args={[0.035, 0.045, 3.6, 8]} />
              <meshStandardMaterial color={SURFACE.steel} roughness={0.6} metalness={0.5} />
            </mesh>
            <mesh position={[-SKID.w / 2 + 0.5, 3.95, 0.9]}>
              <sphereGeometry args={[0.11, 12, 12]} />
              <meshStandardMaterial
                ref={lampMatRef}
                color={CARRIER.feed}
                emissive={CARRIER.feed}
                emissiveIntensity={0.25}
                toneMapped={false}
              />
            </mesh>

            {/* H2 outlet stub toward the tank */}
            <mesh position={[SKID.w / 2 + 0.3, 0.75, 1.0]} rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.09, 0.09, 0.9, 10]} />
              <meshStandardMaterial color={SURFACE.pipe} roughness={0.5} metalness={0.5} />
            </mesh>
          </group>
        )}
      </UnitChrome>
    </group>
  )
}
