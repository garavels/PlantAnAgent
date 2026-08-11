/** The power hub: the DC distribution point between solar and the loads.
 *
 * A low switchgear cabinet with a small busbar mast. The status lamp breathes
 * with total throughput; curtailment leaves the site through a ground grate
 * behind the cabinet (routed in FlowLine, driven by `flows.curtail`).
 */

import { useFrame } from '@react-three/fiber'
import { useRef } from 'react'
import * as THREE from 'three'

import { UNIT_POSITIONS } from './layout'
import { CARRIER, SURFACE } from './palette'
import { UnitChrome, type InspectRow } from './UnitChrome'
import { damp, type PlantVisuals } from './visuals'

interface PowerHubProps {
  flows: PlantVisuals['flows']
}

export function PowerHub({ flows }: PowerHubProps) {
  const lampRef = useRef<THREE.MeshStandardMaterial>(null)
  const stripRef = useRef<THREE.MeshStandardMaterial>(null)
  const activity = useRef(0)

  const throughKw = flows.solarToHub.value

  useFrame((_, delta) => {
    activity.current = damp(activity.current, flows.solarToHub.intensity, delta, 5)
    const a = activity.current
    if (lampRef.current) lampRef.current.emissiveIntensity = 0.25 + a * 1.8
    if (stripRef.current) stripRef.current.emissiveIntensity = 0.15 + a * 1.4
  })

  const rows: InspectRow[] = [
    { label: 'Solar in', value: `${Math.round(throughKw).toLocaleString('en-US')} kW` },
    { label: 'To battery', value: `${Math.round(flows.hubToBattery.value).toLocaleString('en-US')} kW` },
    { label: 'To stack', value: `${Math.round(flows.hubToStack.value).toLocaleString('en-US')} kW` },
    { label: 'Curtailed', value: `${Math.round(flows.curtail.value).toLocaleString('en-US')} kW` },
  ]

  return (
    <group position={[UNIT_POSITIONS.hub.x, 0, UNIT_POSITIONS.hub.z]}>
      <UnitChrome id="hub" name="Power hub" accent={CARRIER.solar} plateHeight={3.1} rows={rows}>
        {(active) => (
          <group>
            {/* plinth */}
            <mesh position={[0, 0.08, 0]} receiveShadow>
              <boxGeometry args={[2.6, 0.16, 1.8]} />
              <meshStandardMaterial color={SURFACE.padLine} roughness={0.95} />
            </mesh>
            {/* switchgear cabinet */}
            <mesh position={[0, 0.75, 0]} castShadow>
              <boxGeometry args={[2.2, 1.35, 1.3]} />
              <meshStandardMaterial
                color={SURFACE.body}
                roughness={0.6}
                metalness={0.35}
                emissive={CARRIER.solar}
                emissiveIntensity={active ? 0.06 : 0}
              />
            </mesh>
            {/* cabinet door seam + activity strip */}
            <mesh position={[0, 0.78, 0.66]}>
              <boxGeometry args={[1.9, 0.05, 0.02]} />
              <meshStandardMaterial
                ref={stripRef}
                color={CARRIER.solar}
                emissive={CARRIER.solar}
                emissiveIntensity={0.15}
                toneMapped={false}
              />
            </mesh>
            {/* busbar mast with three insulators */}
            <mesh position={[-0.7, 1.9, -0.3]} castShadow>
              <cylinderGeometry args={[0.05, 0.06, 1.6, 8]} />
              <meshStandardMaterial color={SURFACE.steel} roughness={0.6} metalness={0.5} />
            </mesh>
            {[0, 1, 2].map((i) => (
              <mesh key={i} position={[-0.7 + i * 0.35, 2.72, -0.3]}>
                <cylinderGeometry args={[0.07, 0.09, 0.22, 8]} />
                <meshStandardMaterial color={SURFACE.steelLight} roughness={0.4} metalness={0.3} />
              </mesh>
            ))}
            <mesh position={[-0.35, 2.85, -0.3]} rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.025, 0.025, 1.1, 6]} />
              <meshStandardMaterial color={SURFACE.steelLight} roughness={0.4} metalness={0.6} />
            </mesh>
            {/* status lamp */}
            <mesh position={[0.95, 1.55, 0.45]}>
              <sphereGeometry args={[0.09, 12, 12]} />
              <meshStandardMaterial
                ref={lampRef}
                color={CARRIER.solar}
                emissive={CARRIER.solar}
                emissiveIntensity={0.25}
                toneMapped={false}
              />
            </mesh>
            <mesh position={[0.95, 1.28, 0.45]}>
              <cylinderGeometry args={[0.03, 0.03, 0.5, 6]} />
              <meshStandardMaterial color={SURFACE.steel} roughness={0.6} metalness={0.5} />
            </mesh>
          </group>
        )}
      </UnitChrome>
    </group>
  )
}
