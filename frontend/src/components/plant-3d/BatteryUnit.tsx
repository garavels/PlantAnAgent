/** The battery: an ISO-style container with a state-of-charge light column.
 *
 * The column on the front face fills with the stored-energy tint as SOC rises
 * -- the 3D analogue of the overview's battery gauge. The roof lamp shows the
 * direction of travel: brightening while charging, a cooler tone while
 * discharging into the stack.
 */

import { useFrame } from '@react-three/fiber'
import { useRef } from 'react'
import * as THREE from 'three'

import { UNIT_POSITIONS } from './layout'
import { CARRIER, SURFACE } from './palette'
import { UnitChrome, type InspectRow } from './UnitChrome'
import { damp, type PlantVisuals } from './visuals'

const BODY = { w: 6.2, h: 2.6, d: 2.4 }
const RIBS = 14
const COLUMN_H = 2.0

interface BatteryUnitProps {
  battery: PlantVisuals['battery']
}

export function BatteryUnit({ battery }: BatteryUnitProps) {
  const fillRef = useRef<THREE.Mesh>(null)
  const fillMatRef = useRef<THREE.MeshStandardMaterial>(null)
  const lampMatRef = useRef<THREE.MeshStandardMaterial>(null)
  const level = useRef(battery.soc)
  const rate = useRef(0)

  useFrame((_, delta) => {
    level.current = damp(level.current, battery.soc, delta, 3.5)
    rate.current = damp(rate.current, Math.min(1, Math.abs(battery.powerKw) / 2500), delta, 5)
    const fill = fillRef.current
    if (fill) {
      const h = Math.max(0.02, level.current * COLUMN_H)
      fill.scale.y = h
      fill.position.y = -COLUMN_H / 2 + h / 2
    }
    if (fillMatRef.current) fillMatRef.current.emissiveIntensity = 0.5 + rate.current * 1.6
    const lamp = lampMatRef.current
    if (lamp) {
      lamp.emissive.set(battery.discharging ? CARRIER.feed : CARRIER.stored)
      lamp.emissiveIntensity = 0.2 + rate.current * 2.2
    }
  })

  const status = battery.charging ? 'Charging' : battery.discharging ? 'Discharging' : 'Idle'
  const rows: InspectRow[] = [
    { label: 'State of charge', value: `${(battery.soc * 100).toFixed(1)} %` },
    { label: 'Power', value: `${battery.powerKw >= 0 ? '+' : ''}${Math.round(battery.powerKw).toLocaleString('en-US')} kW` },
    { label: 'Status', value: status },
    { label: 'Stored', value: `${Math.round(battery.energyKwh).toLocaleString('en-US')} / ${Math.round(battery.capacityKwh).toLocaleString('en-US')} kWh` },
  ]

  return (
    <group position={[UNIT_POSITIONS.battery.x, 0, UNIT_POSITIONS.battery.z]}>
      <UnitChrome id="battery" name="Battery" accent={CARRIER.stored} plateHeight={3.9} rows={rows}>
        {(active) => (
          <group>
            {/* plinth */}
            <mesh position={[0, 0.07, 0]} receiveShadow>
              <boxGeometry args={[BODY.w + 0.5, 0.14, BODY.d + 0.5]} />
              <meshStandardMaterial color={SURFACE.padLine} roughness={0.95} />
            </mesh>
            {/* container body */}
            <mesh position={[0, BODY.h / 2 + 0.14, 0]} castShadow>
              <boxGeometry args={[BODY.w, BODY.h, BODY.d]} />
              <meshStandardMaterial
                color={SURFACE.body}
                roughness={0.62}
                metalness={0.3}
                emissive={CARRIER.stored}
                emissiveIntensity={active ? 0.05 : 0}
              />
            </mesh>
            {/* corrugation ribs on the long faces */}
            {[-1, 1].map((side) =>
              Array.from({ length: RIBS }, (_, i) => (
                <mesh
                  key={`${side}:${i}`}
                  position={[
                    -BODY.w / 2 + ((i + 0.5) * BODY.w) / RIBS,
                    BODY.h / 2 + 0.14,
                    (side * (BODY.d + 0.06)) / 2,
                  ]}
                  castShadow
                >
                  <boxGeometry args={[BODY.w / RIBS - 0.18, BODY.h - 0.24, 0.06]} />
                  <meshStandardMaterial color={SURFACE.bodyDark} roughness={0.66} metalness={0.32} />
                </mesh>
              )),
            )}
            {/* corner posts + top rail */}
            {[-1, 1].map((sx) =>
              [-1, 1].map((sz) => (
                <mesh key={`${sx}:${sz}`} position={[(sx * (BODY.w - 0.12)) / 2, BODY.h / 2 + 0.14, (sz * (BODY.d - 0.12)) / 2]}>
                  <boxGeometry args={[0.18, BODY.h, 0.18]} />
                  <meshStandardMaterial color={SURFACE.steelDark} roughness={0.6} metalness={0.4} />
                </mesh>
              )),
            )}
            <mesh position={[0, BODY.h + 0.2, 0]}>
              <boxGeometry args={[BODY.w + 0.08, 0.12, BODY.d + 0.08]} />
              <meshStandardMaterial color={SURFACE.steelDark} roughness={0.6} metalness={0.4} />
            </mesh>

            {/* SOC light column on the south face */}
            <group position={[BODY.w / 2 - 0.55, BODY.h / 2 + 0.14, BODY.d / 2 + 0.045]}>
              <mesh>
                <boxGeometry args={[0.3, COLUMN_H + 0.14, 0.05]} />
                <meshStandardMaterial color={SURFACE.bodyDark} roughness={0.7} metalness={0.3} />
              </mesh>
              <mesh ref={fillRef} position={[0, -COLUMN_H / 2 + COLUMN_H / 4, 0.02]}>
                <boxGeometry args={[0.18, 1, 0.04]} />
                <meshStandardMaterial
                  ref={fillMatRef}
                  color={CARRIER.stored}
                  emissive={CARRIER.stored}
                  emissiveIntensity={0.5}
                  toneMapped={false}
                />
              </mesh>
              {[0.25, 0.5, 0.75].map((tick) => (
                <mesh key={tick} position={[-0.2, -COLUMN_H / 2 + tick * COLUMN_H, 0.02]}>
                  <boxGeometry args={[0.08, 0.015, 0.04]} />
                  <meshStandardMaterial color={SURFACE.steelLight} roughness={0.6} />
                </mesh>
              ))}
            </group>

            {/* roof status lamp */}
            <mesh position={[-BODY.w / 2 + 0.5, BODY.h + 0.42, BODY.d / 2 - 0.4]}>
              <sphereGeometry args={[0.11, 12, 12]} />
              <meshStandardMaterial
                ref={lampMatRef}
                color={CARRIER.stored}
                emissive={CARRIER.stored}
                emissiveIntensity={0.2}
                toneMapped={false}
              />
            </mesh>
          </group>
        )}
      </UnitChrome>
    </group>
  )
}
