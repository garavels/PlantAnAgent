/** A flow path between two units: the physical conduit plus energy packets.
 *
 * The conduit (a thin tube) is always visible -- it is part of the plant's
 * hardware. Packets ride the route only while flow actually occurs: their
 * speed and number track intensity, and they fade out within a beat when flow
 * stops. Packet motion freezes when the sim is paused or the user prefers
 * reduced motion; opacity still encodes the last known flow.
 */

import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

import type { Vec3 } from './layout'
import { useUnitSelection } from './selection'
import { damp, type FlowVisual } from './visuals'

const MAX_PACKETS = 9
const PACKET_R = 0.085

interface FlowLineProps {
  route: readonly Vec3[]
  flow: FlowVisual
  /** Carrier tint for the packets. */
  accent: string
  /** Conduit (static tube) appearance. */
  conduitRadius?: number
  conduitColor?: string
  packetRadius?: number
}

export function FlowLine({
  route,
  flow,
  accent,
  conduitRadius = 0.055,
  conduitColor = '#333a32',
  packetRadius = PACKET_R,
}: FlowLineProps) {
  const { flowsFrozen } = useUnitSelection()

  const curve = useMemo(
    () =>
      new THREE.CatmullRomCurve3(
        route.map((p) => new THREE.Vector3(p.x, p.y, p.z)),
        false,
        'catmullrom',
        0.08,
      ),
    [route],
  )
  const length = useMemo(() => curve.getLength(), [curve])
  const count = Math.max(3, Math.min(MAX_PACKETS, Math.floor(length / 1.6)))

  const packetsRef = useRef<THREE.InstancedMesh>(null)
  const packetsMatRef = useRef<THREE.MeshStandardMaterial>(null)
  const progress = useRef(0)
  const visibility = useRef(0)
  const dummy = useMemo(() => new THREE.Object3D(), [])

  useFrame((_, delta) => {
    // Packets fade in/out with flow rather than popping.
    visibility.current = damp(visibility.current, flow.live ? 1 : 0, delta, 8)
    if (!flowsFrozen && flow.live) {
      // 0.5 .. 2.6 m/s of packet travel, scaled by intensity.
      progress.current += ((0.5 + 2.1 * flow.intensity) * delta) / length
    }
    const material = packetsMatRef.current
    const mesh = packetsRef.current
    if (!material || !mesh) return
    material.opacity = visibility.current * (0.35 + 0.65 * flow.intensity)
    mesh.visible = visibility.current > 0.02
    if (!mesh.visible) return
    for (let i = 0; i < count; i++) {
      const t = (progress.current + i / count) % 1
      curve.getPointAt(t, dummy.position)
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
    }
    mesh.instanceMatrix.needsUpdate = true
  })

  return (
    <group>
      <mesh castShadow={false} receiveShadow={false}>
        <tubeGeometry args={[curve, 48, conduitRadius, 8, false]} />
        <meshStandardMaterial color={conduitColor} roughness={0.6} metalness={0.35} />
      </mesh>
      <instancedMesh ref={packetsRef} args={[undefined, undefined, count]} frustumCulled={false}>
        <sphereGeometry args={[packetRadius, 12, 12]} />
        <meshStandardMaterial
          ref={packetsMatRef}
          color={accent}
          emissive={accent}
          emissiveIntensity={1.6}
          transparent
          opacity={0}
          toneMapped={false}
        />
      </instancedMesh>
    </group>
  )
}
