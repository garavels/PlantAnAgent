import { Html } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'

import { UNIT_POSITIONS } from './layout'

export type AgentOrbTarget = 'hub' | 'electrolyser' | 'battery' | 'dac' | 'methanator'

export interface AgentOrbProps {
  target: AgentOrbTarget
  playing: boolean
  reducedMotion: boolean
  /** Short label; the full explanation belongs in the replay controls. */
  label: string
}

const CORE = '#eef7f5'
const HALO = '#aac6cb'
// Keep the guide above the equipment inspection labels as well as the units.
const HEIGHT = 7.5
const TRAVEL_SECONDS = 0.6
const NO_RAYCAST = () => null

/** Elliptical footprints sit just beyond each machine's plinth. */
const FOOTPRINTS: Record<AgentOrbTarget, [number, number]> = {
  hub: [1.9, 1.35],
  electrolyser: [4.4, 2.15],
  battery: [3.9, 1.85],
  dac: [3.0, 1.95],
  methanator: [3.3, 2.0],
}

/** A replay guide, separate from the smaller packets that carry energy.
 * Movement follows replay targets, with no autonomous orbit or idle motion.
 */
export function AgentOrb({ target, playing, reducedMotion, label }: AgentOrbProps) {
  const orbRef = useRef<THREE.Group>(null)
  const linkRef = useRef<THREE.BufferAttribute>(null)
  const motion = useRef({
    target: null as AgentOrbTarget | null,
    from: new THREE.Vector3(),
    to: new THREE.Vector3(),
    elapsed: TRAVEL_SECONDS,
  })
  const linkPositions = useMemo(() => new Float32Array(6), [])
  const at = UNIT_POSITIONS[target]
  const footprint = FOOTPRINTS[target]

  useLayoutEffect(() => {
    const orb = orbRef.current
    if (!orb) return

    const travel = motion.current
    const targetChanged = travel.target !== target
    const firstTarget = travel.target === null

    if (targetChanged) {
      const next = UNIT_POSITIONS[target]
      travel.from.copy(orb.position)
      travel.to.set(next.x, next.y + HEIGHT, next.z)
      travel.elapsed = 0
      travel.target = target
    }

    // Pausing alone preserves a flight's exact position. Manual navigation
    // while paused, and reduced-motion navigation, show the new unit at once.
    if (firstTarget || reducedMotion || (targetChanged && !playing)) {
      orb.position.copy(travel.to)
      travel.elapsed = TRAVEL_SECONDS
    }
  }, [target, playing, reducedMotion])

  useFrame((_, delta) => {
    const orb = orbRef.current
    if (!orb) return

    const travel = motion.current
    if (playing && !reducedMotion && travel.elapsed < TRAVEL_SECONDS) {
      travel.elapsed = Math.min(TRAVEL_SECONDS, travel.elapsed + Math.min(delta, 0.1))
      const t = travel.elapsed / TRAVEL_SECONDS
      const eased = t * t * (3 - 2 * t)
      orb.position.lerpVectors(travel.from, travel.to, eased)
    }

    // The fine tether points at the actual destination even during travel.
    // It uses normal depth testing so it disappears inside the equipment.
    const link = linkRef.current
    if (link) {
      link.setXYZ(0, orb.position.x, orb.position.y - 0.62, orb.position.z)
      link.setXYZ(1, at.x, at.y + 0.06, at.z)
      link.needsUpdate = true
    }
  })

  return (
    <group>
      <mesh
        position={[at.x, at.y + 0.04, at.z]}
        rotation={[-Math.PI / 2, 0, 0]}
        scale={[footprint[0], footprint[1], 1]}
        raycast={NO_RAYCAST}
      >
        <ringGeometry args={[0.985, 1, 80]} />
        <meshBasicMaterial color={HALO} transparent opacity={0.4} depthWrite={false} toneMapped={false} />
      </mesh>

      <lineSegments raycast={NO_RAYCAST} frustumCulled={false}>
        <bufferGeometry>
          <bufferAttribute ref={linkRef} attach="attributes-position" args={[linkPositions, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color={HALO} transparent opacity={0.26} depthWrite={false} toneMapped={false} />
      </lineSegments>

      <group ref={orbRef}>
        <mesh raycast={NO_RAYCAST}>
          <sphereGeometry args={[0.85, 24, 20]} />
          <meshBasicMaterial color={HALO} transparent opacity={0.055} depthWrite={false} toneMapped={false} />
        </mesh>
        <mesh raycast={NO_RAYCAST}>
          <sphereGeometry args={[0.63, 24, 20]} />
          <meshBasicMaterial color={HALO} transparent opacity={0.14} depthWrite={false} toneMapped={false} />
        </mesh>
        <mesh raycast={NO_RAYCAST}>
          <sphereGeometry args={[0.4, 24, 20]} />
          <meshStandardMaterial
            color={CORE}
            emissive={CORE}
            emissiveIntensity={1.4}
            roughness={0.2}
            metalness={0.05}
            toneMapped={false}
          />
        </mesh>
        {label && <Html
          position={[0, 1.2, 0]}
          center
          zIndexRange={[15, 0]}
          style={{ pointerEvents: 'none', userSelect: 'none' }}
        >
          <span
            aria-hidden="true"
            style={{
              display: 'block',
              maxWidth: 160,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              padding: '3px 7px',
              borderRadius: 4,
              background: 'rgba(16, 21, 18, 0.8)',
              color: CORE,
              fontFamily: '"Instrument Sans", sans-serif',
              fontSize: 10,
              fontWeight: 600,
              letterSpacing: '0.03em',
              lineHeight: 1.4,
            }}
          >
            Agent · {label}
          </span>
        </Html>}
      </group>
    </group>
  )
}
