/** The site: terrain, gravel pad, perimeter fence, lamps, curtail grate.
 *
 * All dressing is static except the lamp heads, which fade in as daylight
 * fades (driven by `sun.dayFactor`). The grid is deliberately faint -- scale
 * reference, not graph paper.
 */

import { Grid } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'

import { PAD, UNIT_POSITIONS } from './layout'
import { LIGHT, SURFACE } from './palette'
import { damp, type PlantVisuals } from './visuals'

const LAMP_POSITIONS: [number, number][] = [
  [-24, -15],
  [20, -15],
  [-24, 9],
  [20, 9],
]

interface GroundProps {
  sun: PlantVisuals['sun']
}

export function Ground({ sun }: GroundProps) {
  // One shared material so every lamp head brightens together.
  const lampMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#3a4038',
        emissive: LIGHT.lamp,
        emissiveIntensity: 0.05,
        toneMapped: false,
      }),
    [],
  )
  const lightRefs = useRef<THREE.PointLight[]>([])
  const night = useRef(0)

  useFrame((_, delta) => {
    night.current = damp(night.current, 1 - sun.dayFactor, delta, 2.5)
    const n = night.current
    lampMat.emissiveIntensity = 0.05 + n * 2.1
    for (const light of lightRefs.current) light.intensity = n * 44
  })

  const fence = useMemo(() => {
    const posts: [number, number][] = []
    const { w, d, origin } = PAD
    const x0 = origin.x - w / 2
    const x1 = origin.x + w / 2
    const z0 = origin.z - d / 2
    const z1 = origin.z + d / 2
    for (let x = x0; x <= x1 + 0.01; x += 3.25) {
      posts.push([x, z0], [x, z1])
    }
    for (let z = z0 + 3.25; z <= z1 - 0.01; z += 3.25) {
      posts.push([x0, z], [x1, z])
    }
    return { posts, x0, x1, z0, z1 }
  }, [])

  const postsRef = useRef<THREE.InstancedMesh>(null)
  useLayoutEffect(() => {
    const mesh = postsRef.current
    if (!mesh) return
    const dummy = new THREE.Object3D()
    fence.posts.forEach(([x, z], i) => {
      dummy.position.set(x, 1.05, z)
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
    })
    mesh.instanceMatrix.needsUpdate = true
  }, [fence])

  const rails = useMemo(() => {
    const { x0, x1, z0, z1 } = fence
    const spans: { pos: [number, number, number]; len: number; along: 'x' | 'z' }[] = []
    for (const y of [0.7, 1.35, 1.95]) {
      spans.push({ pos: [(x0 + x1) / 2, y, z0], len: x1 - x0, along: 'x' })
      spans.push({ pos: [(x0 + x1) / 2, y, z1], len: x1 - x0, along: 'x' })
      spans.push({ pos: [x0, y, (z0 + z1) / 2], len: z1 - z0, along: 'z' })
      spans.push({ pos: [x1, y, (z0 + z1) / 2], len: z1 - z0, along: 'z' })
    }
    return spans
  }, [fence])

  return (
    <group>
      {/* open terrain */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[-2, -0.02, -4]} receiveShadow>
        <planeGeometry args={[130, 130]} />
        <meshStandardMaterial color={SURFACE.terrain} roughness={1} metalness={0} />
      </mesh>
      {/* gravel pad */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[PAD.origin.x, 0, PAD.origin.z]} receiveShadow>
        <planeGeometry args={[PAD.w, PAD.d]} />
        <meshStandardMaterial color={SURFACE.pad} roughness={0.96} metalness={0} />
      </mesh>
      <Grid
        position={[PAD.origin.x, 0.015, PAD.origin.z]}
        args={[PAD.w, PAD.d]}
        cellSize={2}
        cellThickness={0.6}
        cellColor="#242b22"
        sectionSize={10}
        sectionThickness={1}
        sectionColor="#2e362c"
        fadeDistance={85}
        fadeStrength={2.5}
      />

      {/* perimeter fence */}
      <instancedMesh ref={postsRef} args={[undefined, undefined, fence.posts.length]} castShadow>
        <cylinderGeometry args={[0.045, 0.055, 2.1, 6]} />
        <meshStandardMaterial color={SURFACE.fence} roughness={0.6} metalness={0.5} />
      </instancedMesh>
      {rails.map((rail, i) => (
        <mesh
          key={i}
          position={rail.pos}
          rotation={rail.along === 'x' ? [0, 0, Math.PI / 2] : [Math.PI / 2, 0, 0]}
        >
          <cylinderGeometry args={[0.025, 0.025, rail.len, 6]} />
          <meshStandardMaterial color={SURFACE.fence} roughness={0.6} metalness={0.5} />
        </mesh>
      ))}

      {/* site lamps: emissive heads + real lights that own the night scene */}
      {LAMP_POSITIONS.map(([x, z], i) => (
        <group key={i} position={[x, 0, z]}>
          <mesh position={[0, 2.6, 0]} castShadow>
            <cylinderGeometry args={[0.06, 0.09, 5.2, 8]} />
            <meshStandardMaterial color={SURFACE.lampPole} roughness={0.6} metalness={0.5} />
          </mesh>
          <mesh position={[0, 5.2, 0.45]} rotation={[0.5, 0, 0]} material={lampMat}>
            <boxGeometry args={[0.5, 0.12, 0.3]} />
          </mesh>
          <pointLight
            ref={(el) => {
              if (el) lightRefs.current[i] = el
            }}
            position={[0, 4.9, 0.6]}
            color={LIGHT.lamp}
            intensity={0}
            distance={26}
            decay={2}
          />
        </group>
      ))}

      {/* curtail grate: where refused energy goes */}
      <group position={[UNIT_POSITIONS.curtail.x, 0, UNIT_POSITIONS.curtail.z]}>
        <mesh position={[0, 0.03, 0]} receiveShadow>
          <boxGeometry args={[1.4, 0.06, 1.4]} />
          <meshStandardMaterial color="#141813" roughness={0.9} metalness={0.2} />
        </mesh>
        {[-0.45, -0.15, 0.15, 0.45].map((o) => (
          <mesh key={o} position={[o, 0.065, 0]}>
            <boxGeometry args={[0.08, 0.02, 1.2]} />
            <meshStandardMaterial color={SURFACE.steelDark} roughness={0.7} metalness={0.4} />
          </mesh>
        ))}
      </group>
    </group>
  )
}
