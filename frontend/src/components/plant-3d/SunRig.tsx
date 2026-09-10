/** Sky, sun and site lighting, driven by the simulator's own solar curve.
 *
 * `visuals.sun` is derived from `time_of_day_h` against the config's
 * sunrise/sunset, so the twin's light can never disagree with the plant's
 * generation: panels glow exactly when the light says they should. Night
 * falls back to a low cool fill plus the site lamps (wired up in Ground).
 */

import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

import { LIGHT, SKY } from './palette'
import { damp, type PlantVisuals } from './visuals'

interface SunRigProps {
  sun: PlantVisuals['sun']
  attenuation?: number
}

const SUN_DIST = 120

export function SunRig({ sun, attenuation = 0 }: SunRigProps) {
  const lightRef = useRef<THREE.DirectionalLight>(null)
  const hemiRef = useRef<THREE.HemisphereLight>(null)
  const discRef = useRef<THREE.Mesh>(null)
  const day = useRef(sun.dayFactor)

  const colors = useMemo(
    () => ({
      skyNight: new THREE.Color(SKY.night),
      skyDay: new THREE.Color(SKY.day),
      sunLow: new THREE.Color(LIGHT.sunLow),
      sunNoon: new THREE.Color(LIGHT.sunNoon),
      moon: new THREE.Color(LIGHT.moon),
      scratch: new THREE.Color(),
    }),
    [],
  )

  useFrame(({ scene, gl }, delta) => {
    day.current = damp(day.current, sun.dayFactor, delta, 2.5)
    const d = day.current
    const elev = sun.elevation

    // nights stay legible: a small exposure lift after dusk
    gl.toneMappingExposure = damp(gl.toneMappingExposure, 1.18 - d * 0.2, delta, 2.5)

    // sky + fog follow daylight
    colors.scratch.lerpColors(colors.skyNight, colors.skyDay, d)
    if (scene.background instanceof THREE.Color) scene.background.copy(colors.scratch)
    if (scene.fog instanceof THREE.Fog) {
      scene.fog.color.copy(colors.scratch)
      scene.fog.near = 42 + d * 20
      scene.fog.far = 100 + d * 45
    }

    // the sun rides the same half-sine the simulator generates from
    const azimuth = Math.PI * (0.12 + 0.76 * sun.phase)
    const sx = Math.cos(azimuth) * SUN_DIST
    const sy = Math.max(4, Math.sin(Math.PI * sun.phase) * SUN_DIST * 0.62)
    const sz = -SUN_DIST * 0.45

    const light = lightRef.current
    if (light) {
      if (elev > 0.001) {
        light.position.set(sx * 0.5, Math.max(8, sy * 0.5), sz * 0.5)
        light.color.lerpColors(colors.sunLow, colors.sunNoon, elev)
        light.intensity = 0.25 + elev * 2.6 * (1 - attenuation * .85)
      } else {
        // moonlight: fixed, cool, faint
        light.position.set(-30, 42, -18)
        light.color.copy(colors.moon)
        light.intensity = 0.45
      }
    }
    if (hemiRef.current) hemiRef.current.intensity = 0.36 + d * 0.49
    if (discRef.current) {
      discRef.current.position.set(sx, sy, sz)
      discRef.current.visible = elev > 0.001
      const mat = discRef.current.material as THREE.MeshBasicMaterial
      mat.color.lerpColors(colors.sunLow, colors.sunNoon, elev)
    }
  })

  return (
    <>
      <directionalLight
        ref={lightRef}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-36}
        shadow-camera-right={36}
        shadow-camera-top={30}
        shadow-camera-bottom={-30}
        shadow-camera-far={160}
        shadow-bias={-0.0004}
      />
      <hemisphereLight ref={hemiRef} color="#c8d2bc" groundColor="#1a1f18" intensity={0.3} />
      {/* sun disc: above the bloom threshold at midday, warm smear at the edges */}
      <mesh ref={discRef}>
        <sphereGeometry args={[5, 20, 20]} />
        <meshBasicMaterial color={LIGHT.sunNoon} toneMapped={false} fog={false} />
      </mesh>
    </>
  )
}
