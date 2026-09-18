/** The PV array: an instanced grid of tilted panels.
 *
 * Panel faces carry an emissive sheen that tracks current generation, so the
 * field visibly wakes at dawn and idles at night. Geometry is four
 * InstancedMeshes (two post rows, frames, faces) -- cheap at any panel count.
 */

import { useFrame } from '@react-three/fiber'
import { useLayoutEffect, useRef } from 'react'
import * as THREE from 'three'

import { SOLAR_FIELD } from './layout'
import { CARRIER, SURFACE } from './palette'
import { UnitChrome, type InspectRow } from './UnitChrome'
import { damp, type PlantVisuals } from './visuals'

const PANEL_W = 2.4
const PANEL_D = 1.5
const PANEL_H = 1.05 // centre height above ground

interface SolarFieldProps {
  solar: PlantVisuals['solar']
  peakKw: number
}

export function SolarField({ solar, peakKw }: SolarFieldProps) {
  const northPostsRef = useRef<THREE.InstancedMesh>(null)
  const southPostsRef = useRef<THREE.InstancedMesh>(null)
  const framesRef = useRef<THREE.InstancedMesh>(null)
  const facesRef = useRef<THREE.InstancedMesh>(null)
  const faceMatRef = useRef<THREE.MeshStandardMaterial>(null)
  const glow = useRef(0)

  const { rows, cols, dx, dz, origin, tilt } = SOLAR_FIELD
  const panelCount = rows * cols
  // Panels tilt toward the south (+z): north edge rides higher.
  const halfDrop = (Math.sin(tilt) * PANEL_D) / 2
  const centreX = origin.x + ((cols - 1) * dx) / 2
  const centreZ = origin.z + ((rows - 1) * dz) / 2

  // One matrix per panel, computed once: the field never moves.
  useLayoutEffect(() => {
    const dummy = new THREE.Object3D()
    const fill = (mesh: THREE.InstancedMesh | null, place: (i: number, x: number, z: number) => void) => {
      if (!mesh) return
      let i = 0
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          place(i++, origin.x + c * dx - centreX, origin.z + r * dz - centreZ)
        }
      }
      mesh.instanceMatrix.needsUpdate = true
    }
    fill(northPostsRef.current, (i, x, z) => {
      dummy.position.set(x, (PANEL_H + halfDrop) / 2, z - PANEL_D / 2 + 0.12)
      dummy.rotation.set(0, 0, 0)
      dummy.updateMatrix()
      northPostsRef.current?.setMatrixAt(i, dummy.matrix)
    })
    fill(southPostsRef.current, (i, x, z) => {
      dummy.position.set(x, (PANEL_H - halfDrop) / 2, z + PANEL_D / 2 - 0.12)
      dummy.rotation.set(0, 0, 0)
      dummy.updateMatrix()
      southPostsRef.current?.setMatrixAt(i, dummy.matrix)
    })
    fill(framesRef.current, (i, x, z) => {
      dummy.position.set(x, PANEL_H, z)
      dummy.rotation.set(tilt, 0, 0)
      dummy.updateMatrix()
      framesRef.current?.setMatrixAt(i, dummy.matrix)
    })
    fill(facesRef.current, (i, x, z) => {
      dummy.position.set(x, PANEL_H, z)
      dummy.rotation.set(tilt, 0, 0)
      dummy.translateY(0.045)
      dummy.updateMatrix()
      facesRef.current?.setMatrixAt(i, dummy.matrix)
    })
  }, [rows, cols, dx, dz, origin, tilt, centreX, centreZ, halfDrop])

  useFrame((_, delta) => {
    glow.current = damp(glow.current, solar.frac, delta, 4)
    if (faceMatRef.current) faceMatRef.current.emissiveIntensity = 0.04 + glow.current * 0.56
  })

  const rows_out: InspectRow[] = [
    { label: 'Power', value: `${Math.round(solar.kw).toLocaleString('en-US')} kW` },
    {
      label: 'Status',
      value: solar.status === 'nominal'
        ? (solar.daylight ? 'Generating' : solar.referenceKind === 'forecast_pv' ? 'No PV output' : 'Night')
        : 'Reduced irradiance',
    },
    { label: 'Array peak', value: `${Math.round(peakKw).toLocaleString('en-US')} kW` },
  ]
  if (solar.degradation > 0) {
    rows_out.push({
      label: solar.referenceKind === 'forecast_pv' ? 'Forecast PV' : 'Clear sky',
      value: `${Math.round(solar.potentialKw).toLocaleString('en-US')} kW`,
    })
  }

  return (
    <group position={[centreX, 0, centreZ]}>
      <UnitChrome id="solar" name="Solar array" accent={CARRIER.solar} status={solar.status} plateHeight={3.2} rows={rows_out}>
        {() => (
          <group>
            <instancedMesh ref={northPostsRef} args={[undefined, undefined, panelCount]} castShadow>
              <boxGeometry args={[0.12, PANEL_H + halfDrop, 0.12]} />
              <meshStandardMaterial color={SURFACE.steel} roughness={0.7} metalness={0.4} />
            </instancedMesh>
            <instancedMesh ref={southPostsRef} args={[undefined, undefined, panelCount]} castShadow>
              <boxGeometry args={[0.12, PANEL_H - halfDrop, 0.12]} />
              <meshStandardMaterial color={SURFACE.steel} roughness={0.7} metalness={0.4} />
            </instancedMesh>
            <instancedMesh ref={framesRef} args={[undefined, undefined, panelCount]} castShadow>
              <boxGeometry args={[PANEL_W, 0.07, PANEL_D]} />
              <meshStandardMaterial color={SURFACE.steelDark} roughness={0.55} metalness={0.5} />
            </instancedMesh>
            <instancedMesh ref={facesRef} args={[undefined, undefined, panelCount]}>
              <boxGeometry args={[PANEL_W - 0.12, 0.025, PANEL_D - 0.12]} />
              <meshStandardMaterial
                ref={faceMatRef}
                color="#223028"
                emissive={CARRIER.solar}
                emissiveIntensity={0.04}
                roughness={0.35}
                metalness={0.55}
              />
            </instancedMesh>
          </group>
        )}
      </UnitChrome>
    </group>
  )
}
