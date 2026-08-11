/** The twin itself: one Canvas composing site, units and flow network.
 *
 * Everything in here is a pure function of `PlantVisuals` -- the scene reads
 * the adapter's output and renders it. Selection state (which unit is pinned)
 * is the only state the scene owns.
 */

import { OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { Bloom, EffectComposer, Vignette } from '@react-three/postprocessing'
import { useEffect, useMemo, useState } from 'react'
import * as THREE from 'three'

import type { PlantConfig, PlantState } from '../../types/plant'
import { BatteryUnit } from './BatteryUnit'
import { ElectrolyserUnit } from './ElectrolyserUnit'
import { FlowLine } from './FlowLine'
import { Ground } from './Ground'
import { HydrogenTank } from './HydrogenTank'
import { CAMERA, FLOW_ROUTES } from './layout'
import { useReducedMotion } from './motion'
import { CARRIER, SKY, SURFACE } from './palette'
import { PowerHub } from './PowerHub'
import { SolarField } from './SolarField'
import { SunRig } from './SunRig'
import { UnitSelectionContext, type UnitId } from './selection'
import { deriveVisuals } from './visuals'

interface PlantSceneProps {
  state: PlantState
  config: PlantConfig | null
  /** When the sim clock is stopped, flow packets freeze with it. */
  playing: boolean
}

export function PlantScene({ state, config, playing }: PlantSceneProps) {
  const [selected, setSelected] = useState<UnitId | null>(null)
  const reducedMotion = useReducedMotion()
  const visuals = useMemo(() => deriveVisuals(state, config), [state, config])

  // Escape drops the pinned unit.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setSelected(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const peakKw = config?.solar.peak_kw ?? 5000
  const minLoadFrac = config?.electrolyser.min_load_frac ?? 0.15

  return (
    <UnitSelectionContext.Provider
      value={{ selected, select: setSelected, flowsFrozen: !playing || reducedMotion }}
    >
      <Canvas
        shadows
        dpr={[1, 2]}
        camera={{ position: [CAMERA.position.x, CAMERA.position.y, CAMERA.position.z], fov: CAMERA.fov }}
        gl={{ antialias: false }}
        onPointerMissed={() => setSelected(null)}
        onCreated={({ scene }) => {
          scene.background = new THREE.Color(SKY.night)
          scene.fog = new THREE.Fog(SKY.night, 55, 150)
        }}
      >
        <SunRig sun={visuals.sun} />
        <Ground sun={visuals.sun} />

        <SolarField solar={visuals.solar} peakKw={peakKw} />
        <PowerHub flows={visuals.flows} />
        <BatteryUnit battery={visuals.battery} />
        <ElectrolyserUnit electrolyser={visuals.electrolyser} minLoadFrac={minLoadFrac} />
        <HydrogenTank tank={visuals.tank} />

        {/* the flow network: power in sand/sage/mist, gas in chalk */}
        <FlowLine route={FLOW_ROUTES.solarToHub} flow={visuals.flows.solarToHub} accent={CARRIER.solar} />
        <FlowLine route={FLOW_ROUTES.hubToBattery} flow={visuals.flows.hubToBattery} accent={CARRIER.stored} />
        <FlowLine route={FLOW_ROUTES.hubToStack} flow={visuals.flows.hubToStack} accent={CARRIER.feed} />
        <FlowLine route={FLOW_ROUTES.batteryToStack} flow={visuals.flows.batteryToStack} accent={CARRIER.stored} />
        <FlowLine
          route={FLOW_ROUTES.stackToTank}
          flow={visuals.flows.stackToTank}
          accent={CARRIER.h2}
          conduitRadius={0.09}
          conduitColor={SURFACE.pipe}
        />
        {/* curtailment is a short physical drop from hub to grate */}
        <FlowLine
          route={FLOW_ROUTES.curtail}
          flow={visuals.flows.curtail}
          accent={CARRIER.waste}
          conduitRadius={0.035}
          conduitColor="#262c25"
          packetRadius={0.055}
        />

        <OrbitControls
          target={[CAMERA.target.x, CAMERA.target.y, CAMERA.target.z]}
          enableDamping
          dampingFactor={0.08}
          rotateSpeed={0.55}
          minDistance={9}
          maxDistance={70}
          maxPolarAngle={1.42}
          makeDefault
        />

        <EffectComposer multisampling={4}>
          <Bloom mipmapBlur intensity={0.5} luminanceThreshold={0.72} luminanceSmoothing={0.25} />
          <Vignette offset={0.24} darkness={0.34} />
        </EffectComposer>
      </Canvas>
    </UnitSelectionContext.Provider>
  )
}
