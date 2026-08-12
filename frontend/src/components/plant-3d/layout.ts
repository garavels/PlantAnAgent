/** Where everything stands.
 *
 * One declarative map of the plant's spatial layout: unit origins, flow routes
 * and camera framing. Scene components read their position from here and flow
 * lines follow the routes defined here, so adding another unit means appending
 * to this file plus one component module -- nothing else moves.
 *
 * Units are meters. +x runs east (right from the default camera), +z runs
 * south (toward the default camera).
 */

export interface Vec3 {
  x: number
  y: number
  z: number
}

export const UNIT_POSITIONS = {
  /** East fence of the solar field, where its DC cable leaves. */
  solarField: { x: -13, y: 0, z: -4 },
  hub: { x: 0.5, y: 0, z: -4 },
  battery: { x: 2, y: 0, z: 5.6 },
  electrolyser: { x: 10, y: 0, z: -3.2 },
  h2Tank: { x: 17.6, y: 0, z: 1.6 },
  /** DAC sits on the open north strip, facing the wind. */
  dac: { x: 6.5, y: 0, z: -11 },
  co2Tank: { x: 13, y: 0, z: -11 },
  /** The reactor between the two gas tanks, CH4 tank just east of it. */
  methanator: { x: 13.5, y: 0, z: -6.2 },
  ch4Tank: { x: 18.6, y: 0, z: -6 },
  /** Ground grate that absorbs curtailed energy. */
  curtail: { x: 4.8, y: 0, z: -7.8 },
} as const satisfies Record<string, Vec3>

/** Solar field geometry. */
export const SOLAR_FIELD = {
  rows: 4,
  cols: 6,
  /** panel centre spacing */
  dx: 3.1,
  dz: 3.6,
  /** south-west corner of the array (row 0, col 0 panel centre) */
  origin: { x: -21.5, y: 0, z: -12.5 },
  tilt: 0.38,
} as const

/** Height above ground of the elevated power runs. */
export const CABLE_H = 1.35
/** The battery support line rides higher so the two energy directions separate visually. */
export const SUPPORT_H = 2.3
/** H2 leaves the stack as a low pipe run. */
export const PIPE_H = 0.75

/** Flow routes as polylines; components turn them into smooth curves. */
export const FLOW_ROUTES = {
  solarToHub: [
    { x: -4.5, y: CABLE_H, z: -4 },
    { x: UNIT_POSITIONS.hub.x - 1.1, y: CABLE_H, z: -4 },
  ],
  hubToBattery: [
    { x: UNIT_POSITIONS.hub.x, y: CABLE_H, z: -3 },
    { x: UNIT_POSITIONS.hub.x, y: CABLE_H, z: 3.4 },
    { x: UNIT_POSITIONS.battery.x - 1.4, y: CABLE_H, z: 4.4 },
  ],
  hubToStack: [
    { x: UNIT_POSITIONS.hub.x + 1.1, y: CABLE_H, z: -4 },
    { x: UNIT_POSITIONS.electrolyser.x - 3.6, y: CABLE_H, z: -3.4 },
  ],
  batteryToStack: [
    { x: UNIT_POSITIONS.battery.x + 3.2, y: SUPPORT_H, z: UNIT_POSITIONS.battery.z },
    { x: 14.6, y: SUPPORT_H, z: UNIT_POSITIONS.battery.z },
    { x: 14.6, y: SUPPORT_H, z: -1.2 },
    { x: UNIT_POSITIONS.electrolyser.x + 2.2, y: SUPPORT_H, z: -1.9 },
  ],
  stackToTank: [
    { x: UNIT_POSITIONS.electrolyser.x + 3.6, y: PIPE_H, z: -2.2 },
    { x: 15.4, y: PIPE_H, z: -0.6 },
    { x: UNIT_POSITIONS.h2Tank.x - 1.7, y: PIPE_H, z: UNIT_POSITIONS.h2Tank.z },
  ],
  /** DAC and methanator draw their power off the hub, north side. */
  hubToDac: [
    { x: UNIT_POSITIONS.hub.x - 0.3, y: CABLE_H, z: -4.9 },
    { x: UNIT_POSITIONS.hub.x - 0.3, y: CABLE_H, z: -11 },
    { x: UNIT_POSITIONS.dac.x - 2.1, y: CABLE_H, z: UNIT_POSITIONS.dac.z },
  ],
  hubToMethanator: [
    { x: UNIT_POSITIONS.hub.x + 0.9, y: CABLE_H, z: -4.7 },
    { x: 8.6, y: CABLE_H, z: -6.9 },
    { x: UNIT_POSITIONS.methanator.x - 2.6, y: CABLE_H, z: UNIT_POSITIONS.methanator.z - 0.2 },
  ],
  dacToCo2: [
    { x: UNIT_POSITIONS.dac.x + 2.1, y: PIPE_H, z: UNIT_POSITIONS.dac.z },
    { x: UNIT_POSITIONS.co2Tank.x - 1.7, y: PIPE_H, z: UNIT_POSITIONS.co2Tank.z },
  ],
  co2ToMethanator: [
    { x: UNIT_POSITIONS.co2Tank.x + 0.5, y: PIPE_H, z: UNIT_POSITIONS.co2Tank.z + 1.5 },
    { x: UNIT_POSITIONS.methanator.x + 0.2, y: PIPE_H, z: UNIT_POSITIONS.methanator.z - 1.6 },
  ],
  h2ToMethanator: [
    { x: UNIT_POSITIONS.h2Tank.x - 0.6, y: PIPE_H, z: UNIT_POSITIONS.h2Tank.z - 1.2 },
    { x: 16.6, y: PIPE_H, z: -3.2 },
    { x: UNIT_POSITIONS.methanator.x + 2.6, y: PIPE_H, z: UNIT_POSITIONS.methanator.z + 0.3 },
  ],
  methanatorToCh4: [
    { x: UNIT_POSITIONS.methanator.x + 2.6, y: PIPE_H, z: UNIT_POSITIONS.methanator.z - 0.5 },
    { x: UNIT_POSITIONS.ch4Tank.x - 1.7, y: PIPE_H, z: UNIT_POSITIONS.ch4Tank.z },
  ],
  curtail: [
    { x: UNIT_POSITIONS.hub.x + 0.6, y: CABLE_H, z: -4.9 },
    { x: UNIT_POSITIONS.curtail.x, y: CABLE_H, z: UNIT_POSITIONS.curtail.z },
    { x: UNIT_POSITIONS.curtail.x, y: 0.15, z: UNIT_POSITIONS.curtail.z },
  ],
} as const satisfies Record<string, Vec3[]>

/** The fenced pad everything sits on. */
export const PAD = { w: 46, d: 26, origin: { x: -2, z: -3 } } as const

export const CAMERA = {
  position: { x: 25.5, y: 15, z: 28 },
  target: { x: -0.6, y: 0.6, z: -1.6 },
  fov: 40,
} as const
