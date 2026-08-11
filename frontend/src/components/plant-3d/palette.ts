/** Scene palette: the console's tokens translated into the twin's world.
 *
 * Hex twins of `styles/tokens.css` (CSS custom properties cannot drive WebGL
 * materials). Hue stays rationed: surfaces are near-monochrome olive, and the
 * four energy carriers are the only saturated things in the scene.
 */

export const CARRIER = {
  solar: '#dcc084',
  stored: '#a6c496',
  feed: '#aac6cb',
  h2: '#eef2e6',
  waste: '#6a7166',
  accent: '#b3cb92',
} as const

export const SURFACE = {
  /** open ground beyond the pad */
  terrain: '#181c16',
  /** the fenced gravel pad the plant sits on */
  pad: '#20261e',
  padLine: '#2a3128',
  fence: '#3a4139',
  /** painted steel */
  steelDark: '#232823',
  steel: '#2e342d',
  steelLight: '#3d443b',
  /** container / building bodies */
  body: '#2b322a',
  bodyDark: '#20261f',
  /** the DC cable and H2 pipe when idle */
  conduit: '#333a32',
  pipe: '#3a423a',
  lampPole: '#2b312a',
} as const

export const LIGHT = {
  sunNoon: '#fff3df',
  sunLow: '#ffd9a3',
  moon: '#aebdc9',
  lamp: '#ffd9a8',
} as const

/** Sky and fog, keyed by how much sun the sim says is up. */
export const SKY = {
  night: '#0b0e0b',
  day: '#aebaa4',
  horizonNight: '#10140f',
  horizonDay: '#c3cbb6',
} as const
