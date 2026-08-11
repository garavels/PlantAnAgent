/** Number formatting for instrument readouts. Fixed decimals, no jitter. */

export const kw = (value: number) => Math.round(value).toLocaleString('en-US')

export const kwh = (value: number) => Math.round(value).toLocaleString('en-US')

export const kg = (value: number) => value.toFixed(1)

export const pct = (fraction: number, decimals = 1) => (fraction * 100).toFixed(decimals)

export const clamp = (value: number, low = 0, high = 1) => Math.min(high, Math.max(low, value))

/** Power -> 0..1 intensity, used to drive conduit opacity and flow speed. */
export const intensity = (power: number, reference: number) => clamp(power / Math.max(reference, 1))
