/** Approximate solar geometry for the scene, never a source of simulated PV.
 *
 * NOAA's fractional-year equations give the equation of time and declination:
 * https://gml.noaa.gov/grad/solcalc/solareqns.PDF
 * We omit atmospheric refraction and local terrain horizons. Weather power
 * continues to come exclusively from the backend's downloaded hourly profile.
 */

export interface SiteSunPosition {
  /** Radians above the geometric horizon; negative at night. */
  altitude: number;
  /** Radians clockwise from north. */
  azimuth: number;
  /** Normalized scene direction: +x east, +y up, +z south. */
  direction: [number, number, number];
  /** Fraction of the apparent solar day; 0.5 at local solar noon. */
  solarDayFraction: number;
}

const RAD = Math.PI / 180;
const DAY_MS = 86_400_000;
const wrap = (value: number, period: number) => ((value % period) + period) % period;

export function siteSunPosition(
  latitude: number,
  longitude: number,
  timestamp: string | null | undefined,
): SiteSunPosition | null {
  if (
    !Number.isFinite(latitude) || Math.abs(latitude) > 90 ||
    !Number.isFinite(longitude) || Math.abs(longitude) > 180 ||
    typeof timestamp !== "string" || !/(?:Z|[+-]\d{2}:\d{2})$/i.test(timestamp)
  ) return null;
  const date = new Date(timestamp);
  if (!Number.isFinite(date.getTime())) return null;

  const yearStart = Date.UTC(date.getUTCFullYear(), 0, 1);
  const nextYearStart = Date.UTC(date.getUTCFullYear() + 1, 0, 1);
  const yearDays = (nextYearStart - yearStart) / DAY_MS;
  const dayOfYear = Math.floor((date.getTime() - yearStart) / DAY_MS) + 1;
  const hour = date.getUTCHours() + date.getUTCMinutes() / 60 + date.getUTCSeconds() / 3600;
  const gamma = 2 * Math.PI / yearDays * (dayOfYear - 1 + (hour - 12) / 24);
  const equationMinutes = 229.18 * (
    0.000075 + 0.001868 * Math.cos(gamma) - 0.032077 * Math.sin(gamma)
    - 0.014615 * Math.cos(2 * gamma) - 0.040849 * Math.sin(2 * gamma)
  );
  const declination = 0.006918 - 0.399912 * Math.cos(gamma) + 0.070257 * Math.sin(gamma)
    - 0.006758 * Math.cos(2 * gamma) + 0.000907 * Math.sin(2 * gamma)
    - 0.002697 * Math.cos(3 * gamma) + 0.00148 * Math.sin(3 * gamma);
  const solarMinutes = wrap(hour * 60 + equationMinutes + 4 * longitude, 1440);
  const hourAngle = (solarMinutes / 4 - 180) * RAD;
  const lat = latitude * RAD;
  const east = -Math.cos(declination) * Math.sin(hourAngle);
  const up = Math.sin(lat) * Math.sin(declination)
    + Math.cos(lat) * Math.cos(declination) * Math.cos(hourAngle);
  const north = Math.cos(lat) * Math.sin(declination)
    - Math.sin(lat) * Math.cos(declination) * Math.cos(hourAngle);
  return {
    altitude: Math.asin(Math.max(-1, Math.min(1, up))),
    azimuth: wrap(Math.atan2(east, north), 2 * Math.PI),
    direction: [east, up, -north],
    solarDayFraction: solarMinutes / 1440,
  };
}
