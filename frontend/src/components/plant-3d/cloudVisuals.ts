import type { WeatherView } from "../../types/v2";

export interface CloudVisuals {
  density: number;
  source: "forecast" | "illustrative" | "synthetic" | "none";
  castShadow: boolean;
}

/** Cloud coverage controls the illustrative layer only, never simulated PV.
 *
 * Total cover is a percentage of the sky. Layer-specific covers overlap and
 * must not be summed; irradiance and a plant's stress factor cannot substitute
 * for missing cover. A valid zero is a clear sky, even when fallback is enabled.
 */
export function deriveCloudVisuals(weather: WeatherView, illustrativeClouds = false): CloudVisuals {
  if (weather.reference_kind === "forecast_pv") {
    const cover = weather.site_sample?.cloud_cover_pct;
    if (typeof cover === "number" && Number.isFinite(cover) && cover >= 0 && cover <= 100) {
      return { density: cover / 100, source: "forecast", castShadow: false };
    }
    return illustrativeClouds
      ? { density: 0.55, source: "illustrative", castShadow: false }
      : { density: 0, source: "none", castShadow: false };
  }
  return weather.status === "observed"
    ? { density: weather.attenuation ?? 0, source: "synthetic", castShadow: true }
    : { density: 0, source: "none", castShadow: false };
}
