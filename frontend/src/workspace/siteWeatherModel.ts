import { n } from "./format.ts";
import type { SiteCloudCover } from "./siteTypes";
import type { ScenarioSunlight } from "./scenarioTypes";

/** Plant boundaries retain the preceding one-minute interval's solar sample. */
export function scenarioForecastPower(rawKw: number, elapsedHours: number, periods: ScenarioSunlight[]) {
  const sampleHour = Math.max(0, elapsedHours - 1 / 60);
  return periods.reduce((power, period) => period.forecast_known && period.start_hour <= sampleHour && sampleHour < period.end_hour ? power * period.scale : power, rawKw);
}

export function cloudReadout(sample?: SiteCloudCover) {
  const valid = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 100;
  const percent = (value: unknown) => valid(value) ? `${n(value, 0)}%` : "Unavailable";
  const available = valid(sample?.cloud_cover_pct);
  const total = percent(sample?.cloud_cover_pct);
  return {
    available,
    total,
    caption: available ? `Forecast cloud cover ${total} · illustrative shapes` : "Cloud data unavailable · illustrative clouds",
    layers: [
      { label: "Low", value: percent(sample?.cloud_cover_low_pct) },
      { label: "Mid", value: percent(sample?.cloud_cover_mid_pct) },
      { label: "High", value: percent(sample?.cloud_cover_high_pct) },
    ],
  };
}

interface WeatherReadout {
  reference_kind?: string;
  available_kw: number;
  clear_sky_kw: number;
  status: string;
  attenuation: number | null;
  scenario_name?: string;
  scenario_forecast_scale?: number;
}

/** The compatibility clear_sky field carries a different physical reference for site runs. */
export function weatherReadout(weather: WeatherReadout) {
  if (weather.reference_kind === "forecast_pv") {
    const forecastKw = weather.clear_sky_kw * (weather.scenario_forecast_scale ?? 1);
    const difference = forecastKw - weather.available_kw;
    return {
      heading: forecastKw <= 0.001
        ? "No solar generation in the forecast"
        : difference > 0.001
          ? `${n((difference / forecastKw) * 100, 0)}% below forecast PV`
          : weather.scenario_name ? "Solar follows the scenario forecast" : "Solar follows the downloaded forecast",
      detail: `${n(weather.available_kw, 0)} kW available / ${n(forecastKw, 0)} kW forecast PV`,
    };
  }
  return {
    heading: weather.status === "night" ? "Night at the plant" : weather.attenuation === null ? "Solar reduction unattributed" : `${n(weather.attenuation * 100, 0)}% below clear-sky potential`,
    detail: `${n(weather.available_kw, 0)} kW available${weather.status === "night" ? " · cloud conditions unobserved" : ` / ${n(weather.clear_sky_kw, 0)} kW clear-sky potential`}`,
  };
}

export function missionDays(horizonHours: number) {
  return Math.max(1, Math.ceil(horizonHours / 24));
}

export function utcWeatherTime(value: string) {
  return new Date(value).toLocaleString("en-GB", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", timeZone: "UTC", hour12: false }) + " UTC";
}
