import { useState } from "react";
import { Link } from "react-router-dom";
import { useWorkspace } from "./context";
import countries from "./europeMap.json";
import { projectSite, siteApi } from "./siteApi";
import { n } from "./format";
import { Plot } from "./Plot";
import { Scene } from "./Scene";
import type { SiteForecast } from "./siteTypes";
import { cloudReadout, scenarioForecastPower, utcWeatherTime, weatherReadout } from "./siteWeatherModel";
import "./siteWeather.css";

export function SiteWeatherPanel({ forecast }: { forecast: SiteForecast }) {
  const w = useWorkspace();
  const frame = w.frame!;
  const snapshot = w.snapshot!;
  const [selectedWindow, setSelectedWindow] = useState(0);
  const current = frame.weather.site_sample;
  const clouds = cloudReadout(current);
  const history = snapshot.history.slice(0, w.index + 1);
  const issue = frame.forecasts.issued_at_hours;
  const windows = frame.forecasts.solar;
  const chosen = windows[Math.min(selectedWindow, windows.length - 1)];
  const pin = projectSite(forecast.site.latitude, forecast.site.longitude);
  const readout = weatherReadout(frame.weather);
  const scale = snapshot.site_parameters?.solar_scale ?? 1;
  const scenario = snapshot.site_parameters?.scenario;
  const actualScale = frame.weather.stress_factor ?? scale;
  const knownScale = frame.weather.scenario_forecast_scale ?? 1;
  return <div className="sw-page">
    <div className="sw-site-summary">
      <div className="sw-map-panel">
        <svg className="sw-map" viewBox="0 0 720 608" role="img" aria-label={`Europe map with the active plant at ${forecast.site.name}, latitude ${forecast.site.latitude}, longitude ${forecast.site.longitude}`}>
          <defs><clipPath id="sw-map-clip"><rect width="720" height="608" /></clipPath></defs>
          <g clipPath="url(#sw-map-clip)">{countries.map((country) => <path key={country.name} d={country.path} />)}<g className="sw-map-pin" transform={`translate(${pin.x} ${pin.y})`}><circle r="25" /><circle r="11" /><path d="M-38 0h10M28 0h10M0-38v10M0 28v10" /></g></g>
        </svg>
        <div className="sw-map-caption"><a href="https://www.naturalearthdata.com/about/terms-of-use/" target="_blank" rel="noreferrer">Natural Earth</a></div>
      </div>
      <div className="sw-site-summary-copy">
        <div className="sw-heading"><div><h2>{forecast.site.name}</h2><p className="sw-site-facts">Latitude {n(forecast.site.latitude, 4)}° · Longitude {n(forecast.site.longitude, 4)}° · Elevation {forecast.site.elevation_m === null ? "unavailable" : `${n(forecast.site.elevation_m, 0)} m`}</p><p>{w.time(frame.clock.elapsed_hours)} · shared plant clock</p></div><Link to="/site" className="w-button">Change location</Link></div>
        <div className="sw-connection"><p>Sunlight at this location drives the solar field, plant operation and battery charging.</p><Link className="w-link" to="/plant">Explore plant →</Link></div>
      </div>
    </div>
    <div className="sw-location">
      <div className="sw-overhead">
        <Scene overhead illustrativeClouds />
        <div className="w-map-label"><span>Overhead plant view</span><strong>Plant and solar field</strong><small>{clouds.caption}</small></div>
      </div>
      <section className="sw-conditions" aria-labelledby="sw-current-title">
        <h3 id="sw-current-title">Weather driving the plant</h3><strong className="sw-solar-title">{readout.heading}</strong><p>{readout.detail}</p>
        <div className="sw-cloud-readout"><div className="sw-cloud-total"><span>Forecast cloud cover</span><strong>{clouds.total}</strong></div><dl className="sw-cloud-layers">{clouds.layers.map((layer) => <div key={layer.label}><dt>{layer.label}</dt><dd>{layer.value}</dd></div>)}</dl></div>
        <dl><div><dt>Available solar</dt><dd>{n(frame.weather.available_kw, 0)} kW</dd></div><div><dt>Forecast PV reference</dt><dd>{n(current?.forecast_solar_kw ?? frame.weather.clear_sky_kw, 0)} kW</dd></div>{current && <><div><dt>Forecast irradiance</dt><dd>{n(current.shortwave_w_m2, 0)} W/m²</dd></div><div><dt>Air temperature</dt><dd>{n(current.temperature_c)} °C</dd></div><div><dt>Relative humidity</dt><dd>{n(current.humidity_pct, 0)}%</dd></div><div><dt>Wind speed</dt><dd>{n(current.wind_m_s)} m/s</dd></div></>}</dl>
        {current && <p className="sw-sample-time">Sunlight is the mean over the hour starting {utcWeatherTime(current.interval_start)}. Conditions are forecast for {utcWeatherTime(current.conditions_time ?? new Date(Date.parse(current.interval_start) + 3_600_000).toISOString())}.</p>}
        {!clouds.available && <p className="sw-sample-time">This hour has no cloud-cover data. <Link className="w-link" to="/site">Load a new forecast</Link> to check availability.</p>}
        <p className="sw-scenario">{scenario ? `${scenario.name}: sunlight is ${n(actualScale * 100, 0)}% of the downloaded reference at this step; the planner expects ${n(knownScale * 100, 0)}%. Equipment faults can reduce available power further. Cloud cover above remains the provider's forecast.` : scale < 1 ? `${n((1 - scale) * 100, 0)}% less solar power is being simulated. The controller still receives the original forecast.` : "Simulated solar availability follows the downloaded forecast."}</p>
      </section>
    </div>
    <div className="sw-source"><div><strong>{forecast.source.kind === "cached_forecast" ? "Saved real forecast" : "Downloaded live forecast"}</strong><p><a href={forecast.source.url} target="_blank" rel="noreferrer">{forecast.source.provider}</a> · {forecast.source.model}<br />Retrieved {utcWeatherTime(forecast.source.retrieved_at)}{forecast.source.forecast_issued_at ? ` · Provider issue: ${utcWeatherTime(forecast.source.forecast_issued_at)}` : " · Provider issue time unavailable"}</p></div><a className="w-button" href={siteApi.download(forecast.id)} download>Download weather JSON</a></div>
    <div className="w-two-charts">
      <Plot title="Solar power at completed plant steps" unit="kW" cursor={frame.clock.elapsed_hours} onSelect={(x) => void w.selectFrame(Math.min(w.latestIndex, Math.max(0, Math.round(x * 4))))} series={[{ label: "Available to plant", color: "var(--solar)", points: history.map((point) => ({ x: point.elapsed_hours, y: point.solar_kw })) }, { label: "Forecast PV", color: "var(--text-dim)", dashed: true, points: history.map((point) => ({ x: point.elapsed_hours, y: scenarioForecastPower(point.clear_sky_kw, point.elapsed_hours, scenario?.sunlight ?? []) })) }]} />
      <Plot title="Downloaded hourly irradiance" unit="W/m²" domain={[0, snapshot.environment.horizon_hours]} cursor={frame.clock.elapsed_hours} onSelect={(x) => void w.selectFrame(Math.min(w.latestIndex, Math.max(0, Math.round(x * 4))))} series={[{ label: "Forecast sunlight", color: "var(--solar)", step: true, points: forecast.hours.slice(0, snapshot.environment.horizon_hours).flatMap((hour, index) => [{ x: index, y: hour.shortwave_w_m2 }, { x: index + 1, y: hour.shortwave_w_m2 }]) }]} />
    </div>
    <p className="w-note">Chart hours start at {utcWeatherTime(forecast.start_time)}. Select a completed time to inspect the same step across the workspace. Forecast sunlight beyond the cursor is a projection.</p>
    <section className="sw-window-section"><div className="w-panel-heading"><h2>Scenario forecast for the planner</h2><span>View starting {w.time(issue)}</span></div><p className="w-note">{scenario ? "These windows include sunlight changes marked as known to the planner. Unexpected changes affect the plant only when they occur." : "These windows read the original downloaded forecast."} Moving the plant clock does not download or issue a new weather forecast; no calibrated confidence interval is available.</p>
      <div className="w-forecast-windows">{windows.map((window, index) => <button key={index} aria-pressed={selectedWindow === index} onClick={() => setSelectedWindow(index)}><span>+{window.offset_hours}–{window.end_offset_hours} h</span><strong>{n(window.mean_kw, 0)} <small>kW</small></strong><small>Forecast PV · window mean</small></button>)}</div>
      {chosen && <div className="sw-selected-window"><span>{w.time(issue + chosen.offset_hours)}–{w.time(issue + chosen.end_offset_hours)}</span><strong>{n(chosen.mean_kw, 0)} kW forecast mean</strong><Link to="/planning" className="w-link">Inspect production plan</Link></div>}
    </section>
    <details className="sw-assumptions"><summary>Weather assumptions</summary><p>{forecast.source.note}</p><ul>{forecast.assumptions.map((assumption, index) => <li key={index}>{assumption}</li>)}</ul><p>Forecast cloud cover controls the cloud layer's coverage. Shapes and positions are illustrative. Low, mid and high layers can overlap; their percentages do not add up to the total. When cloud data is unavailable, the overhead view uses a labeled decorative layer. Solar generation uses the downloaded irradiance, with no extra reduction for the cloud display.</p><p>Elevation, temperature, humidity and wind describe the selected site.</p></details>
  </div>;
}
