import { useEffect, useRef, useState } from "react";
import type { FormEvent, MouseEvent } from "react";
import { Link } from "react-router-dom";
import { useWorkspace } from "./context";
import countries from "./europeMap.json";
import { Plot } from "./Plot";
import { EUROPE_BOUNDS, isEuropeanSite, isUnavailableStudy, projectSite, siteApi, STUDY_UNAVAILABLE_MESSAGE, studyDays, unprojectSite } from "./siteApi";
import type { SiteExample, SiteForecast, SiteRunJob, SiteRunRequest, SiteRunResult } from "./siteTypes";
import "./site.css";

const number = (value: number, digits = 0) => value.toLocaleString("en-GB", { maximumFractionDigits: digits });
const date = (value: string, options: Intl.DateTimeFormatOptions = {}) => new Date(value).toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC", ...options });
const dateTime = (value: string) => date(value, { hour: "2-digit", minute: "2-digit", hour12: false });
const intervalEnd = (value: string, hours = 1) => new Date(new Date(value).getTime() + hours * 3_600_000).toISOString();
const message = (error: unknown) => error instanceof Error ? error.message : "The request failed. Try again or choose a saved example.";
const niceName = (name: string) => name.replaceAll("_", " ").replace(/\bdac\b/gi, "CO₂ capture");
const STUDY_STORAGE_KEY = "plant-location-study";
type SavedStudy = { jobId: string; forecastId: string; config?: SiteRunRequest };
function saveStudy(study: SavedStudy | null) {
  try {
    if (study) sessionStorage.setItem(STUDY_STORAGE_KEY, JSON.stringify(study));
    else sessionStorage.removeItem(STUDY_STORAGE_KEY);
  } catch { /* Private browsing can disable storage; simulation still works. */ }
}
function savedStudy(): SavedStudy | null {
  try {
    const study = JSON.parse(sessionStorage.getItem(STUDY_STORAGE_KEY) ?? "null") as SavedStudy | null;
    return study && typeof study.jobId === "string" && typeof study.forecastId === "string" ? study : null;
  } catch { return null; }
}

function EuropeMap({ latitude, longitude, name, examples, disabled, onSelect, onExample }: {
  latitude: number;
  longitude: number;
  name: string;
  examples: SiteExample[];
  disabled: boolean;
  onSelect: (latitude: number, longitude: number) => void;
  onExample: (example: SiteExample) => void;
}) {
  const point = projectSite(latitude, longitude);
  function select(event: MouseEvent<SVGSVGElement>) {
    if (disabled) return;
    const matrix = event.currentTarget.getScreenCTM();
    if (!matrix) return;
    const position = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    if (position.x < 0 || position.x > 720 || position.y < 0 || position.y > 608) return;
    const site = unprojectSite(position.x, position.y);
    onSelect(site.latitude, site.longitude);
  }
  return (
    <div className="site-map-panel">
      <div className="site-map-heading"><strong>Place your plant</strong><span>Click the map or enter coordinates</span></div>
      <svg className={`site-map${disabled ? " site-map--disabled" : ""}`} viewBox="0 0 720 608" onClick={select} role="img" aria-label={`Europe map. Selected site: ${name}, ${number(latitude, 4)}° latitude, ${number(longitude, 4)}° longitude. Use the coordinate fields or example buttons to choose a site with the keyboard.`}>
        <defs><clipPath id="site-map-clip"><rect width="720" height="608" /></clipPath></defs>
        <g clipPath="url(#site-map-clip)">
          {[40, 50, 60, 70].map((lat) => <line key={lat} className="site-map-grid" x1="0" y1={projectSite(lat, 0).y} x2="720" y2={projectSite(lat, 0).y} />)}
          {[-20, -10, 0, 10, 20, 30, 40].map((lon) => <line key={lon} className="site-map-grid" x1={projectSite(0, lon).x} y1="0" x2={projectSite(0, lon).x} y2="608" />)}
          {countries.map((country) => <path key={country.name} d={country.path} className="site-country"><title>{country.name}</title></path>)}
          <text className="site-sea-label" x="50" y="380">Atlantic Ocean</text>
          <text className="site-sea-label" x="358" y="579">Mediterranean Sea</text>
          {[40, 50, 60, 70].map((lat) => <text key={lat} className="site-map-coordinate" x="12" y={projectSite(lat, 0).y - 6}>{lat}°N</text>)}
          {examples.map((example) => {
            const p = projectSite(example.latitude, example.longitude);
            return <g key={example.id} className="site-example-dot" transform={`translate(${p.x} ${p.y})`}><circle r="3" /><text x="8" y="4">{example.name}</text></g>;
          })}
          {isEuropeanSite(latitude, longitude) && <g className="site-map-pin" transform={`translate(${point.x} ${point.y})`}><circle r="17" className="site-pin-halo" /><circle r="6" /><path d="M-25 0h8M17 0h8M0-25v8M0 17v8" /></g>}
        </g>
      </svg>
      <div className="site-map-footer"><span>Europe study area</span><a href="https://www.naturalearthdata.com/about/terms-of-use/" target="_blank" rel="noreferrer">Made with Natural Earth</a></div>
      {examples.length > 0 && <div className="site-examples" aria-label="Saved location examples">
        <span>Saved examples</span>
        {examples.map((example) => <button key={example.id} className="w-button" disabled={disabled || !example.available} onClick={() => onExample(example)}>{example.name}{!example.available ? " (unavailable)" : ""}</button>)}
      </div>}
    </div>
  );
}

function ForecastStrip({ forecast }: { forecast: SiteForecast }) {
  const [selectedDay, setSelectedDay] = useState(0);
  const studyHours = forecast.hours.slice(0, 240);
  const days = studyDays(forecast.hours);
  const selected = days[Math.min(selectedDay, days.length - 1)] ?? [];
  const peak = Math.max(1, ...studyHours.map((hour) => hour.shortwave_w_m2));
  return <section className="site-forecast" aria-labelledby="site-forecast-title">
    <div className="site-section-heading"><div><h2 id="site-forecast-title">Ten days of sunlight</h2><p>Hourly irradiance for {forecast.site.name}. All dates and times are UTC.</p></div><a className="w-button" href={siteApi.download(forecast.id)} download>Download weather JSON</a></div>
    <div className="site-days" aria-label="Select forecast day">
      {days.map((hours, i) => <button key={hours[0].time} className="site-day" aria-pressed={selectedDay === i} onClick={() => setSelectedDay(i)}>
        <strong>Day {i + 1}</strong><small>{date(hours[Math.floor(hours.length / 2)].time)}</small>
        <svg viewBox="0 0 100 48" aria-hidden="true"><path d={`M0 46 ${hours.map((h, j) => `L${(j / Math.max(1, hours.length - 1)) * 100} ${46 - (h.shortwave_w_m2 / peak) * 41}`).join(" ")} L100 46Z`} /></svg>
        <span>{number(hours.reduce((sum, h) => sum + h.shortwave_w_m2, 0) / 1000, 1)} <small>kWh/m²</small></span>
        <small>{number(Math.min(...hours.map((h) => h.temperature_c)))}–{number(Math.max(...hours.map((h) => h.temperature_c)))}°C</small>
      </button>)}
    </div>
    {selected.length > 0 && <div className="site-day-detail"><strong>{dateTime(selected[0].time)}–{dateTime(intervalEnd(selected.at(-1)!.time))} UTC</strong><span>Peak sunlight {number(Math.max(...selected.map((h) => h.shortwave_w_m2)))} W/m²</span><span>Mean wind {number(selected.reduce((sum, h) => sum + h.wind_m_s, 0) / selected.length, 1)} m/s</span><span>Mean humidity {number(selected.reduce((sum, h) => sum + h.humidity_pct, 0) / selected.length)}%</span></div>}
  </section>;
}

function RunResults({ result, jobId }: { result: SiteRunResult; jobId: string }) {
  const { metrics, baseline } = result;
  const [chart, setChart] = useState<"battery" | "production" | "solar">("battery");
  const plotted = (run: SiteRunResult) => run.history.map((h) => ({ x: h.elapsed_hours, y: chart === "battery" ? h.battery_soc * 100 : chart === "production" ? h.methane_total_kg : h.solar_kw }));
  const series = [{ label: result.controller.label, color: "var(--accent)", points: plotted(result) }];
  if (baseline && chart !== "solar") series.push({ label: "Simple baseline", color: "var(--co2)", points: plotted(baseline) });
  if (chart === "solar" && result.parameters.solar_scale < 1) series.push({ label: "Original weather forecast", color: "var(--solar)", points: result.history.map((h) => ({ x: h.elapsed_hours, y: h.forecast_solar_kw })) });
  const comparisons = [
    ["Methane produced", "kg", metrics.methane_produced_kg, baseline?.metrics.methane_produced_kg],
    ["Delivery service", "%", metrics.delivery_service_pct, baseline?.metrics.delivery_service_pct],
    ["Curtailed solar", "kWh", metrics.curtailed_energy_kwh, baseline?.metrics.curtailed_energy_kwh],
    ["Final battery charge", "%", metrics.battery_final_soc * 100, baseline ? baseline.metrics.battery_final_soc * 100 : undefined],
  ] as const;
  return <section className="site-results" aria-labelledby="site-results-title">
    <div className="site-section-heading"><div><h2 id="site-results-title">Projected plant operation</h2><p>{result.site.name} · {date(result.start_time)}–{date(result.end_time, { year: "numeric" })} · {number(result.parameters.solar_kw)} kW solar · {number(result.parameters.battery_kwh)} kWh battery</p></div><a className="w-button" href={`/api/sites/runs/${encodeURIComponent(jobId)}/download`} download>Download study JSON</a></div>
    <div className="site-result-context"><span>{result.controller.label}</span><span>{result.parameters.solar_scale < 1 ? `${number((1 - result.parameters.solar_scale) * 100)}% less sunlight than forecast` : "Weather follows the forecast"}</span><span>{result.fault.enabled ? "CO₂ capture outage injected" : "No injected fault"}</span></div>
    {result.controller.uses_forecast && <p className="site-controller-note">Plans {result.controller.planning_horizon_hours} hours ahead; replans every {result.controller.replanning_interval_hours ?? 3} hours and when an outage or recovery is observed.</p>}
    <div className="site-metrics">
      <div><span>Methane produced</span><strong>{number(metrics.methane_produced_kg)} <small>kg</small></strong><small>Over {number(result.horizon_hours)} hours</small></div>
      <div><span>Solar available</span><strong>{number(metrics.solar_energy_kwh)} <small>kWh</small></strong><small>{number(metrics.curtailment_pct, 1)}% curtailed</small></div>
      <div><span>Delivery service</span><strong>{number(metrics.delivery_service_pct, 1)}<small>%</small></strong><small>{number(metrics.delivery_delivered_kg)} / {number(metrics.delivery_requested_kg)} kg delivered</small></div>
      <div><span>Lowest battery charge</span><strong>{number(metrics.battery_min_soc * 100, 1)}<small>%</small></strong><small>Ends at {number(metrics.battery_final_soc * 100, 1)}%</small></div>
    </div>
    <div className="site-chart-tabs" aria-label="Result chart">
      {([ ["battery", "Battery charge"], ["production", "Methane production"], ["solar", "Solar power"] ] as const).map(([key, label]) => <button key={key} className="w-button" aria-pressed={chart === key} onClick={() => setChart(key)}>{label}</button>)}
    </div>
    <Plot title={chart === "battery" ? "Battery charge" : chart === "production" ? "Cumulative methane production" : "Available solar power"} unit={chart === "battery" ? "%" : chart === "production" ? "kg CH₄" : "kW"} domain={[0, result.horizon_hours]} yDomain={chart === "battery" ? [0, 100] : undefined} series={series} markers={result.fault.enabled ? [{ x: result.fault.start_hours, label: "CO₂ capture outage", color: "var(--alarm)" }, { x: result.fault.start_hours + result.fault.duration_hours, label: "Equipment restored", color: "var(--stored)" }] : []} />
    <div className="site-results-detail">
      {baseline && <div><h3>Controller comparison</h3><p>Both controllers receive the same site, capacities and disturbances.</p><div className="site-table-scroll"><table className="site-table"><thead><tr><th>Outcome</th><th>{result.controller.label}</th><th>Simple baseline</th></tr></thead><tbody>{comparisons.map(([label, unit, value, other]) => <tr key={label}><th>{label}</th><td>{number(value, 1)} {unit}</td><td>{other === undefined ? "—" : `${number(other, 1)} ${unit}`}</td></tr>)}</tbody></table></div></div>}
      <div><h3>What constrained production</h3><p>Most frequent limiting subsystem: <strong>{niceName(metrics.limiting_subsystem)}</strong>.</p><div className="site-utilisation">{Object.entries(metrics.utilisation).map(([key, value]) => <div key={key}><span>{niceName(key)}</span><meter min="0" max="1" value={value} aria-label={`${niceName(key)} average utilisation`} /><strong>{number(value * 100)}%</strong></div>)}</div><p className="site-note">Average utilisation over all ten days, including nights.</p></div>
    </div>
    {result.fault.enabled && <div className="site-fault-result"><strong>Injected CO₂ capture outage · hour {number(result.fault.start_hours)}–{number(result.fault.start_hours + result.fault.duration_hours)}</strong><p>{result.fault.note}</p></div>}
    {result.controller.fallback_count > 0 && <p className="site-notice" role="status">The planner used the reactive heuristic for {number(result.controller.fallback_count)} fifteen-minute intervals after unsuccessful plans. These intervals are included in the reported results.</p>}
    <details className="site-assumptions"><summary>Simulation assumptions & limitations</summary><ul>{result.assumptions.map((assumption, i) => <li key={i}>{assumption}</li>)}</ul></details>
  </section>;
}

export function Site() {
  const workspace = useWorkspace();
  const [activating, setActivating] = useState(false);
  const activeSite = useRef(workspace.siteForecast);
  activeSite.current = workspace.siteForecast;
  const [examples, setExamples] = useState<SiteExample[]>([]);
  const [latitude, setLatitude] = useState("45.6495");
  const [longitude, setLongitude] = useState("13.7768");
  const [name, setName] = useState("Trieste");
  const [forecast, setForecast] = useState<SiteForecast | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [solarKw, setSolarKw] = useState("5000");
  const [batteryKwh, setBatteryKwh] = useState("8000");
  const [controller, setController] = useState<"MPC" | "H">("MPC");
  const [solarScale, setSolarScale] = useState<0.7 | 1>(1);
  const [injectFault, setInjectFault] = useState(false);
  const [job, setJob] = useState<SiteRunJob | null>(null);
  const [starting, setStarting] = useState(false);
  const forecastRequest = useRef<AbortController | null>(null);
  const runRequest = useRef<AbortController | null>(null);
  const studying = starting || job?.status === "queued" || job?.status === "running";
  const running = studying || activating;
  const jobId = job?.id;
  const jobStatus = job?.status;
  const validCoordinates = latitude.trim() !== "" && longitude.trim() !== "" && isEuropeanSite(Number(latitude), Number(longitude));

  function acceptForecast(data: SiteForecast) {
    setForecast(data);
    setLatitude(String(data.site.latitude));
    setLongitude(String(data.site.longitude));
    setName(data.site.name);
    setJob(null);
  }

  useEffect(() => {
    const abort = new AbortController();
    forecastRequest.current = abort;
    setLoading(true);
    void siteApi.examples(abort.signal).then(async (catalog) => {
      if (abort.signal.aborted) return;
      setExamples(catalog);
      if (activeSite.current) return;
      const saved = savedStudy();
      if (saved) {
        try {
          const [data, currentJob] = await Promise.all([siteApi.savedForecast(saved.forecastId, abort.signal), siteApi.job(saved.jobId, abort.signal)]);
          if (abort.signal.aborted || activeSite.current) return;
          acceptForecast(data);
          setJob(currentJob);
          const parameters = currentJob.result?.parameters ?? saved.config;
          if (parameters) {
            setSolarKw(String(parameters.solar_kw));
            setBatteryKwh(String(parameters.battery_kwh));
            setController((currentJob.result?.controller.id ?? saved.config?.controller) === "H" ? "H" : "MPC");
            setSolarScale(parameters.solar_scale < 1 ? 0.7 : 1);
            setInjectFault(parameters.inject_fault);
          }
          if (currentJob.status === "failed") setError(currentJob.error ?? "The previous study failed. You can run it again.");
          return;
        } catch (failure) {
          if (abort.signal.aborted) return;
          if (isUnavailableStudy(failure)) {
            saveStudy(null);
            setError(STUDY_UNAVAILABLE_MESSAGE);
          } else {
            setError(`Could not restore the previous study: ${message(failure)} Refresh the page to try restoring it again.`);
          }
        }
      }
      const first = catalog.find((example) => example.available);
      if (first) {
        const data = await siteApi.example(first.id, abort.signal);
        if (!abort.signal.aborted && !activeSite.current) acceptForecast(data);
      }
    }).catch((failure: unknown) => { if (!abort.signal.aborted) setError(message(failure)); }).finally(() => { if (!abort.signal.aborted) setLoading(false); });
    return () => { abort.abort(); forecastRequest.current?.abort(); runRequest.current?.abort(); };
  }, []);

  const activeForecastId = workspace.siteForecast?.id;
  useEffect(() => {
    const data = activeSite.current;
    if (!data) return;
    acceptForecast(data);
    setLoading(false);
    setError(null);
    const parameters = workspace.snapshot?.site_parameters;
    if (parameters) {
      setSolarKw(String(parameters.solar_kw));
      setBatteryKwh(String(parameters.battery_kwh));
      setSolarScale(parameters.solar_scale < 1 ? 0.7 : 1);
      setInjectFault(parameters.inject_fault);
      setController(workspace.snapshot?.controller.id === "H" ? "H" : "MPC");
    }
    // Restore once per selected forecast, keeping form edits while its clock advances.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeForecastId]);

  useEffect(() => {
    if (!jobId || (jobStatus !== "queued" && jobStatus !== "running")) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await siteApi.job(jobId!, abort.signal);
        if (abort.signal.aborted) return;
        setJob(next);
        setError(null);
        if (next.status === "failed") setError(next.error ?? "The simulation failed. Adjust the configuration and try again.");
        if (next.status === "queued" || next.status === "running") timer = setTimeout(() => void poll(), 1200);
      } catch (failure) {
        if (!abort.signal.aborted) {
          if (isUnavailableStudy(failure)) {
            saveStudy(null);
            setJob((current) => current && current.id === jobId ? { ...current, status: "failed", result: null, error: STUDY_UNAVAILABLE_MESSAGE } : current);
            setError(STUDY_UNAVAILABLE_MESSAGE);
            return;
          }
          setError(`Could not check simulation progress: ${message(failure)} Progress checks will retry automatically.`);
          timer = setTimeout(() => void poll(), 4000);
        }
      }
    }
    timer = setTimeout(() => void poll(), 1000);
    return () => { clearTimeout(timer); abort.abort(); };
  }, [jobId, jobStatus]);

  function changeLocation(lat: string, lon: string, placeName = "Selected location") {
    forecastRequest.current?.abort();
    setLatitude(lat);
    setLongitude(lon);
    setName(placeName);
    setForecast(null);
    setLoading(false);
    setError(null);
    setJob(null);
    saveStudy(null);
  }

  async function loadForecast(example?: SiteExample) {
    forecastRequest.current?.abort();
    const abort = new AbortController();
    forecastRequest.current = abort;
    setLoading(true);
    setError(null);
    setJob(null);
    setForecast(null);
    saveStudy(null);
    if (example) {
      setLatitude(String(example.latitude));
      setLongitude(String(example.longitude));
      setName(example.name);
    }
    try {
      const data = example
        ? await siteApi.example(example.id, abort.signal)
        : await siteApi.forecast({ latitude: Number(latitude), longitude: Number(longitude), name, refresh: true }, abort.signal);
      if (!abort.signal.aborted) acceptForecast(data);
    } catch (failure) { if (!abort.signal.aborted) setError(message(failure)); }
    finally { if (!abort.signal.aborted) setLoading(false); }
  }

  async function startRun(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!forecast || running) return;
    const abort = new AbortController();
    runRequest.current?.abort();
    runRequest.current = abort;
    setStarting(true);
    setError(null);
    setJob(null);
    try {
      const config = { forecast_id: forecast.id, solar_kw: Number(solarKw), battery_kwh: Number(batteryKwh), controller, inject_fault: injectFault, solar_scale: solarScale, compare_baseline: controller === "MPC" };
      const result = await siteApi.run(config, abort.signal);
      if (!abort.signal.aborted) {
        setJob(result);
        saveStudy({ jobId: result.id, forecastId: forecast.id, config });
        if (result.status === "failed") setError(result.error ?? "The simulation failed. Try again.");
      }
    } catch (failure) { if (!abort.signal.aborted) setError(message(failure)); }
    finally { if (!abort.signal.aborted) setStarting(false); }
  }

  async function activate(event: MouseEvent<HTMLButtonElement>) {
    if (!forecast || running || !event.currentTarget.form?.reportValidity()) return;
    setActivating(true);
    setError(null);
    try {
      const ok = await workspace.activateSite({ forecast_id: forecast.id, solar_kw: Number(solarKw), battery_kwh: Number(batteryKwh), controller, inject_fault: injectFault, solar_scale: solarScale });
      if (!ok) setError("Could not activate this site. Try again after checking the service.");
    } finally { setActivating(false); }
  }

  return <div className="site-page">
    <div className="w-page-heading site-page-heading"><div><h1>Location study</h1><p>Move the plant. See what the weather changes.</p></div><Link to="/plant" className="w-button">Explore plant equipment</Link></div>
    <p className="site-intro">Choose a European location and load ten days of weather. Use it across the workspace or run a complete production study.</p>
    {workspace.siteForecast && <div className="site-notice"><strong>Active workspace: {workspace.siteForecast.site.name}</strong><p>The Weather panel, plant view and shared clock use this forecast. Activating a site starts a new simulation from its first hour.</p><div className="site-run-action"><Link className="w-button" to="/weather">View active weather</Link><Link className="w-button" to="/plant">View active plant</Link><button className="w-button" disabled={running || workspace.busy} onClick={() => { setError(null); void workspace.deactivateSite().then((ok) => { if (!ok) setError("Could not restore the synthetic demo."); }); }}>Use synthetic demo</button></div></div>}
    {error && <div className="site-error" role="alert"><span>{error}{workspace.live.error ? ` ${workspace.live.error}` : ""}</span><button className="w-button" onClick={() => setError(null)}>Dismiss</button></div>}
    <div className="site-location-layout">
      <EuropeMap latitude={Number(latitude)} longitude={Number(longitude)} name={name} examples={examples} disabled={running} onSelect={(lat, lon) => changeLocation(String(lat), String(lon))} onExample={(example) => void loadForecast(example)} />
      <section className="site-location-controls" aria-labelledby="site-location-title">
        <div><h2 id="site-location-title">{name}</h2><p>A site for this study</p></div>
        <form onSubmit={(event) => { event.preventDefault(); if (validCoordinates) void loadForecast(); }}>
          <div className="site-coordinate-inputs"><label>Latitude<input type="number" step="any" min={EUROPE_BOUNDS.south} max={EUROPE_BOUNDS.north} required value={latitude} disabled={running} onChange={(event) => changeLocation(event.target.value, longitude)} /></label><label>Longitude<input type="number" step="any" min={EUROPE_BOUNDS.west} max={EUROPE_BOUNDS.east} required value={longitude} disabled={running} onChange={(event) => changeLocation(latitude, event.target.value)} /></label></div>
          {!validCoordinates && <p className="site-input-error">Use latitude 34°–72° and longitude −25°–45°.</p>}
          <button className="w-button w-button--primary site-load-button" type="submit" disabled={running || loading || !validCoordinates}>{loading ? "Loading weather…" : forecast ? "Refresh live forecast" : "Load live forecast"}</button>
        </form>
        {forecast ? <>
          <dl className="site-site-facts"><div><dt>Elevation</dt><dd>{forecast.site.elevation_m === null ? "Unavailable" : `${number(forecast.site.elevation_m)} m`}</dd></div><div><dt>Study period</dt><dd>{dateTime(forecast.start_time)}<br />to {dateTime(intervalEnd(forecast.start_time, 240))}</dd></div><div><dt>Time reference</dt><dd>UTC · {new Date(forecast.start_time).getUTCFullYear()}</dd></div></dl>
          <div className="site-source"><span className="site-data-badge">{forecast.source.kind === "cached_forecast" ? "Saved real forecast" : "Live weather forecast"}</span><p><a href={forecast.source.url} target="_blank" rel="noreferrer">{forecast.source.provider}</a><br />Retrieved {date(forecast.source.retrieved_at, { year: "numeric" })}</p><p>{forecast.source.note}</p></div>
          <p className="site-note">Elevation and weather describe the site. The prototype uses sunlight to drive solar generation; temperature, wind and humidity are shown for context.</p>
        </> : <div className="site-source site-empty-source"><h3>{loading ? "Loading site conditions" : "Ready to explore"}</h3><p>{loading ? "Preparing ten days of hourly weather and site elevation." : "Load a forecast for this point, or choose a saved real-weather example for an offline demonstration."}</p></div>}
      </section>
    </div>
    {forecast && <>
      <ForecastStrip key={forecast.id} forecast={forecast} />
      <details className="site-assumptions"><summary>Weather source & assumptions</summary><p>{forecast.source.provider} · {forecast.source.model}{forecast.source.forecast_issued_at ? ` · Issued ${date(forecast.source.forecast_issued_at, { year: "numeric" })}` : " · Forecast issue time not supplied by provider"}</p>{forecast.hours.length > 240 && <p>The download includes {forecast.hours.length - 240} additional forecast hours for the planner to look beyond the end of the ten-day study.</p>}<ul>{forecast.assumptions.map((assumption, i) => <li key={i}>{assumption}</li>)}</ul></details>
      <section className="site-configuration" aria-labelledby="site-configuration-title"><div className="site-section-heading"><div><h2 id="site-configuration-title">Run the plant here</h2><p>Ten days of operation. The forecast planner looks 24 hours ahead and replans every three hours.</p></div></div>
        <form onSubmit={(event) => void startRun(event)}>
          <div className="site-config-inputs"><label>Solar capacity <span>kW</span><input type="number" min="100" max="50000" step="100" required value={solarKw} disabled={running} onChange={(event) => setSolarKw(event.target.value)} /></label><label>Battery capacity <span>kWh</span><input type="number" min="100" max="100000" step="100" required value={batteryKwh} disabled={running} onChange={(event) => setBatteryKwh(event.target.value)} /></label><label>Controller<select value={controller} disabled={running} onChange={(event) => setController(event.target.value as "MPC" | "H")}><option value="MPC">Forecast planner</option><option value="H">Simple baseline</option></select></label></div>
          <fieldset className="site-sunlight" disabled={running}><legend>Sunlight scenario</legend><label><input type="radio" name="site-sunlight" checked={solarScale === 1} onChange={() => setSolarScale(1)} /><span>Follows the forecast<small>Projected operation under the downloaded weather</small></span></label><label><input type="radio" name="site-sunlight" checked={solarScale === 0.7} onChange={() => setSolarScale(0.7)} /><span>30% less sunlight<small>Simulated forecast error; the plant receives less solar power</small></span></label></fieldset>
          <label className="site-fault-choice"><input type="checkbox" checked={injectFault} disabled={running} onChange={(event) => setInjectFault(event.target.checked)} /><span>Inject a six-hour CO₂ capture outage<small>Equipment stops at hour 108. Compare how the controllers respond.</small></span></label>
          <div className="site-run-action"><button className="w-button w-button--primary" type="button" disabled={running || workspace.busy} onClick={(event) => void activate(event)}>{activating ? "Starting site workspace…" : "Use in workspace"}</button><span>Start the shared clock here and explore Weather, Plant and Planning together.</span></div>
          <div className="site-run-action"><button className="w-button" type="submit" disabled={running}>{studying ? "Simulating ten days…" : "Simulate ten days"}</button><span>{controller === "MPC" ? "Includes the same scenario with a simple baseline controller." : "Runs the simple baseline controller."}</span></div>
        </form>
        {studying && <div className="site-run-progress" role="status"><div><span>{starting ? "Preparing study" : job?.status === "queued" ? "Study queued" : "Simulating plant operation"}</span><strong>{number((job?.progress ?? 0) * 100)}%</strong></div><progress value={job?.progress ?? 0} max="1" aria-label="Ten-day simulation progress" /><p>You can inspect the weather while the simulation runs.</p></div>}
      </section>
    </>}
    {job?.status === "complete" && job.result && <RunResults key={job.id} result={job.result} jobId={job.id} />}
  </div>;
}
