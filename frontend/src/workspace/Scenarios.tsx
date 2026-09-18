import { useEffect, useId, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useWorkspace } from "./context";
import { scenarioApi } from "./scenarioApi";
import { DEFAULT_SCENARIO, FALLBACK_FAULTS, faultLabel, recurringDeliveries, resizeScenario, SAVED_SCENARIO_SITES, SCENARIO_EXAMPLES, scenarioDays, scenarioIssues, scenarioOffset } from "./scenarioModel";
import type { SavedScenarioSite, ScenarioCapabilities, ScenarioController, ScenarioExample, ScenarioFault, ScenarioPreview, ScenarioSpec, ScenarioSunlight } from "./scenarioTypes";
import "./scenarios.css";

const STORAGE_KEY = "plant-scenario-studio-v1";
const format = (value: number, digits = 0) => Number.isFinite(value) ? value.toLocaleString("en-GB", { maximumFractionDigits: digits }) : "—";
const date = (value: string) => new Date(value).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC", hour12: false });
const failureMessage = (error: unknown) => error instanceof Error ? error.message : "The request failed. Your draft is still available; try again.";
const channelLabels: Record<string, string> = { h2_level_kg: "Hydrogen level", co2_level_kg: "CO₂ level", ch4_level_kg: "Methane level", battery_soc: "Battery charge" };

function restoreDraft(): { scenario: ScenarioSpec; prompt: string; restored: boolean } {
  try {
    const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) ?? "null") as { scenario?: ScenarioSpec; prompt?: string } | null;
    const candidate = saved?.scenario;
    if (candidate?.version === "plant-scenario-v1" && candidate.site && candidate.initial && candidate.economics && [candidate.sunlight, candidate.faults, candidate.deliveries, candidate.notes, candidate.unsupported].every(Array.isArray)) {
      return { scenario: candidate, prompt: typeof saved?.prompt === "string" ? saved.prompt : "", restored: true };
    }
  } catch { /* A draft is optional when storage is unavailable. */ }
  return { scenario: structuredClone(DEFAULT_SCENARIO), prompt: "", restored: false };
}

function NumberField({ label, value, onChange, min, max, step = 1, hint, disabled = false }: {
  label: string; value: number; onChange: (value: number) => void; min?: number; max?: number; step?: number | "any"; hint?: ReactNode; disabled?: boolean;
}) {
  const id = useId();
  return <label className="scenario-field" htmlFor={id}><span>{label}</span>
    <input id={id} type="number" value={Number.isFinite(value) ? value : ""} min={min} max={max} step={step} required disabled={disabled} onChange={(event) => onChange(event.target.valueAsNumber)} aria-describedby={hint ? `${id}-hint` : undefined} />
    {hint && <small id={`${id}-hint`}>{hint}</small>}
  </label>;
}

function ScheduleTimeline({ scenario }: { scenario: ScenarioSpec }) {
  const days = scenarioDays(scenario);
  return <section className="scenario-timeline" aria-labelledby="scenario-timeline-title">
    <div className="scenario-section-heading"><h2 id="scenario-timeline-title">Scenario timeline</h2><span>{format(scenario.duration_days * 24)} simulated hours</span></div>
    <div className="scenario-timeline-legend"><span className="scenario-sun-key">Sunlight remaining</span><span className="scenario-fault-key">Equipment faults</span><span className="scenario-delivery-key">Methane deliveries</span></div>
    <div className="scenario-days-scroll"><div className="scenario-days" style={{ gridTemplateColumns: `repeat(${days.length}, minmax(78px, 1fr))` }}>
      {days.map((day) => <div className="scenario-day" key={day.day}>
        <strong>Day {day.day}</strong><small>Hours {day.start}–{day.end}</small>
        <div className="scenario-track scenario-track--sun" aria-label={`Day ${day.day} sunlight changes: ${day.sunlight.length ? day.sunlight.map((period) => `${format(period.scale * 100)}% from hour ${Math.max(period.start_hour, day.start)} to ${Math.min(period.end_hour, day.end)}`).join(", ") : "100% of forecast"}`}>
          <span className="scenario-sun-base" />
          {day.sunlight.map((period, i) => <span key={i} className="scenario-sun-period" style={{ left: `${(Math.max(period.start_hour, day.start) - day.start) / 24 * 100}%`, width: `${(Math.min(period.end_hour, day.end) - Math.max(period.start_hour, day.start)) / 24 * 100}%`, height: `${Math.max(3, period.scale * 28)}px` }} title={`${format(period.scale * 100)}% of forecast sunlight`} />)}
        </div>
        <span className="scenario-day-sun">{day.sunlight.length ? `${format(Math.min(...day.sunlight.map((period) => period.scale)) * 100)}% min.` : "100%"}</span>
        <div className="scenario-track scenario-track--fault" aria-label={`${day.faults.length} equipment faults on day ${day.day}`}>
          {day.faults.map((fault, i) => <span key={i} style={{ left: `${(Math.max(fault.start_hour, day.start) - day.start) / 24 * 100}%`, width: `${Math.max(2, (Math.min(fault.start_hour + (fault.duration_hours ?? scenario.duration_days * 24), day.end) - Math.max(fault.start_hour, day.start)) / 24 * 100)}%`, top: `${Math.min(i, 2) * 4 + 4}px` }} title={`${faultLabel(fault.type)}; hour ${fault.start_hour}`} />)}
        </div>
        <div className="scenario-track scenario-track--delivery" aria-label={`${day.deliveries.length} deliveries on day ${day.day}`}>
          {day.deliveries.map((delivery, i) => <span key={i} style={{ left: `calc(${(delivery.due_hour - day.start) / 24 * 100}% - 2px)` }} title={`${format(delivery.kg)} kg at hour ${delivery.due_hour}`} />)}
        </div>
        <span className="scenario-day-demand">{format(day.deliveries.reduce((total, delivery) => total + delivery.kg, 0))} kg</span>
      </div>)}
    </div></div>
    <p className="scenario-note">Days begin at the weather record’s start. Exact dates appear after checking the scenario. Sunlight percentages are relative to that record.</p>
  </section>;
}

export function Scenarios() {
  const workspace = useWorkspace();
  const navigate = useNavigate();
  const [saved] = useState(restoreDraft);
  const [draft, setDraft] = useState(saved.scenario);
  const [prompt, setPrompt] = useState(saved.prompt);
  const [capabilities, setCapabilities] = useState<ScenarioCapabilities | null>(null);
  const [capabilityStatus, setCapabilityStatus] = useState<"loading" | "ready" | "error">("loading");
  const [reload, setReload] = useState(0);
  const [activity, setActivity] = useState<"generate" | "preview" | "activate" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [preview, setPreview] = useState<ScenarioPreview | null>(null);
  const [controller, setController] = useState<ScenarioController>("MPC");
  const [deliveryKg, setDeliveryKg] = useState(150);
  const [deliveryInterval, setDeliveryInterval] = useState(24);
  const [firstDelivery, setFirstDelivery] = useState(19);
  const dirty = useRef(saved.restored);
  const mounted = useRef(true);
  const request = useRef<AbortController | null>(null);
  const editor = useRef<HTMLFormElement>(null);
  const previewRegion = useRef<HTMLElement>(null);
  const errorRegion = useRef<HTMLDivElement>(null);
  const busy = activity !== null;
  const horizon = draft.duration_days * 24;
  const faults = capabilities?.faults ?? FALLBACK_FAULTS;
  const channels = capabilities?.sensor_channels ?? Object.keys(channelLabels);
  const examples = capabilities?.examples.length ? capabilities.examples : SCENARIO_EXAMPLES;
  const issues = scenarioIssues(draft);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; request.current?.abort(); };
  }, []);

  useEffect(() => {
    const abort = new AbortController();
    setCapabilityStatus("loading");
    void scenarioApi.capabilities(abort.signal).then((result) => {
      if (abort.signal.aborted) return;
      setCapabilities(result);
      setCapabilityStatus("ready");
      if (!dirty.current) setDraft(result.defaults);
    }).catch(() => { if (!abort.signal.aborted) setCapabilityStatus("error"); });
    return () => abort.abort();
  }, [reload]);

  useEffect(() => {
    try {
      // Incomplete numeric input is kept in the form, not persisted as JSON null.
      const json = JSON.stringify({ scenario: draft, prompt }, (_, value: unknown) => {
        if (typeof value === "number" && !Number.isFinite(value)) throw new Error("Incomplete numeric input");
        return value;
      });
      sessionStorage.setItem(STORAGE_KEY, json);
    } catch { /* Editing remains available without browser storage. */ }
  }, [draft, prompt]);

  useEffect(() => { if (error) errorRegion.current?.scrollIntoView({ block: "nearest", behavior: "instant" }); }, [error]);

  function edit(next: ScenarioSpec) {
    dirty.current = true;
    setDraft(next);
    setPreview(null);
    setError(null);
    setNotice(null);
    setWarnings([]);
  }
  function editSunlight(index: number, next: Partial<ScenarioSunlight>) {
    edit({ ...draft, sunlight: draft.sunlight.map((period, i) => i === index ? { ...period, ...next } : period) });
  }
  function editFault(index: number, next: Partial<ScenarioFault>) {
    edit({ ...draft, faults: draft.faults.map((fault, i) => i === index ? { ...fault, ...next } : fault) });
  }
  function loadExample(example: ScenarioExample) {
    edit(structuredClone(example.scenario));
    setPrompt(example.prompt);
    setNotice(`Loaded “${example.title}”. Review the settings below, then check the scenario.`);
  }
  async function generate() {
    const abort = new AbortController();
    request.current?.abort();
    request.current = abort;
    setActivity("generate");
    setError(null);
    setNotice(null);
    try {
      const result = await scenarioApi.generate(prompt.trim(), abort.signal);
      if (abort.signal.aborted) return;
      edit(result.scenario);
      setWarnings(result.warnings);
      setNotice("Description translated. Review the settings and interpretations before running.");
      editor.current?.scrollIntoView({ block: "start", behavior: "instant" });
    } catch (failure) { if (!abort.signal.aborted) setError(failureMessage(failure)); }
    finally { if (!abort.signal.aborted) setActivity(null); }
  }
  async function check(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    // A delayed capabilities refresh must not replace a draft already sent for review.
    dirty.current = true;
    if (issues.length) { setError(issues.join(" ")); return; }
    const abort = new AbortController();
    request.current?.abort();
    request.current = abort;
    setActivity("preview");
    setError(null);
    setNotice(null);
    setPreview(null);
    try {
      const result = await scenarioApi.preview(draft, abort.signal);
      if (abort.signal.aborted) return;
      setDraft(result.scenario);
      setWarnings(result.warnings);
      setPreview(result);
      requestAnimationFrame(() => previewRegion.current?.scrollIntoView({ block: "nearest", behavior: "instant" }));
    } catch (failure) { if (!abort.signal.aborted) setError(failureMessage(failure)); }
    finally { if (!abort.signal.aborted) setActivity(null); }
  }
  async function run() {
    if (!preview) return;
    setActivity("activate");
    setError(null);
    try {
      const accepted = await workspace.activateScenario(preview.preview_id, controller);
      if (!mounted.current) return;
      if (accepted) {
        workspace.setRunning(true);
        navigate("/plant");
      } else {
        setError("The scenario could not start. Check the scenario again to renew its preview, then retry.");
        setPreview(null);
      }
    } catch (failure) { if (mounted.current) setError(failureMessage(failure)); }
    finally { if (mounted.current) setActivity(null); }
  }
  function addSunlight() {
    let start = 0;
    for (const period of [...draft.sunlight].sort((a, b) => a.start_hour - b.start_hour)) {
      if (period.start_hour > start) break;
      start = Math.max(start, period.end_hour);
    }
    if (start >= horizon) { setError("Every hour already has a sunlight adjustment. Edit or remove an existing period."); return; }
    const nextStart = draft.sunlight.filter((period) => period.start_hour > start).sort((a, b) => a.start_hour - b.start_hour)[0]?.start_hour ?? horizon;
    edit({ ...draft, sunlight: [...draft.sunlight, { start_hour: start, end_hour: Math.min(start + 24, nextStart, horizon), scale: 0.5, forecast_known: true }] });
  }
  function replaceDeliveries() {
    const deliveries = recurringDeliveries(draft.duration_days, firstDelivery, deliveryInterval, deliveryKg);
    if (!deliveries.length || !Number.isInteger(firstDelivery * 4) || !Number.isInteger(deliveryInterval * 4)) {
      setError("Choose a positive delivery quantity and 15-minute timing within the run, with no more than 80 deliveries.");
      return;
    }
    edit({ ...draft, deliveries });
    setNotice(`Delivery schedule replaced with ${deliveries.length} deliveries of ${format(deliveryKg)} kg.`);
  }
  function addDelivery() {
    const used = new Set(draft.deliveries.map((delivery) => delivery.due_hour));
    let due = Math.min(horizon, Math.max(0, ...used) + 6);
    if (used.has(due)) {
      due = 6;
      while (used.has(due) && due <= horizon) due += 6;
    }
    if (due > horizon) {
      due = 0.25;
      while (used.has(due) && due <= horizon) due += 0.25;
    }
    edit({ ...draft, deliveries: [...draft.deliveries, { due_hour: due, kg: 150 }] });
  }

  return <div className="scenario-page">
    <div className="scenario-page-heading"><div><h1>Describe a scenario</h1><p>Give the plant a situation to work through. Review the conditions, then watch it unfold in the digital twin.</p></div><Link to="/plant" className="w-button">Back to plant</Link></div>

    <section className="scenario-composer" aria-labelledby="scenario-prompt-label">
      <label id="scenario-prompt-label" htmlFor="scenario-prompt">What should happen?</label>
      <textarea id="scenario-prompt" rows={4} maxLength={4000} value={prompt} disabled={busy} onChange={(event) => {
        setPrompt(event.target.value);
        setPreview(null);
        setNotice("Generate scenario to apply your description, or review and check the current editable settings below.");
      }} placeholder={SCENARIO_EXAMPLES[0].prompt} aria-describedby="scenario-prompt-hint" />
      <div className="scenario-compose-actions"><button className="w-button w-button--primary" type="button" disabled={busy || !prompt.trim() || !capabilities?.llm_available} onClick={() => void generate()}>{activity === "generate" ? "Translating description…" : "Generate scenario"}</button><button className="w-button" type="button" disabled={busy} onClick={() => editor.current?.scrollIntoView({ block: "start", behavior: "instant" })}>Edit settings manually</button><span id="scenario-prompt-hint">Location, sunlight, starting stores, faults and deliveries</span></div>
      <div className="scenario-examples"><span>Prepared examples</span>{examples.map((example) => <button type="button" key={example.title} disabled={busy} onClick={() => loadExample(example)}>{example.title}</button>)}</div>
      {capabilityStatus === "loading" ? <p className="scenario-availability" role="status">Checking natural-language generation… Manual editing is available.</p> : !capabilities?.llm_available || capabilityStatus === "error" ? <p className="scenario-availability" role="status">{capabilityStatus === "error" ? "The scenario service could not be reached." : "Natural-language generation is not configured."} You can still edit a prepared example.{capabilityStatus === "error" && <button className="w-button" type="button" onClick={() => setReload((value) => value + 1)}>Retry connection</button>}</p> : null}
      {capabilityStatus === "ready" && !capabilities?.llm_available && <details className="scenario-details scenario-generation-help"><summary>Enable text generation with OpenRouter</summary><p className="scenario-note">Set <code>OPENROUTER_API_KEY</code> on the server, then restart the backend. The default model is <code>openai/gpt-4.1-mini</code> through OpenRouter. Set <code>PLANT_SCENARIO_MODEL</code> to choose another OpenRouter model that supports strict JSON schema output. The key stays on the server.</p><button type="button" className="w-button" onClick={() => setReload((value) => value + 1)}>Check availability</button></details>}
    </section>

    {notice && <p className="scenario-notice" role="status">{notice}</p>}
    {error && <div ref={errorRegion} className="scenario-error" role="alert"><strong>Check the scenario</strong><p>{error}</p></div>}

    <form ref={editor} className="scenario-editor" onSubmit={(event) => void check(event)} aria-busy={busy}>
      <fieldset disabled={busy} className="scenario-editor-body">
        <legend className="scenario-editor-title">Review and edit</legend>
        <div className="scenario-draft-heading"><label className="scenario-field scenario-name"><span>Scenario name</span><input value={draft.name} maxLength={80} required onChange={(event) => edit({ ...draft, name: event.target.value })} /></label><NumberField label="Duration (days)" value={draft.duration_days} min={1} max={10} onChange={(value) => {
          if (Number.isInteger(value) && value >= 1 && value <= 10) {
            edit(resizeScenario(draft, value));
            if (value < draft.duration_days) setNotice("The shorter duration removes later deliveries and faults, and trims periods at the new end. Review the schedule below.");
          } else edit({ ...draft, duration_days: value });
        }} /></div>
        <label className="scenario-field scenario-description"><span>Scenario summary</span><textarea rows={2} value={draft.description} maxLength={800} required onChange={(event) => edit({ ...draft, description: event.target.value })} /></label>

        {draft.unsupported.length > 0 && <section className="scenario-unsupported" aria-labelledby="scenario-unsupported-title"><h2 id="scenario-unsupported-title">These requests need a different model</h2><ul>{draft.unsupported.map((item, i) => <li key={i}>{item}</li>)}</ul><p>They cannot run in this twin. Edit the description and generate again, or continue with only the supported settings shown below.</p><button type="button" className="w-button" onClick={() => {
          edit({ ...draft, unsupported: [] });
          setNotice("Continuing with the supported settings. The excluded requests will not be simulated.");
        }}>Use supported settings</button></section>}

        <div className="scenario-layout"><div className="scenario-settings">
          <section className="scenario-section"><h2>Location & plant</h2><div className="scenario-fields">
            <label className="scenario-field scenario-full"><span>Weather source</span><select value={draft.site.source === "live" ? "live" : draft.site.example_id ?? "trieste"} onChange={(event) => {
              const value = event.target.value;
              edit({ ...draft, site: value === "live" ? { ...draft.site, source: "live", example_id: null } : { ...SAVED_SCENARIO_SITES[value as SavedScenarioSite] } });
            }}>{Object.entries(SAVED_SCENARIO_SITES).map(([id, site]) => <option key={id} value={id}>Saved weather · {site.name}</option>)}<option value="live">Live forecast · custom location</option></select></label>
            {draft.site.source === "live" ? <><label className="scenario-field scenario-full"><span>Location name</span><input value={draft.site.name} maxLength={100} required onChange={(event) => edit({ ...draft, site: { ...draft.site, name: event.target.value } })} /></label><NumberField label="Latitude (°N)" value={draft.site.latitude} step="any" min={34} max={72} onChange={(value) => edit({ ...draft, site: { ...draft.site, latitude: value } })} /><NumberField label="Longitude (°E)" value={draft.site.longitude} step="any" min={-25} max={45} onChange={(value) => edit({ ...draft, site: { ...draft.site, longitude: value } })} /><p className="scenario-note scenario-full">Live forecasts support European locations. Saved weather also works when the weather service is unavailable.</p></> : <p className="scenario-note scenario-full">{draft.site.name} · {format(draft.site.latitude, 4)}°N, {format(draft.site.longitude, 4)}°E. Saved weather gives a repeatable run.</p>}
            <NumberField label="Solar capacity (kW)" value={draft.solar_kw} min={100} max={50000} onChange={(value) => edit({ ...draft, solar_kw: value })} />
            <NumberField label="Battery capacity (kWh)" value={draft.battery_kwh} min={100} max={100000} onChange={(value) => edit({ ...draft, battery_kwh: value })} />
          </div></section>

          <section className="scenario-section"><h2>Starting inventories</h2><div className="scenario-fields">
            <NumberField label="Battery charge (%)" value={Math.round(draft.initial.battery_soc * 10000) / 100} min={0} max={100} step="any" hint={`${format(draft.initial.battery_soc * draft.battery_kwh)} kWh stored`} onChange={(value) => edit({ ...draft, initial: { ...draft.initial, battery_soc: value / 100 } })} />
            {([ ["h2_kg", "Hydrogen", 2000], ["co2_kg", "CO₂", 6000], ["ch4_kg", "Methane", 3000] ] as const).map(([key, name, capacity]) => <NumberField key={key} label={`${name} (kg)`} value={draft.initial[key]} min={0} max={capacity} step="any" hint={`${format(draft.initial[key] / capacity * 100, 1)}% of ${format(capacity)} kg capacity`} onChange={(value) => edit({ ...draft, initial: { ...draft.initial, [key]: value } })} />)}
          </div></section>

          <section className="scenario-section"><h2>Economic assumptions</h2><div className="scenario-fields"><NumberField label="Delivery value (€/kg)" value={draft.economics.delivered_service_value_eur_per_kg} min={0} max={10000} step="any" onChange={(value) => edit({ ...draft, economics: { ...draft.economics, delivered_service_value_eur_per_kg: value } })} /><NumberField label="Missed delivery cost (€/kg)" value={draft.economics.unmet_delivery_penalty_eur_per_kg} min={0.01} max={100000} step="any" onChange={(value) => edit({ ...draft, economics: { ...draft.economics, unmet_delivery_penalty_eur_per_kg: value } })} /></div><p className="scenario-note">These values change the economic results. The controller prioritizes meeting deliveries. This plant operates off-grid.</p></section>
          <details className="scenario-details"><summary>Repeatability & model limits</summary><NumberField label="Simulation seed" value={draft.seed} min={0} max={2147483647} onChange={(value) => edit({ ...draft, seed: value })} /><p className="scenario-note">Use the same settings, saved weather and seed to reproduce a scenario. Weather changes production through sunlight; temperature, wind and humidity do not add equipment effects.</p></details>
        </div><div className="scenario-schedule">
          <ScheduleTimeline scenario={draft} />

          <section className="scenario-section"><div className="scenario-section-heading"><div><h2>Sunlight schedule</h2><p>Unchanged hours follow the selected weather.</p></div><button className="w-button" type="button" disabled={draft.sunlight.length >= 16} onClick={addSunlight}>Add period</button></div>
            {draft.sunlight.length === 0 && <p className="scenario-empty">Sunlight follows the weather throughout this run.</p>}
            {draft.sunlight.map((period, i) => <div className="scenario-schedule-row" key={i}><div className="scenario-row-heading"><strong>Period {i + 1}</strong><button className="scenario-remove" type="button" aria-label={`Remove sunlight period ${i + 1}`} onClick={() => edit({ ...draft, sunlight: draft.sunlight.filter((_, index) => index !== i) })}>Remove</button></div><div className="scenario-row-fields"><NumberField label="From hour" value={period.start_hour} min={0} max={horizon - 1} hint={scenarioOffset(period.start_hour)} onChange={(value) => editSunlight(i, { start_hour: value })} /><NumberField label="Until hour" value={period.end_hour} min={1} max={horizon} hint={scenarioOffset(period.end_hour)} onChange={(value) => editSunlight(i, { end_hour: value })} /><NumberField label="Sunlight remaining (%)" value={Math.round(period.scale * 10000) / 100} min={0} max={100} step="any" hint={`${format((1 - period.scale) * 100)}% less than forecast`} onChange={(value) => editSunlight(i, { scale: value / 100 })} /></div><label className="scenario-check"><input type="checkbox" checked={period.forecast_known} onChange={(event) => editSunlight(i, { forecast_known: event.target.checked })} /><span>Include this change in the controller’s forecast<small>Uncheck to make the sunlight reduction an unexpected disturbance.</small></span></label></div>)}
          </section>

          <section className="scenario-section"><div className="scenario-section-heading"><div><h2>Equipment faults</h2><p>Scheduled from the start of the run.</p></div><button className="w-button" type="button" disabled={draft.faults.length >= 12} onClick={() => edit({ ...draft, faults: [...draft.faults, { type: "dac_outage", severity: 1, start_hour: Math.min(72, Math.max(0, horizon - 6)), duration_hours: Math.min(6, horizon), channel: null }] })}>Add fault</button></div>
            {draft.faults.length === 0 && <p className="scenario-empty">Equipment begins healthy, with no scheduled faults.</p>}
            {draft.faults.map((fault, i) => <div className="scenario-schedule-row" key={i}><div className="scenario-row-heading"><strong>Fault {i + 1}</strong><button className="scenario-remove" type="button" aria-label={`Remove fault ${i + 1}`} onClick={() => edit({ ...draft, faults: draft.faults.filter((_, index) => index !== i) })}>Remove</button></div><label className="scenario-field"><span>Fault type</span><select value={fault.type} onChange={(event) => {
              const kind = faults.find((item) => item.type === event.target.value)!;
              editFault(i, { type: kind.type, severity: kind.default_severity, channel: kind.type === "sensor_drift" ? channels[0] : null });
            }}>{faults.map((kind) => <option value={kind.type} key={kind.type}>{faultLabel(kind.type)}</option>)}</select></label><div className="scenario-row-fields"><NumberField label="Starts at hour" value={fault.start_hour} min={0} max={horizon - 0.25} step={0.25} hint={scenarioOffset(fault.start_hour)} onChange={(value) => editFault(i, { start_hour: value })} /><NumberField label="Duration (hours)" value={fault.duration_hours ?? Math.max(0.25, horizon - fault.start_hour)} min={0.25} max={horizon - fault.start_hour} step={0.25} disabled={fault.duration_hours === null} onChange={(value) => editFault(i, { duration_hours: value })} /><NumberField label={fault.type === "sensor_drift" ? "Reading bias (%)" : "Severity (%)"} value={Math.round(fault.severity * 10000) / 100} min={0} max={fault.type === "sensor_drift" || fault.type.endsWith("_outage") ? 100 : 95} step="any" disabled={fault.type.endsWith("_outage")} onChange={(value) => editFault(i, { severity: value / 100 })} /></div>{fault.type === "sensor_drift" && <label className="scenario-field"><span>Sensor</span><select value={fault.channel ?? channels[0]} onChange={(event) => editFault(i, { channel: event.target.value })}>{channels.map((channel) => <option key={channel} value={channel}>{channelLabels[channel] ?? channel}</option>)}</select></label>}<label className="scenario-check"><input type="checkbox" checked={fault.duration_hours === null} onChange={(event) => editFault(i, { duration_hours: event.target.checked ? null : Math.min(6, horizon - fault.start_hour) })} /><span>Lasts until the end of the run</span></label>{fault.type === "sensor_drift" && <p className="scenario-note">Bias adds a percentage of the sensor’s full scale to its reading. Stored quantities are unchanged.</p>}</div>)}
          </section>

          <section className="scenario-section"><div className="scenario-section-heading"><div><h2>Methane deliveries</h2><p>{draft.deliveries.length} deliveries · {format(draft.deliveries.reduce((total, delivery) => total + delivery.kg, 0))} kg requested</p></div><button className="w-button" type="button" disabled={draft.deliveries.length >= 80} onClick={addDelivery}>Add delivery</button></div>
            <details className="scenario-details"><summary>Set a recurring delivery schedule</summary><div className="scenario-row-fields"><NumberField label="Each delivery (kg)" value={deliveryKg} min={0.01} max={100000} step="any" onChange={setDeliveryKg} /><NumberField label="First at hour" value={firstDelivery} min={0.25} max={horizon} step={0.25} onChange={setFirstDelivery} /><NumberField label="Repeat every (hours)" value={deliveryInterval} min={0.25} max={horizon} step={0.25} onChange={setDeliveryInterval} /></div><button className="w-button" type="button" onClick={replaceDeliveries}>Replace delivery schedule</button><p className="scenario-note">Replaces the list below, through the end of this run.</p></details>
            {draft.deliveries.length === 0 ? <p className="scenario-empty">No methane deliveries are requested.</p> : <div className="scenario-delivery-list"><table><thead><tr><th scope="col">Due at hour</th><th scope="col">Methane (kg)</th><th scope="col"><span className="scenario-sr-only">Action</span></th></tr></thead><tbody>{draft.deliveries.map((delivery, i) => <tr key={i}><td><input type="number" aria-label={`Delivery ${i + 1} due at hour`} value={Number.isFinite(delivery.due_hour) ? delivery.due_hour : ""} min={0.25} max={horizon} step={0.25} required onChange={(event) => edit({ ...draft, deliveries: draft.deliveries.map((item, index) => index === i ? { ...item, due_hour: event.target.valueAsNumber } : item) })} /><small>{scenarioOffset(delivery.due_hour)}</small></td><td><input type="number" aria-label={`Delivery ${i + 1} methane kilograms`} value={Number.isFinite(delivery.kg) ? delivery.kg : ""} min={0.01} max={100000} step="any" required onChange={(event) => edit({ ...draft, deliveries: draft.deliveries.map((item, index) => index === i ? { ...item, kg: event.target.valueAsNumber } : item) })} /></td><td><button className="scenario-remove" type="button" aria-label={`Remove delivery ${i + 1}`} onClick={() => edit({ ...draft, deliveries: draft.deliveries.filter((_, index) => index !== i) })}>Remove</button></td></tr>)}</tbody></table></div>}
          </section>
        </div></div>
      </fieldset>

      {(draft.notes.length > 0 || warnings.length > 0) && <section className="scenario-interpretations"><h2>Interpretations & assumptions</h2><ul>{[...new Set([...draft.notes, ...warnings])].map((note, i) => <li key={i}>{note}</li>)}</ul></section>}
      <section ref={previewRegion} className={`scenario-review${preview ? " scenario-review--ready" : ""}`} aria-labelledby="scenario-review-title">
        <div className="scenario-section-heading"><div><h2 id="scenario-review-title">{preview ? "Ready to run" : "Check before running"}</h2><p>{preview ? "The weather and settings below are fixed for this run." : "Validate the settings and prepare the weather for your scenario."}</p></div><button type="submit" className={`w-button${preview ? "" : " w-button--primary"}`} disabled={busy || issues.length > 0}>{activity === "preview" ? "Checking scenario…" : preview ? "Check again" : "Check scenario"}</button></div>
        {!preview && issues.length > 0 && <ul className="scenario-validation" role="status">{issues.map((issue) => <li key={issue}>{issue}</li>)}</ul>}
        {preview && <><dl className="scenario-preview-facts"><div><dt>Location</dt><dd>{preview.forecast.site.name}</dd></div><div><dt>Simulated dates (UTC)</dt><dd>{date(preview.summary.start_time)} – {date(preview.summary.end_time)}</dd></div><div><dt>Delivery commitment</dt><dd>{format(preview.summary.total_delivery_kg)} kg in {preview.summary.delivery_count} deliveries</dd></div><div><dt>Weather</dt><dd>{preview.scenario.site.source === "saved" ? "Saved weather" : "Downloaded forecast"} · {preview.forecast.source.provider}</dd></div></dl><div className="scenario-launch"><label className="scenario-field"><span>Controller</span><select value={controller} disabled={busy} onChange={(event) => setController(event.target.value as ScenarioController)}><option value="MPC">Forecast planner (MPC)</option><option value="H">Reactive baseline</option></select></label><button type="button" className="w-button w-button--primary" disabled={busy} onClick={() => void run()}>{activity === "activate" ? "Starting simulation…" : "Run in digital twin"}</button><p>Starts a new simulation in the workspace with these initial inventories and scheduled events.</p></div></>}
      </section>
    </form>
  </div>;
}
