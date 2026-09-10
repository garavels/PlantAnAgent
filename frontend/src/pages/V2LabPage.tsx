import { Component, lazy, Suspense, useEffect, useId, useRef, useState } from 'react'
import type { ReactNode } from 'react'

import { PlantFlow } from '../components/overview/PlantFlow'
import { PageHeader } from '../components/shell/PageHeader'
import { useV2Demo } from '../hooks/useV2Demo'
import type { V2DemoFrame, V2DemoSnapshot } from '../types/v2'
import './plant-3d.css'
import './v2-lab.css'

const PlantScene = lazy(() => import('../components/plant-3d/PlantScene').then((module) => ({ default: module.PlantScene })))
const number = (value: number, digits = 1) => value.toLocaleString('en-GB', { maximumFractionDigits: digits })
const signed = (value: number, digits = 1) => {
  const magnitude = Math.round(Math.abs(value) * 10 ** digits) / 10 ** digits
  return `${magnitude === 0 ? '' : value > 0 ? '+' : '−'}${number(magnitude, digits)}`
}
const euro = (value: number) => {
  const magnitude = Math.round(Math.abs(value) * 100) / 100
  return `${magnitude > 0 && value < 0 ? '−' : ''}€${number(magnitude, 2)}`
}
const elapsed = (hours: number) => {
  const minutes = Math.round(hours * 60)
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`
}

interface Series {
  label: string
  color: string
  points: { x: number; y: number }[]
  dashed?: boolean
  step?: boolean
}

/** Data-only SVG; no smoothing that would invent intermediate measurements. */
function Plot({ title, unit, series, band, xLabel = 'Hours after issue', compact = false }: {
  title: string
  unit: string
  series: Series[]
  band?: { x: number; low: number; high: number }[]
  xLabel?: string
  compact?: boolean
}) {
  const titleId = useId()
  const values = series.flatMap((line) => line.points)
  const hasValues = values.length > 0
  const container = useRef<HTMLElement | null>(null)
  const [width, setWidth] = useState(600)
  useEffect(() => {
    const element = container.current
    if (!element) return
    const observer = new ResizeObserver((entries) => {
      const measured = entries[0]?.contentRect.width
      if (measured) setWidth(Math.max(240, Math.round(measured)))
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [hasValues])
  if (!hasValues) return <p className="v2__empty">No {title.toLowerCase()} published at this boundary.</p>
  const minX = Math.min(...values.map((point) => point.x))
  const maxX = Math.max(minX + 0.25, ...values.map((point) => point.x))
  const low = Math.min(0, ...values.map((point) => point.y), ...(band?.map((point) => point.low) ?? []))
  const high = Math.max(1, ...values.map((point) => point.y), ...(band?.map((point) => point.high) ?? []))
  const pad = (high - low) * 0.08
  const left = 62
  const right = width - 20
  const top = 18
  const bottom = compact ? 108 : 166
  const x = (value: number) => left + ((value - minX) / (maxX - minX)) * (right - left)
  const y = (value: number) => bottom - ((value - low + pad) / (high - low + 2 * pad)) * (bottom - top)
  const path = (points: Series['points'], step = false) => points.map((point, index) =>
    `${index === 0 ? 'M' : step ? `H${x(point.x)} V` : 'L'}${index > 0 && step ? y(point.y) : `${x(point.x)},${y(point.y)}`}`,
  ).join(' ')
  const bandPath = band?.length ? `${path(band.map((point) => ({ x: point.x, y: point.high })))} ${band.slice().reverse().map((point) => `L${x(point.x)},${y(point.low)}`).join(' ')} Z` : null
  const yTicks = [low, (low + high) / 2, high]
  const xTicks = [minX, (minX + maxX) / 2, maxX]
  return (
    <figure ref={container} className={`v2__plot${compact ? ' v2__plot--compact' : ''}`}>
      <svg viewBox={`0 0 ${width} ${bottom + 50}`} role="img" aria-labelledby={titleId}>
        <title id={titleId}>{title}. Vertical axis: {unit}. Horizontal axis: {xLabel}. {series.map((line) => line.label).join(', ')}.</title>
        {yTicks.map((tick, index) => <g key={index}>
          <line x1={left} x2={right} y1={y(tick)} y2={y(tick)} className="v2__gridline" />
          <text x={left - 10} y={y(tick) + 4} textAnchor="end">{number(tick, 0)}</text>
        </g>)}
        {bandPath && <path d={bandPath} className="v2__forecast-band" />}
        {series.map((line) => <g key={line.label}>
          <path d={path(line.points, line.step)} fill="none" stroke={line.color} strokeWidth={2.2} strokeDasharray={line.dashed ? '5 4' : undefined} vectorEffect="non-scaling-stroke" />
          {line.points.length === 1 && <circle cx={x(line.points[0].x)} cy={y(line.points[0].y)} r={4} fill={line.color} />}
        </g>)}
        {xTicks.map((tick, index) => <text key={index} x={x(tick)} y={bottom + 23} textAnchor={index === 0 ? 'start' : index === 2 ? 'end' : 'middle'}>{number(tick, 1)}</text>)}
        <text x={right} y={bottom + 43} textAnchor="end">{xLabel}</text>
        <text x={left} y={12}>{unit}</text>
      </svg>
      <figcaption className="v2__legend">
        {series.map((line) => <span key={line.label}><i style={{ background: line.color, opacity: line.dashed ? 0.5 : 1 }} />{line.label}</span>)}
        {band && <span><i className="v2__legend-band" />p10–p90 range</span>}
      </figcaption>
    </figure>
  )
}

class SceneBoundary extends Component<{ children: ReactNode; fallback: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() {
    return this.state.failed ? <><p className="v2__scene-warning">3D is unavailable. The process diagram below uses the same snapshot.</p>{this.props.fallback}</> : this.props.children
  }
}

function Forecasts({ frame }: { frame: V2DemoFrame }) {
  const { forecasts } = frame
  const solarSeries = (key: 'mean_kw' | 'p10_kw') => forecasts.solar.flatMap((point) => [
    { x: point.offset_hours, y: point[key] }, { x: point.end_offset_hours, y: point[key] },
  ])
  return (
    <section className="v2__forecast-section" aria-labelledby="v2-forecasts">
      <div className="v2__section-head">
        <div><h3 id="v2-forecasts">What is known ahead</h3><p>Public forecasts issued at {elapsed(forecasts.issued_at_hours)} elapsed. All offsets start at that issue time.</p></div>
        <span className="v2__annotation">Forecasts, not future observations</span>
      </div>
      <div className="v2__forecast-pair">
        <article className="v2__forecast">
          <header><h4>Solar availability</h4><span>Aggregate forecast intervals</span></header>
          <Plot title="Public solar forecast" unit="kW, interval mean" series={[
            { label: 'Mean', color: 'var(--solar)', points: solarSeries('mean_kw') },
            { label: 'p10 lower estimate', color: 'var(--text-dim)', points: solarSeries('p10_kw'), dashed: true },
          ]} />
          <p className="v2__note">The public solar surface supplies a mean and p10 for each interval. No upper quantile is published.</p>
          <details className="v2__data"><summary>Solar forecast values</summary><div className="v2__table-scroll"><table><thead><tr><th>Offset after issue</th><th>Mean kW</th><th>p10 kW</th></tr></thead><tbody>{forecasts.solar.map((point) => <tr key={point.offset_hours}><th>+{number(point.offset_hours)}–{number(point.end_offset_hours)} h</th><td>{number(point.mean_kw)}</td><td>{number(point.p10_kw)}</td></tr>)}</tbody></table></div></details>
        </article>
        <article className="v2__forecast">
          <header><h4>Public market</h4><span>Current price {euro(forecasts.price_current_eur_mwh)} / MWh</span></header>
          <Plot title="Public market price forecast" unit="EUR / MWh" series={[
            { label: 'Mean price', color: 'var(--co2)', points: forecasts.prices.map((point) => ({ x: point.offset_hours, y: point.mean })) },
          ]} band={forecasts.prices.map((point) => ({ x: point.offset_hours, low: point.p10, high: point.p90 }))} />
          <p className="v2__note">The shaded range shows the published p10 and p90 estimates. Future realized prices are not exposed.</p>
          <details className="v2__data"><summary>Market forecast values</summary><div className="v2__table-scroll"><table><thead><tr><th>Offset after issue</th><th>Mean</th><th>p10</th><th>p90</th></tr></thead><tbody>{forecasts.prices.map((point) => <tr key={point.offset_hours}><th>+{number(point.offset_hours)} h</th><td>{number(point.mean, 2)}</td><td>{number(point.p10, 2)}</td><td>{number(point.p90, 2)}</td></tr>)}</tbody></table></div></details>
        </article>
      </div>
    </section>
  )
}

function Schedule({ snapshot }: { snapshot: V2DemoSnapshot }) {
  const { frame, controller } = snapshot
  const plan = frame.plan
  const lines = [
    { key: 'electrolyser_kw', label: 'Electrolyser', color: 'var(--feed)' },
    { key: 'dac_kw', label: 'DAC', color: 'var(--co2)' },
    { key: 'methanator_kw', label: 'Methanator', color: 'var(--ch4)' },
    { key: 'battery_charge_kw', label: 'Battery charge', color: 'var(--stored)' },
  ] as const
  return (
    <section className="v2__schedule" aria-labelledby="v2-schedule">
      <div className="v2__section-head"><div><h3 id="v2-schedule">The controller’s forward schedule</h3><p>{plan ? `Retained MPC plan issued at ${elapsed(plan.issued_at_hours)} elapsed. Targets can change at the next decision.` : controller.id === 'H' ? 'H is forecast-blind. It acts on the current observation and publishes no forward schedule.' : snapshot.status === 'ready' ? 'No MPC plan is available at this boundary. Advance a decision to request the controller’s first plan.' : 'No retained MPC plan is available in this snapshot.'}</p></div><span className="v2__annotation">{controller.label}</span></div>
      {plan && <Plot title="Actual retained MPC plan" unit="kW target" series={lines.map((line) => ({ label: line.label, color: line.color, step: true, points: plan.points.map((point) => ({ x: point.offset_hours, y: point[line.key] })) }))} />}
      {plan && <details className="v2__data"><summary>Published schedule values in kW</summary><div className="v2__table-scroll"><table><thead><tr><th>Offset after issue</th>{lines.map((line) => <th key={line.key}>{line.label}</th>)}</tr></thead><tbody>{plan.points.map((point) => <tr key={point.offset_hours}><th>+{number(point.offset_hours)} h</th>{lines.map((line) => <td key={line.key}>{number(point[line.key])}</td>)}</tr>)}</tbody></table></div></details>}
    </section>
  )
}

function Deliveries({ snapshot }: { snapshot: V2DemoSnapshot }) {
  const { delivery, clock, cycle_settlements } = snapshot.frame
  const [showAll, setShowAll] = useState(false)
  const tableId = useId()
  const settled = delivery.commitments.filter((item) => item.status !== 'pending')
  const pending = delivery.commitments.filter((item) => item.status === 'pending')
  const visible = showAll ? delivery.commitments : [...settled.slice(-4), ...pending.slice(0, 4)]
  const shortfalls = delivery.commitments.filter((item) => item.status === 'shortfall').length
  return (
    <section className="v2__deliveries" aria-labelledby="v2-deliveries">
      <div className="v2__section-head"><div><h3 id="v2-deliveries">Delivery commitments</h3><p>{number(delivery.delivered_kg)} kg delivered; {number(delivery.unmet_kg)} kg settled shortfall across {shortfalls} {shortfalls === 1 ? 'delivery' : 'deliveries'}.</p></div><span className="v2__annotation">Due times are elapsed hours</span></div>
      <div className="v2__delivery-ruler" aria-label={`${elapsed(clock.elapsed_hours)} elapsed of ${snapshot.environment.horizon_hours} hours`}>
        <span style={{ width: `${Math.min(100, (clock.elapsed_hours / snapshot.environment.horizon_hours) * 100)}%` }} />
        {delivery.commitments.map((item, index) => <i key={index} className={`v2__deadline v2__deadline--${item.status}`} style={{ left: `${(item.due_hours / snapshot.environment.horizon_hours) * 100}%` }} title={`${number(item.due_hours)} h: ${item.status}`} />)}
      </div>
      <div className="v2__ruler-labels"><span>0 h</span><span>{snapshot.environment.horizon_hours} h</span></div>
      <div className="v2__delivery-window"><span>{showAll ? `All ${delivery.commitments.length} commitments, including every settlement.` : `Showing ${Math.min(4, settled.length)} recent settlements and ${Math.min(4, pending.length)} next commitments.`}</span><button className="pill" aria-expanded={showAll} aria-controls={tableId} onClick={() => setShowAll(!showAll)}>{showAll ? 'Show nearest and recent' : `Show all ${delivery.commitments.length} commitments`}</button></div>
      <div id={tableId} className="v2__table-scroll v2__delivery-table"><table><thead><tr><th>Due at</th><th>Required</th><th>Delivered</th><th>Shortfall</th><th>Settlement</th></tr></thead><tbody>
        {visible.map((item, index) => <tr key={index} className={item.status === 'pending' && item.due_hours === delivery.next_due_hours ? 'v2__next-row' : undefined}><th>{number(item.due_hours)} h{item.due_hours === delivery.next_due_hours && <span className="v2__inline-note"> next</span>}</th><td>{number(item.requested_kg)} kg</td><td>{item.delivered_kg === null ? '—' : `${number(item.delivered_kg)} kg`}</td><td>{item.unmet_kg === null ? '—' : `${number(item.unmet_kg)} kg`}</td><td><span className={`v2__settlement v2__settlement--${item.status}`}>{item.status === 'met' ? 'Met' : item.status === 'shortfall' ? 'Shortfall' : 'Pending'}</span></td></tr>)}
      </tbody></table></div>
      <div className="v2__cycles"><span>Settled daily returns</span>{cycle_settlements.length ? cycle_settlements.map((cycle) => <span key={cycle.cycle_index} className={cycle.feasible ? 'v2__positive' : 'v2__failure'}>Cycle {cycle.cycle_index}: {cycle.feasible ? 'met' : 'failed'}</span>) : <span>No cycle has settled yet.</span>}</div>
    </section>
  )
}

function Decision({ frame }: { frame: V2DemoFrame }) {
  const decision = frame.decision
  return (
    <section className="v2__decision" aria-labelledby="v2-decision">
      <div className="v2__section-head"><div><h3 id="v2-decision">Requested, dispatched, guarded</h3><p>{decision ? `Last decision at ${elapsed(decision.at_hours)} elapsed. Applied power is the mean over its physics steps.` : 'No controller decision has executed in this run.'}</p></div></div>
      {decision ? <>
        <div className="v2__table-scroll"><table><thead><tr><th>Target</th><th>Requested kW</th><th>Applied mean kW</th></tr></thead><tbody>{decision.targets.map((target) => <tr key={target.id}><th>{target.label}</th><td>{number(target.requested_kw)}</td><td>{number(target.applied_mean_kw)}</td></tr>)}</tbody></table></div>
        <div className="v2__decision-facts"><span><strong>{number(decision.guard_adjustments, 0)}</strong> guard adjustments</span><span><strong>{number(decision.computation_ms, 1)} ms</strong> controller computation</span></div>
        <p className="v2__note">Battery charging can include recovered spill above the requested charge ceiling. A dispatch difference alone does not imply a guard intervention.</p>
        {decision.notes.length > 0 && <><p className="v2__note">Controller intent and execution notes. Applied quantities above are authoritative.</p><ul className="v2__decision-notes">{decision.notes.map((note, index) => <li key={index}>{note}</li>)}</ul></>}
      </> : <p className="v2__empty">Advance 15 minutes to inspect a real command and its applied result.</p>}
    </section>
  )
}

function Economics({ frame }: { frame: V2DemoFrame }) {
  return (
    <section className="v2__economics" aria-labelledby="v2-economics">
      <div className="v2__section-head"><div><h3 id="v2-economics">Realized economics</h3><p>Signed contributions since this run began.</p></div><strong className="v2__net mono">{euro(frame.economics.net_value_eur)}</strong></div>
      <dl className="v2__ledger">{frame.economics.entries.map((entry) => <div key={entry.id}><dt>{entry.label}</dt><dd className="mono">{entry.value_eur > 0 ? '+' : ''}{euro(entry.value_eur)}</dd></div>)}<div className="v2__ledger-total"><dt>Net simulated value</dt><dd className="mono">{euro(frame.economics.net_value_eur)}</dd></div></dl>
      <p className="v2__note">{frame.economics.assumption_note}</p>
    </section>
  )
}

function Evidence({ snapshot }: { snapshot: V2DemoSnapshot }) {
  const { environment, frame } = snapshot
  return (
    <details className="v2__evidence">
      <summary><span>Run evidence and research status</span><span className="v2__annotation">Public identifiers only</span></summary>
      <p>This interactive fixture is for inspection. It is not a registered experiment, controller comparison or claim of commercial savings.</p>
      <dl className="v2__evidence-grid">
        <div><dt>Run</dt><dd>{snapshot.run_id}</dd></div><div><dt>Revision</dt><dd>{snapshot.revision}</dd></div>
        <div><dt>Environment</dt><dd>{environment.id}</dd></div><div><dt>Fixture seed</dt><dd>{environment.fixture_seed}</dd></div>
        <div><dt>Interface</dt><dd>{environment.observation_count} observations; {environment.action_count} actions</dd></div><div><dt>Schema</dt><dd>{snapshot.schema}</dd></div>
        <div><dt>Interface digest</dt><dd>{environment.interface_digest}</dd></div><div><dt>Task digest</dt><dd>{environment.task_digest}</dd></div>
        {frame.decision && <><div><dt>Command digest</dt><dd>{frame.decision.command_digest}</dd></div><div><dt>Execution digest</dt><dd>{frame.decision.execution_digest}</dd></div></>}
      </dl>
      <div className="v2__research-status"><p><strong>PPO unavailable in this lab.</strong> No registered beta policy is presented as trained, selected or validated here.</p><p><strong>HEPA unavailable in V2 control.</strong> No learned risk probabilities or remaining-life predictions are shown.</p></div>
    </details>
  )
}

export function V2LabPage() {
  const { snapshot, pending, error, stale, refresh, advance, reset } = useV2Demo()
  const [view, setView] = useState<'diagram' | '3d'>('diagram')
  const [selected, setSelected] = useState<V2DemoSnapshot['controller']['id'] | null>(null)
  const busy = pending !== null

  if (!snapshot) return <div className="v2 v2--loading">
    <PageHeader title="V2 Lab" />
    <div className="v2__notice"><span>Synthetic simulation</span><span>Not a registered benchmark</span></div>
    <section className="v2__initial" aria-busy={busy}><h3>{busy ? 'Opening the V2 plant' : 'The V2 snapshot is unavailable'}</h3><p>{busy ? 'Reading the current fixture. The clock advances only when you request it.' : error}</p>{!busy && <button className="pill pill--on" onClick={() => void refresh()}>Retry snapshot</button>}</section>
  </div>

  const { frame, environment, controller } = snapshot
  const { clock, delivery } = frame
  const nextController = selected ?? controller.id
  const selectedOption = snapshot.available_controllers.find((option) => option.id === nextController)
  const remaining = clock.decision_count - clock.decision_index
  const canAdvance = !busy && !stale && snapshot.status === 'ready'
  const hoursToDelivery = delivery.next_due_hours === null ? null : Math.max(0, delivery.next_due_hours - clock.elapsed_hours)
  const diagram = <div className="v2__diagram"><PlantFlow state={frame.state} config={snapshot.config} /></div>

  return (
    <div className="v2" aria-busy={busy}>
      <PageHeader title="V2 Lab"><span className="v2__clock mono">{elapsed(clock.elapsed_hours)} <small>elapsed</small></span><span className="v2__header-cycle">Cycle {clock.cycle_index} / {clock.cycle_count}</span></PageHeader>
      <div className="v2__notice"><span>Synthetic simulation</span><span>Not a registered benchmark</span><span>{environment.decision_minutes}-minute decisions / {environment.physics_minutes}-minute physics</span></div>
      <section className="v2__transport" aria-label="Manual simulation controls">
        <div className="v2__advance"><div><strong>{controller.label}</strong><span>{controller.uses_forecast ? 'Uses the public forecasts' : 'Forecast-blind controller'}</span></div><button className="pill pill--on" disabled={!canAdvance || remaining < 1} onClick={() => void advance(1)}>Advance {environment.decision_minutes} min</button><button className="pill" disabled={!canAdvance || remaining < 4} onClick={() => void advance(4)}>Advance 1 hour</button></div>
        <div className="v2__reset"><label htmlFor="v2-controller">Controller for new run</label><select id="v2-controller" value={nextController} disabled={busy} onChange={(event) => setSelected(event.target.value as V2DemoSnapshot['controller']['id'])}>{snapshot.available_controllers.map((option) => <option key={option.id} value={option.id} disabled={!option.available}>{option.label}{!option.available ? ' (unavailable)' : ''}</option>)}</select><button className="pill" disabled={busy || stale || !selectedOption?.available} onClick={() => void reset(nextController)}>Reset and start new run</button></div>
        <div className="v2__transport-note" role="status" aria-live="polite"><span>{pending === 'step' ? 'Computing the requested decisions…' : pending === 'reset' ? 'Starting a new fixture run…' : pending === 'refresh' ? 'Refreshing the confirmed state…' : `Decision ${clock.decision_index} of ${clock.decision_count}. The clock is paused.`}</span><span>{selectedOption?.reason ?? 'Changing controller takes effect only when you reset.'}</span></div>
      </section>
      {(error || snapshot.error) && <div className="v2__error" role="alert"><div><strong>{stale ? 'Refresh needed before continuing' : 'The run needs attention'}</strong><p>{error ?? snapshot.error}</p>{stale && <p>The displayed snapshot is the last confirmed state. A failed request may have changed the server state.</p>}</div><button className="pill" disabled={busy} onClick={() => void refresh()}>Refresh snapshot</button></div>}
      {snapshot.status === 'complete' && <div className="v2__completion" role="status">The {environment.horizon_hours}-hour fixture is complete. Inspect its deliveries, cycle settlements and realized economics below, or reset to start a new run.</div>}
      <section className="v2__hero" aria-label="Current plant and production obligation">
        <div className="v2__plant">
          <div className="v2__plant-head"><div><h3>Power to methane</h3><p>Plant clock {frame.state.clock}</p></div><div className="v2__view-switch" aria-label="Plant presentation"><button className="pill" aria-pressed={view === 'diagram'} onClick={() => setView('diagram')}>Process diagram</button><button className="pill" aria-pressed={view === '3d'} onClick={() => setView('3d')}>Explore in 3D</button></div></div>
          <div className="v2__plant-stage">{view === 'diagram' ? diagram : <SceneBoundary fallback={diagram}><Suspense fallback={<div className="v2__scene-loading">Loading the 3D view…</div>}><PlantScene state={frame.state} config={snapshot.config} playing={false} /></Suspense></SceneBoundary>}</div>
          <div className="v2__plant-readouts"><span>Solar <strong>{number(frame.state.solar_available_kw, 0)} kW</strong></span><span>Battery <strong>{number(frame.state.battery_soc * 100, 1)}%</strong></span><span>Methane stored <strong>{number(frame.state.ch4_level_kg)} kg</strong></span></div>
          <p className="v2__plant-caption">{view === '3d' ? 'Drag to orbit. Scroll to zoom. Select equipment to inspect the current snapshot.' : 'Current physical state. Flow marks are stationary while the simulation is paused.'}</p>
        </div>
        <aside className="v2__obligation">
          <div className="v2__delivery-head"><span>Next methane delivery</span><span className="v2__annotation">{hoursToDelivery === null ? 'All due times passed' : `In ${elapsed(hoursToDelivery)}`}</span></div>
          <div className="v2__delivery-amount">{hoursToDelivery === null ? 'No pending' : <>{number(delivery.next_requested_kg)}<span>kg CH₄</span></>}</div>
          <p>{delivery.next_due_hours === null ? 'See the settled delivery record below.' : `Due at ${elapsed(delivery.next_due_hours)} elapsed. ${number(frame.state.ch4_level_kg)} kg is currently stored.`}</p>
          <div className="v2__return-head"><h4>Daily resource returns</h4><span>Due in {elapsed(clock.hours_to_cycle_end)}</span></div>
          <p className="v2__return-note">Current / minimum, then the signed return margin.</p>
          <dl className="v2__resources">{frame.resources.map((resource) => <div key={resource.id}><dt>{resource.label}<small>{number(resource.current)} / {number(resource.required)} {resource.unit}</small></dt><dd><strong className={`mono ${resource.margin < 0 ? 'v2__restore' : 'v2__positive'}`}>{signed(resource.margin)} <small>{resource.unit}</small></strong><span>{resource.margin < 0 ? 'to restore by cycle end' : resource.margin > 0 ? 'above return minimum' : 'at return minimum'}</span></dd></div>)}</dl>
          <p className="v2__note">A margin below zero before settlement is a restoration requirement, not a failed cycle.</p>
        </aside>
      </section>

      <Forecasts frame={frame} />
      <Schedule snapshot={snapshot} />
      <Deliveries key={snapshot.run_id} snapshot={snapshot} />
      <div className="v2__audit-pair"><Economics frame={frame} /><Decision frame={frame} /></div>
      <section className="v2__history" aria-labelledby="v2-history"><div className="v2__section-head"><div><h3 id="v2-history">What happened in this run</h3><p>{snapshot.history.length} retained decision boundaries. These trends summarize observations; they do not replay past plant frames.</p></div></div><div className="v2__history-pair"><Plot compact title="Observed methane inventory" unit="kg CH₄ stored" xLabel="Elapsed hours" series={[{ label: 'Methane inventory', color: 'var(--ch4)', points: snapshot.history.map((point) => ({ x: point.elapsed_hours, y: point.ch4_kg })) }]} /><Plot compact title="Realized net value" unit="EUR net value" xLabel="Elapsed hours" series={[{ label: 'Net simulated value', color: 'var(--stored)', points: snapshot.history.map((point) => ({ x: point.elapsed_hours, y: point.net_value_eur })) }]} /></div></section>
      <Evidence snapshot={snapshot} />
    </div>
  )
}
