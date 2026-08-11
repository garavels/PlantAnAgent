import { BarTrend } from './BarTrend'
import './metric-card.css'

interface MetricCardProps {
  label: string
  value: string
  unit: string
  footer: string
  trend: number[]
  trendMax: number
  /** 0..1 -- shown as a percentage of the unit's own ceiling. */
  fill?: number
  index?: number
}

export function MetricCard({ label, value, unit, footer, trend, trendMax, fill, index = 0 }: MetricCardProps) {
  return (
    <article className="card metric" style={{ animationDelay: `${index * 60}ms` }}>
      <header className="metric__head">
        <span className="label">{label}</span>
        {fill !== undefined && <span className="metric__gauge mono">{Math.round(fill * 100)}%</span>}
      </header>

      <div className="metric__trend">
        <BarTrend values={trend} max={trendMax} />
      </div>

      <div className="metric__readout">
        <span className="metric__value mono">{value}</span>
        <span className="metric__unit">{unit}</span>
      </div>

      <footer className="metric__foot">{footer}</footer>
    </article>
  )
}
