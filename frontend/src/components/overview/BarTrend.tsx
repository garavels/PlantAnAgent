/** Fine-line trend: one hairline column per sample, newest at the right.
 *
 * Deliberately monochrome and dependency-free -- older columns fade back so the
 * eye lands on the present without needing a second colour.
 */

const COLUMNS = 44
const H = 34

export function BarTrend({ values, max }: { values: number[]; max: number }) {
  const stride = Math.max(1, Math.floor(values.length / COLUMNS))
  const points = values.filter((_, index) => index % stride === 0).slice(-COLUMNS)
  const ceiling = Math.max(max, ...points) || 1
  const pitch = 100 / COLUMNS

  return (
    <svg className="trend" viewBox={`0 0 100 ${H}`} preserveAspectRatio="none" aria-hidden>
      {points.map((value, index) => {
        const height = Math.max(0.6, (Math.max(0, value) / ceiling) * H)
        return (
          <rect
            key={index}
            className="trend__bar"
            x={index * pitch}
            y={H - height}
            width={pitch * 0.42}
            height={height}
            opacity={0.3 + 0.7 * (index / Math.max(1, points.length - 1))}
          />
        )
      })}
    </svg>
  )
}
