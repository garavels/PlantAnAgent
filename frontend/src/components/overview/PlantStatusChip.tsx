/** The plant's operational status, as derived by the simulator.
 *
 * The frontend never decides this: it renders whatever `plant_status` the
 * telemetry carries.
 */

import type { PlantStatus } from '../../types/plant'

const CHIP: Record<PlantStatus, { className: string; text: string }> = {
  normal: { className: 'chip chip--live', text: 'Normal' },
  degraded: { className: 'chip chip--warn', text: 'Degraded' },
  critical: { className: 'chip chip--alarm', text: 'Critical' },
  offline: { className: 'chip chip--idle', text: 'Offline' },
}

export function PlantStatusChip({ status }: { status: PlantStatus }) {
  const chip = CHIP[status] ?? CHIP.normal
  return (
    <span className={chip.className} title="Plant status, derived from simulator conditions">
      <i className="chip__led" />
      {chip.text}
    </span>
  )
}
