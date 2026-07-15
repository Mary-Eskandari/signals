import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import type { BeatFeatures } from '../lib/types'

interface Props {
  recordId: string
}

const QUALITY_CLASS: Record<string, string> = {
  good: 'quality-good',
  questionable: 'quality-questionable',
  excluded: 'quality-excluded',
}

function fmt(n: number | null, digits = 1): string {
  return n === null || n === undefined ? '—' : n.toFixed(digits)
}

export function BeatFeaturesTable({ recordId }: Props) {
  const [beats, setBeats] = useState<BeatFeatures[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setBeats(null)
    setError(null)
    api.getRecordBeats(recordId).then(setBeats).catch((e) => setError(String(e)))
  }, [recordId])

  if (error) return <div className="panel-status panel-error">{error}</div>
  if (!beats) return <div className="panel-status">Loading beat features…</div>

  return (
    <div className="table-scroll">
      <table className="feature-table">
        <thead>
          <tr>
            <th>Beat</th>
            <th>Onset (s)</th>
            <th>Systolic</th>
            <th>Diastolic</th>
            <th>Mean</th>
            <th>Pulse P.</th>
            <th>RR (ms)</th>
            <th>SQI</th>
            <th>Quality</th>
            <th>AO time (s)</th>
            <th>AO amp</th>
            <th>AC time (s)</th>
            <th>AC amp</th>
          </tr>
        </thead>
        <tbody>
          {beats.map((b) => (
            <tr key={b.beat_id}>
              <td>{b.beat_id.split('-beat')[1]}</td>
              <td>{fmt(b.onset_time_s, 2)}</td>
              <td>{fmt(b.pap_systolic_mmhg)}</td>
              <td>{fmt(b.pap_diastolic_mmhg)}</td>
              <td>{fmt(b.pap_mean_mmhg)}</td>
              <td>{fmt(b.pulse_pressure_mmhg)}</td>
              <td>{fmt(b.rr_interval_ms, 0)}</td>
              <td>{fmt(b.sqi_score, 2)}</td>
              <td>
                <span className={`quality-pill ${QUALITY_CLASS[b.quality_flag]}`}>{b.quality_flag}</span>
              </td>
              <td>{fmt(b.scg_ao_time_s, 2)}</td>
              <td>{fmt(b.scg_ao_amplitude)}</td>
              <td>{fmt(b.scg_ac_time_s, 2)}</td>
              <td>{fmt(b.scg_ac_amplitude)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
