import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import type { DailyTelemetry } from '../lib/types'

interface Props {
  patientId: string
}

function fmt(n: number | null | undefined, digits = 1): string {
  return n === null || n === undefined || Number.isNaN(n) ? '—' : n.toFixed(digits)
}

export function DailyTelemetryTable({ patientId }: Props) {
  const [rows, setRows] = useState<DailyTelemetry[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setRows(null)
    setError(null)
    api.getPatientDaily(patientId).then(setRows).catch((e) => setError(String(e)))
  }, [patientId])

  if (error) return <div className="panel-status panel-error">{error}</div>
  if (!rows) return <div className="panel-status">Loading daily telemetry…</div>

  return (
    <div className="table-scroll">
      <table className="feature-table">
        <thead>
          <tr>
            <th>Date</th>
            <th>Weight (kg)</th>
            <th>Systolic BP</th>
            <th>Diastolic BP</th>
            <th>SpO2 (%)</th>
            <th>HR (bpm)</th>
            <th>Activity</th>
            <th>Flags</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.date} className={r.flags.length > 0 ? 'flagged-row' : undefined}>
              <td>{r.date}</td>
              <td>{fmt(r.weight_kg)}</td>
              <td>{fmt(r.systolic_bp_mmhg, 0)}</td>
              <td>{fmt(r.diastolic_bp_mmhg, 0)}</td>
              <td>{fmt(r.spo2_pct, 0)}</td>
              <td>{fmt(r.hr_bpm, 0)}</td>
              <td>{fmt(r.activity_score, 2)}</td>
              <td>
                {r.flags.map((f) => (
                  <span key={f} className="quality-pill quality-questionable">
                    {f}
                  </span>
                ))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
