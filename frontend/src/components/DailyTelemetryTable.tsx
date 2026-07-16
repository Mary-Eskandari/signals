import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { DAILY_FIELDS, FLAG_EXPLANATIONS, FLAG_LABELS } from '../lib/glossary'
import type { DailyTelemetry } from '../lib/types'

interface Props {
  patientId: string
}

const COLUMNS: (keyof typeof DAILY_FIELDS)[] = [
  'weight_kg',
  'systolic_bp_mmhg',
  'diastolic_bp_mmhg',
  'spo2_pct',
  'hr_bpm',
  'activity_score',
]

function fmt(n: number | null | undefined, digits = 1): string {
  return n === null || n === undefined || Number.isNaN(n) ? '—' : n.toFixed(digits)
}

function Th({ field }: { field: string }) {
  const meta = DAILY_FIELDS[field]
  return <th title={meta.tooltip}>{meta.label}</th>
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
            <th title="Calendar date of this reading (mapped from the dataset's study-relative day index).">Date</th>
            {COLUMNS.map((c) => (
              <Th key={c} field={c} />
            ))}
            <Th field="flags" />
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
                  <span key={f} className="quality-pill quality-questionable" title={FLAG_EXPLANATIONS[f] ?? f}>
                    {FLAG_LABELS[f] ?? f}
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
