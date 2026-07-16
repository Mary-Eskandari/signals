import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { BEAT_FIELDS } from '../lib/glossary'
import type { BeatFeatures } from '../lib/types'

interface Props {
  recordId: string
}

const QUALITY_CLASS: Record<string, string> = {
  good: 'quality-good',
  questionable: 'quality-questionable',
  excluded: 'quality-excluded',
}

const COLUMNS: (keyof typeof BEAT_FIELDS)[] = [
  'onset_time_s',
  'pap_systolic_mmhg',
  'pap_diastolic_mmhg',
  'pap_mean_mmhg',
  'pulse_pressure_mmhg',
  'rr_interval_ms',
  'sqi_score',
  'quality_flag',
  'scg_ao_time_s',
  'scg_ao_amplitude',
  'scg_ac_time_s',
  'scg_ac_amplitude',
]

function fmt(n: number | null, digits = 1): string {
  return n === null || n === undefined ? '—' : n.toFixed(digits)
}

function Th({ field }: { field: string }) {
  const meta = BEAT_FIELDS[field]
  return <th title={meta.tooltip}>{meta.label}</th>
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
            <th title="Sequential beat index within this recording window.">Beat</th>
            {COLUMNS.map((c) => (
              <Th key={c} field={c} />
            ))}
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
              <td title={BEAT_FIELDS.quality_flag.tooltip}>
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
