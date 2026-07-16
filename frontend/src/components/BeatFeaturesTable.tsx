import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { BEAT_FIELDS, SOTA_BEAT_FIELDS } from '../lib/glossary'
import type { BeatFeatures } from '../lib/types'

interface Props {
  recordId: string
}

const QUALITY_CLASS: Record<string, string> = {
  good: 'quality-good',
  questionable: 'quality-questionable',
  excluded: 'quality-excluded',
}

function fmt(n: number | null | undefined, digits = 1): string {
  return n === null || n === undefined ? '—' : n.toFixed(digits)
}

// Each column pairs a BEAT_FIELDS key with how to render that value — adding a
// field here is the only step needed to surface it in the table, so the column
// list can't silently drift out of sync with what the pipeline actually computes.
const COLUMNS: { field: keyof typeof BEAT_FIELDS; render: (b: BeatFeatures) => React.ReactNode }[] = [
  { field: 'onset_time_s', render: (b) => fmt(b.onset_time_s, 2) },
  { field: 'pap_systolic_mmhg', render: (b) => fmt(b.pap_systolic_mmhg) },
  { field: 'pap_diastolic_mmhg', render: (b) => fmt(b.pap_diastolic_mmhg) },
  { field: 'pap_mean_mmhg', render: (b) => fmt(b.pap_mean_mmhg) },
  { field: 'pulse_pressure_mmhg', render: (b) => fmt(b.pulse_pressure_mmhg) },
  { field: 'rr_interval_ms', render: (b) => fmt(b.rr_interval_ms, 0) },
  { field: 'sqi_score', render: (b) => fmt(b.sqi_score, 2) },
  {
    field: 'quality_flag',
    render: (b) => (
      <span className={`quality-pill ${QUALITY_CLASS[b.quality_flag]}`} title={BEAT_FIELDS.quality_flag.tooltip}>
        {b.quality_flag}
      </span>
    ),
  },
  { field: 'scg_ao_time_s', render: (b) => fmt(b.scg_ao_time_s, 2) },
  { field: 'scg_ao_amplitude', render: (b) => fmt(b.scg_ao_amplitude) },
  { field: 'scg_ac_time_s', render: (b) => fmt(b.scg_ac_time_s, 2) },
  { field: 'scg_ac_amplitude', render: (b) => fmt(b.scg_ac_amplitude) },
  { field: 'scg_detection_confidence', render: (b) => fmt(b.scg_detection_confidence, 2) },
  { field: 'dicrotic_notch_time_s', render: (b) => fmt(b.dicrotic_notch_time_s, 2) },
  { field: 'dicrotic_notch_pressure_mmhg', render: (b) => fmt(b.dicrotic_notch_pressure_mmhg) },
  { field: 'upstroke_slope_mmhg_s', render: (b) => fmt(b.upstroke_slope_mmhg_s, 0) },
  { field: 'beat_auc_mmhg_s', render: (b) => fmt(b.beat_auc_mmhg_s, 2) },
  { field: 'beat_skewness', render: (b) => fmt(b.beat_skewness, 2) },
  { field: 'beat_kurtosis', render: (b) => fmt(b.beat_kurtosis, 2) },
]

function Th({ field }: { field: keyof typeof BEAT_FIELDS }) {
  const meta = BEAT_FIELDS[field]
  return (
    <th title={meta.tooltip}>
      {meta.label}
      {SOTA_BEAT_FIELDS.has(field) && <sup className="sota-marker" title="SOTA feature">SOTA</sup>}
    </th>
  )
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
              <Th key={c.field} field={c.field} />
            ))}
          </tr>
        </thead>
        <tbody>
          {beats.map((b) => (
            <tr key={b.beat_id}>
              <td>{b.beat_id.split('-beat')[1]}</td>
              {COLUMNS.map((c) => (
                <td key={c.field}>{c.render(b)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
