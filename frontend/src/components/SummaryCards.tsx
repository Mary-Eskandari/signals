import { SUMMARY_FIELDS } from '../lib/glossary'
import type { ProcedureSummary } from '../lib/types'

interface Props {
  summary: ProcedureSummary
}

function Card({
  field,
  value,
  unit,
  sub,
}: {
  field: keyof typeof SUMMARY_FIELDS
  value: string
  unit?: string
  sub?: string
}) {
  const meta = SUMMARY_FIELDS[field]
  return (
    <div className="stat-card" title={meta.tooltip}>
      <div className="stat-label">{meta.label}</div>
      <div className="stat-value">
        {value}
        {unit && <span className="stat-unit">{unit}</span>}
      </div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  )
}

export function SummaryCards({ summary }: Props) {
  return (
    <div className="stat-grid">
      <Card
        field="pap_systolic"
        value={summary.pap_systolic_median_mmhg.toFixed(1)}
        unit="mmHg"
        sub={`IQR ${summary.pap_systolic_iqr[0].toFixed(1)}–${summary.pap_systolic_iqr[1].toFixed(1)}`}
      />
      <Card field="pap_diastolic" value={summary.pap_diastolic_median_mmhg.toFixed(1)} unit="mmHg" />
      <Card field="pap_mean" value={summary.pap_mean_median_mmhg.toFixed(1)} unit="mmHg" />
      <Card field="hrv_sdnn" value={summary.hrv_sdnn_ms.toFixed(1)} unit="ms" />
      <Card field="hrv_rmssd" value={summary.hrv_rmssd_ms.toFixed(1)} unit="ms" />
      {summary.scg_ao_ac_interval_ms !== null && (
        <Card field="scg_ao_ac_interval" value={summary.scg_ao_ac_interval_ms.toFixed(0)} unit="ms" />
      )}
      <Card
        field="beats_included"
        value={`${summary.n_beats_included}/${summary.n_beats_total}`}
        sub="after quality filtering"
      />
    </div>
  )
}
