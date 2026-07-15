import type { ProcedureSummary } from '../lib/types'

interface Props {
  summary: ProcedureSummary
}

function Card({ label, value, unit, sub }: { label: string; value: string; unit?: string; sub?: string }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
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
        label="PA Systolic (median)"
        value={summary.pap_systolic_median_mmhg.toFixed(1)}
        unit="mmHg"
        sub={`IQR ${summary.pap_systolic_iqr[0].toFixed(1)}–${summary.pap_systolic_iqr[1].toFixed(1)}`}
      />
      <Card label="PA Diastolic (median)" value={summary.pap_diastolic_median_mmhg.toFixed(1)} unit="mmHg" />
      <Card label="PA Mean (median)" value={summary.pap_mean_median_mmhg.toFixed(1)} unit="mmHg" />
      <Card label="HRV SDNN" value={summary.hrv_sdnn_ms.toFixed(1)} unit="ms" />
      <Card label="HRV RMSSD" value={summary.hrv_rmssd_ms.toFixed(1)} unit="ms" />
      {summary.scg_ao_ac_interval_ms !== null && (
        <Card label="SCG AO-AC interval" value={summary.scg_ao_ac_interval_ms.toFixed(0)} unit="ms" />
      )}
      <Card
        label="Beats included"
        value={`${summary.n_beats_included}/${summary.n_beats_total}`}
        sub="after quality filtering"
      />
    </div>
  )
}
