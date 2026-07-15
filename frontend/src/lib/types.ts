export interface RecordListItem {
  record_id: string
  patient_id: string
}

export interface PatientListItem {
  patient_id: string
  n_days: number
  day_start: number
  day_end: number
}

export interface BeatFeatures {
  beat_id: string
  record_id: string
  patient_id: string
  onset_time_s: number
  pap_systolic_mmhg: number
  pap_diastolic_mmhg: number
  pap_mean_mmhg: number
  pulse_pressure_mmhg: number
  rr_interval_ms: number
  sqi_score: number
  quality_flag: 'good' | 'questionable' | 'excluded'
  scg_ao_time_s: number | null
  scg_ao_amplitude: number | null
  scg_ac_time_s: number | null
  scg_ac_amplitude: number | null
}

export interface ProcedureSummary {
  record_id: string
  patient_id: string
  n_beats_total: number
  n_beats_included: number
  pap_systolic_median_mmhg: number
  pap_systolic_iqr: [number, number]
  pap_diastolic_median_mmhg: number
  pap_mean_median_mmhg: number
  hrv_sdnn_ms: number
  hrv_rmssd_ms: number
  scg_ao_amplitude_mean: number | null
  scg_ac_amplitude_mean: number | null
  scg_ao_ac_interval_ms: number | null
}

export interface WaveformResponse {
  record_id: string
  fs_effective_hz: number
  time_s: number[]
  channels: Record<string, number[]>
}

export interface ChamberEvents {
  RA: number
  RV: number
  PA: number
  PCW: number
}

export interface DailyTelemetry {
  patient_id: string
  date: string
  weight_kg: number
  systolic_bp_mmhg: number
  diastolic_bp_mmhg: number
  spo2_pct: number
  hr_bpm: number
  activity_score: number | null
  symptom_score: number | null
  flags: string[]
}

export interface PatientTrendSummary {
  patient_id: string
  window_start: string
  window_end: string
  weight_slope_kg_per_day: number
  bp_trend_flags: string[]
  flagged_events: string[]
}

export interface ReportSection {
  title: string
  findings: string[]
  unavailable: string[]
}

export interface ClinicalReport {
  summary: string
  pa_pressure_findings: ReportSection
  rhythm_hrv_findings: ReportSection
  scg_findings: ReportSection | null
  trend_findings: ReportSection | null
  flags: string[]
  disclaimer: string
}

export interface ModelsResponse {
  allowed_models: string[]
  default: string
}
