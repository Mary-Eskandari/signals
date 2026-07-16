export interface FieldMeta {
  label: string
  tooltip: string
}

export const BEAT_FIELDS: Record<string, FieldMeta> = {
  onset_time_s: {
    label: 'Onset Time (s)',
    tooltip: 'Time within the recording when PA pressure begins its systolic upstroke — marks the start of this heartbeat.',
  },
  pap_systolic_mmhg: {
    label: 'Systolic (mmHg)',
    tooltip: 'Peak pulmonary artery pressure during ventricular contraction.',
  },
  pap_diastolic_mmhg: {
    label: 'Diastolic (mmHg)',
    tooltip: 'Lowest pulmonary artery pressure between heartbeats.',
  },
  pap_mean_mmhg: {
    label: 'Mean Pressure (mmHg)',
    tooltip: 'Time-averaged pulmonary artery pressure across the full cardiac cycle.',
  },
  pulse_pressure_mmhg: {
    label: 'Pulse Pressure (mmHg)',
    tooltip: 'Systolic minus diastolic pressure — reflects arterial stiffness and stroke volume.',
  },
  rr_interval_ms: {
    label: 'RR Interval (ms)',
    tooltip: 'Time between this beat and the next, measured from ECG R-peaks. Determines instantaneous heart rate.',
  },
  sqi_score: {
    label: 'Signal Quality',
    tooltip: 'Automated 0-1 quality score for this beat; low scores indicate noise or catheter artifact.',
  },
  quality_flag: {
    label: 'Quality Flag',
    tooltip: '"good" / "questionable" / "excluded", derived from the signal quality score and physiological plausibility checks.',
  },
  scg_ao_time_s: {
    label: 'Aortic Opening (s)',
    tooltip: 'Time of aortic valve opening, detected from the seismocardiogram — marks the start of systolic ejection.',
  },
  scg_ao_amplitude: {
    label: 'Aortic Opening Amp.',
    tooltip: 'Seismocardiogram signal amplitude at the moment of aortic valve opening (arbitrary units).',
  },
  scg_ac_time_s: {
    label: 'Aortic Closure (s)',
    tooltip: 'Time of aortic valve closure, detected from the seismocardiogram — marks the end of systolic ejection.',
  },
  scg_ac_amplitude: {
    label: 'Aortic Closure Amp.',
    tooltip: 'Seismocardiogram signal amplitude at the moment of aortic valve closure (arbitrary units).',
  },
}

export const DAILY_FIELDS: Record<string, FieldMeta> = {
  weight_kg: { label: 'Weight (kg)', tooltip: 'Daily home scale reading.' },
  systolic_bp_mmhg: { label: 'Systolic BP (mmHg)', tooltip: 'Systolic (peak) blood pressure from a home cuff reading.' },
  diastolic_bp_mmhg: { label: 'Diastolic BP (mmHg)', tooltip: 'Diastolic (resting) blood pressure from a home cuff reading.' },
  spo2_pct: { label: 'Blood Oxygen (%)', tooltip: 'Blood oxygen saturation (SpO₂) from a pulse oximeter; below ~90% is considered low.' },
  hr_bpm: { label: 'Heart Rate (bpm)', tooltip: 'Average daily heart rate from the wearable ECG patch.' },
  activity_score: {
    label: 'Activity Score',
    tooltip: 'Daily physical activity / energy-expenditure estimate from the wearable accelerometer (arbitrary units).',
  },
  flags: { label: 'Flags', tooltip: "Rule-based early-warning flags computed from this patient's trend." },
}

export const FLAG_EXPLANATIONS: Record<string, string> = {
  weight_gain_3d_gt_2kg:
    'Weight increased more than 2kg within a 3-day window — a classic early-warning pattern for heart-failure fluid retention/decompensation.',
  low_spo2: 'Blood oxygen saturation (SpO₂) fell below 90% — a hypoxemia threshold.',
}

export const FLAG_LABELS: Record<string, string> = {
  weight_gain_3d_gt_2kg: 'Rapid weight gain',
  low_spo2: 'Low blood oxygen',
}

export const SUMMARY_FIELDS: Record<string, FieldMeta> = {
  pap_systolic: {
    label: 'PA Systolic (median)',
    tooltip: 'Median peak pulmonary artery pressure across all good-quality beats in this window.',
  },
  pap_diastolic: {
    label: 'PA Diastolic (median)',
    tooltip: 'Median lowest pulmonary artery pressure (between beats) across all good-quality beats.',
  },
  pap_mean: {
    label: 'PA Mean (median)',
    tooltip: 'Median time-averaged pulmonary artery pressure per beat. Mean PA pressure >20mmHg is the clinical threshold for pulmonary hypertension.',
  },
  hrv_sdnn: {
    label: 'HRV SDNN',
    tooltip: 'Standard deviation of beat-to-beat (NN) intervals — reflects overall heart rate variability.',
  },
  hrv_rmssd: {
    label: 'HRV RMSSD',
    tooltip: 'Root mean square of successive differences between beats — reflects short-term, parasympathetically-driven heart rate variability.',
  },
  scg_ao_ac_interval: {
    label: 'SCG AO-AC Interval',
    tooltip: 'Time between aortic valve opening and closure, detected from the seismocardiogram — a heuristic proxy for left-ventricular ejection time.',
  },
  beats_included: {
    label: 'Beats Included',
    tooltip: 'Beats retained after quality filtering, out of the total detected in this window.',
  },
}
