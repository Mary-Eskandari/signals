"""Shared Pydantic contract: pipeline output schema, API response models, and LLM grounding payload."""

from datetime import date
from typing import Literal

from pydantic import BaseModel


class BeatFeatures(BaseModel):
    beat_id: str
    record_id: str
    patient_id: str
    onset_time_s: float
    pap_systolic_mmhg: float
    pap_diastolic_mmhg: float
    pap_mean_mmhg: float
    pulse_pressure_mmhg: float
    rr_interval_ms: float
    sqi_score: float
    quality_flag: Literal["good", "questionable", "excluded"]
    scg_ao_time_s: float | None = None
    scg_ao_amplitude: float | None = None
    scg_ac_time_s: float | None = None
    scg_ac_amplitude: float | None = None
    # Agreement between this beat's AO/AC timing and the R-peak-aligned ensemble-average
    # template (see scg_features.py's ensemble-averaging cross-check) — 1.0 = tight
    # agreement, 0.0 = no agreement, None = not computable. Independent of pap sqi_score.
    scg_detection_confidence: float | None = None
    # Dicrotic notch: second-derivative fiducial point on the diastolic runoff (see
    # beat_features._dicrotic_notch_idx for citation). None when no sufficiently
    # prominent notch was found (e.g. an overdamped catheter trace).
    dicrotic_notch_time_s: float | None = None
    dicrotic_notch_pressure_mmhg: float | None = None
    # Pulse-wave-analysis morphological features (see beat_features._upstroke_slope /
    # _beat_shape_stats for citations).
    upstroke_slope_mmhg_s: float | None = None
    beat_auc_mmhg_s: float | None = None
    beat_skewness: float | None = None
    beat_kurtosis: float | None = None


class ProcedureSummary(BaseModel):
    record_id: str
    patient_id: str
    n_beats_total: int
    n_beats_included: int
    pap_systolic_median_mmhg: float
    pap_systolic_iqr: tuple[float, float]
    pap_diastolic_median_mmhg: float
    pap_mean_median_mmhg: float
    hrv_sdnn_ms: float
    hrv_rmssd_ms: float
    # Frequency-domain and nonlinear HRV (nk.hrv_frequency / nk.hrv_nonlinear) need
    # longer/steadier RR-interval series than SDNN/RMSSD to be numerically stable —
    # None when the window was too short or NeuroKit2 returned a non-finite value.
    hrv_lf_hf_ratio: float | None = None
    hrv_sd1_ms: float | None = None
    hrv_sd2_ms: float | None = None
    hrv_sample_entropy: float | None = None
    scg_ao_amplitude_mean: float | None = None
    scg_ac_amplitude_mean: float | None = None
    scg_ao_ac_interval_ms: float | None = None


class DailyTelemetry(BaseModel):
    patient_id: str
    date: date
    weight_kg: float
    systolic_bp_mmhg: float
    diastolic_bp_mmhg: float
    spo2_pct: float
    hr_bpm: float
    activity_score: float | None = None
    symptom_score: float | None = None
    flags: list[str] = []


class PatientTrendSummary(BaseModel):
    patient_id: str
    window_start: date
    window_end: date
    weight_slope_kg_per_day: float
    bp_trend_flags: list[str] = []
    flagged_events: list[str] = []


class ReportSection(BaseModel):
    title: str
    findings: list[str]
    unavailable: list[str] = []


class ClinicalReport(BaseModel):
    summary: str
    pa_pressure_findings: ReportSection
    rhythm_hrv_findings: ReportSection
    scg_findings: ReportSection | None = None
    trend_findings: ReportSection | None = None
    flags: list[str] = []
    disclaimer: str = (
        "Demonstration only, generated from public de-identified research data "
        "(PhysioNet SCG-RHC / Chiron CHF telemonitoring). Not a clinical tool, "
        "not FDA-cleared, not for real patient use."
    )
